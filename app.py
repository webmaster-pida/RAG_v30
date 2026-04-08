import os
import json
import logging
import tempfile
import warnings
import asyncio
import re
from typing import Dict, Any, List
from contextlib import asynccontextmanager

# --- 1. CONFIGURACIÓN GLOBAL Y CLIENTES ---
# Diccionario global para mantener las conexiones a GCP activas
clients = {}
COLLECTION_NAME = "pida_knowledge_base_v2"

# Silenciamos advertencias de versiones de las librerías de Google
warnings.filterwarnings("ignore")

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from tenacity import retry, wait_exponential, stop_after_attempt

# Librerías de LangChain y Google Cloud
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_google_firestore import FirestoreVectorStore
from google.cloud import firestore, storage
from google.cloud.firestore_v1.base_query import FieldFilter

# SDK de Google GenAI y VertexAI
from google import genai
from google.genai.types import EmbedContentConfig
from vertexai.generative_models import GenerativeModel, GenerationConfig
import vertexai

# Configuración de Logging para ver el progreso en Cloud Run
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. CLASES DE SOPORTE ---

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    fetch_k: int = 15

class ModernGeminiEmbeddings(Embeddings):
    """
    Genera vectores de 2048 dimensiones. 
    Firestore NO acepta 3072, por lo que aquí forzamos la compatibilidad.
    """
    def __init__(self, model_name="gemini-embedding-001", dimensionality=2048):
        self.model_name = model_name
        self.dimensionality = dimensionality
        self.client = genai.Client()

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
    def _get_embeddings_with_retry(self, texts: List[str], task_type: str) -> List[List[float]]:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.dimensionality
            )
        )
        return [emb.values for emb in response.embeddings]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results = self._get_embeddings_with_retry(batch, "RETRIEVAL_DOCUMENT")
            embeddings.extend(results)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        results = self._get_embeddings_with_retry([text], "RETRIEVAL_QUERY")
        return results[0]

# --- 3. LÓGICA DE EXTRACCIÓN Y LIMPIEZA AUTOMÁTICA ---

def _extract_metadata_robust(raw_text: str, filename: str):
    """
    Identifica título y autor sin intervención humana.
    Prioriza IA, luego patrones de texto y finalmente el nombre del archivo.
    """
    # Intentar con IA primero (Analizando los primeros 8000 caracteres)
    model = clients['metadata_model']
    t_ai, a_ai = None, None
    try:
        prompt = (
            "Analiza el inicio de este documento legal. Extrae el Título y el Autor Reales. "
            "Ignora placeholders como '[Author Name]' o '[Exact Book Title]'. "
            "Responde solo con el JSON: {'title': '...', 'author': '...'}\n\n"
            f"Texto:\n{raw_text[:8000]}"
        )
        res = model.generate_content(prompt, generation_config=GenerationConfig(response_mime_type="application/json", temperature=0))
        meta = json.loads(res.text)
        t_ai = meta.get("title")
        a_ai = meta.get("author")
    except:
        pass

    # Si la IA falla, buscar por patrones de texto (Regex)
    t_reg = re.search(r'Título:\s*([^\[\n\r]+)', raw_text, re.I)
    a_reg = re.search(r'Autor:\s*([^\[\n\r]+)', raw_text, re.I)
    
    final_title = t_ai or (t_reg.group(1).strip() if t_reg else filename.replace(".md", ""))
    final_author = a_ai or (a_reg.group(1).strip() if a_reg else "Fabián Salvioli")

    # Limpiar posibles corchetes residuales de LlamaParse
    final_title = re.sub(r'\[.*?\]', '', str(final_title)).strip()
    final_author = re.sub(r'\[.*?\]', '', str(final_author)).strip()
    
    return final_title, final_author

def _deep_clean_text(text: str) -> str:
    """Elimina basura visual y etiquetas HTML del contenido de los fragmentos."""
    # Eliminar etiquetas <sup> (notas al pie) y HTML
    text = re.sub(r'<sup>.*?</sup>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    # Eliminar las líneas de Título/Autor inyectadas por LlamaParse para no duplicar
    text = re.sub(r'# Título:.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Autor:.*', '', text, flags=re.IGNORECASE)
    # Limpiar espacios
    text = re.sub(r' +', ' ', text)
    return text.strip()

def _full_process_workflow(file_path: str, filename: str):
    """Workflow de automatización total de PDF/MD a Firestore."""
    vector_store = clients['vector_store']
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        raw_content = f.read()

    # 1. Metadatos robustos
    title, author = _extract_metadata_robust(raw_content, filename)
    logger.info(f"Procesando: {title} | Autor: {author}")

    # 2. Limpieza del contenido
    clean_content = _deep_clean_text(raw_content)

    # 3. Chunking Jerárquico (Headers Markdown + Tamaño)
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","H1"),("##","H2"),("###","H3")])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)
    
    header_docs = md_splitter.split_text(clean_content)
    chunks = text_splitter.split_documents(header_docs)
    
    docs_to_db = []
    for i, chunk in enumerate(chunks):
        # Filtro: Omitir fragmentos de índice o muy cortos
        if len(chunk.page_content) < 150: continue
        if "ÍNDICE" in chunk.page_content.upper() and len(chunk.page_content) < 800: continue

        chunk.metadata.update({
            "source": filename,
            "title": title,
            "author": author,
            "chunk_index": i
        })
        docs_to_db.append(Document(page_content=chunk.page_content, metadata=chunk.metadata))
    
    # 4. Guardado en lotes (Batch)
    if docs_to_db:
        batch_size = 50
        for i in range(0, len(docs_to_db), batch_size):
            vector_store.add_documents(docs_to_db[i:i + batch_size])
        logger.info(f"✅ Libro indexado: {title} ({len(docs_to_db)} fragmentos)")

# --- 4. LIFESPAN E INICIALIZACIÓN ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    PROJECT_ID = os.environ.get("PROJECT_ID")
    vertexai.init(project=PROJECT_ID, location="us-central1")
    
    clients['firestore'] = firestore.Client()
    clients['storage'] = storage.Client()
    clients['embedding'] = ModernGeminiEmbeddings()
    clients['metadata_model'] = GenerativeModel("gemini-1.5-flash")
    clients['vector_store'] = FirestoreVectorStore(
        collection=COLLECTION_NAME,
        embedding_service=clients['embedding'],
        client=clients['firestore']
    )
    logger.info("--- RAG v30: Sistema de Automatización Inicializado ---")
    yield
    clients.clear()

app = FastAPI(lifespan=lifespan)

# --- 5. ENDPOINTS DE COMUNICACIÓN ---

@app.post("/")
async def handle_gcs_notification(request: Request):
    """Endpoint que recibe el archivo desde Google Cloud Storage."""
    event = await request.json()
    file_id = event.get("name")
    bucket_name = event.get("bucket")
    
    if not file_id or not file_id.lower().endswith(".md"):
        return {"status": "ignored"}

    logger.info(f"Archivo detectado en bucket: {file_id}")
    blob = clients['storage'].bucket(bucket_name).blob(file_id)
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        blob.download_to_filename(tmp.name)
        # Síncrono para Cloud Run (Mantiene CPU al 100%)
        await asyncio.to_thread(_full_process_workflow, tmp.name, file_id)
        os.unlink(tmp.name)
        
    return {"status": "success"}

@app.post("/query")
async def handle_query(request: QueryRequest):
    """Endpoint para buscar información en la base de conocimientos."""
    vector_store = clients['vector_store']
    docs = vector_store.max_marginal_relevance_search(
        query=request.query, 
        k=request.top_k, 
        fetch_k=request.fetch_k
    )
    return {
        "results": [
            {
                "content": d.page_content, 
                "title": d.metadata.get("title"), 
                "author": d.metadata.get("author"),
                "source": d.metadata.get("source")
            } for d in docs
        ]
    }
