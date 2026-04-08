import os
import json
import logging
import tempfile
import warnings
import asyncio
import re
from typing import Dict, Any, List
from contextlib import asynccontextmanager

# --- 1. CONFIGURACIÓN Y CLIENTES GLOBALES ---
clients = {}
COLLECTION_NAME = "pida_knowledge_base_v2"

# Silenciar advertencias innecesarias de las librerías de Google
warnings.filterwarnings("ignore", "Support for google-cloud-storage", category=FutureWarning)

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from tenacity import retry, wait_exponential, stop_after_attempt

# LangChain y Google Cloud
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_google_firestore import FirestoreVectorStore
from google.cloud import firestore, storage
from google.cloud.firestore_v1.base_query import FieldFilter

# SDK google-genai y VertexAI
from google import genai
from google.genai.types import EmbedContentConfig
from vertexai.generative_models import GenerativeModel, GenerationConfig
import vertexai

# Configuración de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. MODELOS DE DATOS (PYDANTIC) ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    fetch_k: int = 15

# --- 3. CLASE DE EMBEDDINGS (2048d) ---
class ModernGeminiEmbeddings(Embeddings):
    """
    Genera embeddings de 2048 dimensiones usando el modelo moderno de Gemini.
    Firestore requiere exactamente 2048 o menos.
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
            logger.info(f"Lote de embeddings generado ({len(embeddings)} procesados)")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        results = self._get_embeddings_with_retry([text], "RETRIEVAL_QUERY")
        return results[0]

# --- 4. FUNCIONES DE PROCESAMIENTO Y LIMPIEZA ---

def _extract_clean_metadata(raw_text: str, filename: str):
    """
    Extrae Título y Autor asumiendo un Markdown limpio (1 solo H1 y autor debajo),
    pero utilizando IA para garantizar precisión y manejar variaciones.
    """
    model = clients['metadata_model']
    try:
        prompt = (
            "Lee el inicio de este libro jurídico en Markdown. Extrae el Título Principal y el Autor.\n"
            "REGLAS:\n"
            "1. El título suele ser el único H1 (#).\n"
            "2. El autor suele estar en negritas inmediatamente debajo del título.\n"
            "3. Ignora textos como 'Prólogo', 'A quienes piensan...', etc.\n"
            f"TEXTO A ANALIZAR:\n{raw_text[:5000]}\n\n"
            "Responde estrictamente con este formato JSON:\n"
            "{'title': 'Título exacto', 'author': 'Nombre del Autor'}"
        )
        res = model.generate_content(
            prompt, 
            generation_config=GenerationConfig(response_mime_type="application/json", temperature=0)
        )
        meta = json.loads(res.text)
        
        title = meta.get("title", filename.replace(".md", ""))
        author = meta.get("author", "Autor Desconocido")
        
        # Salvavidas de seguridad final
        if not author or "Desconocido" in author or "Author Name" in author: 
            author = "Fabián Salvioli"
        if not title or len(title) < 5 or "Exact Book Title" in title: 
            title = filename.replace(".md", "").replace("_", " ").title()
            
        return title.strip(), author.strip()
    except Exception as e:
        logger.error(f"Error IA Metadatos: {e}")
        return filename.replace(".md", ""), "Fabián Salvioli"

def _clean_content_for_rag(text: str) -> str:
    """
    Red de seguridad: Elimina basura visual (HTML) y normaliza espacios.
    Aunque LlamaParse mejore, esto asegura que el RAG nunca ingeste basura.
    """
    text = re.sub(r'<sup>.*?</sup>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def _process_workflow(file_path: str, filename: str):
    """Flujo de ingesta lineal, robusto y preparado para producción."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        raw_content = f.read()

    # 1. Extracción de Metadatos sobre el texto crudo
    title, author = _extract_clean_metadata(raw_content, filename)
    logger.info(f"--- INGESTANDO: {title} | AUTOR: {author} ---")

    # 2. Limpieza de seguridad para los vectores
    clean_content = _clean_content_for_rag(raw_content)

    # 3. Chunking Jerárquico Natural
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","H1"),("##","H2"),("###","H3")])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)
    
    chunks = text_splitter.split_documents(md_splitter.split_text(clean_content))
    
    docs_to_db = []
    for i, chunk in enumerate(chunks):
        # Filtro de calidad: omitir fragmentos inútiles o índices
        if len(chunk.page_content) < 150: continue
        if "ÍNDICE" in chunk.page_content.upper() and len(chunk.page_content) < 1000: continue

        chunk.metadata.update({
            "source": filename,
            "title": title,
            "author": author,
            "chunk_index": i
        })
        docs_to_db.append(Document(page_content=chunk.page_content, metadata=chunk.metadata))
    
    # 4. Guardado en Firestore
    if docs_to_db:
        vector_store = clients['vector_store']
        batch_size = 50
        for i in range(0, len(docs_to_db), batch_size):
            vector_store.add_documents(docs_to_db[i:i + batch_size])
        logger.info(f"✅ Éxito: {len(docs_to_db)} fragmentos guardados para '{title}'.")

# --- 5. LIFESPAN E INICIALIZACIÓN ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    PROJECT_ID = os.environ.get("PROJECT_ID")
    LOCATION = os.environ.get("VERTEX_AI_LOCATION", "us-central1")
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    clients['firestore'] = firestore.Client()
    clients['storage'] = storage.Client()
    clients['embedding'] = ModernGeminiEmbeddings()
    clients['metadata_model'] = GenerativeModel("gemini-2.0-flash")
    clients['vector_store'] = FirestoreVectorStore(
        collection=COLLECTION_NAME, 
        embedding_service=clients['embedding'], 
        client=clients['firestore']
    )
    logger.info("--- Backend RAG v30 (Producción Completa) Inicializado ---")
    yield
    clients.clear()

app = FastAPI(lifespan=lifespan)

# --- 6. ENDPOINTS ---

@app.post("/")
async def handle_gcs_event(request: Request):
    """Webhook de Eventarc para procesar archivos subidos al Storage."""
    event = await request.json()
    file_id = event.get("name")
    bucket_name = event.get("bucket")
    
    if not file_id or not bucket_name:
        return {"status": "ignored", "reason": "Faltan datos en el evento"}
        
    if not file_id.lower().endswith(".md"):
        logger.info(f"Ignorado por no ser Markdown: {file_id}")
        return {"status": "ignored"}

    logger.info(f"Iniciando descarga y proceso: {file_id}")
    blob = clients['storage'].bucket(bucket_name).blob(file_id)
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        blob.download_to_filename(tmp.name)
        # Hilo separado síncrono para mantener CPU en Cloud Run
        await asyncio.to_thread(_process_workflow, tmp.name, file_id)
        os.unlink(tmp.name)
        
    return {"status": "success"}

@app.post("/query")
async def handle_query(request_data: QueryRequest):
    """
    Endpoint para el Chat Frontend. Devuelve los fragmentos estructurados
    con los metadatos precisos para generar las citas bibliográficas.
    """
    vector_store = clients['vector_store']
    
    try:
        docs = vector_store.max_marginal_relevance_search(
            query=request_data.query, 
            k=request_data.top_k,
            fetch_k=request_data.fetch_k
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
    except Exception as e:
        logger.error(f"Error en la búsqueda vectorial: {e}")
        raise HTTPException(status_code=500, detail="Error procesando la búsqueda en la base de conocimientos")
