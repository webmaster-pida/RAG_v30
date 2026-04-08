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

# --- 2. CLASES DE SOPORTE ---

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    fetch_k: int = 15

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

# --- 3. FUNCIONES DE LIMPIEZA Y PROCESAMIENTO ---

def _deep_clean(text: str) -> str:
    """Borra automáticamente la basura de LlamaParse y etiquetas HTML."""
    # Eliminar placeholders inyectados por error de LlamaParse
    text = re.sub(r'# Título:.*\[Exact Book Title\].*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\*\*Autor:\*\*.*\[Author Name\].*', '', text, flags=re.IGNORECASE)
    # Eliminar etiquetas <sup>...</sup> (números de notas al pie en HTML)
    text = re.sub(r'<sup>.*?</sup>', '', text)
    # Eliminar cualquier otra etiqueta HTML residual
    text = re.sub(r'<[^>]+>', '', text)
    # Normalizar espacios y saltos de línea excesivos
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def _extract_real_metadata(text_content: str, filename: str):
    """Extrae metadatos reales usando Gemini 2.0 Flash."""
    model = clients['metadata_model']
    try:
        # Analizamos el inicio del texto para identificar título y autor reales
        prompt = (
            "Analiza el siguiente texto de un libro. Extrae el Título Real y el Autor Real. "
            "Ignora cualquier texto genérico o placeholders. Responde estrictamente en JSON.\n\n"
            f"Texto:\n{text_content[:5000]}"
        )
        res = model.generate_content(
            prompt, 
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        meta = json.loads(res.text)
        title = meta.get("title", filename)
        author = meta.get("author", "Autor Desconocido")
        return title, author
    except Exception as e:
        logger.warning(f"Fallo al extraer metadatos con IA: {e}")
        return filename, "Autor Desconocido"

def _process_and_embed_text_file(file_path: str, filename: str):
    """Proceso principal de transformación de archivo a vectores."""
    vector_store = clients['vector_store']
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        raw_content = f.read()
    
    # 1. Limpieza automática
    text_content = _deep_clean(raw_content)
    if not text_content:
        logger.error(f"El archivo {filename} quedó vacío tras la limpieza.")
        return

    # 2. Extracción de metadatos
    doc_title, doc_author = _extract_real_metadata(text_content, filename)
    logger.info(f"Metadatos: Título='{doc_title}', Autor='{doc_author}'")

    # 3. Chunking Jerárquico
    # Primero dividimos por encabezados Markdown (#, ##, ###)
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")]
    )
    # Luego dividimos en fragmentos de 2000 caracteres para asegurar consistencia
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=250
    )
    
    md_header_docs = md_splitter.split_text(text_content)
    chunks = text_splitter.split_documents(md_header_docs)
    
    documents_to_upload = []
    for i, chunk in enumerate(chunks):
        # Filtro automático: Si el fragmento es solo el Índice o muy corto, lo saltamos
        content_upper = chunk.page_content.upper()
        if "ÍNDICE" in content_upper and len(chunk.page_content) < 600:
            continue
            
        metadata = chunk.metadata.copy()
        metadata.update({
            "source": filename,
            "title": doc_title,
            "author": doc_author,
            "chunk_index": i
        })
        documents_to_upload.append(Document(page_content=chunk.page_content, metadata=metadata))
    
    # 4. Inserción en Firestore (Batch de 50 para evitar límites de API)
    logger.info(f"Subiendo {len(documents_to_upload)} fragmentos a Firestore...")
    batch_size = 50
    for i in range(0, len(documents_to_upload), batch_size):
        batch = documents_to_upload[i:i + batch_size]
        vector_store.add_documents(batch)
        logger.info(f"Progreso {filename}: Lote {i//batch_size + 1} completado.")

    logger.info(f"--- ÉXITO: {filename} indexado correctamente ---")

# --- 4. LIFESPAN Y APLICACIÓN ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicialización al arrancar el contenedor
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
    logger.info("--- Microservicio RAG v30 (Automatización Total) Inicializado ---")
    yield
    clients.clear()

app = FastAPI(lifespan=lifespan)

# --- 5. ENDPOINTS ---

@app.post("/")
async def handle_gcs_event(request: Request):
    """Recibe notificaciones de Eventarc cuando subes un archivo al Bucket."""
    event = await request.json()
    bucket_name = event.get("bucket")
    file_id = event.get("name")
    
    if not bucket_name or not file_id:
        return {"status": "ignored", "reason": "No bucket/name"}
        
    if not (file_id.lower().endswith(".md") or file_id.lower().endswith(".txt")):
        logger.info(f"Ignorando archivo por extensión: {file_id}")
        return {"status": "ignored", "reason": "Extension not supported"}

    logger.info(f"Evento recibido: {file_id} en bucket {bucket_name}")
    
    storage_client = clients.get('storage')
    blob = storage_client.bucket(bucket_name).blob(file_id)
    
    if not blob.exists():
        return {"status": "error", "reason": "File not found"}

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        blob.download_to_filename(tmp.name)
        # Ejecución síncrona en hilo separado para mantener vivo el CPU de Cloud Run
        await asyncio.to_thread(_process_and_embed_text_file, tmp.name, file_id)
        os.unlink(tmp.name)
        
    return {"status": "success", "file": file_id}

@app.post("/query")
async def query_rag_handler(request_data: QueryRequest):
    """Endpoint para que el Chat haga las consultas vectoriales."""
    vector_store = clients.get('vector_store')
    
    # max_marginal_relevance_search para evitar resultados redundantes
    found_docs = vector_store.max_marginal_relevance_search(
        query=request_data.query,
        k=request_data.top_k,
        fetch_k=request_data.fetch_k,
        lambda_mult=0.5
    )
    
    results = []
    for d in found_docs:
        results.append({
            "content": d.page_content,
            "source": d.metadata.get("source"),
            "title": d.metadata.get("title"),
            "author": d.metadata.get("author")
        })
        
    return {"results": results}
