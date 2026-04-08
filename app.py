import os
import json
import logging
import tempfile
import warnings
import asyncio
from typing import Dict, Any, List
from contextlib import asynccontextmanager

warnings.filterwarnings("ignore", "Support for google-cloud-storage", category=FutureWarning)

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from tenacity import retry, wait_exponential, stop_after_attempt

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_google_firestore import FirestoreVectorStore
from google.cloud import firestore, storage
from google.cloud.firestore_v1.base_query import FieldFilter

from google import genai
from google.genai.types import EmbedContentConfig
from vertexai.generative_models import GenerativeModel, GenerationConfig
import vertexai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COLLECTION_NAME = "pida_knowledge_base_v2" 

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    fetch_k: int = 15

clients = {}

class ModernGeminiEmbeddings(Embeddings):
    def __init__(self, model_name="gemini-embedding-001", dimensionality=2048):
        self.model_name = model_name
        self.dimensionality = dimensionality
        self.client = genai.Client() 

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
    def _get_embeddings_with_retry(self, texts: List[str], task_type: str) -> List[List[float]]:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=EmbedContentConfig(task_type=task_type, output_dimensionality=self.dimensionality)
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
        return self._get_embeddings_with_retry([text], "RETRIEVAL_QUERY")[0]

@asynccontextmanager
async def lifespan(app: FastAPI):
    PROJECT_ID = os.environ.get("PROJECT_ID")
    VERTEX_AI_LOCATION = os.environ.get("VERTEX_AI_LOCATION", "us-central1")
    vertexai.init(project=PROJECT_ID, location=VERTEX_AI_LOCATION)
    
    clients['firestore'] = firestore.Client()
    clients['storage'] = storage.Client()
    clients['embedding'] = ModernGeminiEmbeddings()
    clients['metadata_model'] = GenerativeModel("gemini-2.5-flash")
    clients['vector_store'] = FirestoreVectorStore(
        collection=COLLECTION_NAME, 
        embedding_service=clients['embedding'], 
        client=clients['firestore']
    )
    logger.info("--- RAG v30 (2048d) CARGADO ---")
    yield
    clients.clear()

app = FastAPI(lifespan=lifespan)

def _process_and_embed_text_file(file_path: str, filename: str):
    db = clients.get('firestore')
    vector_store = clients.get('vector_store')
    meta_model = clients.get('metadata_model')
    
    logger.info(f"[DEBUG 1] Leyendo archivo: {filename}")
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        text_content = f.read()
    
    if not text_content.strip():
        logger.error("[ERROR] El archivo está vacío.")
        return

    # --- NUEVO: LIMPIEZA PREVIA ---
    logger.info(f"[DEBUG 2] Limpiando rastro anterior de {filename} en {COLLECTION_NAME}...")
    docs_ref = db.collection(COLLECTION_NAME)
    docs_to_delete = docs_ref.where(filter=FieldFilter("metadata.source", "==", filename)).stream()
    deleted_count = 0
    async def delete_docs(): # Helper para limpiar
        nonlocal deleted_count
        async for doc in docs_to_delete:
            await doc.reference.delete()
            deleted_count += 1
    # Nota: Firestore SDK en este wrapper es sincrónico, usamos loop directo
    for doc in db.collection(COLLECTION_NAME).where(filter=FieldFilter("metadata.source", "==", filename)).stream():
        doc.reference.delete()
        deleted_count += 1
    logger.info(f"[DEBUG 2.1] Se borraron {deleted_count} fragmentos antiguos.")

    logger.info("[DEBUG 3] Extrayendo metadatos con Gemini...")
    doc_title, doc_author = filename, "Autor Desconocido"
    try:
        meta_response = meta_model.generate_content(
            f"Extrae Título y Autor: {text_content[:2000]}",
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                response_schema={"type":"OBJECT","properties":{"title":{"type":"STRING"},"author":{"type":"STRING"}},"required":["title","author"]},
                temperature=0.1
            )
        )
        meta = json.loads(meta_response.text)
        doc_title = meta.get("title", filename)
        doc_author = meta.get("author", "Autor Desconocido")
        logger.info(f"[DEBUG 3.1] Título: {doc_title} | Autor: {doc_author}")
    except Exception as e:
        logger.warning(f"[ALERTA] Fallo metadatos IA: {e}")

    logger.info("[DEBUG 4] Iniciando Chunking (División de texto)...")
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","H1"),("##","H2"),("###","H3")])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)
    
    md_docs = md_splitter.split_text(text_content)
    chunks = text_splitter.split_documents(md_docs)
    logger.info(f"[DEBUG 4.1] Texto dividido en {len(chunks)} fragmentos.")
    
    documents = []
    for i, chunk in enumerate(chunks):
        m = chunk.metadata.copy()
        m.update({"source": filename, "title": doc_title, "author": doc_author, "index": i})
        documents.append(Document(page_content=chunk.page_content, metadata=m))
    
    logger.info(f"[DEBUG 5] Iniciando guardado en Firestore (Dimensiones: 2048)...")
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        vector_store.add_documents(batch)
        logger.info(f"[PROGRESS] Guardado lote {i//batch_size + 1}")
    
    logger.info(f"--- ¡ÉXITO TOTAL! {filename} indexado correctamente en {COLLECTION_NAME} ---")

@app.post("/")
async def handle_gcs_event(request: Request):
    event = await request.json()
    bucket_name, file_id = event.get("bucket"), event.get("name")
    if not bucket_name or not file_id: return {"status": "ignored"}
    if not (file_id.endswith(".md") or file_id.endswith(".txt")):
        logger.info(f"Ignorado por extensión: {file_id}")
        return {"status": "ignored"}

    logger.info(f"--- EVENTO RECIBIDO: {file_id} ---")
    storage_client = clients.get('storage')
    blob = storage_client.bucket(bucket_name).blob(file_id)
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        blob.download_to_filename(tmp.name)
        tmp_name = tmp.name
    
    try:
        # Ejecución síncrona real para bloquear el hilo y mantener el CPU
        await asyncio.to_thread(_process_and_embed_text_file, tmp_name, file_id)
    except Exception as e:
        logger.error(f"!!! ERROR CRÍTICO EN PROCESO !!!: {e}", exc_info=True)
        return {"status": "error", "detail": str(e)}
    finally:
        if os.path.exists(tmp_name): os.unlink(tmp_name)
        
    return {"status": "success"}

@app.post("/query")
async def query_rag_handler(request_data: QueryRequest):
    vector_store = clients.get('vector_store')
    found_docs = vector_store.max_marginal_relevance_search(
        query=request_data.query, k=request_data.top_k, fetch_k=request_data.fetch_k, lambda_mult=0.5
    )
    return {"results": [{"source": d.metadata.get("source"), "content": d.page_content, "title": d.metadata.get("title"), "author": d.metadata.get("author")} for d in found_docs]}
