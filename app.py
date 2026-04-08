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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mantenemos v2 para no mezclar con la v1 antigua si ya hiciste pruebas, 
# pero ahora con la dimensión correcta (2048)
COLLECTION_NAME = "pida_knowledge_base_v2" 

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    fetch_k: int = 15

clients = {}

# --- CLASE DE EMBEDDINGS CORREGIDA (2048 DIMENSIONES) ---
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
            config=EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.dimensionality, # <--- FORZADO A 2048 PARA FIRESTORE
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
            logger.info(f"Embeddings (2048d) generados: lote {i//batch_size + 1}")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        results = self._get_embeddings_with_retry([text], "RETRIEVAL_QUERY")
        return results[0]

@asynccontextmanager
async def lifespan(app: FastAPI):
    PROJECT_ID = os.environ.get("PROJECT_ID")
    VERTEX_AI_LOCATION = os.environ.get("VERTEX_AI_LOCATION", "us-central1")
    vertexai.init(project=PROJECT_ID, location=VERTEX_AI_LOCATION)
    
    clients['firestore'] = firestore.Client()
    clients['storage'] = storage.Client()
    clients['embedding'] = ModernGeminiEmbeddings(dimensionality=2048)
    clients['metadata_model'] = GenerativeModel("gemini-2.5-flash")
    clients['vector_store'] = FirestoreVectorStore(
        collection=COLLECTION_NAME, 
        embedding_service=clients['embedding'], 
        client=clients['firestore']
    )
    logger.info("--- Microservicio RAG v30 (2048d) Inicializado ---")
    yield
    clients.clear()

app = FastAPI(lifespan=lifespan)

def _process_and_embed_text_file(file_path: str, filename: str):
    db = clients.get('firestore')
    vector_store = clients.get('vector_store')
    meta_model = clients.get('metadata_model')
    
    # 1. Metadatos inteligentes
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        text_content = f.read()
    
    doc_title, doc_author = filename, "Autor Desconocido"
    try:
        meta_response = meta_model.generate_content(
            f"Extrae Título y Autor de este texto: {text_content[:3000]}",
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                response_schema={"type":"OBJECT","properties":{"title":{"type":"STRING"},"author":{"type":"STRING"}},"required":["title","author"]},
                temperature=0.1
            )
        )
        meta = json.loads(meta_response.text)
        doc_title, doc_author = meta.get("title", filename), meta.get("author", "Autor Desconocido")
    except Exception: pass

    # 2. Chunking
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","H1"),("##","H2"),("###","H3")])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)
    chunks = text_splitter.split_documents(md_splitter.split_text(text_content))
    
    documents = []
    for i, chunk in enumerate(chunks):
        m = chunk.metadata.copy()
        m.update({"source": filename, "title": doc_title, "author": doc_author, "index": i})
        documents.append(Document(page_content=chunk.page_content, metadata=m))
    
    # 3. Guardado (Firestore limitará esto a 2048d)
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        vector_store.add_documents(documents[i:i + batch_size])
    logger.info(f"Éxito: {filename} indexado con 2048 dimensiones.")

@app.post("/")
async def handle_gcs_event(request: Request):
    event = await request.json()
    bucket_name, file_id = event.get("bucket"), event.get("name")
    if not bucket_name or not file_id or not (file_id.endswith(".md") or file_id.endswith(".txt")):
        return {"status": "ignored"}

    storage_client = clients.get('storage')
    blob = storage_client.bucket(bucket_name).blob(file_id)
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        blob.download_to_filename(tmp.name)
        # Síncrono para mantener vivo el CPU de Cloud Run
        await asyncio.to_thread(_process_and_embed_text_file, tmp.name, file_id)
        os.unlink(tmp.name)
        
    return {"status": "ok"}

@app.post("/query")
async def query_rag_handler(request_data: QueryRequest):
    vector_store = clients.get('vector_store')
    found_docs = vector_store.max_marginal_relevance_search(
        query=request_data.query, k=request_data.top_k, fetch_k=request_data.fetch_k, lambda_mult=0.5
    )
    return {"results": [{"source": d.metadata.get("source"), "content": d.page_content, "title": d.metadata.get("title"), "author": d.metadata.get("author")} for d in found_docs]}
