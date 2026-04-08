import os
import json
import logging
import tempfile
import warnings
import asyncio
import re  # <--- Librería para limpieza de texto
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
    logger.info("--- RAG v30: LIMPIEZA ACTIVA (2048d) ---")
    yield
    clients.clear()

app = FastAPI(lifespan=lifespan)

def _clean_text(text: str) -> str:
    """Elimina etiquetas HTML y ruido de formato."""
    # 1. Eliminar etiquetas <sup>...</sup> y su contenido (números de nota)
    text = re.sub(r'<sup>.*?</sup>', '', text)
    # 2. Eliminar cualquier otra etiqueta HTML residual
    text = re.sub(r'<[^>]+>', '', text)
    # 3. Normalizar espacios en blanco y saltos de línea
    text = re.sub(r'\n{3,}', '\n\n', text) # Máximo 2 saltos de línea
    text = re.sub(r' +', ' ', text) # Eliminar espacios dobles
    return text.strip()

def _process_and_embed_text_file(file_path: str, filename: str):
    db = clients.get('firestore')
    vector_store = clients.get('vector_store')
    meta_model = clients.get('metadata_model')
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        raw_content = f.read()
    
    # LIMPIEZA DE "COSAS EXTRAÑAS"
    text_content = _clean_text(raw_content)
    
    # 1. Metadatos
    doc_title, doc_author = filename, "Autor Desconocido"
    try:
        meta_response = meta_model.generate_content(
            f"Extrae Título y Autor: {text_content[:2000]}",
            generation_config=GenerationConfig(response_mime_type="application/json", temperature=0.1)
        )
        meta = json.loads(meta_response.text)
        doc_title = meta.get("title", filename)
        doc_author = meta.get("author", "Autor Desconocido")
    except: pass

    # 2. Chunking (Jerárquico por Markdown)
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","H1"),("##","H2"),("###","H3")])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)
    
    chunks = text_splitter.split_documents(md_splitter.split_text(text_content))
    
    documents = []
    for i, chunk in enumerate(chunks):
        m = chunk.metadata.copy()
        m.update({"source": filename, "title": doc_title, "author": doc_author, "index": i})
        documents.append(Document(page_content=chunk.page_content, metadata=m))
    
    # 3. Guardado Síncrono
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        vector_store.add_documents(documents[i:i + batch_size])
        logger.info(f"Progreso {filename}: Lote {i//batch_size + 1} guardado.")

@app.post("/")
async def handle_gcs_event(request: Request):
    event = await request.json()
    bucket_name, file_id = event.get("bucket"), event.get("name")
    if not bucket_name or not file_id or not file_id.endswith(".md"): return {"status": "ignored"}

    logger.info(f"Procesando: {file_id}")
    blob = clients.get('storage').bucket(bucket_name).blob(file_id)
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        blob.download_to_filename(tmp.name)
        await asyncio.to_thread(_process_and_embed_text_file, tmp.name, file_id)
        os.unlink(tmp.name)
        
    return {"status": "success"}
