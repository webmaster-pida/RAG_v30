import os
import json
import logging
import tempfile
import warnings
import asyncio  # <--- Movido aquí, al inicio
from typing import Dict, Any, List
from contextlib import asynccontextmanager

# Filtramos advertencias de librerías
warnings.filterwarnings("ignore", "Support for google-cloud-storage", category=FutureWarning)

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
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

COLLECTION_NAME = "pida_knowledge_base_v2" # Sugiero subir a v2 dado el cambio de dimensiones (incompatibles con v1)

# --- MODELOS DE DATOS ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    fetch_k: int = 15

clients = {}

# --- CLASE CUSTOM OPTIMIZADA CON EL NUEVO SDK Y 3072 DIMENSIONES ---
class ModernGeminiEmbeddings(Embeddings):
    def __init__(self, model_name="gemini-embedding-001", dimensionality=3072):
        self.model_name = model_name
        self.dimensionality = dimensionality
        # Inicializa el nuevo cliente GenAI (Requiere GOOGLE_GENAI_USE_VERTEXAI=True)
        self.client = genai.Client() 

    # Reintento exponencial en caso de límite de cuotas de API
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
    def _get_embeddings_with_retry(self, texts: List[str], task_type: str) -> List[List[float]]:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.dimensionality,
            )
        )
        return [emb.values for emb in response.embeddings]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        batch_size = 20 # Procesamiento en lotes para seguridad
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                results = self._get_embeddings_with_retry(batch, "RETRIEVAL_DOCUMENT")
                embeddings.extend(results)
                logger.info(f"Embeddings generados: lote {i//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error generando embeddings con nuevo SDK: {e}")
                raise e
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        results = self._get_embeddings_with_retry([text], "RETRIEVAL_QUERY")
        return results[0]

# --- LIFESPAN: INICIALIZACIÓN SEGURA ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("--- Inicializando clientes de GCP... ---")
    PROJECT_ID = os.environ.get("PROJECT_ID")
    VERTEX_AI_LOCATION = os.environ.get("VERTEX_AI_LOCATION", "us-central1")
    
    # Inicializamos Vertex clásico (aún necesario para el GenerativeModel de metadatos)
    vertexai.init(project=PROJECT_ID, location=VERTEX_AI_LOCATION)
    
    clients['firestore'] = firestore.Client()
    clients['storage'] = storage.Client()
    
    # Instanciamos nuestra nueva clase con 3072 dimensiones
    clients['embedding'] = ModernGeminiEmbeddings(dimensionality=3072)
    clients['metadata_model'] = GenerativeModel("gemini-2.5-flash")
    
    clients['vector_store'] = FirestoreVectorStore(
        collection=COLLECTION_NAME, 
        embedding_service=clients['embedding'], 
        client=clients['firestore']
    )
    logger.info("--- Clientes inicializados correctamente ---")
    yield
    clients.clear()

app = FastAPI(lifespan=lifespan)

# --- LÓGICA CORE DE INGESTA ---
def _process_and_embed_text_file(file_path: str, filename: str) -> Dict[str, Any]:
    try:
        logger.info(f"Iniciando procesamiento de archivo: {filename}")
        db = clients.get('firestore')
        vector_store = clients.get('vector_store')
        meta_model = clients.get('metadata_model')
        
        docs_ref = db.collection(COLLECTION_NAME)
        existing_docs = docs_ref.where(filter=FieldFilter("metadata.source", "==", filename)).limit(1).stream()
        if any(existing_docs):
            logger.warning(f"El archivo {filename} ya existe. Cancelando indexación.")
            return {"status": "skipped", "message": "Archivo ya indexado previamente."}

        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            text_content = f.read()
        
        if not text_content.strip():
            return {"status": "error", "reason": "El archivo está vacío."}

        # Extracción estricta de metadatos
        doc_title = filename
        doc_author = "Autor Desconocido"
        
        try:
            sample_text = text_content[:3000]
            prompt_meta = f"""Eres un bibliotecario experto. Analiza el fragmento de texto y extrae el Título y el Autor.
            TEXTO: {sample_text}"""
            
            response_schema = {
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING"},
                    "author": {"type": "STRING"}
                },
                "required": ["title", "author"]
            }
            
            meta_response = meta_model.generate_content(
                prompt_meta,
                generation_config=GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=0.1
                )
            )
            
            metadata_extracted = json.loads(meta_response.text)
            doc_title = metadata_extracted.get("title", filename)
            doc_author = metadata_extracted.get("author", "Autor Desconocido")
            logger.info(f"Metadatos extraídos: Título='{doc_title}', Autor='{doc_author}'")
            
        except Exception as e:
            logger.warning(f"Error extrayendo metadatos, usando defaults. Detalle: {e}")

        # Procesamiento y Chunking
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(text_content)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=250,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(md_header_splits)
        
        documents = []
        for i, chunk in enumerate(chunks):
            meta = chunk.metadata.copy()
            meta.update({
                "source": filename,
                "title": doc_title,
                "author": doc_author,
                "chunk_index": i,
                "dimensions": 3072 # Etiqueta útil para debugging en Firestore
            })
            documents.append(Document(page_content=chunk.page_content, metadata=meta))
        
        batch_size = 50 
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vector_store.add_documents(batch)
        
        logger.info(f"Finalizado: {filename}. {len(documents)} chunks insertados (3072d).")
        return {"status": "ok", "message": f"Archivo procesado: {doc_title}"}
        
    except Exception as e:
        logger.error(f"Error procesando documento: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}

# --- ENDPOINTS ---

@app.post("/")
async def handle_gcs_event(request: Request):
    event = await request.json()
    if not event: raise HTTPException(status_code=400, detail="Sin body")

    bucket_name = event.get("bucket")
    file_id = event.get("name")
    
    if not bucket_name or not file_id: return {"status": "ignored"}
    if not (file_id.endswith(".txt") or file_id.endswith(".md")): return {"status": "ignored"}

    storage_client = clients.get('storage')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_id)
    if not blob.exists() or blob.size == 0: return {"status": "ignored"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        blob.download_to_filename(temp_file.name)
        temp_path = temp_file.name
    try:
        logger.info(f"Descarga completada. Iniciando procesamiento SÍNCRONO de {file_id}")
        
        # OBLIGAMOS a Cloud Run a esperar y mantener el CPU al 100%
        await asyncio.to_thread(_process_and_embed_text_file, temp_path, file_id)
        
    finally:
        if os.path.exists(temp_path): os.unlink(temp_path)

    return {"status": "ok", "message": f"Procesamiento 100% finalizado para {file_id}"}


@app.post("/query")
async def query_rag_handler(request_data: QueryRequest):
    try:
        user_query = request_data.query
        top_k = request_data.top_k
        fetch_k = request_data.fetch_k
        vector_store: FirestoreVectorStore = clients.get('vector_store')
        
        logger.info(f"MMR Search (3072d) para: '{user_query}'")
        
        found_docs = vector_store.max_marginal_relevance_search(
            query=user_query, 
            k=top_k,
            fetch_k=fetch_k,
            lambda_mult=0.5 
        )
        
        results = []
        for doc in found_docs:
            raw_meta = doc.metadata
            inner_meta = raw_meta.get("metadata", {})
            data_source = inner_meta if isinstance(inner_meta, dict) and inner_meta else raw_meta
            
            results.append({
                "source": data_source.get("source", "Desconocido"),
                "content": doc.page_content,
                "title": data_source.get("title", data_source.get("Title", "Sin título")),
                "author": data_source.get("author", data_source.get("Author", "Autor Desconocido"))
            })
        
        return {"results": results, "count": len(results)}

    except Exception as e:
        logger.error(f"Error query MMR: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
