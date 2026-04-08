# Dockerfile
FROM python:3.12-slim

# Evita que Python genere archivos .pyc y habilita logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Activa el enrutamiento de Vertex AI para el nuevo SDK google-genai
ENV GOOGLE_GENAI_USE_VERTEXAI=True

WORKDIR /app

# Instalar dependencias primero para aprovechar caché de Docker
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY . .

# Exponer puerto para Cloud Run
EXPOSE 8080

# Usamos Uvicorn estándar para FastAPI, ideal para cargas de trabajo RAG asíncronas
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2", "--timeout-keep-alive", "120"]
