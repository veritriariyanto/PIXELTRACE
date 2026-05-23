from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn
from app.core.config import settings
from app.core.ml_model import clip_vectorizer
from app.api.v1 import endpoints, vectorizer

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Logika sewaktu aplikasi dinyalakan (Startup)
    clip_vectorizer.load_model()
    yield
    # Logika sewaktu aplikasi dimatikan (Shutdown) bisa ditaruh di sini jika ada

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    lifespan=lifespan
)

# Daftarkan Router (Include Routers)
app.include_router(endpoints.router, prefix="/api/v1")
# Router vectorizer otomatis mendapatkan prefix /api/v1/vectorize dari file modulnya
app.include_router(vectorizer.router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",  # Menggunakan string import agar hot-reload bekerja maksimal
        host="0.0.0.0",
        port=8000,
        workers=2
    )