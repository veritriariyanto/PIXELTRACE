from fastapi import APIRouter
from app.core.config import logger, settings
from app.core.ml_model import clip_vectorizer

router = APIRouter()


@router.get("/info")
def get_info():
    """Endpoint untuk mendapatkan informasi service dan status model."""
    return {
        "title": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "model": settings.MODEL_NAME,
        "device": clip_vectorizer.device,
        "model_loaded": clip_vectorizer.model is not None,
        "vector_dimension": settings.VECTOR_DIMENSION
    }