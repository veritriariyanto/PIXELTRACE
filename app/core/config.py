import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redam noise dari library eksternal
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

class Settings:
    PROJECT_NAME: str = "Image Vectorizer API"
    DESCRIPTION: str = "High-performance API untuk mengubah gambar menjadi vektor embedding menggunakan CLIP"
    VERSION: str = "1.0.0"
    MODEL_NAME: str = "openai/clip-vit-base-patch32"
    VECTOR_DIMENSION: int = 512
    MAX_IMAGE_SIZE: str = "1024x1024"
    SUPPORTED_FORMATS: list[str] = ["jpeg", "png", "webp", "bmp"]
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    MAX_BATCH_SIZE: int = 20

settings = Settings()