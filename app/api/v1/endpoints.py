import io
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from fastapi import APIRouter
from fastapi import HTTPException
from app.core.config import logger, settings

router = APIRouter()

class CLIPVectorizer:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cpu"

    def load_model(self):
        """Memuat model CLIP ke memori."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Memuat model {settings.MODEL_NAME} pada device: {self.device}...")
        
        self.processor = CLIPProcessor.from_pretrained(settings.MODEL_NAME)
        self.model = CLIPModel.from_pretrained(settings.MODEL_NAME)
        self.model = self.model.to(self.device)
        self.model.eval()  # Set ke mode evaluasi
        logger.info("Model berhasil dimuat dan siap menerima request!")

    def vectorize(self, image_bytes: bytes) -> dict:
        """Proses ekstraksi fitur gambar menjadi vektor."""
        if self.model is None or self.processor is None:
            raise HTTPException(status_code=503, detail="Model belum siap atau gagal dimuat.")
            
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.vision_model(**{k: v for k, v in inputs.items() if k == 'pixel_values'})
                image_embeds = outputs.pooler_output
                image_features = F.normalize(image_embeds, p=2, dim=-1)
            
            vector_list = image_features.squeeze().detach().cpu().tolist()
            
            if not isinstance(vector_list, list):
                vector_list = [vector_list]
                
            return {
                "success": True,
                "dimensions": len(vector_list),
                "vector": vector_list
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Gagal memproses gambar: {str(e)}"
            )

# Inisialisasi objek tunggal (Singleton) untuk digunakan lintas router
clip_vectorizer = CLIPVectorizer()


@router.get("/info")
def get_info():
    """Endpoint untuk mendapatkan informasi service dan status model."""
    return {
        "title": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "model": settings.MODEL_NAME,
        "device": clip_vectorizer.device,
        "model_loaded": clip_vectorizer.model is not None,
        "vector_dimension": 512
    }