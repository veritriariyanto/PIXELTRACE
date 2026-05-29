import io
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from fastapi import HTTPException
from app.core.config import logger, settings

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
                # Prefer the model helper that returns projected image features
                # (CLIPModel.get_image_features applies the visual projection to match
                # the configured embedding dimension, e.g., 512).
                if hasattr(self.model, "get_image_features"):
                    image_embeds = self.model.get_image_features(pixel_values=inputs['pixel_values'])
                else:
                    # Fallback: run vision backbone and apply visual projection if present
                    outputs = self.model.vision_model(pixel_values=inputs['pixel_values'])
                    pooled = outputs.pooler_output
                    if hasattr(self.model, "visual_projection"):
                        image_embeds = self.model.visual_projection(pooled)
                    else:
                        image_embeds = pooled

                # Ensure we have a Tensor (some transformers versions return ModelOutput)
                if not isinstance(image_embeds, torch.Tensor):
                    if hasattr(image_embeds, "pooler_output"):
                        image_embeds = image_embeds.pooler_output
                    elif hasattr(image_embeds, "image_embeds"):
                        image_embeds = image_embeds.image_embeds
                    elif hasattr(image_embeds, "last_hidden_state"):
                        # take the pooled representation if only last_hidden_state exists
                        image_embeds = image_embeds.last_hidden_state
                        # if it's sequence, try mean pooling over seq dim
                        if image_embeds.dim() == 3:
                            image_embeds = image_embeds.mean(dim=1)
                    elif isinstance(image_embeds, (list, tuple)) and len(image_embeds) > 0:
                        image_embeds = image_embeds[0]
                    else:
                        raise HTTPException(status_code=500, detail="Unable to extract tensor from model output")

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