from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from app.core.config import settings, logger
from app.core.ml_model import clip_vectorizer

router = APIRouter(prefix="/vectorize", tags=["Vectorizer"])

async def validate_file(file: UploadFile):
    """Fungsi pembantu untuk validasi tipe dan ukuran file."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="File yang diunggah harus berupa gambar (jpeg, png, webp, bmp)"
        )
    
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Ukuran file {file.filename} terlalu besar. Maksimal 10MB."
        )

@router.post("")
async def vectorize_single_image(file: UploadFile = File(...)):
    """Endpoint untuk mengubah satu gambar menjadi vektor."""
    await validate_file(file)
    
    try:
        image_bytes = await file.read()
        result = await run_in_threadpool(clip_vectorizer.vectorize, image_bytes)
        
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "vector_dimension": result["dimensions"],
            "vector": result["vector"],
            "message": "Gambar berhasil diubah menjadi vektor"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error tidak terduga: {str(e)}")

@router.post("/batch")
async def vectorize_batch_images(files: list[UploadFile] = File(...)):
    """Endpoint untuk mengubah banyak gambar sekaligus (Batch)."""
    if len(files) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Maksimal {settings.MAX_BATCH_SIZE} file per request"
        )
    
    results = []
    errors = []
    
    for file in files:
        try:
            await validate_file(file)
            image_bytes = await file.read()
            
            result = await run_in_threadpool(clip_vectorizer.vectorize, image_bytes)
            results.append({
                "filename": file.filename,
                "vector_dimension": result["dimensions"],
                "vector": result["vector"]
            })
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })
            
    return {
        "total_files": len(files),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors if errors else None
    }