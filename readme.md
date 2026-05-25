---
title: PixelTrace
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# PixelTrace API

# Image Vectorizer API - Dokumentasi Lengkap

## 📋 Daftar Isi
1. [Penjelasan Masalah & Solusi](#penjelasan-masalah--solusi)
2. [Instalasi](#instalasi)
3. [Menjalankan API](#menjalankan-api)
4. [Dokumentasi Endpoint](#dokumentasi-endpoint)
5. [Contoh Penggunaan](#contoh-penggunaan)
6. [Handle Multiple Requests](#handle-multiple-requests)
7. [Troubleshooting](#troubleshooting)

---

## 🔧 Penjelasan Masalah & Solusi

### Masalah Original
```json
{
    "detail": "Gagal memproses gambar: 'BaseModelOutputWithPooling' object has no attribute 'norm'"
}
```

### Root Cause
Masalah ini terjadi karena:

1. **Incompatible API Usage**: `model.get_image_features()` tidak tersedia atau tidak kompatibel dengan versi transformers yang Anda gunakan
2. **Normalisasi Implicit**: Model CLIP modern sudah melakukan normalisasi otomatis, jadi mencoba akses `.norm` yang tidak ada menyebabkan error
3. **Version Mismatch**: Versi `transformers==4.45.0` mungkin memiliki breaking changes

### Solusi yang Diterapkan

#### 1. **Menggunakan Vision Model Langsung**
```python
# BEFORE (Error)
image_features = model.get_image_features(**inputs)

# AFTER (Bekerja)
outputs = model.vision_model(**inputs)
image_embeds = outputs.pooler_output
image_features = F.normalize(image_embeds, p=2, dim=-1)
```

#### 2. **Async/Await untuk Concurrency**
```python
# Menggunakan async def untuk better request handling
@app.post("/api/v1/vectorize")
async def vectorize_image(file: UploadFile = File(...)):
    # Proses berjalan di thread pool (non-blocking)
    result = await run_in_threadpool(process_image_vectorization, image_bytes)
```

#### 3. **Model Evaluation Mode**
```python
model.eval()  # Set ke mode evaluasi untuk inference
```

#### 4. **Better Error Handling**
- Logging yang proper
- Validasi file size
- Error messages yang informatif

---

## 📦 Instalasi

### Prasyarat
- Python 3.8+
- pip atau conda

### Step 1: Clone/Setup Project
```bash
mkdir clip-vectorizer && cd clip-vectorizer
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Catatan untuk GPU (optional)**:
Jika ingin menggunakan GPU CUDA, ubah torch installation:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Verify Installation
```bash
python -c "import torch; import transformers; print('OK')"
```

---

## 🚀 Menjalankan API

### Development Mode (dengan auto-reload)
```bash
uvicorn clip_vectorizer_fixed:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode (single worker)
```bash
uvicorn clip_vectorizer_fixed:app --host 0.0.0.0 --port 8000 --workers 1
```

### Production Mode (multiple workers)
```bash
uvicorn clip_vectorizer_fixed:app --host 0.0.0.0 --port 8000 --workers 4
```

**Output yang diharapkan:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Memuat model openai/clip-vit-base-patch32 pada device: cpu...
INFO:     Model berhasil dimuat dan siap menerima request!
```

---

## 📚 Dokumentasi Endpoint

### 1. Health Check
```http
GET /
```

**Response:**
```json
{
  "status": "healthy",
  "device": "cpu",
  "model_loaded": true
}
```

---

### 2. Get API Info
```http
GET /api/v1/info
```

**Response:**
```json
{
  "title": "Image Vectorizer API",
  "version": "1.0.0",
  "vector_dimension": 512,
  "device": "cpu",
  "max_image_size": "1024x1024",
  "supported_formats": ["jpeg", "png", "webp", "bmp"]
}
```

---

### 3. Vectorize Single Image ⭐
```http
POST /api/v1/vectorize
Content-Type: multipart/form-data

file: <binary_image_data>
```

**Parameters:**
- `file` (required): Image file (jpeg, png, webp, bmp)

**Response (Success):**
```json
{
  "filename": "photo.jpg",
  "content_type": "image/jpeg",
  "vector_dimension": 512,
  "vector": [0.123, -0.456, 0.789, ...],
  "message": "Gambar berhasil diubah menjadi vektor"
}
```

**Response (Error):**
```json
{
  "detail": "Ukuran file terlalu besar. Maksimal 10MB."
}
```

---

### 4. Vectorize Batch (Multiple Images)
```http
POST /api/v1/vectorize-batch
Content-Type: multipart/form-data

files: [<image1>, <image2>, <image3>, ...]
```

**Parameters:**
- `files` (required): List of image files (max 20 files)

**Response:**
```json
{
  "total_files": 3,
  "successful": 3,
  "failed": 0,
  "results": [
    {
      "filename": "image1.jpg",
      "vector_dimension": 512,
      "vector": [...]
    },
    {
      "filename": "image2.jpg",
      "vector_dimension": 512,
      "vector": [...]
    }
  ],
  "errors": null
}
```

---

### 5. Get Stats
```http
GET /api/v1/stats
```

**Response:**
```json
{
  "device": "cpu",
  "model": "openai/clip-vit-base-patch32",
  "vector_dimension": 512,
  "status": "operational"
}
```

---

## 💻 Contoh Penggunaan

### 1. Menggunakan cURL
```bash
# Single image
curl -X POST "http://localhost:8000/api/v1/vectorize" \
  -F "file=@/path/to/image.jpg"

# Batch images
curl -X POST "http://localhost:8000/api/v1/vectorize-batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

### 2. Menggunakan Python (requests)
```python
import requests
import json

# Single image
with open("photo.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/api/v1/vectorize",
        files=files
    )

data = response.json()
print(f"Vector dimension: {data['vector_dimension']}")
print(f"Vector sample: {data['vector'][:10]}")

# Simpan vector untuk nanti
vector = data['vector']
```

### 3. Menggunakan JavaScript (fetch)
```javascript
async function vectorizeImage(imageFile) {
  const formData = new FormData();
  formData.append("file", imageFile);

  const response = await fetch("http://localhost:8000/api/v1/vectorize", {
    method: "POST",
    body: formData,
  });

  const data = await response.json();
  console.log(`Vector dimension: ${data.vector_dimension}`);
  console.log(`Vector sample: ${data.vector.slice(0, 10)}`);
  
  return data.vector;
}
```

### 4. Menggunakan Server Lain ke Server Ini

**Server A (yang mengirim gambar ke Server B):**
```python
import requests

def send_image_for_vectorization(image_path, vectorizer_url):
    """
    Mengirim gambar ke server vectorizer untuk diubah menjadi vektor.
    Vector disimpan tanpa perlu menyimpan gambar original.
    """
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(
            f"{vectorizer_url}/api/v1/vectorize",
            files=files
        )
    
    if response.status_code == 200:
        data = response.json()
        vector = data['vector']
        
        # Simpan vector ke database (bukan gambar)
        # Contoh: save_to_database(vector)
        
        return {
            "success": True,
            "vector": vector,
            "dimension": data['vector_dimension']
        }
    else:
        return {
            "success": False,
            "error": response.json()
        }

# Usage
result = send_image_for_vectorization(
    "photo.jpg",
    "http://vectorizer-server:8000"
)

if result['success']:
    vector = result['vector']
    # Gunakan vector untuk similarity search, clustering, dll
```

---

## 🔄 Handle Multiple Requests

### Concurrency Model

FastAPI + Uvicorn bisa handle banyak concurrent requests:

```
┌─────────────────────────────────────────┐
│         Client Requests (banyak)         │
└────────────────┬────────────────────────┘
                 │
        ┌────────▼────────┐
        │  Uvicorn Event  │
        │      Loop       │
        └────────┬────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
 Thread1      Thread2      Thread3
    │            │            │
    └────────────┬────────────┘
                 │
        ┌────────▼────────┐
        │  GPU/CPU Pool   │
        │  (Model Load)   │
        └─────────────────┘
```

### Konfigurasi Production

**File: `config.py`**
```python
import multiprocessing

# Hitung optimal worker count
cpu_count = multiprocessing.cpu_count()
optimal_workers = (cpu_count * 2) + 1

# Contoh: 4 CPU cores → 9 workers
WORKERS = optimal_workers
MAX_CONNECTIONS = 100
TIMEOUT = 300  # 5 minutes
```

**Run dengan multiple workers:**
```bash
uvicorn clip_vectorizer_fixed:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --timeout-keep-alive 5 \
  --timeout-notify 30
```

### Load Testing
```bash
# Install Apache Bench
apt-get install apache2-utils

# Test dengan 100 concurrent requests, 1000 total requests
ab -n 1000 -c 100 -p image.jpg -T image/jpeg \
  http://localhost:8000/api/v1/vectorize
```

---

## 🧪 Testing

### Run Test Suite
```bash
# Pastikan API sudah jalan
python test_api.py
```

**Output:**
```
============================================================
IMAGE VECTORIZER API - TEST SUITE
============================================================

✓ Test image created: /tmp/test_image.png

============================================================
TEST 1: Health Check
============================================================
Status Code: 200
Response: {
  "status": "healthy",
  "device": "cpu",
  "model_loaded": true
}

... [more tests] ...

============================================================
TEST SUMMARY
============================================================
health_check.................................. ✓ PASSED
get_info..................................... ✓ PASSED
vectorize_single............................ ✓ PASSED
invalid_file................................ ✓ PASSED
stats...................................... ✓ PASSED
concurrent.................................. ✓ PASSED

Total: 6/6 tests passed
```

---

## 🐛 Troubleshooting

### 1. "Model not loaded" Error
```
Status: not_loaded
```

**Solusi:**
- Tunggu model selesai loading (lihat logs)
- Restart API server
- Check disk space untuk model cache

### 2. "CUDA out of memory"
```
RuntimeError: CUDA out of memory
```

**Solusi:**
```python
# Gunakan CPU mode di startup
device = "cpu"  # Force CPU

# Atau set limit GPU memory
import torch
torch.cuda.set_per_process_memory_fraction(0.5)
```

### 3. "Timeout" untuk Request Besar
```
RequestTimeout: The server did not send any data
```

**Solusi:**
- Kurangi ukuran image
- Tingkatkan timeout di client:
  ```python
  response = requests.post(url, files=files, timeout=60)
  ```

### 4. "File corrupted"
```
OSError: cannot identify image file
```

**Solusi:**
- Pastikan file adalah valid image
- Test dengan image simple dulu
- Check format: JPEG, PNG, WebP, BMP

### 5. Model Loading Lambat

**Solusi:**
- First load akan slow (download model)
- Gunakan SSD untuk cache
- Enable GPU untuk faster inference

---

## 📊 Performance Tips

### Optimization Checklist
- [ ] Gunakan GPU jika available
- [ ] Resize image besar sebelum upload
- [ ] Batch processing untuk multiple images
- [ ] Cache vectors di database
- [ ] Monitor memory usage

### Memory Optimization
```python
# Reduce model precision untuk CPU
import torch
model = CLIPModel.from_pretrained(..., torch_dtype=torch.float16)
```

### Speed Optimization
```bash
# Gunakan multiple workers
uvicorn clip_vectorizer_fixed:app --workers 4

# Disable dev logging in production
--log-level warning
```

---

## 📝 License & Attribution

- Model: OpenAI CLIP (ViT-B/32)
- Framework: FastAPI, PyTorch, Hugging Face Transformers

---

## 🤝 Support

Untuk issues atau pertanyaan:
1. Check logs: `uvicorn` console output
2. Test endpoint: `GET /api/v1/stats`
3. Run test suite: `python test_api.py`

---

**Happy Vectorizing! 🎉**