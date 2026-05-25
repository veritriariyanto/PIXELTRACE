FROM python:3.11-slim

# Set working directory di /app container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Force install PyTorch CPU version first supaya build ringan
RUN pip install --no-cache-dir torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu

# Copy dan install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 \
    -i https://pypi.org/simple \
    -r requirements.txt
# COPY seluruh folder project ke dalam container
# Pastikan di local kamu menjalankan `docker build` dari root project (sejajar folder app/)
COPY . .

# Expose port FastAPI
EXPOSE 8000

# Jalankan uvicorn dengan pattern module path yang sesuai dengan if __name__ == "__main__" kamu
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]