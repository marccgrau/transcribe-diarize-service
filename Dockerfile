FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# OS deps for audio I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) install deps first to leverage Docker layer cache
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 2) copy code
COPY . /app

# Optional: make 'src' importable without --app-dir
ENV PYTHONPATH=/app

EXPOSE 8000

# A) using --app-dir (no PYTHONPATH needed):
# CMD ["uvicorn", "main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]

# B) using fully-qualified module path (PYTHONPATH=/app):
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]