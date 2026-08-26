FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-nvidia.txt requirements-sam.txt ./
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      "torch>=2.5.1" "torchvision>=0.20.1" \
    && pip install --no-cache-dir -r requirements-sam.txt
COPY . .
RUN pip install --no-cache-dir --no-build-isolation --no-deps -e ./third_party/sam2
RUN mkdir -p data uploads evidence
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
