# Standalone chatbot image. Build from this directory:
#   docker build -t chatbot-prebuilt:latest -f Dockerfile .
# Expects `frontend_server.py` (FastAPI + Gradio `app`) and related modules in the build context.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

RUN mkdir -p /app/imgs /app/slides

EXPOSE 7860

# NVIDIA_NIM_API_KEY is mapped to NVIDIA_API_KEY for langchain-nvidia endpoints.
CMD ["bash", "-c", "if [ -n \"${NVIDIA_NIM_API_KEY:-}\" ] && [ -z \"${NVIDIA_API_KEY:-}\" ]; then export NVIDIA_API_KEY=\"$NVIDIA_NIM_API_KEY\"; fi; exec gunicorn --worker-class uvicorn.workers.UvicornWorker frontend_server:app --workers 1 --bind 0.0.0.0:${PORT:-7860} --log-level info"]
