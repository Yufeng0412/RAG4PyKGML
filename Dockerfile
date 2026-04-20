FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install runtime dependencies only.
COPY requirements.txt /app/requirements.txt
COPY frontend/requirements.txt /app/frontend/requirements.txt
COPY config_LangGraph/requirements.txt /app/config_LangGraph/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy only runtime source and data (not notebooks/dev folders).
COPY backend_server.py /app/backend_server.py
COPY frontend /app/frontend
COPY config_LangGraph /app/config_LangGraph
COPY pykgml_vector_db /app/pykgml_vector_db
COPY composer/start.sh /app/composer/start.sh

RUN mkdir -p /app/imgs /app/slides \
    && chmod +x /app/composer/start.sh

EXPOSE 7860

# start.sh boots backend (9012) + frontend (7860).
CMD ["bash", "/app/composer/start.sh"]
