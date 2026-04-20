FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY requirements.txt /app/requirements.txt
COPY frontend/requirements.txt /app/frontend/requirements.txt
COPY config_LangGraph/requirements.txt /app/config_LangGraph/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy project source code (see .dockerignore: excludes unused/, PyKGML/, notebooks, etc.).
COPY . /app

EXPOSE 7860

# start.sh uses bash-only `wait -n`; must not be invoked as `sh script`.
CMD ["bash", "/app/composer/start.sh"]
