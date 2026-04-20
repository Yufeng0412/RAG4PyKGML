# Chatbot Docker Distribution Guide

This guide shows how to package and distribute the chatbot as a prebuilt Docker image so end users can run it without source code.

## Before you start

- Install Docker: [Docker Desktop](https://www.docker.com/products/docker-desktop/) or the official setup docs at [Get Docker](https://docs.docker.com/get-docker/).
- Docker command reference and guides: [Docker Docs](https://docs.docker.com/).
- If you use Docker Desktop (macOS/Windows), make sure Docker Desktop is running before executing `docker` commands in your terminal.

## Source layout

The image expects a runnable Gradio/FastAPI app in this folder, including at least:

- `frontend_server.py` — must expose a FastAPI `app` (e.g. `app = gr.mount_gradio_app(...)`).
- Other modules your UI imports (e.g. `frontend_block.py`, tools, assets).

If those files are missing, the image can still build, but the container will fail at startup
with errors like `ModuleNotFoundError: No module named 'frontend_server'`.

## Important note (recommended path)

If your runnable app code lives in the repository root (`frontend/`, `backend_server.py`,
`config_LangGraph/`, `pykgml_vector_db/`) and not inside `chatbot/`, use the **root**
`Dockerfile` instead of `chatbot/Dockerfile`.

## For Maintainers: Build and Export Image

Use the **root `Dockerfile`** as the canonical build path.
The current runtime app wiring lives in repository root (`frontend/`, `backend_server.py`,
`config_LangGraph/`, `pykgml_vector_db/`), so building from `chatbot/Dockerfile` is not the
recommended path for this project.

**Build from repository root**

```bash
docker build -t chatbot-prebuilt:latest -f Dockerfile .
```

Then export:

```bash
docker save -o chatbot-prebuilt_latest.tar chatbot-prebuilt:latest
```

Optional: compress the tar file for transfer:

```bash
gzip -9 chatbot-prebuilt_latest.tar
```

Share either:
- `chatbot-prebuilt_latest.tar`, or
- `chatbot-prebuilt_latest.tar.gz`

## For Users: Load and Run (No Source Needed)

### 1) Load image

If you received `.tar`:

```bash
docker load -i chatbot-prebuilt_latest.tar
```

If you received `.tar.gz`:

```bash
gunzip -c chatbot-prebuilt_latest.tar.gz | docker load
```

### 2) Run container

```bash
docker run --rm -p 7860:7860 \
  -e NVIDIA_NIM_API_KEY=your_api_key_here \
  chatbot-prebuilt:latest
```

Notes:
- `NVIDIA_NIM_API_KEY` is required for LLM access.
- The container maps the web UI to host port `7860`.

### 3) Open the app

Go to:
- http://localhost:7860

### 4) Stop the app

Press `Ctrl+C` in the terminal running Docker.

## Troubleshooting

- Port already in use:
  - Run with a different host port, e.g. `-p 7861:7860`, then open `http://localhost:7861`.
- API/auth errors:
  - Confirm `NVIDIA_NIM_API_KEY` is valid and active.
- Check logs:
  - If detached mode is used (`-d`), inspect with `docker logs <container_id>`.
- `ModuleNotFoundError: frontend_server`:
  - You likely built from `chatbot/Dockerfile` without `frontend_server.py` in `chatbot/`.
    Rebuild using the root `Dockerfile`, or restore the required runtime files inside `chatbot/`.
