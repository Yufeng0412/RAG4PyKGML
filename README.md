# RAG PyKGML Chatbot

Python chatbot project with:
- FastAPI/LangServe backend
- Gradio frontend
- External LLM API integration (bring your own API key)

## Run With Docker (Quickstart)

### 1) Build image

```bash
docker build -f composer/Dockerfile -t chatbot-app .
```

### 2) Run container

```bash
docker run --rm -p 7860:7860 -e NVIDIA_NIM_API_KEY=your_key chatbot-app
```

Then open:
- http://localhost:7860

`NVIDIA_NIM_API_KEY` is supported and automatically mapped to `NVIDIA_API_KEY` inside the container.

## Run With Docker Compose

### Option A: Inline key

```bash
NVIDIA_NIM_API_KEY=your_key docker compose -f composer/docker-compose.yml up --build
```

### Option B: `.env` file

Create `.env` in the project root:

```env
NVIDIA_NIM_API_KEY=your_key
```

Then run:

```bash
docker compose -f composer/docker-compose.yml up --build
```

Open:
- http://localhost:7860

## Notes

- Frontend runs on port `7860` (container and host mapping).
- Backend runs internally on port `9012`.
- Logs stream to your terminal (`docker logs` / `docker compose logs -f`).
- Stop cleanly with `Ctrl+C` (Compose) or `docker stop <container_id>`.

## Related Docs

- `config_LangGraph/README.md`
- `PyKGML/README.md`
- `chatbot/README_docker.md` (prebuilt image packaging and end-user run guide)