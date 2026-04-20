
## NOTE: THIS SERVER IS RUNNING PERPETUALLY FOR THIS COURSE.
## DO NOT CHANGE CODE HERE; INSTEAD, INTERFACE WITH IT VIA USER INTERFACE
## AND BY DEPLOYING ON PORT :9012

import gradio as gr
import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import logging
import uvicorn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

route = os.environ.get("APP_ROOT_PATH")
if route in ("", "/"):
    route = None

#####################################################################
## Final App Deployment
from frontend_block_agent import get_demo

def create_app() -> FastAPI:
    logger.warning("Creating Gradio demo and mounting on FastAPI")
    try:
        demo = get_demo()
        demo.queue()
        fastapi_app = FastAPI()

        # Register before Gradio mount so "/" mount does not shadow /health.
        @fastapi_app.get("/health")
        async def health():
            return {"success": True}

        @fastapi_app.get("/")
        async def root_redirect():
            return RedirectResponse(url="/gradio", status_code=307)

        return gr.mount_gradio_app(
            fastapi_app,
            demo,
            path="/gradio",
            root_path=route,
        )
    except Exception:
        logger.exception("Frontend app startup failed")
        raise

app = create_app()
port = int(os.getenv("PORT", "7860"))
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)
