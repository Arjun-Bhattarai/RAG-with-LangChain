from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes.chat import router as chat_router


app = FastAPI(
    title="Advanced RAG API",
    version="1.0.0"
)


# API
app.include_router(
    chat_router,
    prefix="/chat",
    tags=["chat"]
)


# Frontend
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static"
)


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}