import os
import mimetypes
from pathlib import Path
from typing import List, Optional
import sys

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchAny

# Ensure project root is on sys.path for `src` imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

# Reuse shared embedding utility
from src.utils.embeddings import get_audio_embedding

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_SOURCES = ["english-song-snippet", "youtube-royalty-free"]
COLLECTION_NAME = "audio_collection"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# -----------------------------------------------------------------------------
# Initialize client (singleton for performance)
# -----------------------------------------------------------------------------
from src.client.connection import get_client
client = get_client()


# -----------------------------------------------------------------------------
# FastAPI app setup
# -----------------------------------------------------------------------------
app = FastAPI(title="Qdrant Audio Search", version="1.0")

ASSETS_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"

app.mount("/static", StaticFiles(directory=str(ASSETS_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def build_audio_path(zip_source: str, filename: str) -> Path:
	candidate = (DATA_DIR / zip_source / filename).resolve()
	if not str(candidate).startswith(str((DATA_DIR).resolve())):
		raise HTTPException(status_code=400, detail="Invalid path")
	return candidate


def to_audio_mime(path: Path) -> str:
	mime, _ = mimetypes.guess_type(str(path))
	return mime or "audio/mpeg"






# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
	return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
	return templates.TemplateResponse(
		"index.html", {"request": request, "results": None, "sources": DEFAULT_SOURCES}
	)


@app.post("/search", response_class=HTMLResponse)
async def upload_and_search(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
	# Save upload to a temp file
	tmp_dir = Path(".tmp")
	tmp_dir.mkdir(exist_ok=True)
	upload_path = tmp_dir / file.filename
	with open(upload_path, "wb") as out:
		out.write(await file.read())

	# Embed and search
	embedding = get_audio_embedding(upload_path)
	print("[DEBUG] Query embedding shape:", getattr(embedding, 'shape', None))
	search_result = client.search(
		collection_name=COLLECTION_NAME,
		query_vector=embedding.tolist(),
		limit=5,
	)
	print("[DEBUG] Raw search_result length:", len(search_result))
	for hit in search_result:
		print(f"Score: {hit.score}, Payload: {hit.payload}")

	# Keep UI working: map to results list
	results: List[dict] = []
	for r in search_result:
		payload = r.payload or {}
		filename = payload.get("filename")
		zip_source = payload.get("zip_source")
		if not filename or not zip_source:
			continue
		audio_url = f"/audio/{zip_source}/{filename}"
		results.append(
			{
				"id": r.id,
				"score": float(r.score),
				"filename": filename,
				"zip_source": zip_source,
				"session": payload.get("session"),
				"duration": payload.get("duration"),
				"collection": COLLECTION_NAME,
				"audio_url": audio_url,
			}
		)

	return templates.TemplateResponse(
		"index.html",
		{
			"request": request,
			"results": results,
			"uploaded_filename": file.filename,
			"sources": DEFAULT_SOURCES,
		},
	)


@app.get("/audio/{zip_source}/{filename}")
async def serve_audio(zip_source: str, filename: str) -> FileResponse:
	path = build_audio_path(zip_source, filename)
	if not path.exists():
		raise HTTPException(status_code=404, detail="Audio not found")
	return FileResponse(str(path), media_type=to_audio_mime(path))
