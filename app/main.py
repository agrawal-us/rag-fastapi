from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from openai import OpenAI

from app.config import settings
from app.store import load_faiss, load_chunks, search

app = FastAPI(title="RAG Demo (FastAPI + FAISS + OpenAI)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Serve the UI from / (index.html)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

client = OpenAI(api_key=settings.openai_api_key)
_index = None
_chunks = None

def ensure_index():
    global _index, _chunks
    if _index is None or _chunks is None:
        if not settings.faiss_path.exists() or not settings.chunks_path.exists():
            raise HTTPException(status_code=500, detail="Index not built. Run `python -m app.ingest` first.")
        _index = load_faiss(settings.faiss_path)
        _chunks = load_chunks(settings.chunks_path)

class AskReq(BaseModel):
    question: str
    top_k: int | None = None

def embed_query(q: str) -> np.ndarray:
    vec = client.embeddings.create(model=settings.embedding_model, input=[q]).data[0].embedding
    return np.array([vec], dtype="float32")

def build_context(ids):
    lines, cites = [], []
    for i in ids:
        ch = _chunks[i]
        lines.append(ch.text)
        cites.append({"id": ch.id, "source": ch.meta.get("source"), "section": ch.meta.get("section")})
    return "\n\n---\n\n".join(lines), cites

@app.post("/ask")
def ask(req: AskReq):
    ensure_index()
    k = req.top_k or settings.top_k
    qv = embed_query(req.question)
    D, I = search(_index, qv, k)
    ctx, cites = build_context(I[0].tolist())

    system = ("You are a helpful assistant. Use ONLY the provided context to answer. "
              "If the context is insufficient, say you don't know. Always include citations.")
    prompt = f"Context:\n{ctx}\n\nQuestion: {req.question}\n"

    chat = client.chat.completions.create(
        model=settings.chat_model,
        temperature=0.2,
        messages=[{"role":"system","content":system},
                  {"role":"user","content":prompt}]
    )
    return {"answer": chat.choices[0].message.content, "citations": cites}