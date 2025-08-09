import re
from pathlib import Path
from typing import List
import numpy as np
from openai import OpenAI
from pypdf import PdfReader
import tiktoken

from app.config import settings
from app.store import Chunk, save_chunks, build_faiss, save_faiss

client = OpenAI(api_key=settings.openai_api_key)

def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    return path.read_text(encoding="utf-8", errors="ignore")

def simple_clean(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

_enc = tiktoken.get_encoding("cl100k_base")
def token_len(s: str) -> int:
    return len(_enc.encode(s))

def chunk_text(text: str, max_tokens=600, overlap=80) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], []
    cur = 0
    for p in paras:
        parts = [p] if token_len(p) <= max_tokens else re.findall(r".{1,1200}(?:\s+|$)", p, flags=re.S)
        for part in parts:
            t = token_len(part)
            if cur + t > max_tokens and buf:
                joined = "\n\n".join(buf)
                chunks.append(joined)
                tail = joined[-overlap*4:]
                buf = [tail, part] if tail else [part]
                cur = token_len("\n\n".join(buf))
            else:
                buf.append(part); cur += t
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks

def embed_texts(texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(model=settings.embedding_model, input=texts)
    return np.array([d.embedding for d in resp.data], dtype="float32")

def run_ingest():
    docs = [p for p in settings.data_dir.glob("**/*") if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"}]
    if not docs:
        raise SystemExit("No documents found in ./data")

    all_chunks: List[Chunk] = []
    for p in docs:
        raw = read_file(p)
        clean = simple_clean(raw)
        for i, c in enumerate(chunk_text(clean, settings.max_chunk_tokens, settings.chunk_overlap)):
            all_chunks.append(Chunk(id=f"{p.name}::#{i}", text=c, meta={"source": str(p), "section": i}))

    embs = embed_texts([c.text for c in all_chunks])
    index = build_faiss(embs)
    save_chunks(all_chunks, settings.chunks_path)
    save_faiss(index, settings.faiss_path)
    print(f"Ingested {len(all_chunks)} chunks from {len(docs)} docs into {settings.faiss_path}")

if __name__ == "__main__":
    run_ingest()