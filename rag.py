import os
import json
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Config & client
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise SystemExit("Missing OPENAI_API_KEY. Put it in .env")

client = OpenAI()

# -----------------------------
# Simple corpus (swap with your own)
# -----------------------------
DOCS = [
    {
        "id": "guide-embeddings",
        "title": "Embeddings guide (toy)",
        "text": (
            "Embeddings turn text into dense vectors that capture semantic meaning. "
            "They enable similarity search, clustering, and RAG retrieval. "
            "Choose smaller, cheaper models for fast retrieval and larger ones for accuracy."
        ),
    },
    {
        "id": "chunking-notes",
        "title": "Chunking notes",
        "text": (
            "Chunk text into overlapping windows so the retriever can match user queries. "
            "Common sizes are 300-800 characters with overlaps of 50-200. "
            "Keep chunks self-contained and include source metadata for citation."
        ),
    },
    {
        "id": "prompting-tips",
        "title": "Prompting tips",
        "text": (
            "When generating answers from retrieved context, instruct the model to use only the provided chunks. "
            "Ask for citations and avoid fabricating facts that are not present in the context."
        ),
    },
]

# -----------------------------
# Utilities
# -----------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0: start = 0
    return chunks

# -----------------------------
# Index types
# -----------------------------
@dataclass
class Chunk:
    doc_id: str
    title: str
    text: str

@dataclass
class Index:
    embeddings: np.ndarray        # shape: (n_chunks, dim)
    chunks: List[Chunk]           # metadata

# -----------------------------
# Embedding helpers
# -----------------------------
def embed_texts(texts: List[str]) -> np.ndarray:
    # Returns np.array of shape (len(texts), dim)
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [item.embedding for item in resp.data]
    return np.array(vecs, dtype=np.float32)

def embed_query(q: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    return np.array(resp.data[0].embedding, dtype=np.float32)

# -----------------------------
# Build index in-memory
# -----------------------------
def build_index(docs=DOCS, chunk_size=500, overlap=100) -> Index:
    all_chunks: List[Chunk] = []
    for d in docs:
        for c in chunk_text(d["text"], chunk_size, overlap):
            all_chunks.append(Chunk(doc_id=d["id"], title=d["title"], text=c))

    embeddings = embed_texts([c.text for c in all_chunks])
    return Index(embeddings=embeddings, chunks=all_chunks)

# -----------------------------
# Retrieval
# -----------------------------
def retrieve(index: Index, query: str, k: int = 4) -> List[Tuple[float, Chunk]]:
    qvec = embed_query(query)
    sims = []
    for i, vec in enumerate(index.embeddings):
        sims.append((cosine_sim(qvec, vec), index.chunks[i]))
    sims.sort(key=lambda x: x[0], reverse=True)
    return sims[:k]

# -----------------------------
# Generation
# -----------------------------
SYSTEM_PROMPT = (
    "You are a precise assistant. Use ONLY the provided context to answer. "
    "If the answer isn't in the context, say you don't know. Include brief citations like [title]."
)

def answer_with_context(query: str, context_items: List[Tuple[float, Chunk]]) -> str:
    # Build context block with simple provenance
    context_texts = []
    for score, ch in context_items:
        context_texts.append(f"[{ch.title}] {ch.text}")

    user_prompt = (
        f"Question: {query}\n\n"
        f"Context:\n" + "\n---\n".join(context_texts) + "\n\n"
        "Answer succinctly and cite sources like [title]."
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()

# -----------------------------
# Console REPL
# -----------------------------
def repl():
    print("Indexing corpus... (first run will create embeddings)")
    index = build_index()
    print(f"Indexed {len(index.chunks)} chunks. Ask questions! Type 'exit' to quit.\n")

    while True:
        q = input("Q> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        top = retrieve(index, q, k=4)
        ans = answer_with_context(q, top)

        print("\nAnswer:")
        print(ans)
        print("\nTop matches (score • title • doc_id):")
        for score, ch in top:
            print(f"  {score:.3f} • {ch.title} • {ch.doc_id}")
        print()

if __name__ == "__main__":
    repl()