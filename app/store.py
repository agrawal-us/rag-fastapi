import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple
import faiss
import numpy as np

@dataclass
class Chunk:
    id: str
    text: str
    meta: Dict[str, Any]

def save_chunks(chunks: List[Chunk], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([{"id": c.id, "text": c.text, "meta": c.meta} for c in chunks], f, ensure_ascii=False)

def load_chunks(path: Path) -> List[Chunk]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**d) for d in data]

def build_faiss(embs: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)
    return index

def save_faiss(index: faiss.Index, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))

def load_faiss(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))

def search(index: faiss.Index, qvec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    return index.search(qvec, k)