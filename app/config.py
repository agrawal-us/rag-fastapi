from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    index_dir: Path = Path("./.index")
    faiss_path: Path = index_dir / "index.faiss"
    chunks_path: Path = index_dir / "chunks.json"
    data_dir: Path = Path("./data")
    max_chunk_tokens: int = 600
    chunk_overlap: int = 80
    top_k: int = 10

    class Config:
        env_file = ".env"

settings = Settings()
settings.index_dir.mkdir(parents=True, exist_ok=True)