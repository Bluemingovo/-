from pathlib import Path
from typing import Optional
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent


class Config(BaseModel):
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    temperature: float = 0.2
    embedding_backend: str = "hash"  # "hash" | "openai" | ...
    embedding_model: str = ""
    vector_dir: Path = BASE_DIR / "data" / "vector_store"
    docs_dir: Path = BASE_DIR / "data" / "docs"


def load_config() -> Config:
    """从环境变量读取配置；调用前请先 load_dotenv()。"""
    import os
    return Config(
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        llm_model=os.getenv("LLM_MODEL", "deepseek-chat"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "hash"),
        embedding_model=os.getenv("EMBEDDING_MODEL", ""),
        vector_dir=Path(os.getenv("VECTOR_DIR", str(BASE_DIR / "data" / "vector_store"))),
        docs_dir=Path(os.getenv("DOCS_DIR", str(BASE_DIR / "data" / "docs"))),
    )
