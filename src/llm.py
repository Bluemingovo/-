"""LLM 客户端（DeepSeek，OpenAI 兼容）。"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from .config import Config


def build_llm(cfg: Config) -> ChatOpenAI:
    if not cfg.api_key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY，请在 .env 中配置。")
    return ChatOpenAI(
        model=cfg.llm_model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        timeout=60,
        max_retries=2,
    )
