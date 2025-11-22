from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()


@dataclass(kw_only=True)
class Config:
    host: str = "localhost"
    port: int = 8000
    openai_llm_model: str = "gpt-4o"
    _raw = os.getenv("OPENAI_API_KEY")
    openai_api_key: Optional[SecretStr] = SecretStr(_raw) if _raw else None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"
