from __future__ import annotations

import os
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()


class Settings:
    """Default values for the example configuration.

    Place API keys in the .env file beforehand.
    """

    RAGGIFY_BASE_URL: str = "http://localhost:8000/v1"
    OPENAI_LLM_MODEL: str = "gpt-4-turbo"
    _raw = os.getenv("OPENAI_API_KEY")
    OPENAI_API_KEY: Optional[SecretStr] = SecretStr(_raw) if _raw else None
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"
