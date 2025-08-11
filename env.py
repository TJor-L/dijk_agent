from __future__ import annotations
import os
from typing import Iterable, Optional
from dotenv import load_dotenv

def load_env(dotenv_path: Optional[str] = None,
             require_keys: Iterable[str] = ("OPENAI_API_KEY",)) -> None:

    load_dotenv(dotenv_path)
    missing = [k for k in require_keys if not os.getenv(k)]
    if missing:
        hint = ",".join(missing)
        raise RuntimeError(
            f"Environment variables missing: {hint}. Please create a .env file in the project root or set them in the system environment."
        )
