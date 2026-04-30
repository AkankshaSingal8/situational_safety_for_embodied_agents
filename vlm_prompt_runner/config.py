"""
vlm_prompt_runner/config.py

Load API keys from .env or .envfile at import time.
Searches for .env / .envfile in the project root (two levels up from this file).
Does NOT override variables already set in the environment.
"""
from pathlib import Path


def load_env(env_path: str | None = None) -> None:
    """Load key=value pairs from env_path into os.environ (no-override)."""
    try:
        from dotenv import load_dotenv
        _use_dotenv = True
    except ImportError:
        _use_dotenv = False

    candidates = (
        [Path(env_path)] if env_path
        else [
            Path(__file__).resolve().parents[1] / ".env",
            Path(__file__).resolve().parents[1] / ".envfile",
        ]
    )

    for path in candidates:
        if path.exists():
            if _use_dotenv:
                load_dotenv(dotenv_path=path, override=False)
            else:
                _manual_load(path)
            break


def _manual_load(path: Path) -> None:
    import os
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


# Load automatically on import so backends pick up keys without extra wiring.
load_env()
