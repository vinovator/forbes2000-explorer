# src/llm.py
from __future__ import annotations
import json, math, os
from typing import Any, Dict, List
from .config import SETTINGS

# Try streamlit (for st.secrets on Streamlit Cloud)
try:
    import streamlit as st
except Exception:
    st = None

# Load .env locally (does nothing on Cloud)
try:
    from dotenv import load_dotenv
    load_dotenv()  # looks for .env in CWD
except Exception:
    pass

# Optional backends
try:
    import google.generativeai as genai  # Gemini
except Exception:
    genai = None

try:
    import ollama  # local model
except Exception:
    ollama = None


def load_prompt(name: str) -> str:
    path = SETTINGS.prompts_dir / name
    if not path.exists():
        return "You are a concise analyst. Use only the JSON provided."
    text = path.read_text(encoding="utf-8")
    return (
        text
        .replace("{YEAR}", str(SETTINGS.data_year))
        .replace("{CURRENCY}", SETTINGS.data_currency)
    )


def _to_jsonable(x: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python; replace non-finite floats with None."""
    import numpy as np
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    return x


def _get_google_api_key() -> str | None:
    """Prefer Streamlit secrets (Cloud), else environment var (loaded from .env locally)."""
    # Streamlit Cloud
    if st is not None:
        try:
            key = st.secrets.get("GOOGLE_API_KEY", None)
            if key:
                return str(key)
        except Exception:
            pass
    # Local env (dotenv makes this available)
    return os.getenv("GOOGLE_API_KEY")


def _gemini_generate(prompt: str) -> str:
    if SETTINGS.llm_backend != "gemini":
        return "LLM disabled."
    if genai is None:
        return "LLM disabled (google-generativeai not installed)."
    api_key = _get_google_api_key()
    if not api_key:
        return "LLM disabled (missing GOOGLE_API_KEY)."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(SETTINGS.gemini_model)
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": SETTINGS.llm_temperature,
                "max_output_tokens": SETTINGS.llm_max_tokens,
            },
        )
        return (resp.text or "").strip()
    except Exception as e:
        return f"LLM error: {e}"


def _ollama_generate(prompt: str) -> str:
    if SETTINGS.llm_backend != "ollama":
        return "LLM disabled."
    if ollama is None:
        return "LLM disabled (ollama not installed)."
    try:
        resp = ollama.generate(
            model=SETTINGS.ollama_model,
            prompt=prompt,
            options={"temperature": SETTINGS.llm_temperature, "num_predict": SETTINGS.llm_max_tokens},
        )
        return resp.get("response", "").strip()
    except Exception as e:
        return f"LLM error: {e}"


def _generate(prompt: str) -> str:
    backend = getattr(SETTINGS, "llm_backend", "gemini")
    if backend == "gemini":
        return _gemini_generate(prompt)
    if backend == "ollama":
        return _ollama_generate(prompt)
    return "LLM disabled."


def call_with_json(prompt_file: str, payload: Dict[str, Any], whitelist_numbers: List[float] | None = None) -> Any:
    template = load_prompt(prompt_file)
    payload = _to_jsonable(payload)

    body = json.dumps(payload, ensure_ascii=False)

    validator = ""
    if whitelist_numbers:
        safe_nums = []
        for n in whitelist_numbers:
            try:
                x = float(n)
                if math.isfinite(x):
                    safe_nums.append(x)
            except Exception:
                pass
        if safe_nums:
            validator = f"\n\n# Validator: Only reuse numbers from this set:\n{';'.join(map(str, safe_nums))}"

    prompt = f"{template}\n\nINPUT_JSON:\n{body}{validator}"
    text = _generate(prompt)

    # Try JSON parse; else return raw text (your callers already handle both)
    try:
        return json.loads(text)
    except Exception:
        return text
