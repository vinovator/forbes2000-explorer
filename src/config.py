# src/config.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    data_dir: Path = ROOT / "data" / "parquet"
    models_dir: Path = ROOT / "models" / "faiss"
    prompts_dir: Path = ROOT / "prompts"

    # Data metadata
    data_year: int = 2025
    data_currency: str = "USD"

    # LLM backend selector: "gemini" | "ollama" | "disabled"
    llm_backend: str = "gemini"

    # Gemini (no key here!)
    gemini_model: str = "gemini-1.5-flash-latest"

    # Ollama
    ollama_model: str = "llama3.1:8b-instruct"

    # Common generation params
    llm_temperature: float = 0.2
    llm_max_tokens: int = 400

    # Compare defaults
    compare_limit: int = 5
    default_same_industry: bool = True
    default_same_region: bool = False

SETTINGS = Settings()
ROOT_DIR = ROOT
