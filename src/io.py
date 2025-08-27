# src/io.py
from functools import lru_cache
from pathlib import Path
import pandas as pd
from .config import SETTINGS

@lru_cache(maxsize=1)
def load_master() -> pd.DataFrame:
    path = SETTINGS.data_dir / "companies_master.parquet"
    return pd.read_parquet(path)

@lru_cache(maxsize=1)
def load_timeseries() -> pd.DataFrame:
    path = SETTINGS.data_dir / "companies_timeseries.parquet"
    return pd.read_parquet(path)

@lru_cache(maxsize=1)
def load_slice_stats() -> pd.DataFrame:
    path = SETTINGS.data_dir / "slice_stats.parquet"
    return pd.read_parquet(path)

@lru_cache(maxsize=1)
def load_slice_topics() -> pd.DataFrame:
    path = SETTINGS.data_dir / "slice_topics.parquet"
    return pd.read_parquet(path)

def ensure_dirs() -> None:
    """Create expected folders on first run; harmless if they exist."""
    for p in [SETTINGS.data_dir, SETTINGS.models_dir, SETTINGS.prompts_dir]:
        Path(p).mkdir(parents=True, exist_ok=True)
