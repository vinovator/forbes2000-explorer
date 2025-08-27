# Usage: python scripts/03_build_embeddings.py
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"          # don't import TensorFlow at all
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1" # avoid torchvision image ops path

import faiss, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Anchor to repo root
ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "parquet" / "companies_master.parquet"
OUT_DIR = ROOT / "models" / "faiss"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME="BAAI/bge-small-en-v1.5"

df=pd.read_parquet(PARQUET)

# Stable ordering for embeddings + lookup
df = df.sort_values("naturalId").reset_index(drop=True)

def make_text(r):
    parts=[str(r.get("company","")), str(r.get("industry","")), str(r.get("country","")), "::", str(r.get("description",""))]
    return " ".join([p.strip() for p in parts if p])

model=SentenceTransformer(MODEL_NAME)
texts=df.apply(make_text, axis=1).tolist()
emb=np.asarray(model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True), dtype="float32")

index=faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
faiss.write_index(index, str(OUT_DIR/"desc.index"))

df[["naturalId","company","industry","country"]].to_parquet(OUT_DIR/"desc_lookup.parquet", index=False)
print("Built FAISS:", OUT_DIR/"desc.index", "vectors:", emb.shape[0])
