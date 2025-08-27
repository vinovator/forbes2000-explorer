# Usage: python scripts/01_fetch_normalize.py --year 2025 --outdir data/parquet
import argparse
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]

def fetch_forbes(year: int) -> pd.DataFrame:
    url = f"https://www.forbes.com/forbesapi/org/global2000/{year}/position/true.json?limit=2000"
    headers = {"accept": "application/json, text/plain, */*", "user-agent": "Mozilla/5.0"}
    cookies = {"notice_behavior": "expressed,eu", "notice_gdpr_prefs": "0,1,2:1a8b5228dd7ff0717196863a5d28ce6c"}
    r = requests.get(url, headers=headers, cookies=cookies, timeout=60)
    r.raise_for_status()
    data = r.json()["organizationList"]["organizationsLists"]
    return pd.json_normalize(data).sort_values("position")

def _norm(u):
    if not isinstance(u, str): return None
    s = u.strip()
    if not s: return None
    if s.startswith("//"): s = "https:" + s
    s = s.replace("http://http:", "http:").replace("https://http:", "http:")
    if not (s.startswith("http://") or s.startswith("https://")):
        s = "https://" + s.lstrip("/")
    return s

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # Rename a few keys; keep 'region' and 'country' exactly as provided by Forbes
    df = df.rename(columns={
        "organizationName": "company",
        "ceoName": "ceo",
        "ceoTitle": "ceoTitle",
        "webSite": "website",
        "description": "description",
        "naturalId": "naturalId",
    })

    # Normalize common URL fields (optional but harmless)
    for col in ["image","squareImage","portraitImage","landscapeImage",
                "organization.image","organization.squareImage","website","webSite"]:
        if col in df.columns:
            df[col] = df[col].apply(_norm)

    # Numerics
    for c in ["revenue","profits","assets","marketValue","employees",
              "profitsRank","assetsRank","marketValueRank","revenueRank",
              "position","rank","year","month"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Lists
    for c in ["revenueList","profitList","assetsList","employeesList"]:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: x if isinstance(x, list) else [])
        else:
            df[c] = [[]] * len(df)

    # Strings
    for c in ["naturalId","company","industry","country","region","city","ceo","ceoTitle","website","description"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    return df

def resolve(path_like: str | Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (ROOT / p)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--outdir", default="data/parquet")
    args = ap.parse_args()

    outdir = resolve(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = fetch_forbes(args.year)
    df = coerce_types(df).reset_index(drop=True)

    out_path = outdir / "companies_master.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Wrote: {out_path}  rows: {len(df)}  year: {args.year}")
