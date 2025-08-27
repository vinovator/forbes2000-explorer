# scripts/02_enrich_and_split.py
# Usage: python scripts/02_enrich_and_split.py
from pathlib import Path
import json
import numpy as np
import pandas as pd

# ----------------------------- config ---------------------------------
FALLBACK_IF_NO_HISTORY = True   # set to False if you prefer no timeseries when history is absent
# ----------------------------------------------------------------------

# Anchor to repo root
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "parquet"
IN = DATA_DIR / "companies_master.parquet"
OUT_DIR = DATA_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CSV mapping for country -> region
META_DIR = ROOT / "data" / "meta"
REGION_CSV = META_DIR / "country_region.csv"
MISSING_TXT = META_DIR / "_missing_countries.txt"


def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return np.where((b == 0) | (pd.isna(b)), np.nan, a / b)


def cagr_from_list(seq):
    if not isinstance(seq, list) or len(seq) < 2:
        return np.nan
    arr = [x for x in seq if x is not None]
    if len(arr) < 2:
        return np.nan
    start = float(arr[-10]) if len(arr) >= 10 else float(arr[0])
    end = float(arr[-1])
    n = max(1, len(arr) - 1)
    if start <= 0 or end <= 0:
        return np.nan
    return (end / start) ** (1 / n) - 1.0


# ---------- NEW: robust history normalization (works even if columns are missing/strings) ----------
def _to_num_list(x):
    """Return a list[float] parsed from x. Accepts list/tuple/ndarray/JSON-string; else []."""
    # already a list-like
    if isinstance(x, list):
        vals = x
    elif isinstance(x, tuple):
        vals = list(x)
    elif hasattr(x, "tolist"):
        vals = list(x.tolist())
    # JSON-ish string like "[1,2,3]"
    elif isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                vals = json.loads(s)
            except Exception:
                return []
        else:
            return []
    else:
        return []

    out = []
    for v in vals:
        try:
            if v is None:
                continue
            out.append(float(v))
        except Exception:
            # skip non-numeric junk
            pass
    return out


def ensure_history_lists(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee the 4 history columns exist and are normalized as list[float]."""
    df = df.copy()
    for c in ["revenueList", "profitList", "assetsList", "employeesList"]:
        if c in df.columns:
            df[c] = df[c].apply(_to_num_list)
        else:
            df[c] = [[]] * len(df)
    # make sure naturalId is a plain string key
    if "naturalId" in df.columns:
        df["naturalId"] = df["naturalId"].astype(str)
    return df
# ---------------------------------------------------------------------------------------------------


def add_region_via_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Left-join on country to add 'region'. Unmapped -> 'Other'. Also logs missing."""
    if not REGION_CSV.exists():
        df = df.copy()
        df["region"] = "Other"
        print(f"[warn] {REGION_CSV} not found. Set region='Other' for all rows.")
        return df

    # Read as strings and trim whitespace
    map_df = pd.read_csv(
        REGION_CSV,
        dtype={"country": "string", "region": "string"},
        keep_default_na=False  # don't turn empty strings into NaN
    )
    map_df["country"] = map_df["country"].str.strip()
    map_df["region"]  = map_df["region"].str.strip()

    df = df.copy()
    if "country" not in df.columns:
        raise KeyError("Input dataframe has no 'country' column.")
    df["country"] = df["country"].astype("string").str.strip()

    # Merge; if 'region' already exists, leave as-is (idempotent)
    if "region" in df.columns:
        base = df
    else:
        base = df.merge(map_df, on="country", how="left")
        base["region"] = base["region"].fillna("Other")

    # Log any countries present in data but missing in the mapping CSV
    present = set(base["country"].dropna().astype(str).str.strip().unique())
    known   = set(map_df["country"].dropna().astype(str).str.strip().unique())
    missing = sorted(present - known)
    if missing:
        MISSING_TXT.parent.mkdir(parents=True, exist_ok=True)
        MISSING_TXT.write_text("\n".join(missing))
        print(f"[info] Missing countries written to {MISSING_TXT} ({len(missing)})")

    return base


def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    # Core ratios
    df["profit_margin"]  = safe_div(df["profits"], df["revenue"]).astype(float)
    df["roa"]            = safe_div(df["profits"], df["assets"]).astype(float)
    df["asset_turnover"] = safe_div(df["revenue"], df["assets"]).astype(float)
    df["rev_per_emp"]    = safe_div(df["revenue"], df["employees"]).astype(float)
    df["profit_per_emp"] = safe_div(df["profits"], df["employees"]).astype(float)

    # Growth from history lists
    df["rev_cagr"]    = df.get("revenueList",  pd.Series([[]] * len(df))).apply(cagr_from_list)
    df["profit_cagr"] = df.get("profitList",   pd.Series([[]] * len(df))).apply(cagr_from_list)

    # Fundamental score (percentile ranks within industry)
    comps = ["profit_margin", "roa", "asset_turnover", "rev_cagr", "profit_cagr"]
    ranks = df.groupby("industry", dropna=False)[comps].rank(pct=True)
    df["fund_score"] = (
        0.30 * ranks["profit_margin"]
        + 0.30 * ranks["roa"]
        + 0.10 * ranks["asset_turnover"]
        + 0.15 * ranks["rev_cagr"]
        + 0.15 * ranks["profit_cagr"]
    ) * 100
    return df


def build_timeseries_long(df: pd.DataFrame) -> pd.DataFrame:
    """Build (naturalId, metric, t_index, value) from history arrays.
    If no history exists and FALLBACK_IF_NO_HISTORY is True, add a single t_index=0 point
    from the current scalar metric (revenue/profits/assets/employees)."""
    rows = []
    for _, r in df.iterrows():
        nid = str(r.get("naturalId", ""))
        if not nid:
            continue
        for metric, col in [
            ("revenue",   "revenueList"),
            ("profits",   "profitList"),
            ("assets",    "assetsList"),
            ("employees", "employeesList"),
        ]:
            seq = r[col] if col in df.columns else []
            if isinstance(seq, list) and len(seq) > 0:
                tail = seq[-10:]
                for i, val in enumerate(tail):
                    try:
                        rows.append({
                            "naturalId": nid,
                            "metric": metric,
                            "t_index": int(i),
                            "value": float(val),
                        })
                    except Exception:
                        pass
            elif FALLBACK_IF_NO_HISTORY:
                # fallback to a single current point if present
                v = r.get(metric, None)
                try:
                    if pd.notna(v):
                        rows.append({
                            "naturalId": nid,
                            "metric": metric,
                            "t_index": 0,
                            "value": float(v),
                        })
                except Exception:
                    pass
    return pd.DataFrame(rows, columns=["naturalId","metric","t_index","value"])


def slice_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["industry", "region"] if "region" in df.columns else ["industry"]

    agg = (
        df.groupby(group_cols, dropna=False)
          .agg(
              n=("naturalId", "count"),
              total_revenue=("revenue", "sum"),
              total_profits=("profits", "sum"),
              total_mktcap=("marketValue", "sum"),
              median_margin=("profit_margin", "median"),
              median_roa=("roa", "median"),
          )
          .reset_index()
    )

    # Top-5 share
    if "marketValue" in df.columns:
        df_sorted = df.sort_values("marketValue", ascending=False)
        top5 = (
            df_sorted.groupby(group_cols, dropna=False, as_index=False)
            .head(5)
            .groupby(group_cols, dropna=False, as_index=False)["marketValue"]
            .sum()
            .rename(columns={"marketValue": "top5_mv"})
        )
        total_mv = (
            df.groupby(group_cols, dropna=False, as_index=False)["marketValue"]
            .sum()
            .rename(columns={"marketValue": "total_mv"})
        )
        share = total_mv.merge(top5, on=group_cols, how="left")
        share["top5_mv_share"] = np.where(
            share["total_mv"] > 0, share["top5_mv"].fillna(0) / share["total_mv"], np.nan
        )
        agg = agg.merge(share[group_cols + ["top5_mv_share"]], on=group_cols, how="left")

    # Leaders
    rec = []
    gobj = df.groupby(group_cols, dropna=False)
    for key, g in gobj:
        def topn(col, asc=False):
            if col not in g.columns:
                return []
            d = g[["naturalId", "company", col]].dropna().sort_values(col, ascending=asc)
            d = d.head(5) if asc else d.tail(5)[::-1]
            return d.to_dict(orient="records")
        rec.append({
            **({"industry": key[0], "region": key[1]} if len(group_cols) == 2 else {"industry": key}),
            "leaders_roa":    topn("roa", asc=False),
            "laggards_roa":   topn("roa", asc=True),
            "leaders_margin": topn("profit_margin", asc=False),
            "leaders_mktcap": topn("marketValue", asc=False),
        })
    leaders_df = pd.DataFrame(rec)
    return agg, leaders_df


def topics(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception:
        return pd.DataFrame(columns=["industry", "region", "top_terms"])

    rows = []
    group_cols = ["industry", "region"] if "region" in df.columns else ["industry"]
    for key, g in df.groupby(group_cols, dropna=False):
        texts = [str(x) for x in g["description"].fillna("")] if "description" in g.columns else []
        label = {"industry": key[0], "region": key[1]} if len(group_cols) == 2 else {"industry": key}
        if not any(texts):
            rows.append({**label, "top_terms": []})
            continue
        tf = TfidfVectorizer(stop_words="english", max_features=2000)
        X = tf.fit_transform(texts)
        means = X.mean(axis=0).A1
        idx = means.argsort()[-10:][::-1]
        terms = [t for i, t in enumerate(tf.get_feature_names_out()) if i in idx]
        rows.append({**label, "top_terms": terms})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_parquet(IN)

    # Ensure history arrays are present/normalized BEFORE anything else
    df = ensure_history_lists(df)

    # Add region via mapping CSV (idempotent if region already exists)
    df = add_region_via_csv(df)

    # KPIs + write back the enriched master
    df = compute_kpis(df)
    df.to_parquet(OUT_DIR / "companies_master.parquet", index=False)

    # Time-series
    ts = build_timeseries_long(df)
    ts.to_parquet(OUT_DIR / "companies_timeseries.parquet", index=False)

    # Slice stats
    agg, leaders = slice_stats(df)
    on_cols = ["industry", "region"] if "region" in df.columns else ["industry"]
    out = agg.merge(leaders, on=on_cols, how="left")
    out.to_parquet(OUT_DIR / "slice_stats.parquet", index=False)

    # Topics
    tdf = topics(df)
    tdf.to_parquet(OUT_DIR / "slice_topics.parquet", index=False)

    print("Enriched parquet files written to", OUT_DIR)
    print("Timeseries rows:", len(ts))
