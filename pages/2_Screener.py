# pages/2_Screener.py
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.io import load_master
from src.llm import call_with_json
from src.config import SETTINGS

from src.ui import render_banner, render_footer

render_banner()     # shows the image

st.set_page_config(page_title="Screener", layout="wide")
st.title("Screener — Region × Country × Industry")
st.caption(f"All figures from Forbes Global 2000 ({SETTINGS.data_year}); {SETTINGS.data_currency}.")

# ---- Load master once
@st.cache_data(show_spinner=False)
def get_master():
    return load_master()

df = get_master().copy()

# ---- Filters (main screen, not sidebar)
st.subheader("Filters")

# Regions
region_opts = sorted(df["region"].dropna().astype(str).unique().tolist()) if "region" in df.columns else []
sel_regions = st.multiselect("Region", options=region_opts, default=[])

# Countries depend on Regions (if any)
df_reg = df[df["region"].isin(sel_regions)] if sel_regions and "region" in df.columns else df
country_opts = sorted(df_reg["country"].dropna().astype(str).unique().tolist()) if "country" in df_reg.columns else []
sel_countries = st.multiselect("Country", options=country_opts, default=[])

# Industries depend on Region+Country (if any)
df_rc = df_reg[df_reg["country"].isin(sel_countries)] if sel_countries and "country" in df_reg.columns else df_reg
industry_opts = sorted(df_rc["industry"].dropna().astype(str).unique().tolist()) if "industry" in df_rc.columns else []
sel_industries = st.multiselect("Industry", options=industry_opts, default=[])

# Value guard(s)
rev_min = st.number_input("Revenue ≥ (USD Millions)", value=0, step=100)

# Commentary style
mode = st.radio("Commentary mode", ["concise", "extended"], index=0, horizontal=True)

st.divider()

# ---- Apply filters
f = df.copy()

if sel_regions and "region" in f.columns:
    f = f[f["region"].isin(sel_regions)]

if sel_countries and "country" in f.columns:
    f = f[f["country"].isin(sel_countries)]

if sel_industries and "industry" in f.columns:
    f = f[f["industry"].isin(sel_industries)]

# numeric guard
if "revenue" in f.columns:
    f = f[f["revenue"].fillna(0) >= float(rev_min)]

# ---- KPI cards
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("# Companies", int(len(f)))

median_margin = float(f["profit_margin"].median()) if "profit_margin" in f.columns and len(f) else np.nan
median_roa = float(f["roa"].median()) if "roa" in f.columns and len(f) else np.nan
c2.metric("Median Margin", f"{median_margin*100:.1f}%" if pd.notna(median_margin) else "—")
c3.metric("Median ROA", f"{median_roa*100:.1f}%" if pd.notna(median_roa) else "—")

mv_total = float(f["marketValue"].sum()) if "marketValue" in f.columns and len(f) else 0.0
c4.metric("Total Market Value (Bn)", f"{mv_total/1e3:.1f}")

def top5_share(g: pd.DataFrame) -> float:
    if g.empty or "marketValue" not in g.columns:
        return np.nan
    s = g["marketValue"].dropna().to_numpy()
    if s.size == 0 or s.sum() <= 0:
        return np.nan
    s.sort()
    return float(s[-5:].sum() / s.sum())

mv_share = top5_share(f)
c5.metric("Top-5 MV share", f"{mv_share*100:.0f}%" if pd.notna(mv_share) else "—")

# ---- Results table
view_cols = [c for c in ["company", "region", "country", "industry", "revenue", "profits", "marketValue", "employees", "fund_score"] if c in f.columns]
st.dataframe(
    f[view_cols],
    use_container_width=True,
    height=420,
    column_config={
        "revenue": st.column_config.NumberColumn("Revenue (USD M)", format="%,.0f"),
        "profits": st.column_config.NumberColumn("Profits (USD M)", format="%,.0f"),
        "marketValue": st.column_config.NumberColumn("Market Value (USD M)", format="%,.0f"),
        "employees": st.column_config.NumberColumn("Employees", format="%,.0f"),
        "fund_score": st.column_config.NumberColumn("Fundamental Score (0–100)", format="%.1f"),
    },
)

# ---- Optional quick chart: top 10 by market cap in the slice
if not f.empty and "marketValue" in f.columns:
    top10 = f.sort_values("marketValue", ascending=False).head(10)
    if not top10.empty:
        st.plotly_chart(
            px.bar(top10, x="company", y="marketValue", title="Top 10 Market Value (USD M)"),
            use_container_width=True,
        )

st.divider()

# ---- Sector commentary (LLM)
st.subheader("Sector commentary")

def topn(df_, col, n=5, asc=False):
    if col not in df_.columns or df_.empty:
        return []
    d = df_[["company", col]].dropna().sort_values(col, ascending=asc)
    d = d.head(n) if asc else d.tail(n)[::-1]
    out = []
    for _, r in d.iterrows():
        try:
            out.append({"company": str(r["company"]), "value": float(r[col])})
        except Exception:
            continue
    return out

def nice_money_m(v):
    if v is None or not np.isfinite(v):
        return "n/a"
    v = float(v)
    if v >= 1000:
        return f"${v/1e3:.1f}B"
    return f"${v:.1f}M"

def pct1(x):
    return f"{x*100:.1f}%" if x is not None and np.isfinite(x) else "n/a"

# Slice label uses current selections
lab_regions = " + ".join(sel_regions) if sel_regions else "All Regions"
lab_countries = ", ".join(sel_countries) if sel_countries else "All Countries"
lab_inds = ", ".join(sel_industries) if sel_industries else "All Industries"

# Build payload (what the LLM sees)
payload = {
    "slice_label": f"{lab_regions} × {lab_countries} × {lab_inds}",
    "mode": mode,
    "metrics": {
        "n": int(len(f)),
        "total_revenue": float(f["revenue"].sum()) if "revenue" in f.columns else 0.0,
        "total_profits": float(f["profits"].sum()) if "profits" in f.columns else 0.0,
        "total_mktcap": mv_total,
        "median_margin": float(median_margin) if pd.notna(median_margin) else None,
        "median_roa": float(median_roa) if pd.notna(median_roa) else None,
        "top5_mv_share": float(mv_share) if pd.notna(mv_share) else None,
    },
    "leaders": {
        "roa": topn(f, "roa", asc=False),
        "margin": topn(f, "profit_margin", asc=False),
        "mktcap": topn(f, "marketValue", asc=False),
    },
    "topics": [],
}

# Numeric-echo validator list
nums = []
for v in payload["metrics"].values():
    if isinstance(v, (int, float)) and v is not None:
        nums.append(v)
for arr in payload["leaders"].values():
    for item in arr:
        try:
            nums.append(float(item["value"]))
        except Exception:
            pass

def fallback_bullets(p):
    m = p["metrics"]
    L = p["leaders"]
    bullets = []
    bullets.append(
        f"{p['slice_label']}: {m['n']} companies; "
        f"revenue {nice_money_m(m['total_revenue'])}, "
        f"profits {nice_money_m(m['total_profits'])}, "
        f"market cap {nice_money_m(m['total_mktcap'])}."
    )
    if m.get("median_margin") is not None or m.get("median_roa") is not None:
        bullets.append(
            f"Medians — margin {pct1(m.get('median_margin'))}, ROA {pct1(m.get('median_roa'))}."
        )
    if m.get("top5_mv_share") is not None:
        bullets.append(f"Concentration: top-5 hold {pct1(m['top5_mv_share'])} of market value.")
    top_mv = [x["company"] for x in (L.get("mktcap") or [])[:3]]
    if top_mv:
        bullets.append("Top by market value: " + ", ".join(top_mv) + ".")
    top_roa = [x["company"] for x in (L.get("roa") or [])[:3]]
    if top_roa:
        bullets.append("Best ROA: " + ", ".join(top_roa) + ".")
    top_margin = [x["company"] for x in (L.get("margin") or [])[:3]]
    if top_margin:
        bullets.append("Best margins: " + ", ".join(top_margin) + ".")
    return bullets[:6]

def render_llm_output(out_obj):
    # 1) If model returned a clean list of bullets
    if isinstance(out_obj, list):
        return [str(x).strip() for x in out_obj if str(x).strip()]

    # 2) If model returned a dict or JSON string → treat as “echoed input”; use fallback
    try:
        if isinstance(out_obj, str) and (out_obj.strip().startswith("{") or out_obj.strip().startswith("[")):
            return fallback_bullets(payload)
        if isinstance(out_obj, dict):
            return fallback_bullets(payload)
        # Try to parse JSON inside a string
        import json, re
        if isinstance(out_obj, str):
            try:
                parsed = json.loads(out_obj)
                if isinstance(parsed, (dict, list)):
                    return fallback_bullets(payload)
            except Exception:
                # Try to locate a JSON block inside
                m = re.search(r"(\{.*\}|\[.*\])", out_obj, flags=re.S)
                if m:
                    try:
                        parsed = json.loads(m.group(1))
                        if isinstance(parsed, (dict, list)):
                            return fallback_bullets(payload)
                    except Exception:
                        pass
    except Exception:
        pass

    # 3) Plain text → split into tidy bullets
    s = str(out_obj).strip()
    lines = [ln.strip(" -•") for ln in s.splitlines() if ln.strip()]
    return lines if lines else fallback_bullets(payload)

if st.button("Generate commentary"):
    with st.spinner("Summarizing slice…"):
        out = call_with_json("sector_commentary.md", payload, nums)

    lines = render_llm_output(out)
    if lines:
        st.markdown("\n".join(f"- {ln}" for ln in lines))
    else:
        st.info("No commentary returned.")

render_footer()