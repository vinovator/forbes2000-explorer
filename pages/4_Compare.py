# pages/4_Compare.py
import json
import math
import pandas as pd
import plotly.express as px
import streamlit as st

from src.io import load_master
from src.config import SETTINGS
from src.llm import call_with_json

from src.ui import render_banner, render_footer

render_banner()     # shows the image

st.set_page_config(page_title="Compare", layout="wide")
st.title(f"Compare up to {SETTINGS.compare_limit} companies")

df = load_master()

# --- Controls ---
same_ind = st.toggle(
    "Restrict to same industry (recommended)",
    value=SETTINGS.default_same_industry,
)

company_options = (
    df["company"]
    .dropna()
    .astype(str)
    .drop_duplicates()
    .sort_values()
    .tolist()
)
sel = st.multiselect(
    "Pick companies",
    options=company_options,
    max_selections=SETTINGS.compare_limit,
)

cmp = df[df["company"].isin(sel)].copy()

# If restricting to one industry, keep the dominant industry safely
if same_ind and len(cmp) >= 2:
    top_inds = cmp["industry"].dropna()
    if not top_inds.empty:
        ind = top_inds.mode().iloc[0]
        cmp = cmp[cmp["industry"] == ind]
        st.caption(f"Industry locked to: **{ind}**")
    else:
        st.warning("Industry is missing for some selections; can't restrict to same industry.")

if len(cmp) < 2:
    st.info("Pick at least 2 companies.")
    st.stop()

# --- Charts ---
pretty = {
    "revenue": "Revenue (USD M)",
    "profits": "Profits (USD M)",
    "marketValue": "Market Value (USD M)",
    "employees": "Employees",
}
for metric in ["revenue", "profits", "marketValue", "employees"]:
    if metric in cmp.columns:
        st.plotly_chart(
            px.bar(cmp, x="company", y=metric, title=pretty[metric]),
            use_container_width=True,
        )

# --- Efficiency snapshot ---
eff_cols = [c for c in ["company", "profit_margin", "roa", "asset_turnover"] if c in cmp.columns]
st.subheader("Efficiency snapshot")
st.dataframe(cmp[eff_cols], use_container_width=True)

# --- Head-to-head LLM brief ---
st.subheader("Head-to-head brief")

def to_num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

if st.button("Generate brief"):
    # Build minimal, clean payload (LLM prompt expects a list of companies)
    companies_payload = []
    for _, r in cmp.iterrows():
        companies_payload.append({
            "company": str(r["company"]),
            "metrics": {
                "revenue": to_num(r.get("revenue")),
                "profits": to_num(r.get("profits")),
                "marketValue": to_num(r.get("marketValue")),
            },
        })

    payload = {
        "companies": companies_payload,
        "max_items": SETTINGS.compare_limit,
    }

    # Collect only finite numbers for numeric-echo
    nums = set()
    for c in payload["companies"]:
        for v in c["metrics"].values():
            if isinstance(v, (int, float)) and v is not None and math.isfinite(v):
                nums.add(float(v))

    out = call_with_json("compare_head_to_head.md", payload, sorted(nums))

    # Render as plain markdown. Handle a few common variants defensively.
    st.subheader("Head-to-head brief")

    def show_markdown(x):
        if isinstance(x, str):
            st.markdown(x)
            return
        if isinstance(x, list):
            for line in x:
                st.markdown(f"- {str(line).strip()}")
            return
        if isinstance(x, dict):
            # If a model still returns a dict with bullets/points, render them
            for k in ("bullets", "points", "lines"):
                v = x.get(k)
                if isinstance(v, list) and v:
                    for line in v:
                        st.markdown(f"- {str(line).strip()}")
                    return
            # Fallback: stringify compactly (still not raw JSON)
            st.markdown(" • ".join(f"**{k}**: {v}" for k, v in x.items()))
            return
        st.markdown(str(x))

    show_markdown(out)

render_footer()