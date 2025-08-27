# pages/1_Home.py
import pandas as pd
import streamlit as st

from src.io import load_master
from src.config import SETTINGS

from src.ui import render_banner, render_footer

render_banner()     # shows the image

st.set_page_config(page_title="Home", layout="wide")
st.title("Forbes 2000 Explorer — Home")
st.caption(f"All figures from Forbes Global 2000 ({SETTINGS.data_year}); {SETTINGS.data_currency}.")

# ---- Load data
@st.cache_data(show_spinner=False)
def get_master():
    return load_master()

df = get_master().copy()

# ---- Controls
q = st.text_input("Search (company / country / industry / description)", "")

regions = sorted(df["region"].dropna().astype(str).unique().tolist()) if "region" in df.columns else []
countries = sorted(df["country"].dropna().astype(str).unique().tolist()) if "country" in df.columns else []
industries = sorted(df["industry"].dropna().astype(str).unique().tolist()) if "industry" in df.columns else []

sel_regions = st.multiselect("Region", options=regions, default=regions if regions else [])
sel_countries = st.multiselect("Country", options=countries)
sel_industries = st.multiselect("Industry", options=industries)

# Visible columns (we’ll add an "Open" link column ourselves)
default_cols = [
    "company", "country", "region", "industry",
    "position", "revenue", "profits", "marketValue", "employees", "fund_score",
]
default_cols = [c for c in default_cols if c in df.columns]
show_cols = st.multiselect("Columns to show", options=sorted(df.columns), default=default_cols)

# Sort
sort_col = st.selectbox(
    "Sort by",
    options=show_cols,
    index=show_cols.index("position") if "position" in show_cols else 0
)
sort_asc = st.toggle("Ascending", value=True)

# ---- Filtering logic
f = df

if q:
    s = q.strip().lower()
    mask = pd.Series(False, index=f.index)
    for col in ["company", "industry", "country", "description"]:
        if col in f.columns:
            mask |= f[col].fillna("").astype(str).str.lower().str.contains(s, na=False)
    f = f[mask]

if sel_regions and "region" in f.columns:
    f = f[f["region"].isin(sel_regions)]

if sel_countries and "country" in f.columns:
    f = f[f["country"].isin(sel_countries)]

if sel_industries and "industry" in f.columns:
    f = f[f["industry"].isin(sel_industries)]

# ---- Sort
if sort_col in f.columns:
    f = f.sort_values(sort_col, ascending=sort_asc, kind="mergesort")

# ---- Build view + prepend "Open" link column (no header)
view = f[show_cols].copy()

# Build a URL per row to the Company page; 3_Company.py accepts ?naturalId=…
if "naturalId" in f.columns:
    open_urls = f["naturalId"].astype(str).map(lambda nid: f"./Company?naturalId={nid}")
    view.insert(0, "Open", open_urls)  # first column
else:
    view.insert(0, "Open", "")  # graceful fallback

# ---- Column formatting
col_config = {}

# Clickable link column: empty header
col_config["Open"] = st.column_config.LinkColumn(
    label="",
    help="Open company profile",
    display_text="Open",
)

if "position" in view.columns:
    col_config["position"] = st.column_config.NumberColumn("Rank", help="Forbes Global 2000 rank (1 is best)", format="%d")
if "revenue" in view.columns:
    col_config["revenue"] = st.column_config.NumberColumn("Revenue (USD)", format="%,.0f")
if "profits" in view.columns:
    col_config["profits"] = st.column_config.NumberColumn("Profits (USD)", format="%,.0f")
if "marketValue" in view.columns:
    col_config["marketValue"] = st.column_config.NumberColumn("Market Value (USD)", format="%,.0f")
if "assets" in view.columns:
    col_config["assets"] = st.column_config.NumberColumn("Assets (USD)", format="%,.0f")
if "employees" in view.columns:
    col_config["employees"] = st.column_config.NumberColumn("Employees", format="%,.0f")
if "fund_score" in view.columns:
    col_config["fund_score"] = st.column_config.NumberColumn("Fundamental Score (0–100)", format="%.1f")

# ---- Export current slice (CSV) — exclude the "Open" link
csv_bytes = f[show_cols].to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Export current slice (CSV)", data=csv_bytes, file_name="forbes2000_slice.csv", mime="text/csv")

st.write(f"Showing {len(view):,} rows")

# ---- Scrollable, interactive grid
st.dataframe(
    view,
    use_container_width=True,
    hide_index=True,
    column_config=col_config,
    height=560,  # scrolls instead of rendering all 2000 rows
)

render_footer()