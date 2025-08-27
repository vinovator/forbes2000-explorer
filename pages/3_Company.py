# pages/3_Company.py
import math
import json, re
import pandas as pd
import plotly.express as px
import streamlit as st

from src.io import load_master, load_timeseries
from src.llm import call_with_json
from src.config import SETTINGS

from src.ui import render_banner, render_footer

render_banner()     # shows the image

st.set_page_config(page_title="Company", layout="wide")
st.title("Company Profile")

# ---------- helpers ----------
def get_query_params():
    if hasattr(st, "query_params"):
        qp = st.query_params
        return {k: (v[0] if isinstance(v, list) else v) for k, v in qp.items()}
    return {k: (v[0] if isinstance(v, list) else v) for k, v in st.experimental_get_query_params().items()}

def set_query_params(**kwargs):
    try:
        st.query_params.update(kwargs)
    except Exception:
        st.experimental_set_query_params(**kwargs)

def safe_div(a, b):
    try:
        a = float(a); b = float(b)
        if b == 0: return None
        return a / b
    except Exception:
        return None

def cagr_from_list(seq):
    """CAGR from a list (uses last 10 if longer). Returns float or None."""
    if not isinstance(seq, list) or len(seq) < 2:
        return None
    tail = [x for x in seq[-10:] if x is not None]
    if len(tail) < 2: return None
    start, end = float(tail[0]), float(tail[-1])
    n = len(tail) - 1
    if start <= 0 or end <= 0 or n <= 0:
        return None
    return (end / start) ** (1 / n) - 1.0

def fmt_pct(x, digits=1):
    return f"{x:.{digits}%}" if x is not None and not math.isnan(x) else "—"

def fmt_num(x):
    return f"{x:,.0f}" if x is not None and not math.isnan(x) else "—"

def mk_link(url: str | None):
    if isinstance(url, str) and url.strip():
        st.link_button("Website", url)

# ---------- load data ----------
df = load_master().copy()
try:
    ts = load_timeseries()
except Exception:
    ts = None

# ensure essential columns exist to avoid KeyError
for col in ["fund_score", "profit_margin", "roa", "asset_turnover", "rev_per_emp", "profit_per_emp", "rev_cagr", "profit_cagr", "region"]:
    if col not in df.columns:
        df[col] = pd.NA

params = get_query_params()
# Accept both keys; Home currently uses "naturalId"
nid = params.get("nid") or params.get("naturalId")

# ---------- fallback UX if nid missing ----------
if not nid or nid not in set(df["naturalId"]):
    st.info("Pick a company to view its profile.")
    choice = st.selectbox("Company", df["company"].dropna().astype(str).sort_values().tolist())
    open_btn = st.button("Open profile", type="primary")
    if open_btn and choice:
        nid_val = df.loc[df["company"] == choice, "naturalId"].iloc[0]
        set_query_params(nid=nid_val)
        st.rerun()
    st.stop()

# ---------- fetch the selected row ----------
row = df.loc[df["naturalId"] == nid].iloc[0]

# ---------- header ----------
left, right = st.columns([0.70, 0.30])
with left:
    st.subheader(row.get("company", "—"))
    bits = []
    if isinstance(row.get("city"), str) and row["city"]:
        bits.append(row["city"])
    if isinstance(row.get("country"), str) and row["country"]:
        bits.append(row["country"])
    if isinstance(row.get("industry"), str) and row["industry"]:
        bits.append(row["industry"])
    if bits:
        st.caption(" • ".join(bits))
    mk_link(row.get("website"))
with right:
    img = row.get("squareImage") or row.get("image")
    if isinstance(img, str) and img.strip():
        st.image(img, use_container_width=True)

# ---------- KPI tiles ----------
pm  = safe_div(row.get("profits"), row.get("revenue"))
roa = safe_div(row.get("profits"), row.get("assets"))
at  = safe_div(row.get("revenue"), row.get("assets"))

# Data is USD **millions**. Convert per-employee to USD for display.
rpe_musd = safe_div(row.get("revenue"), row.get("employees"))   # million USD / employee
ppe_musd = safe_div(row.get("profits"), row.get("employees"))   # million USD / employee

def to_usd(x_musd):
    return (x_musd * 1_000_000) if (x_musd is not None and not math.isnan(x_musd)) else None

rpe_usd = to_usd(rpe_musd)
ppe_usd = to_usd(ppe_musd)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Profit Margin", fmt_pct(pm, 1))
k2.metric("ROA", fmt_pct(roa, 2))
k3.metric("Asset Turnover", f"{at:.2f}" if at is not None and not math.isnan(at) else "—")
k4.metric("Rev / Employee (USD)", f"${fmt_num(rpe_usd)}")
k5.metric("Profit / Employee (USD)", f"${fmt_num(ppe_usd)}")

# ---------- description ----------
if isinstance(row.get("description"), str) and row["description"].strip():
    st.markdown("### Description")
    st.write(row["description"])

st.divider()

# ---------- trends (sparklines + simple CAGR labels) ----------
st.markdown("### Trends (10-year sparklines)")
grid = st.columns(2)

def plot_metric(metric_key: str, title: str, col):
    # Guard: no ts, or required cols missing
    if ts is None or getattr(ts, "empty", True):
        return
    required = {"naturalId", "metric", "t_index", "value"}
    if not required.issubset(set(ts.columns)):
        return

    sub = ts[(ts["naturalId"] == nid) & (ts["metric"] == metric_key)].sort_values("t_index")
    if sub.empty:
        return

    # compute simple CAGR on values
    seq = sub["value"].tolist()
    cg = cagr_from_list(seq)
    subtitle = f" (CAGR {fmt_pct(cg, 1)})" if cg is not None else ""

    fig = px.line(sub, x="t_index", y="value", markers=False, title=f"{title}{subtitle}")
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=35, b=10))
    col.plotly_chart(fig, use_container_width=True)

# ✅ actually render the four sparklines
with grid[0]:
    plot_metric("revenue", "Revenue", grid[0])
    plot_metric("assets", "Assets", grid[0])
with grid[1]:
    plot_metric("profits", "Profits", grid[1])
    plot_metric("employees", "Employees", grid[1])

st.divider()

# ---------- AI Brief (LLM) ----------
st.markdown("### AI Brief")

def money_fmt_musd(x):
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
    except Exception:
        return None
    return f"${v/1000:.1f}B" if v >= 1000 else f"${v:.1f}M"

if st.button("Generate AI brief"):
    with st.spinner("Generating AI brief…"):
        industry = row.get("industry") or "Unknown"
        region = row.get("region") or "Other"

        # Percentiles (robust to missing keys)
        def pct_within(group_col: str, series_name: str):
            try:
                ranks = df.groupby(group_col, dropna=False)[series_name].rank(pct=True)
                val = ranks.loc[row.name]
                return float(val) if pd.notna(val) else None
            except Exception:
                return None

        percentiles_industry = {
            "profit_margin": pct_within("industry", "profit_margin"),
            "roa": pct_within("industry", "roa"),
            "asset_turnover": pct_within("industry", "asset_turnover"),
            "fund_score": pct_within("industry", "fund_score"),
        }
        percentiles_region = {
            "profit_margin": pct_within("region", "profit_margin"),
            "roa": pct_within("region", "roa"),
            "asset_turnover": pct_within("region", "asset_turnover"),
            "fund_score": pct_within("region", "fund_score"),
        }

        # On-screen metrics: keep per-employee in USD only
        pm  = safe_div(row.get("profits"), row.get("revenue"))
        roa = safe_div(row.get("profits"), row.get("assets"))
        at  = safe_div(row.get("revenue"), row.get("assets"))
        rpe_usd = ppe_usd = None
        if row.get("employees"):
            rpe_usd = safe_div(row.get("revenue"), row.get("employees"))
            ppe_usd = safe_div(row.get("profits"), row.get("employees"))
            if rpe_usd is not None: rpe_usd *= 1_000_000
            if ppe_usd is not None: ppe_usd *= 1_000_000

        metrics_payload = {
            "revenue": row.get("revenue"),
            "profits": row.get("profits"),
            "assets": row.get("assets"),
            "market_value": row.get("marketValue"),
            "employees": row.get("employees"),
            "profit_margin": pm,
            "roa": roa,
            "asset_turnover": at,
            "rev_per_employee_usd": rpe_usd,
            "profit_per_employee_usd": ppe_usd,
            "rev_cagr": row.get("rev_cagr"),
            "profit_cagr": row.get("profit_cagr"),
            "fund_score": row.get("fund_score"),
            "rank": row.get("rank") or row.get("position"),
        }

        payload = {
            "company": row.get("company"),
            "industry": industry,
            "region": region,
            "metrics": metrics_payload,
            "percentiles": {
                "within_industry": percentiles_industry,
                "within_region": percentiles_region,
            },
            "notes": "Use only the values provided here; do not invent numbers.",
        }

        # numeric-echo for call_with_json
        def collect_numbers(obj):
            out = []
            if isinstance(obj, dict):
                for v in obj.values(): out.extend(collect_numbers(v))
            elif isinstance(obj, (list, tuple)):
                for v in obj: out.extend(collect_numbers(v))
            else:
                try:
                    x = float(obj)
                    if math.isfinite(x): out.append(x)
                except Exception:
                    pass
            return out

        nums = sorted(collect_numbers(payload))
        out = call_with_json("company_brief.md", payload, nums)

        # ---- Render: sanitize to plain sentences, then bulletize ----
        def sanitize(s: str) -> str:
            s = s.replace("∗", "*")
            s = re.sub(r"[{}()\[\]“”\"`]+", "", s)          # remove braces/quotes
            s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)     # zero-width chars
            s = re.sub(r"\s+", " ", s).strip()
            return s

        def flatten_to_text(x) -> str:
            if isinstance(x, list):   return " ".join(str(i) for i in x)
            if isinstance(x, dict):
                for k in ("bullets","points","lines"):
                    v = x.get(k)
                    if isinstance(v, list): return " ".join(str(i) for i in v)
                return " ".join(f"{k}: {v}" for k, v in x.items())
            return str(x)

        text = sanitize(flatten_to_text(out))
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
        bullets = parts[:6] if parts else []

        if not bullets:
            st.info("AI brief not available.")
        else:
            for b in bullets:
                st.markdown(f"- {b}")
else:
    st.caption("Click the button to generate a short AI summary of this company.")

render_footer()
