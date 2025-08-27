# Forbes 2000 Explorer

A fast, local-first Streamlit app to explore the **Forbes Global 2000** universe.  
Filter by country/industry, screen slices, open a company profile with KPIs & 10-year trends, and (optionally) generate short AI commentary.

> **Units:** unless noted otherwise, money fields are **USD millions**. Per-employee metrics are shown in **USD**.

---

## ✨ Features

- **Home** — searchable, filterable table with a one-click **Open** link per company (first column), native scrolling.
- **Screener** — slice by **Region × Country × Industry**, quick KPIs & top-10 chart, optional AI sector commentary.
- **Company** — header, KPI tiles (margin, ROA, asset turnover, *per-employee in USD*), 10-year sparklines, optional AI brief.
- **Compare** — pick up to *N* companies (configurable) for side-by-side charts and a readable head-to-head summary.
- **LLM backend switch** — **Gemini 1.5 Flash** (default; cloud-friendly) or **Ollama** (local) via settings/env.
- **Local embeddings** (FAISS) for quick semantic lookups (optional).
- Clear separation of **data build scripts** vs **UI pages**.

---

## 🗂 Repository layout

```

forbes2000-explorer/
├─ app.py                       # Entry point; immediately redirects to Home
├─ pages/
│  ├─ 1\_Home.py                 # Open link column + scrollable grid
│  ├─ 2\_Screener.py             # Region/Country/Industry filters, KPIs, commentary
│  ├─ 3\_Company.py              # KPIs, sparklines, AI brief button
│  └─ 4\_Compare.py              # Up to N companies + summary
├─ src/
│  ├─ config.py                 # SETTINGS (paths, year, LLM backend, branding)
│  ├─ io.py                     # Cached parquet readers
│  └─ llm.py                    # Gemini/Ollama routing + prompt loading
├─ scripts/
│  ├─ 01\_fetch\_normalize.py     # Fetch Forbes JSON → normalized master parquet
│  ├─ 02\_enrich\_and\_split.py    # KPIs, region join, time-series, slice stats/topics
│  └─ 03\_build\_embeddings.py    # SentenceTransformers → FAISS index
├─ data/
│  ├─ parquet/                  # Generated parquet outputs
│  └─ meta/
│     └─ country\_region.csv     # Country→Region mapping (header: country,region)
├─ models/
│  └─ faiss/                    # FAISS index + lookup parquet
├─ prompts/
│  ├─ company\_brief.md
│  ├─ sector\_commentary.md
│  └─ compare\_commentary.md
├─ img/
│  └─ banner.png                # (optional) header image if you use it
├─ requirements.txt
├─ LICENSE
└─ README.md

```

---

## 🚀 Quick start

### 1) Environment

```bash
# Python 3.10–3.12 recommended
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
````

> If you hit NumPy 2.x binary issues, use: `pip install "numpy<2"`.

### 2) Configure (local `.env` or Streamlit secrets)

This app reads config from `src/config.py` (defaults) and environment variables.
For local runs, create a `.env` at the repo root (do **not** commit secrets):

```env
# Paths (optional; defaults are fine)
DATA_DIR=data/parquet
FAISS_DIR=models/faiss
PROMPTS_DIR=prompts

# Data meta
DATA_YEAR=2025
DATA_CURRENCY=USD

# LLM backend: gemini (default) | ollama | disabled
LLM_BACKEND=gemini
GEMINI_MODEL=gemini-1.5-flash-latest
OLLAMA_MODEL=llama3.1:8b

# Model controls
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=400

# Compare page
COMPARE_LIMIT=5
DEFAULT_SAME_INDUSTRY=true

# Gemini API key (local only; on Streamlit use Secrets)
GOOGLE_API_KEY=your_key_here
```

On **Streamlit Cloud**, set **Secrets** instead of `.env`:

* Add `GOOGLE_API_KEY` under **App secrets**.

### 3) Build data (one-time per year/update)

```bash
# 1) Fetch & normalize
python scripts/01_fetch_normalize.py --year 2025 --outdir data/parquet

# 2) Enrich (KPIs), add Region via CSV, build time-series & slice outputs
python scripts/02_enrich_and_split.py

# 3) (Optional) Build embeddings index for semantic search
python scripts/03_build_embeddings.py
```

**Region mapping CSV** must be a simple two-column file:

```csv
country,region
United States,Americas
United Kingdom,Europe
India,APAC
...
```

### 4) Run

```bash
streamlit run app.py
```

`app.py` immediately redirects to the **Home** page so the “app” tab isn’t exposed.

---

## ⚙️ Configuration

`src/config.py` (`Settings`) controls:

* Paths: `data_dir`, `models_dir`, `prompts_dir`
* Data meta: `data_year`, `data_currency`
* LLM: `llm_backend` (`gemini`/`ollama`/`disabled`), `gemini_model`, `ollama_model`,
  `llm_temperature`, `llm_max_tokens`
* Compare defaults: `compare_limit`, `default_same_industry`

All of these can be overridden via environment variables (see `.env` example above).

---

## 🧠 AI commentary backends

The app can generate small, readable summaries using either backend:

### Gemini (default; cloud-friendly)

* Set `LLM_BACKEND=gemini`.
* Provide `GOOGLE_API_KEY` via local `.env` or Streamlit **Secrets**.
* Model: `gemini-1.5-flash-latest` (change with `GEMINI_MODEL`).

### Ollama (local)

* Install [Ollama](https://ollama.com/).
* Pull a model (e.g.):

  ```bash
  ollama pull llama3.1:8b
  ```
* Set `LLM_BACKEND=ollama` and `OLLAMA_MODEL=llama3.1:8b` (or `:8b-instruct`).
* No cloud key required.

If `LLM_BACKEND=disabled`, the app runs without AI text.

---

## 📦 Deploying to Streamlit Community Cloud

* **Main file:** `app.py`
  (it redirects to `pages/1_Home.py` so users land on Home)
* **Secrets:** set `GOOGLE_API_KEY` for Gemini.
* **Data artifacts:** commit `data/parquet/*.parquet` and `models/faiss/*`
  (the `.gitignore` in this repo keeps them **tracked**), *or* run the build scripts at startup.

---

## 🛠 Troubleshooting

**Timeseries charts are empty**

* Ensure `01_fetch_normalize.py` pulled list fields: `revenueList`, `profitList`, `assetsList`, `employeesList`.
* Re-run `02_enrich_and_split.py` to rebuild `companies_timeseries.parquet`.

**Open button doesn’t navigate**

* The table builds links using `naturalId`. Make sure `naturalId` exists in `companies_master.parquet`.

**Region join failed**

* Check `data/meta/country_region.csv` header is exactly `country,region` (no extra whitespace/quotes).

**LLM says “not available”**

* Set `LLM_BACKEND` and (for Gemini) provide `GOOGLE_API_KEY`. For Ollama, pull the model name you configured.

---

## 📄 License & attribution

* **License:** MIT (see `LICENSE`)
* **Data source:** Forbes Global 2000. Use responsibly; verify insights before making decisions.

### Acknowledgement

This project uses public data exposed via **forbes.com** endpoints to construct the dataset.
We are not affiliated with Forbes. Data is used for informational, non-commercial purposes with attribution.
