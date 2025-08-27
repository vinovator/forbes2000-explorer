# ui.py
from __future__ import annotations
from pathlib import Path
import streamlit as st

# Optional: read from your SETTINGS if available, else fall back to a sensible default.
try:
    from src.config import SETTINGS
    _DEFAULT_YEAR = getattr(SETTINGS, "data_year", "")
    _DEFAULT_CCY  = getattr(SETTINGS, "data_currency", "USD")
    _BANNER_CFG   = getattr(SETTINGS, "brand_banner", "img/banner.png")
    _FOOTER_NOTE  = getattr(SETTINGS, "footer_note", None)
except Exception:
    _DEFAULT_YEAR = ""
    _DEFAULT_CCY  = "USD"
    _BANNER_CFG   = "img/banner.png"
    _FOOTER_NOTE  = None

def _banner_source() -> str | None:
    """
    Returns a valid image source or None if not found.
    Accepts: http(s) URLs, local files (Path), or a string path.
    """
    src = _BANNER_CFG
    if not src:
        return None
    if isinstance(src, str) and (src.startswith("http://") or src.startswith("https://")):
        return src
    p = Path(str(src))
    return str(p) if p.exists() else None

def render_banner(caption: bool = False) -> None:
    """
    Renders a full-width banner image at the top of the page.
    Set caption=True if you want a tiny line under the image.
    """
    src = _banner_source()
    if src:
        st.image(src, use_container_width=True)
        if caption and _DEFAULT_YEAR:
            st.caption(f"All figures from Forbes Global 2000 ({_DEFAULT_YEAR}); {_DEFAULT_CCY}.")
    else:
        # If no image available, do nothing (keeps things clean).
        pass

def render_footer() -> None:
    """
    Subtle, grey, footnote-style footer. Keep it small and out of the way.
    """
    note = (
        _FOOTER_NOTE
        if _FOOTER_NOTE
        else (f"All figures from Forbes Global 2000 ({_DEFAULT_YEAR}); {_DEFAULT_CCY}."
              if _DEFAULT_YEAR else "All figures from Forbes Global 2000.")
    )
    st.markdown(
        f"""
        <div style="
            margin-top: 1rem;
            padding-top: .5rem;
            border-top: 1px solid rgba(0,0,0,.08);
            color: #6b7280;       /* Tailwind gray-500 */
            font-size: 12px;      /* small, footnote-like */
            line-height: 1.3;
        ">
            {note}
        </div>
        """,
        unsafe_allow_html=True,
    )
