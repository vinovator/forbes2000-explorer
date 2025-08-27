# app.py
import streamlit as st

st.set_page_config(page_title="Forbes 2000 Explorer", layout="wide")

if __name__ == "__main__":
    try:
        st.switch_page("pages/1_Home.py")
    except Exception:
        st.title("Forbes 2000 Explorer")
        st.page_link("pages/1_Home.py", label="Home", icon="🏠")
