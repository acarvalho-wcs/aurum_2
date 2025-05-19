import streamlit as st

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Aurum Dashboard", layout="wide")

# --- TRADUÇÃO ---
from translations import TRANSLATIONS

def t(key):
    lang = st.session_state.get("lang", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)

# --- SELETOR DE IDIOMA NA SIDEBAR ---
with st.sidebar:
    st.markdown("### 🌐 Idioma / Language / Idioma")
    lang_choice = st.selectbox("", ["Português", "English", "Español"], key="lang_sidebar")
    lang_map = {"Português": "pt", "English": "en", "Español": "es"}
    st.session_state["lang"] = lang_map.get(lang_choice, "en")

# --- TÍTULO PRINCIPAL ---
st.title(t("dashboard_title"))
