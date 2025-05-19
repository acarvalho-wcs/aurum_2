import streamlit as st

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Aurum Dashboard", layout="wide")

# --- TRADUÇÃO E SELETOR DE IDIOMA ---
from translations import TRANSLATIONS

def t(key):
    lang = st.session_state.get("lang", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)

# Ordem correta: título à esquerda (maior), seletor à direita (menor)
col_title, col_lang = st.columns([5, 1])

with col_lang:
    lang_choice = st.selectbox("", ["Português", "English", "Español"])
    lang_map = {"Português": "pt", "English": "en", "Español": "es"}
    st.session_state["lang"] = lang_map.get(lang_choice, "en")

with col_title:
    st.title(t("dashboard_title"))
