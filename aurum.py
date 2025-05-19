import streamlit as st

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Aurum Dashboard", layout="wide")

# --- TRADUÇÃO ---
from translations import TRANSLATIONS

def t(key):
    lang = st.session_state.get("lang", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)

col_spacer, col_title, col_lang = st.columns([4, 8, 2])

with col_lang:
    lang_choice = st.selectbox("🌐", ["Português", "English", "Español"])
    lang_map = {"Português": "pt", "English": "en", "Español": "es"}
    st.session_state["lang"] = lang_map.get(lang_choice, "en")

with col_title:
    st.markdown(f"<h1 style='text-align: center;'>{t('dashboard_title')}</h1>", unsafe_allow_html=True)
