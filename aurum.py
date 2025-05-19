import streamlit as st

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Aurum Dashboard", layout="wide")

# --- TRADUÇÃO ---
from translations import TRANSLATIONS

def t(key):
    lang = st.session_state.get("lang", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)

# --- CSS para forçar posicionamento absoluto no topo direito ---
st.markdown("""
    <style>
    div.fixed-lang {
        position: fixed;
        top: 15px;
        right: 20px;
        z-index: 9999;
        background-color: white;
        padding: 2px 8px;
        border-radius: 8px;
        box-shadow: 0px 0px 6px rgba(0,0,0,0.1);
    }
    </style>
    <div class="fixed-lang">
        <span id="lang-placeholder"></span>
    </div>
""", unsafe_allow_html=True)

# --- Selectbox real (posicionado via placeholder) ---
lang_placeholder = st.empty()
with lang_placeholder.container():
    lang_choice = st.selectbox("", ["Português", "English", "Español"], key="lang_selector_final")
    lang_map = {"Português": "pt", "English": "en", "Español": "es"}
    st.session_state["lang"] = lang_map.get(lang_choice, "en")

# --- TÍTULO DO APP ---
st.title(t("dashboard_title"))

