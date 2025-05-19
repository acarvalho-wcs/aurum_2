import streamlit as st

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Aurum Dashboard", layout="wide")

# --- TRADUÇÃO ---
from translations import TRANSLATIONS

def t(key):
    lang = st.session_state.get("lang", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)

# --- CSS para posicionar seletor no topo direito ---
st.markdown("""
    <style>
        .lang-selector {
            position: absolute;
            top: 15px;
            right: 25px;
            z-index: 9999;
            width: 180px;
        }
        .stSelectbox label {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Container para seletor em posição fixa ---
lang_container = st.empty()
with lang_container.container():
    st.markdown('<div class="lang-selector">', unsafe_allow_html=True)
    lang_choice = st.selectbox("Idioma", ["Português", "English", "Español"], key="lang_selector_fixed")
    lang_map = {"Português": "pt", "English": "en", "Español": "es"}
    st.session_state["lang"] = lang_map.get(lang_choice, "en")
    st.markdown("</div>", unsafe_allow_html=True)

# --- TÍTULO CENTRALIZADO OU À ESQUERDA ---
st.title(t("dashboard_title"))
