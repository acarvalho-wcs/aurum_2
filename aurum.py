import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import re
import requests
import unicodedata
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from networkx.algorithms.community import greedy_modularity_communities
from io import BytesIO
import base64
from itertools import combinations
from scipy.stats import chi2_contingency
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
import urllib.parse
from uuid import uuid4
from datetime import datetime
import pytz
from streamlit_shadcn_ui import tabs
from streamlit_shadcn_ui import button
import streamlit.components.v1 as components
brt = pytz.timezone("America/Sao_Paulo")

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Aurum Dashboard", layout="wide")

# --- TRADUÇÃO ---
from translations import TRANSLATIONS

def t(key):
    lang = st.session_state.get("lang", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)

# --- SELETOR DE IDIOMA NA SIDEBAR ---
with st.sidebar:
    st.markdown("### Idioma / Language / Idioma")
    lang_choice = st.selectbox("", ["Português", "English", "Español"], key="lang_sidebar")
    lang_map = {"Português": "pt", "English": "en", "Español": "es"}
    st.session_state["lang"] = lang_map.get(lang_choice, "en")

# --- TÍTULO PRINCIPAL ---
st.title(t("dashboard_title"))
