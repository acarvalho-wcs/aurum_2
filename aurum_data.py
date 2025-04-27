# --- aurum_data.py ---

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import streamlit as st

# Constantes
SHEET_ID = "1HVYbot3Z9OBccBw7jKNw5acodwiQpfXgavDTIptSKic"
SCOPE = ["https://www.googleapis.com/auth/spreadsheets"]

# --- CONEXÃO ---

def connect_to_sheets():
    """Conecta de forma segura ao Google Sheets."""
    try:
        credentials = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=SCOPE
        )
        client = gspread.authorize(credentials)
        sheets = client.open_by_key(SHEET_ID)
        return sheets
    except Exception as e:
        st.error(f"❌ Erro ao conectar com Google Sheets: {e}")
        return None

# --- CARGA DE DADOS ---

def load_sheet_dataframe(sheet_name):
    """Carrega uma aba como DataFrame."""
    sheets = connect_to_sheets()
    if sheets:
        try:
            worksheet = sheets.worksheet(sheet_name)
            records = worksheet.get_all_records()
            return pd.DataFrame(records)
        except Exception as e:
            st.error(f"❌ Erro ao carregar a aba '{sheet_name}': {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

# --- FUNÇÕES ESPECÍFICAS ---

def load_aurum_data():
    return load_sheet_dataframe("Aurum_data")

def load_users():
    return load_sheet_dataframe("Users")

def load_requests():
    return load_sheet_dataframe("Access Requests")

def load_alerts():
    return load_sheet_dataframe("Alerts")

def load_alert_updates():
    return load_sheet_dataframe("Alert Updates")

def get_worksheet(sheet_name="Aurum_data"):
    """Retorna diretamente uma worksheet específica."""
    sheets = connect_to_sheets()
    if sheets:
        try:
            return sheets.worksheet(sheet_name)
        except Exception as e:
            st.error(f"❌ Erro ao acessar aba '{sheet_name}': {e}")
            return None
    else:
        return None
