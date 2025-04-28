# --- IMPORTS ---
import streamlit as st
import pandas as pd
import gspread
from PIL import Image
from uuid import uuid4
from datetime import datetime
import pytz
from google.oauth2.service_account import Credentials
from streamlit_shadcn_ui import button, tabs

# --- CONFIG INICIAL ---
SHEET_ID = "1HVYbot3Z9OBccBw7jKNw5acodwiQpfXgavDTIptSKic"
ALERTS_SHEET = "Alerts"
UPDATES_SHEET = "Alert Updates"
USERS_SHEET = "Users"
brt = pytz.timezone("America/Sao_Paulo")

scope = ["https://www.googleapis.com/auth/spreadsheets"]
credentials = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
client = gspread.authorize(credentials)
sheets = client.open_by_key(SHEET_ID)

# --- FUNÇÕES AUXILIARES ---
@st.cache_data(ttl=600)
def load_alerts():
    return pd.DataFrame(sheets.worksheet(ALERTS_SHEET).get_all_records())

@st.cache_data(ttl=600)
def load_alert_updates():
    try:
        return pd.DataFrame(sheets.worksheet(UPDATES_SHEET).get_all_records())
    except gspread.exceptions.WorksheetNotFound:
        return pd.DataFrame(columns=["Alert ID", "Timestamp", "User", "Update Text"])

@st.cache_data(ttl=600)
def load_users():
    return pd.DataFrame(sheets.worksheet(USERS_SHEET).get_all_records())

def submit_new_alert(alert_row):
    worksheet = sheets.worksheet(ALERTS_SHEET)
    worksheet.append_row(alert_row, value_input_option="USER_ENTERED")

def submit_alert_update(update_row):
    try:
        worksheet = sheets.worksheet(UPDATES_SHEET)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sheets.add_worksheet(title=UPDATES_SHEET, rows="1000", cols="4")
        worksheet.append_row(["Alert ID", "Timestamp", "User", "Update Text"])
    worksheet.append_row(update_row, value_input_option="USER_ENTERED")

def status_color(status):
    status = status.lower()
    if status == "ongoing": return "#2ecc71"
    elif status == "closed": return "#e74c3c"
    elif status == "investigating": return "#f1c40f"
    elif status == "resolved": return "#3498db"
    else: return "#95a5a6"

# --- SIDEBAR (Login) ---
with st.sidebar:
    st.markdown("## Aurum 2.0")
    st.caption("Wildlife Trafficking Intelligence Platform")
    st.markdown("---")

    users_df = load_users()

    if "user" not in st.session_state:
        st.markdown("### 🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.button("Login")

        if login_btn and username and password:
            user_row = users_df[users_df["Username"] == username]
            if not user_row.empty and str(user_row.iloc[0]["Approved"]).strip().lower() == "true":
                stored_password = user_row.iloc[0]["Password"]
                if password == stored_password:
                    st.session_state["user"] = username
                    st.session_state["user_email"] = user_row.iloc[0]["E-mail"]
                    st.session_state["is_admin"] = str(user_row.iloc[0].get("Is_Admin", "")).strip().lower() == "true"
                    st.success(f"✅ Logged in as {username}")
                    st.rerun()
                else:
                    st.error("❌ Incorrect password.")
            else:
                st.error("❌ User not approved or does not exist.")
    else:
        st.success(f"✅ Logged in as {st.session_state['user']}")
        if st.button("Logout"):
            for key in ["user", "user_email", "is_admin"]:
                st.session_state.pop(key, None)
            st.rerun()

    st.markdown("---")
    st.caption("© Wildlife Conservation Society - Brazil, 2025")

# --- INTERFACE PRINCIPAL ---
st.title("Wildlife Trafficking Alerts")

current_tab = tabs(
    options=["Public Alerts", "Submit New Alert", "Update Alert"],
    default_value="Public Alerts",
)

df_alerts = load_alerts()
df_updates = load_alert_updates()

# --- PUBLIC ALERTS ---
if current_tab == "Public Alerts":
    st.subheader("Recent Alerts")

    if df_alerts.empty:
        st.info("No alerts found.")
    else:
        public_alerts = df_alerts[df_alerts["Public"].astype(str).str.upper() == "TRUE"]
        cols = st.columns(3)

        for idx, (_, a) in enumerate(public_alerts.iterrows()):
            with cols[idx % 3]:
                with st.container():
                    st.markdown(f"""
                        <div style='background-color: #fff; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                            <h4>{a['Title']}</h4>
                            <p style='color: #666; font-size: 0.9rem;'>{a.get('Description', '')}</p>
                            <p><b>Location:</b> {a.get('Country', 'Unknown')}</p>
                            <p><b>Date:</b> {a.get('Created At', 'Unknown')}</p>
                            <p><b>Status:</b> <code>{a.get('Risk Level', 'Unknown')}</code></p>
                        </div>
                    """, unsafe_allow_html=True)

                with st.expander("View Updates Timeline"):
                    updates = df_updates[df_updates["Alert ID"] == a["Alert ID"]]
                    if updates.empty:
                        st.info("No updates available.")
                    else:
                        for _, upd in updates.sort_values("Timestamp", ascending=False).iterrows():
                            color = status_color(upd.get("Status", ""))
                            st.markdown(f"""
                                <div style='display: flex; align-items: flex-start; margin-bottom: 0.8rem;'>
                                    <div style='margin-right: 10px; margin-top: 6px;'>
                                        <div style='width: 10px; height: 10px; background-color: {color}; border-radius: 50%;'></div>
                                    </div>
                                    <div>
                                        <strong>{upd['Timestamp']}</strong> — <span style='color: {color};'><b>{upd['User']}</b></span><br>
                                        <span style='font-size: 0.9em; color: #666;'>{upd['Update Text']}</span>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

# --- SUBMIT NEW ALERT ---
elif current_tab == "Submit New Alert":
    st.subheader("Submit a New Alert")
    title = st.text_input("Alert Title")
    description = st.text_area("Description")
    species = st.text_input("Species involved (optional)")
    country = st.text_input("Location")
    risk = st.selectbox("Risk Level", ["Low", "Medium", "High"])
    submit = button(text="Submit Alert", variant="primary")

    if submit and title and description:
        alert_id = str(uuid4())
        created_at = datetime.now(brt).strftime("%Y-%m-%d %H:%M:%S (BRT)")
        created_by = st.session_state.get("user_email", "Anonymous")
        display_as = st.session_state.get("user", "Anonymous")
        category = "Species"
        source_link = ""

        new_alert = [
            alert_id, created_at, created_by, display_as, title, description,
            category, species, country, risk, source_link, "TRUE"
        ]
        submit_new_alert(new_alert)
        st.cache_data.clear()
        st.success("✅ Alert submitted successfully!")
        st.rerun()

# --- UPDATE ALERT ---
elif current_tab == "Update Alert":
    st.subheader("Update an Existing Alert")

    if df_alerts.empty:
        st.info("No alerts found.")
    else:
        alert_titles = df_alerts["Title"].tolist()
        selected_alert = st.selectbox("Select Alert to Update", options=alert_titles)

        new_update = st.text_area("Update Notes")
        update = button(text="Submit Update", variant="secondary")

        if update and new_update:
            alert_id = df_alerts[df_alerts["Title"] == selected_alert].iloc[0]["Alert ID"]
            timestamp = datetime.now(brt).strftime("%Y-%m-%d %H:%M:%S (BRT)")
            user = st.session_state.get("user", "Anonymous")

            update_row = [alert_id, timestamp, user, new_update]
            submit_alert_update(update_row)
            st.cache_data.clear()
            st.success("✅ Update submitted successfully!")
            st.rerun()

# --- RODAPÉ ---
st.markdown("\n---\n")
st.caption("Powered by Aurum 2.0")
