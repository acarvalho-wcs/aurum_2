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

# Upload do arquivo
from PIL import Image
logo = Image.open("logo.png")
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.markdown("## Welcome to Aurum")
st.sidebar.markdown("Log in below to unlock multi-user tools.")
show_about = st.sidebar.button("**About Aurum**")
if show_about:
    st.markdown("## About Aurum")
    st.markdown("""
**Aurum** is a modular and interactive platform for **criminal intelligence in wildlife trafficking**. Developed by the Wildlife Conservation Society (WCS) – Brazil, it empowers analysts, researchers, and enforcement professionals with data-driven insights through a user-friendly interface.

The platform enables the upload and processing of case-level data and provides a suite of analytical tools, including:

- **Trend Analysis**: Explore directional changes in seizure patterns using segmented regression (TCS) and detect significant deviations from historical averages with cumulative sum control charts (CUSUM).
- **Species Co-occurrence**: Identify statistically significant co-trafficking relationships between species using chi-square tests and network-based visualizations.
- **Anomaly Detection**: Detect atypical or high-impact cases using multiple outlier detection methods (Isolation Forest, LOF, DBSCAN, Mahalanobis distance, and Z-Score).
- **Criminal Network Analysis**: Reveal connections between cases based on shared attributes such as species or offender countries to infer coordination and logistical convergence.
- **Interactive Visualization**: Build customized plots and dashboards based on selected variables to support real-time analysis and reporting.

**Aurum** bridges conservation data and investigative workflows, offering a scalable and field-ready platform for intelligence-led responses to wildlife crime.
""")

st.sidebar.markdown("## 📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("**Upload your Excel file (.xlsx).**", type=["xlsx"])

st.sidebar.markdown("**Download Template**")
with open("Aurum_template.xlsx", "rb") as f:
    st.sidebar.download_button(
        label="Download a data template for wildlife trafficking analysis in Aurum",
        data=f,
        file_name="aurum_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --- Mensagem inicial caso nenhum arquivo tenha sido enviado e usuário não esteja logado ---
if uploaded_file is None:
    st.markdown("""
    **Aurum** is a criminal intelligence platform developed to support the monitoring and investigation of **wildlife trafficking**.
    By integrating advanced statistical methods and interactive visualizations, Aurum enables researchers, enforcement agencies, and conservation organizations to identify operational patterns and support data-driven responses to illegal wildlife trade.

    **Upload your XLSX data file in the sidebar to begin.**  
    For the full Aurum experience, please request access or log in if you already have an account.  
    Click **About Aurum** to learn more about each analysis module.
    """)

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
    default_value="None",
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

st.title("Analyses")
current_tab = tabs(
    options=["Import xlsx", "Visualization", "Trend Analysis", "Co-occurrence Analysis", "Anomaly Detection", "Network Analysis"],
    default_value="",
)

df = None
df_selected = None
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)

        df.columns = df.columns.str.strip().str.replace('\xa0', '', regex=True)
        if 'Year' in df.columns:
            df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})').astype(float)

        def expand_multi_species_rows(df):
            expanded_rows = []
            for _, row in df.iterrows():
                matches = re.findall(r'(\d+)\s*([A-Z][a-z]+(?:_[a-z]+)+)', str(row.get('N seized specimens', '')))
                if matches:
                    for qty, species in matches:
                        new_row = row.copy()
                        new_row['N_seized'] = float(qty)
                        new_row['Species'] = species
                        expanded_rows.append(new_row)
                else:
                    expanded_rows.append(row)
            return pd.DataFrame(expanded_rows)

        df = expand_multi_species_rows(df).reset_index(drop=True)

        # Aplicar valores numéricos aos países se o arquivo estiver disponível
        import os
        country_score_path = "country_offenders_values.csv"
        if os.path.exists(country_score_path):
            df_country_score = pd.read_csv(country_score_path, encoding="ISO-8859-1")
            country_map = dict(zip(df_country_score["Country"].str.strip(), df_country_score["Value"]))

            def score_countries(cell_value, country_map):
                if not isinstance(cell_value, str):
                    return 0
                countries = [c.strip() for c in cell_value.split("+")]
                return sum(country_map.get(c, 0) for c in countries)

            # País de origem dos infratores
            if "Country of offenders" in df.columns:
                df["Offender_value"] = df["Country of offenders"].apply(lambda x: score_countries(x, country_map))

            # País de apreensão ou envio
            if "Country of seizure or shipment" in df.columns:
                df["Seizure_value"] = df["Country of seizure or shipment"].apply(lambda x: score_countries(x, country_map))

        else:
            st.warning("⚠️ File country_offenders_values.csv not found. Country scoring skipped.")

        # Marcação de convergência logística
        if 'Case #' in df.columns and 'Species' in df.columns:
            species_per_case = df.groupby('Case #')['Species'].nunique()
            df['Logistic Convergence'] = df['Case #'].map(lambda x: "1" if species_per_case.get(x, 0) > 1 else "0")

        def normalize_text(text):
            if not isinstance(text, str):
                text = str(text)
            text = text.strip().lower()
            text = unicodedata.normalize("NFKD", text)
            text = re.sub(r'\\s+', ' ', text)
            return text

        def infer_stage(row):
            seizure = normalize_text(row.get("Seizure Status", ""))
            transit = normalize_text(row.get("Transit Feature", ""))
            logistic = row.get("Logistic Convergence", "0")
            if any(k in seizure for k in ["planned", "trap", "attempt"]):
                return "Preparation"
            elif "captivity" in transit or "breeding" in transit:
                return "Captivity"
            elif any(k in transit for k in ["airport", "border", "highway", "port"]):
                return "Transport"
            elif logistic == "1":
                return "Logistic Consolidation"
            else:
                return "Unclassified"

        df["Inferred Stage"] = df.apply(infer_stage, axis=1)

        st.success("✅ File uploaded and cleaned successfully!")

        st.sidebar.markdown("## Select Species")
        species_options = sorted(df['Species'].dropna().unique())
        selected_species = st.sidebar.multiselect("Select one or more species:", species_options)

        if selected_species:
            df_selected = df[df['Species'].isin(selected_species)]

            show_viz = st.sidebar.checkbox("Data Visualization", value=False)
            if show_viz:
                st.markdown("## Data Visualization")
                if st.sidebar.checkbox("Preview data"):
                    st.write("### Preview of cleaned data:")
                    st.dataframe(df_selected.head())

                chart_type = st.sidebar.selectbox("Select chart type:", ["Bar", "Line", "Scatter", "Pie"])
                x_axis = st.sidebar.selectbox("X-axis:", df_selected.columns, index=0)
                y_axis = st.sidebar.selectbox("Y-axis:", df_selected.columns, index=1)

                import plotly.express as px
                st.markdown("### Custom Chart")
                if chart_type == "Bar":
                    fig = px.bar(df_selected, x=x_axis, y=y_axis, color='Species')
                elif chart_type == "Line":
                    fig = px.line(df_selected, x=x_axis, y=y_axis, color='Species')
                elif chart_type == "Scatter":
                    fig = px.scatter(df_selected, x=x_axis, y=y_axis, color='Species')
                elif chart_type == "Pie":
                    fig = px.pie(df_selected, names=x_axis, values=y_axis)
                st.plotly_chart(fig)

# --- RODAPÉ ---
st.markdown("\n---\n")
st.caption("Powered by Aurum 2.0")
