# --- IMPORTS PADRÃO DA BIBLIOTECA PYTHON ---
import os
import re
import base64
import unicodedata
from io import BytesIO
from uuid import uuid4
from itertools import combinations
from datetime import datetime

# --- IMPORTS DE TERCEIROS (BIBLIOTECAS INSTALADAS) ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pytz
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from scipy.stats import chi2_contingency
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# --- IMPORTS INTERNOS DO AURUM ---
from aurum_data import (
    load_aurum_data,
    load_users,
    load_requests,
    load_alerts,
    load_alert_updates
)

# --- CONFIGURAÇÃO DE FUSO HORÁRIO ---
brt = pytz.timezone("America/Sao_Paulo")

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Aurum Dashboard", layout="wide")
st.title("Aurum - Criminal Intelligence in Wildlife Trafficking")

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

# --- CARGA DOS DADOS USUÁRIOS E SOLICITAÇÕES ---
users_df = load_users()
requests_df = load_requests()

# --- Função para acessar worksheet de dados principais ---
def get_worksheet(name="Aurum_data"):
    return sheets.worksheet(name)

# --- Mensagem inicial caso nenhum arquivo tenha sido enviado e usuário não esteja logado ---
if uploaded_file is None:
    st.markdown("""
    **Aurum** is a criminal intelligence platform developed to support the monitoring and investigation of **wildlife trafficking**.
    By integrating advanced statistical methods and interactive visualizations, Aurum enables researchers, enforcement agencies, and conservation organizations to identify operational patterns and support data-driven responses to illegal wildlife trade.

    **Upload your XLSX data file in the sidebar to begin.**  
    For the full Aurum experience, please request access or log in if you already have an account.  
    Click **About Aurum** to learn more about each analysis module.
    """)

# --- ALERTAS PÚBLICOS (visível para todos, inclusive sem login) ---
def display_public_alerts_section(sheet_id):
    with st.container():
        st.markdown("## 🌍 Alert Board")
        st.caption("These alerts are publicly available and updated by verified users of the Aurum system.")
        st.markdown("### Wildlife Trafficking Alerts")

        # Carregamento dos dados via aurum_data
        df_alerts = load_alerts()
        df_updates = load_alert_updates()

        if df_alerts.empty or "Public" not in df_alerts.columns:
            st.info("No public alerts available.")
            return

        df_alerts = df_alerts[df_alerts["Public"].astype(str).str.strip().str.upper() == "TRUE"]

        if df_alerts.empty:
            st.info("No public alerts available.")
            return

        df_alerts = df_alerts.sort_values("Created At", ascending=False)

        alert_cols = st.columns(3)
        for idx, (_, row) in enumerate(df_alerts.iterrows()):
            col = alert_cols[idx % 3]
            with col:
                with st.expander(f"**🚨 {row['Title']} ({row['Risk Level']})**", expanded=False):
                    st.markdown(f"**Description:** {row['Description']}")
                    st.markdown(f"**Category:** {row['Category']}")
                    if row.get("Species"):
                        st.markdown(f"**Species:** {row['Species']}")
                    if row.get("Country"):
                        st.markdown(f"**Country:** {row['Country']}")

                    if row.get("Source Link"):
                        link = row['Source Link']
                        st.markdown(
                            f"🔗 **Source:** <a href='{link}' target='_blank'>{link}</a>",
                            unsafe_allow_html=True
                        )

                    display_name = row.get("Display As", row.get("Created By", "Unknown"))
                    st.caption(f"Submitted on {row['Created At']} by *{display_name}*")

                    st.markdown(
                        """
                        <div style='font-size: 12px; color: #666; margin-top: 6px;'>
                            <em>This alert was published by verified users on <strong>AURUM</strong>, the intelligence system against wildlife trafficking.</em>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Botão de remoção (apenas para o criador)
                    user_email = st.session_state.get("user_email", "").strip().lower()
                    creator = row.get("Created By", "").strip().lower()
                    if creator == user_email:
                        if st.button(f"🗑️ Remove alert from public board", key=f"delete_{row['Alert ID']}"):
                            try:
                                # Recarregar a planilha para editar
                                sheets = connect_to_sheets()
                                if sheets:
                                    sheet = sheets.worksheet("Alerts")
                                    cell = sheet.find(str(row["Alert ID"]))
                                    public_col = df_alerts.columns.get_loc("Public") + 1  # gspread é 1-based
                                    sheet.update_cell(cell.row, public_col, "FALSE")
                                    st.success("Alert removed from public board (still stored in the system).")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"❌ Failed to update visibility: {e}")

                    # Timeline de atualizações
                    if not df_updates.empty and "Alert ID" in df_updates.columns:
                        updates = df_updates[df_updates["Alert ID"] == row["Alert ID"]].sort_values("Timestamp")
                        if not updates.empty:
                            st.markdown("**Update Timeline**")
                            for _, upd in updates.iterrows():
                                st.markdown(f"🕒 **{upd['Timestamp']}** – *{upd['User']}*: {upd['Update Text']}")


# Executa antes do login
display_public_alerts_section(SHEET_ID)

# --- DASHBOARD RESUMO INICIAL (sem login, baseado no Google Sheets) ---
if uploaded_file is None and not st.session_state.get("user"):
    try:
        worksheet = get_worksheet()
        records = worksheet.get_all_records()
        df_dashboard = pd.DataFrame(records)

        if not df_dashboard.empty and "N seized specimens" in df_dashboard.columns:
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

            df_dashboard = expand_multi_species_rows(df_dashboard)
            df_dashboard = df_dashboard[df_dashboard["Species"].notna()]
            df_dashboard["N_seized"] = pd.to_numeric(df_dashboard["N_seized"], errors="coerce").fillna(0)

            st.markdown("## Summary Dashboard")

            available_species = sorted(df_dashboard["Species"].unique())
            selected_species_dash = st.selectbox("Select a species to view:", ["All species"] + available_species)

            filtered_df = df_dashboard.copy()
            
            # Extended metrics in two rows: 3 columns per row (only for All species)
            if selected_species_dash == "All species":
                total_species = df_dashboard["Species"].nunique()
                total_cases_all = df_dashboard["Case #"].nunique()
                total_individuals_all = int(df_dashboard["N_seized"].sum())
                total_countries_all = df_dashboard["Country of seizure or shipment"].nunique() if "Country of seizure or shipment" in df_dashboard.columns else 0

                # Extract estimated weight in kg
                df_dashboard["kg_seized"] = df_dashboard["N seized specimens"].str.extract(r'(\d+(?:\.\d+)?)\s*kg', expand=False)[0]
                total_kg = pd.to_numeric(df_dashboard["kg_seized"], errors="coerce").fillna(0).sum()

                # Extract number of parts
                df_dashboard["parts_seized"] = df_dashboard["N seized specimens"].str.extract(r'(\d+(?:\.\d+)?)\s*(part|parts)', expand=False)[0]
                total_parts = pd.to_numeric(df_dashboard["parts_seized"], errors="coerce").fillna(0).sum()

                st.markdown("---\n### Global Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Species seized", total_species)
                col2.metric("Total cases", total_cases_all)
                col3.metric("Countries involved", total_countries_all)

                col4, col5, col6 = st.columns(3)
                col4.metric("Individuals seized", total_individuals_all)
                col5.metric("Estimated weight (kg)", f"{total_kg:.1f}")
                col6.metric("Animal parts seized", int(total_parts))
                
            if selected_species_dash == "All species":
                # --- DISTRIBUIÇÃO TEMPORAL E GEOGRÁFICA ---
                st.markdown("## Temporal and Geographic Distribution of Recorded Seizures")

                import plotly.express as px
                import pycountry
                from collections import Counter

                # Layout lado a lado
                col1, col2 = st.columns([1, 1.4])

                with col1:
                    st.markdown("#### Cases per Year")
                    if "Year" in df_dashboard.columns:
                        df_dashboard["Year"] = pd.to_numeric(df_dashboard["Year"], errors="coerce")
                        df_years = df_dashboard.groupby("Year", as_index=False)["Case #"].nunique()
                        fig_years = px.bar(
                            df_years,
                            x="Year",
                            y="Case #",
                            labels={"Case #": "Number of Cases", "Year": "Year"},
                            height=450
                        )
                        fig_years.update_layout(margin=dict(t=30, b=30, l=10, r=10))
                        st.plotly_chart(fig_years, use_container_width=True)
                    else:
                        st.info("Year column not available in data.")

                with col2:
                    st.markdown("#### Countries with Recorded Seizures")

                    country_lookup = {country.name: country.alpha_3 for country in pycountry.countries}
                    iso_to_name = {country.alpha_3: country.name for country in pycountry.countries}

                    custom_iso = {
                        "French Guiana": "GUF", "Hong Kong": "HKG", "Macau": "MAC", "Puerto Rico": "PRI",
                        "Palestine": "PSE", "Kosovo": "XKX", "Taiwan": "TWN", "Réunion": "REU",
                        "Guadeloupe": "GLP", "Martinique": "MTQ", "New Caledonia": "NCL"
                    }
                    custom_name = {v: k for k, v in custom_iso.items()}
                    all_iso_codes = list(country_lookup.values()) + list(custom_name.keys())

                    if "Country of seizure or shipment" in df_dashboard.columns:
                        countries_raw = df_dashboard["Country of seizure or shipment"].dropna()

                        iso_codes = []
                        for name in countries_raw:
                            name_clean = name.strip()
                            try:
                                match = pycountry.countries.lookup(name_clean)
                                iso_codes.append(match.alpha_3)
                            except:
                                if name_clean in custom_iso:
                                    iso_codes.append(custom_iso[name_clean])

                        country_counts = Counter(iso_codes)

                        df_map = pd.DataFrame({"ISO": all_iso_codes})
                        df_map["Cases"] = df_map["ISO"].apply(lambda x: country_counts.get(x, 0))
                        df_map["Country"] = df_map["ISO"].apply(lambda x: iso_to_name.get(x, custom_name.get(x, "Unknown")))

                        color_scale = [
                            [0.0, "#ffffff"],
                            [0.01, "#a0c4e8"],
                            [0.25, "#569fd6"],
                            [0.5, "#2171b5"],
                            [1.0, "#08306b"],
                        ]

                        fig_map = px.choropleth(
                            df_map,
                            locations="ISO",
                            color="Cases",
                            hover_name="Country",
                            color_continuous_scale=color_scale,
                            range_color=(0, max(df_map["Cases"].max(), 1)),
                            height=450
                        )

                        fig_map.update_layout(
                            geo=dict(showframe=False, showcoastlines=False, projection_type="natural earth"),
                            coloraxis_colorbar=dict(title="Number of Cases"),
                            margin=dict(l=10, r=10, t=30, b=0),
                        )

                        st.plotly_chart(fig_map, use_container_width=True)
                    else:
                        st.info("No country information available to display the map.")
            
            # Gráficos lado a lado: scatter + barras (somente para uma espécie selecionada)
            if selected_species_dash != "All species":
                df_species = df_dashboard[df_dashboard["Species"] == selected_species_dash]

                if "Year" in df_species.columns and not df_species.empty:
                    try:
                        df_species["Year"] = pd.to_numeric(df_species["Year"], errors="coerce")

                        n_cases = df_species["Case #"].nunique()
                        n_countries = df_species["Country of seizure or shipment"].nunique()
                        if not df_species.empty and df_species["N_seized"].max() > 0:
                            idx_max = df_species["N_seized"].idxmax()
                            max_row = df_species.loc[idx_max]
                            max_apreensao = f"{max_row['Country of seizure or shipment']} in {int(max_row['Year'])}"
                        else:
                            max_apreensao = "No data"

                        st.markdown("### Key Indicators for Selected Species")
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Cases recorded", n_cases)
                        col_b.metric("Countries with seizures", n_countries)
                        col_c.metric("Largest seizure", max_apreensao)

                        # Gráficos
                        col1, col2 = st.columns(2)

                        with col1:
                            import plotly.express as px
                            fig_scatter = px.scatter(
                                df_species,
                                x="Year",
                                y="N_seized",
                                title="Individuals Seized per Case",
                                labels={"N_seized": "Individuals", "Year": "Year"}
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)

                        with col2:
                            df_bar = df_species.groupby("Year", as_index=False)["N_seized"].sum()
                            fig_bar = px.bar(
                                df_bar,
                                x="Year",
                                y="N_seized",
                                title="Total Individuals per Year",
                                labels={"N_seized": "Total Individuals", "Year": "Year"}
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)

                    except Exception as e:
                        st.warning(f"Could not render plots: {e}")
                        
            # Co-occurrence with other species in the same cases
            if selected_species_dash != "All species":
                st.markdown("### Species co-occurring in same cases")

                # Encontra casos que contêm a espécie selecionada
                cases_with_selected = df_dashboard[df_dashboard["Species"] == selected_species_dash]["Case #"].unique()

                # Filtra o dataframe apenas para esses casos
                coocurrence_df = df_dashboard[df_dashboard["Case #"].isin(cases_with_selected)]

                # Remove a espécie selecionada
                co_species = coocurrence_df[coocurrence_df["Species"] != selected_species_dash]["Species"].unique()

                if len(co_species) > 0:
                    st.write(", ".join(sorted(co_species)))
                else:
                    st.info("No other species recorded with the selected species.")

    except Exception as e:
        st.error(f"❌ Failed to load dashboard summary: {e}")

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

            
            show_trend = st.sidebar.checkbox("Trend Analysis", value=False)
            if show_trend:
                st.markdown("## Trend Analysis")

                breakpoint_year = st.number_input("Breakpoint year (split the trend):", 1990, 2030, value=2015)

                def trend_component(df, year_col='Year', count_col='N_seized', breakpoint=2015):
                    df_pre = df[df[year_col] <= breakpoint]
                    df_post = df[df[year_col] > breakpoint]

                    if len(df_pre) < 2 or len(df_post) < 2:
                        return 0.0, "Insufficient data for segmented regression"

                    X_pre = sm.add_constant(df_pre[[year_col]])
                    y_pre = df_pre[count_col]
                    model_pre = sm.OLS(y_pre, X_pre).fit()
                    slope_pre = model_pre.params[year_col]

                    X_post = sm.add_constant(df_post[[year_col]])
                    y_post = df_post[count_col]
                    model_post = sm.OLS(y_post, X_post).fit()
                    slope_post = model_post.params[year_col]

                    tcs = (slope_post - slope_pre) / (abs(slope_pre) + 1)
                    log = f"TCS = {tcs:.2f}"
                    return tcs, log

                tcs, tcs_log = trend_component(df_selected, breakpoint=breakpoint_year)
                st.markdown(f"**Trend Coordination Score (TCS):** `{tcs:.2f}`")
                st.info(tcs_log)

                with st.expander("ℹ️ Learn more about this analysis"):
                    st.markdown("""
                    ### About Trend Analysis

                    The *Trend Analysis* section helps identify shifts in wildlife seizure patterns over time for the selected species.

                    - The analysis uses segmented linear regressions based on a user-defined **breakpoint year**.
                    - For each species, a regression is computed before and after the breakpoint to estimate the slope (i.e., the trend) of increase or decrease.
                    - These slopes are used to calculate the **Trend Coordination Score (TCS)**, which measures the relative change between the two periods:
                      - `TCS > 0` indicates an increase in trend after the breakpoint.
                      - `TCS < 0` indicates a decrease.
                      - `TCS ≈ 0` suggests stability.

                    - The score is normalized to reduce instability when the pre-breakpoint slope is close to zero. While TCS has no strict bounds, in practice it typically falls between −1 and +1. 
                    - Extreme values may indicate sharp shifts in trend intensity or imbalances in the temporal distribution of data. Although wildlife trafficking patterns are rarely linear in reality, this method adopts linear segments as a practical approximation to detect directional shifts. 
                    - It does not assume true linear behavior, but rather uses regression slopes as a comparative metric across time intervals. The analysis requires at least two observations on each side of the breakpoint to produce meaningful estimates. 
                    - The score can be sensitive to outliers or sparsely populated time ranges, and should be interpreted in light of the broader case context.
                    - The section also generates a plot showing data points and trend lines for each species, making it easier to visualize changes over time.
                    - Find more details in the ReadMe file and/or in Carvalho (2025).
                    """)

                st.markdown("### Trend Plot")
                fig, ax = plt.subplots(figsize=(8, 5))

                for species in selected_species:
                    subset = df_selected[df_selected['Species'] == species]
                    ax.scatter(subset['Year'], subset['N_seized'], label=species, alpha=0.6)

                    df_pre = subset[subset['Year'] <= breakpoint_year]
                    df_post = subset[subset['Year'] > breakpoint_year]

                    if len(df_pre) > 1:
                        model_pre = sm.OLS(df_pre['N_seized'], sm.add_constant(df_pre['Year'])).fit()
                        ax.plot(df_pre['Year'], model_pre.predict(sm.add_constant(df_pre['Year'])), linestyle='--')

                    if len(df_post) > 1:
                        model_post = sm.OLS(df_post['N_seized'], sm.add_constant(df_post['Year'])).fit()
                        ax.plot(df_post['Year'], model_post.predict(sm.add_constant(df_post['Year'])), linestyle='-.')

                ax.axvline(breakpoint_year, color='red', linestyle=':', label=f"Breakpoint = {breakpoint_year}")
                ax.set_title("Seizure Trend by Species")
                ax.set_xlabel("Year")
                ax.set_ylabel("Individuals Seized")
                ax.legend()
                st.pyplot(fig)

                with st.expander("📉 Show regression details by species"):
                    for species in selected_species:
                        subset = df_selected[df_selected['Species'] == species]
                        df_pre = subset[subset['Year'] <= breakpoint_year]
                        df_post = subset[subset['Year'] > breakpoint_year]

                        st.markdown(f"#### {species}")

                        if len(df_pre) > 1:
                            X_pre = sm.add_constant(df_pre['Year'])
                            y_pre = df_pre['N_seized']
                            model_pre = sm.OLS(y_pre, X_pre).fit()
                            slope_pre = model_pre.params['Year']
                            r2_pre = model_pre.rsquared
                            pval_pre = model_pre.pvalues['Year']
                            st.markdown(f"- Pre-breakpoint slope: β = `{slope_pre:.2f}`")
                            st.markdown(f"- R² = `{r2_pre:.2f}`")
                            st.markdown(f"- p-value = `{pval_pre:.4f}`")
                        else:
                            st.info("Not enough data before breakpoint.")

                        if len(df_post) > 1:
                            X_post = sm.add_constant(df_post['Year'])
                            y_post = df_post['N_seized']
                            model_post = sm.OLS(y_post, X_post).fit()
                            slope_post = model_post.params['Year']
                            r2_post = model_post.rsquared
                            pval_post = model_post.pvalues['Year']
                            st.markdown(f"- Post-breakpoint slope: β = `{slope_post:.2f}`")
                            st.markdown(f"- R² = `{r2_post:.2f}`")
                            st.markdown(f"- p-value = `{pval_post:.4f}`")
                        else:
                            st.info("Not enough data after breakpoint.")

                # Optional CUSUM
                if st.checkbox("Activate CUSUM analysis"):
                    st.subheader("CUSUM - Trend Change Detection")

                    cusum_option = st.radio(
                        "Choose the metric to analyze:",
                        ["Total individuals seized per year", "Average individuals per seizure (per year)"]
                    )

                    if cusum_option == "Total individuals seized per year":
                        df_cusum = df_selected.groupby("Year")["N_seized"].sum().reset_index()
                        col_data = "N_seized"
                        col_time = "Year"
                    else:
                        df_cusum = df_selected.groupby("Year")["N_seized"].mean().reset_index()
                        col_data = "N_seized"
                        col_time = "Year"

                    def plot_cusum_trend(df, col_data, col_time, species_name="Selected Species"):
                        df_sorted = df.sort_values(by=col_time).reset_index(drop=True)
                        years = df_sorted[col_time]
                        values = df_sorted[col_data]

                        mean_val = values.mean()
                        std_dev = values.std()

                        cusum_pos = [0]
                        cusum_neg = [0]

                        for i in range(1, len(values)):
                            delta = values.iloc[i] - mean_val
                            cusum_pos.append(max(0, cusum_pos[-1] + delta))
                            cusum_neg.append(min(0, cusum_neg[-1] + delta))

                        fig, ax = plt.subplots(figsize=(10, 6))

                        ax.plot(years, values, color='black', marker='o', label='Trend')
                        ax.plot(years, cusum_pos, color='green', linestyle='--', label='CUSUM+')
                        ax.plot(years, cusum_neg, color='orange', linestyle='--', label='CUSUM-')

                        # Highlight years with significant deviation
                        highlight_years = [
                            i for i, val in enumerate(values)
                            if abs(val - mean_val) > 1.5 * std_dev
                        ]

                        ax.scatter(
                            [years.iloc[i] for i in highlight_years],
                            [values.iloc[i] for i in highlight_years],
                            color='red', marker='x', s=100, label='Significant Deviation'
                        )

                        ax.set_title(f"{species_name} - Trend & CUSUM", fontsize=14)
                        ax.set_xlabel("Year")
                        ax.set_ylabel("Seized Specimens")
                        ax.grid(True, linestyle='--', linewidth=0.5, color='lightgray')
                        ax.legend()
                        st.pyplot(fig)

                        # Interpretation
                        st.subheader("Automated Interpretation")
                        cusum_range = max(cusum_pos) - min(cusum_neg)

                        if cusum_range > 2 * std_dev:
                            # Detect all years with significant deviation from the mean
                            change_years = [
                                years.iloc[i]
                                for i, val in enumerate(values)
                                if abs(val - mean_val) > 1.5 * std_dev
                            ]

                            if change_years:
                                formatted_years = " and ".join(str(y) for y in change_years)
                                st.markdown(f"Significant trend changes detected in: **{formatted_years}** (based on deviations from the historical average).")
                            else:
                                st.markdown("CUSUM suggests change, but no single year shows strong deviation from the mean.")
                        else:
                            st.markdown("✅ No significant trend change detected.")


                        # Explanation toggle
                        with st.expander("ℹ️ Learn more about this analysis"):
                            st.markdown("""
                            ### About CUSUM Analysis

                            The *CUSUM Analysis* section is designed to detect significant changes in the temporal trend of wildlife seizures by evaluating how yearly values deviate from the historical average.

                            - The method is based on **Cumulative Sum (CUSUM)** analysis, which tracks the cumulative deviation of observed values from their overall mean.
                            - Two cumulative paths are calculated:
                              - **CUSUM+** accumulates positive deviations (above the mean).
                              - **CUSUM−** accumulates negative deviations (below the mean).
                            - This dual-track approach highlights the **direction and magnitude of long-term deviations**, making it easier to identify sustained changes.

                            - Unlike methods that directly model trends (e.g., segmented regression), CUSUM reacts **only when there is consistent deviation**, amplifying the signal of real change over time.
                            - The method does **not identify the exact year of change by itself**, but instead signals that a shift in the distribution has occurred—often **triggered by a single or small set of high-impact events**.
                            - To estimate the timing more precisely, the analysis **identifies years where the seizure counts deviate sharply from the historical mean** (greater than 1.5 standard deviations).
                            - These years are reported as **likely points of trend change**.

                            - CUSUM is especially useful when changes are not gradual or linear, and when a single anomalous year can drive broader shifts.
                            - The results should be interpreted in light of species-specific context and enforcement history, as well as any known conservation events, policy changes, or trafficking routes.

                            - The section also generates a plot combining:
                              - Observed seizure counts (black line),
                              - CUSUM+ and CUSUM− paths (green and orange dashed lines),
                              - Highlighted years with significant deviations (when present).

                            - For more details, refer to the ReadMe file and/or Carvalho (2025).
                            """)

                    plot_cusum_trend(df_cusum, col_data=col_data, col_time=col_time)

            show_cooc = st.sidebar.checkbox("Species Co-occurrence", value=False)
            if show_cooc:
                st.markdown("## Species Co-occurrence Analysis")

                with st.expander("ℹ️ Learn more about this analysis"):
                    st.markdown("""
                    ### About Species Co-occurrence

                    The Species Co-occurrence section identifies pairs of species that tend to be trafficked together within the same cases, revealing potential patterns of coordinated extraction, transport, or market demand.

                    This analysis uses a binary presence-absence matrix for each selected species across all case IDs. For every species pair, a **chi-square test of independence** is performed to evaluate whether the observed co-occurrence is statistically significant beyond what would be expected by chance.

                    - A **2×2 contingency table** is generated for each pair, indicating joint presence or absence across cases.
                    - The **Chi² statistic** quantifies the degree of association: higher values suggest stronger deviation from independence (i.e., a stronger link between the species).
                    - The associated **p-value** indicates whether this deviation is statistically significant. A p-value below 0.05 typically means the co-occurrence is unlikely to be due to chance.

                    **Interpretation**:
                    - **High Chi² values** signal that the two species co-occur more (or less) than expected — implying possible ecological overlap, shared trafficking routes, or joint market targeting.
                    - **Low Chi² values** suggest weak or no association, even if species occasionally appear together.

                    This method is particularly useful for identifying species that may be captured, transported, or traded together due to logistical, ecological, or commercial drivers.

                    The results are displayed in an interactive table showing co-occurrence counts, the Chi² statistic, and p-values for each species pair — helping analysts prioritize combinations for deeper investigation.
                    """)

                def general_species_cooccurrence(df, species_list, case_col='Case #'):
                    presence = pd.DataFrame()
                    presence[case_col] = df[case_col].unique()
                    presence.set_index(case_col, inplace=True)

                    for sp in species_list:
                        sp_df = df[df['Species'] == sp][[case_col]]
                        sp_df['present'] = 1
                        grouped = sp_df.groupby(case_col)['present'].max()
                        presence[sp] = grouped

                    presence.fillna(0, inplace=True)
                    presence = presence.astype(int)

                    results = []
                    for sp_a, sp_b in combinations(species_list, 2):
                        table = pd.crosstab(presence[sp_a], presence[sp_b])
                        if table.shape == (2, 2):
                            chi2, p, _, _ = chi2_contingency(table)
                            results.append((sp_a, sp_b, chi2, p, table))
                    return results

                def interpret_cooccurrence(table, chi2, p):
                    a = table.iloc[0, 0]
                    b = table.iloc[0, 1]
                    c = table.iloc[1, 0]
                    d = table.iloc[1, 1]

                    threshold = 0.05

                    if p >= threshold:
                        st.info("No statistically significant association between these species was found (p ≥ 0.05).")
                        return

                    if d == 0:
                        st.warning("⚠️ These species were never trafficked together. This pattern suggests **mutual exclusivity**, possibly due to distinct trafficking chains or ecological separation.")
                    elif b + c == 0:
                        st.success("✅ These species always appear together. This indicates a **perfect positive association**, potentially reflecting joint capture, transport, or market demand.")
                    elif d > b + c:
                        st.success("🔗 These species frequently appear together and are **positively associated** in trafficking records. The co-occurrence is unlikely to be due to chance.")
                    elif d < min(b, c):
                        st.error("❌ These species are almost always recorded **separately**, suggesting a **strong negative association** or operational separation in trafficking routes.")
                    else:
                        st.info("ℹ️ A statistically significant association was detected. While co-occurrence exists, it is not dominant — suggesting **partial overlap** in trafficking patterns.")

                co_results = general_species_cooccurrence(df_selected, selected_species)

                if co_results:
                    st.markdown("### Co-occurrence Results")
                    for sp_a, sp_b, chi2, p, table in co_results:
                        st.markdown(f"**{sp_a} × {sp_b}**")
                        st.dataframe(table)
                        st.markdown(f"Chi² = `{chi2:.2f}` | p = `{p:.4f}`")
                        interpret_cooccurrence(table, chi2, p)
                        st.markdown("---")
                else:
                    st.info("No co-occurrence data available for selected species.")

                st.markdown("### Co-trafficked Species Cases")

                grouped = df_selected.groupby('Case #')
                multi_species_cases = grouped.filter(lambda x: x['Species'].nunique() > 1)

                if multi_species_cases.empty:
                    st.info("No multi-species trafficking cases found for the selected species.")
                else:
                    summary = multi_species_cases[['Case #', 'Country of offenders', 'Species', 'N_seized']].sort_values(by='Case #')
                    st.dataframe(summary)

            show_anomaly = st.sidebar.checkbox("Anomaly Detection", value=False)
            if show_anomaly:
                st.markdown("## Anomaly Detection")

                with st.expander("ℹ️ Learn more about this analysis"):
                    st.markdown("""
                    ### About Anomaly Detection

                    The Anomaly Detection section helps identify wildlife trafficking cases that deviate significantly from typical patterns based on selected numerical features.

                    The analysis applies multiple outlier detection algorithms to highlight cases that may involve unusual species combinations, unusually large quantities of individuals, recurring offender countries, or rare time periods.

                    By default, the following methods are applied in parallel:

                    - **Isolation Forest**: an ensemble method that detects anomalies by isolating data points in a randomly partitioned feature space.
                    - **Local Outlier Factor (LOF)**: detects anomalies based on the local density of data points. Cases that lie in areas of significantly lower density than their neighbors are flagged.
                    - **DBSCAN**: a clustering algorithm that marks low-density or unclustered points as outliers.
                    - **Z-Score**: a statistical approach that flags values deviating more than 3 standard deviations from the mean in any feature.
                    - **Mahalanobis Distance**: a multivariate distance measure that accounts for correlations between features to identify statistically distant points.

                    Each method produces a binary vote (outlier or not). The final score reflects how many methods agree in classifying a case as anomalous — forming a **consensus-based ranking**.

                    A high consensus score suggests stronger evidence of atypical behavior, but anomalies do not necessarily imply criminality. They may reflect rare events, data entry issues, or unique but legitimate circumstances.

                    This module is most effective when the user selects a meaningful combination of numerical features, such as year, number of individuals, or offender-related values. The output highlights the top-ranked outliers and their anomaly vote count, supporting investigation and prioritization.
                    """)

                numeric_cols = [col for col in df_selected.columns if pd.api.types.is_numeric_dtype(df_selected[col])]
                selected_features = st.multiselect("Select numeric features for anomaly detection:", numeric_cols, default=["N_seized", "Year", "Seizure_value", "Offender_value"])

                if selected_features:
                    X = StandardScaler().fit_transform(df_selected[selected_features])

                    models = {
                        "Isolation Forest": IsolationForest(random_state=42).fit_predict(X),
                        "LOF": LocalOutlierFactor().fit_predict(X),
                        "DBSCAN": DBSCAN(eps=1.2, min_samples=2).fit_predict(X),
                        "Z-Score": np.where(np.any(np.abs(X) > 3, axis=1), -1, 1)
                    }

                    try:
                        cov = np.cov(X, rowvar=False)
                        inv_cov = np.linalg.inv(cov)
                        mean = np.mean(X, axis=0)
                        diff = X - mean
                        md = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
                        threshold_md = np.percentile(md, 97.5)
                        models["Mahalanobis"] = np.where(md > threshold_md, -1, 1)
                    except np.linalg.LinAlgError:
                        models["Mahalanobis"] = np.ones(len(X))

                    vote_df = pd.DataFrame(models)
                    vote_df["Outlier Votes"] = (vote_df == -1).sum(axis=1)
                    vote_df["Case #"] = df_selected["Case #"].values

                    consensus_ratio = (vote_df["Outlier Votes"] > 2).sum() / len(vote_df)
                    st.markdown(f"**Consensus Outlier Ratio:** `{consensus_ratio:.2%}`")

                    st.markdown("### Most anomalous cases")
                    top_outliers = vote_df.sort_values(by="Outlier Votes", ascending=False).head(10)
                    st.dataframe(top_outliers.set_index("Case #"))

            show_network = st.sidebar.checkbox("Network Analysis", value=False)
            if show_network:
                st.markdown("## Network Analysis")

                import networkx as nx
                import plotly.graph_objects as go

                st.markdown("This network connects cases that share attributes like species or offender country.")

                default_features = ["Species", "Country of offenders"]
                network_features = st.multiselect(
                    "Select features to compare across cases:", 
                    options=[col for col in df_selected.columns if col != "Case #"],
                    default=default_features
                )

                if network_features:
                    # Prepare feature sets for each Case #
                    case_feature_sets = (
                        df_selected
                        .groupby("Case #")[network_features]
                        .agg(lambda x: set(x.dropna()))
                        .apply(lambda row: set().union(*row), axis=1)
                    )

                    G = nx.Graph()

                    # Create nodes
                    for case_id in case_feature_sets.index:
                        G.add_node(case_id)

                    # Create edges between cases that share features
                    case_ids = list(case_feature_sets.index)
                    for i in range(len(case_ids)):
                        for j in range(i + 1, len(case_ids)):
                            shared = case_feature_sets[case_ids[i]].intersection(case_feature_sets[case_ids[j]])
                            if shared:
                                G.add_edge(case_ids[i], case_ids[j], weight=len(shared))

                    if G.number_of_edges() == 0:
                        st.info("No connections were found between cases using the selected features.")
                    else:
                        pos = nx.spring_layout(G, seed=42)

                        edge_x = []
                        edge_y = []
                        for edge in G.edges():
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])

                        edge_trace = go.Scatter(
                            x=edge_x, y=edge_y,
                            line=dict(width=1, color='#888'),
                            hoverinfo='text',
                            mode='lines',
                            text=[f"Shared features: {G[u][v]['weight']}" for u, v in G.edges()]
                        )

                        node_x = []
                        node_y = []
                        node_text = []
                        for node in G.nodes():
                            x, y = pos[node]
                            node_x.append(x)
                            node_y.append(y)
                            degree = G.degree[node]
                            node_text.append(f"Case #: {node} ({degree} connections)")

                        node_trace = go.Scatter(
                            x=node_x, y=node_y,
                            mode='markers+text',
                            text=node_text,
                            textposition='top center',
                            hoverinfo='text',
                            marker=dict(
                                showscale=False,
                                color='lightblue',
                                size=[4 + 0.5*G.degree[node] for node in G.nodes()],
                                line_width=1
                            )
                        )

                        fig = go.Figure(
                            data=[edge_trace, node_trace],
                            layout=go.Layout(
                                title='Case Connectivity Network',
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20, l=5, r=5, t=40)
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("### Network Metrics")

                        num_nodes = G.number_of_nodes()
                        num_edges = G.number_of_edges()
                        density = nx.density(G)
                        components = nx.number_connected_components(G)
                        degrees = dict(G.degree())
                        avg_degree = sum(degrees.values()) / num_nodes if num_nodes else 0
                        top_central = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]

                        st.write(f"- **Nodes:** {num_nodes}")
                        st.write(f"- **Edges:** {num_edges}")
                        st.write(f"- **Density:** `{density:.3f}`")
                        st.write(f"- **Connected components:** {components}")
                        st.write(f"- **Average degree:** `{avg_degree:.2f}`")

                        st.markdown("**Top central cases by degree:**")
                        for case_id, degree in top_central:
                            st.markdown(f"- Case `{case_id}`: {degree} connections")
                        
                        with st.expander("ℹ️ Learn more about this analysis"):
                            st.markdown("""
                            ### About Case Network Analysis

                            This section visualizes a network of wildlife trafficking cases based on shared attributes such as species, offender countries, or other relevant fields.

                            - **Each node in the network represents a unique case** (`Case #`).
                            - **An edge between two cases indicates that they share one or more selected attributes**, such as:
                                - The same species involved,
                                - The same offender country,
                                - Other user-selected fields (e.g., seizure location, transport method).

                            - The more attributes two cases have in common, the **stronger their connection** (i.e., higher edge weight).
                            - **Edge weight** represents the number of shared elements between the two cases, and is displayed interactively when hovering over connections.

                            - This type of network helps to:
                                - **Identify clusters of related cases**, which may signal recurrent patterns, shared trafficking routes, or organizational links.
                                - **Visualize potential case consolidation** (e.g., repeated behavior by the same actors or coordinated multi-species trafficking).
                                - Reveal connections that may not be obvious in tabular data.

                            - Node size reflects the number of connections (degree), helping to identify central or highly connected cases.
                            - The analysis is dynamic: users can choose which attributes to include, allowing flexible exploration of the data.

                            - For example:
                                - If two cases both involve *Panthera onca* and occurred with offenders from Brazil, a connection is drawn.
                                - If a third case shares only the species but not the country, it will also connect, but with a lower weight.

                            - For more information on network methods in wildlife trafficking analysis, refer to the ReadMe file and Carvalho (2025).
                            """)
                else:
                    st.info("Please select at least one feature to define connections between cases.")
                    
        else:
            st.warning("⚠️ Please select at least one species to explore the data.")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def place_logo_bottom_right(image_path, width=100):
    img_base64 = get_base64_image(image_path)
    st.markdown(
        f"""
        <style>
        .custom-logo {{
            position: fixed;
            bottom: 40px;
            right: 10px;
            z-index: 9999;
        }}
        </style>
        <div class="custom-logo">
            <img src="data:image/png;base64,{img_base64}" width="{width}"/>
        </div>
        """,
        unsafe_allow_html=True
    )

# Chamada da função para exibir a logo
place_logo_bottom_right("wcs.jpg")

st.sidebar.markdown("## Export Options")
export_xlsx = st.sidebar.button("Export Cleaned data.xlsx")
export_html = st.sidebar.button("Export Analysis Report (.html)")

if export_xlsx and df_selected is not None:
    from io import BytesIO
    towrite = BytesIO()
    df_selected.to_excel(towrite, index=False, engine='openpyxl')
    towrite.seek(0)
    st.download_button(
        label="Download Cleaned Excel File",
        data=towrite,
        file_name="aurum_cleaned_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
if export_html and df_selected is not None:
    from datetime import datetime
    now = datetime.now(brt).strftime("%Y-%m-%d %H:%M:%S (BRT)")

    html_sections = []
    html_sections.append(f"<h1>Aurum Wildlife Trafficking Report</h1>")
    html_sections.append(f"<p><strong>Generated:</strong> {now}</p>")
    html_sections.append(f"<p><strong>Selected Species:</strong> {', '.join(selected_species)}</p>")

    # Tabela de dados
    html_sections.append("<h2>Data Sample</h2>")
    html_sections.append(df_selected.head(10).to_html(index=False))

    # Resultados de tendência
    if show_trend:
        html_sections.append("<h2>Trend Analysis</h2>")
        html_sections.append(f"<p><strong>TCS:</strong> {tcs:.2f}</p>")

        # Salvar figura
        trend_buf = BytesIO()
        fig.savefig(trend_buf, format="png", bbox_inches="tight")
        trend_buf.seek(0)
        trend_base64 = base64.b64encode(trend_buf.read()).decode("utf-8")
        html_sections.append(f'<img src="data:image/png;base64,{trend_base64}" width="700">')

    # Coocorrência
    if show_cooc and co_results:
        html_sections.append("<h2>Species Co-occurrence</h2>")
        for sp_a, sp_b, chi2, p, table in co_results:
            html_sections.append(f"<h4>{sp_a} × {sp_b}</h4>")
            html_sections.append(table.to_html())
            html_sections.append(f"<p>Chi² = {chi2:.2f} | p = {p:.4f}</p>")

    # Anomalias
    if show_anomaly and 'vote_df' in locals():
        html_sections.append("<h2>Anomaly Detection</h2>")
        html_sections.append(f"<p><strong>Consensus Outlier Ratio:</strong> {consensus_ratio:.2%}</p>")
        html_sections.append("<h4>Top Anomalies</h4>")
        html_sections.append(top_outliers.to_html(index=False))

    # Finaliza o HTML
    html_report = f"""
    <html>
    <head><meta charset='utf-8'><title>Aurum Report</title></head>
    <body>{''.join(html_sections)}</body>
    </html>
    """

    report_bytes = BytesIO()
    report_bytes.write(html_report.encode("utf-8"))
    report_bytes.seek(0)

    st.download_button(
        label="Download HTML Report",
        data=report_bytes,
        file_name="aurum_report.html",
        mime="text/html"
    )

# --- LOGIN ---
st.sidebar.markdown("---")

if "user" in st.session_state:
    st.sidebar.markdown(f"✅ **{st.session_state['user']}** is connected.")
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

else:
    st.sidebar.markdown("## 🔐 Login to Aurum")

    # Usando keys explícitos para controlar valores via session_state
    if "login_username" not in st.session_state:
        st.session_state["login_username"] = ""
    if "login_password" not in st.session_state:
        st.session_state["login_password"] = ""

    username = st.sidebar.text_input("Username", key="login_username")
    password = st.sidebar.text_input("Password", type="password", key="login_password")

    login_col, _ = st.sidebar.columns([1, 1])
    login_button = login_col.button("Login")

    def verify_password(password, hashed):
        return password == hashed_pw

    if login_button and username and password:
        user_row = users_df[users_df["Username"] == username]
        if not user_row.empty and str(user_row.iloc[0]["Approved"]).strip().lower() == "true":
            hashed_pw = user_row.iloc[0]["Password"].strip()

            if verify_password(password, hashed_pw):
                st.session_state["user"] = username
                st.session_state["user_email"] = user_row.iloc[0]["E-mail"]
                st.session_state["is_admin"] = str(user_row.iloc[0]["Is_Admin"]).strip().lower() == "true"

                # Limpa os campos após login bem-sucedido
                st.session_state.pop("login_username", None)
                st.session_state.pop("login_password", None)

                st.rerun()
            else:
                st.error("Incorrect password.")
        else:
            st.error("User not approved or does not exist.")

# --- FORMULÁRIO DE ACESSO (REQUISIÇÃO) ---
# Inicializa estado
if "show_sidebar_request" not in st.session_state:
    st.session_state["show_sidebar_request"] = False

# Botão na sidebar
if st.sidebar.button("📩 Request Access"):
    st.session_state["show_sidebar_request"] = True

# Exibe o formulário de solicitação na sidebar se o botão foi clicado
if st.session_state["show_sidebar_request"]:
    with st.sidebar.form("sidebar_request_form"):
        new_username = st.text_input("Choose a username", key="sidebar_user")
        new_password = st.text_input("Choose a password", type="password", key="sidebar_pass")
        institution = st.text_input("Institution", key="sidebar_inst")
        email = st.text_input("E-mail", key="sidebar_email")
        reason = st.text_area("Why do you want access to Aurum?", key="sidebar_reason")
        submit_request = st.form_submit_button("Submit Request")

        if submit_request:
            if not new_username or not new_password or not reason:
                st.sidebar.warning("All fields are required.")
            else:
                timestamp = datetime.now(brt).strftime("%Y-%m-%d %H:%M:%S (BRT)")
                requests_ws.append_row([
                    timestamp,
                    new_username,
                    new_password,
                    institution,
                    email,
                    reason
                ])
                st.sidebar.success("✅ Request submitted!")
                st.session_state["show_sidebar_request"] = False

if st.session_state.get("is_admin"):
    st.markdown("## 🛡️ Admin Panel - Approve Access Requests")
    request_df = pd.DataFrame(requests_ws.get_all_records())
    if not request_df.empty:
        st.dataframe(request_df)

        with st.form("approve_form"):
            new_user = st.selectbox("Select username to approve:", request_df["Username"].unique())
            new_password = st.text_input("Set initial password", type="password")
            is_admin = st.checkbox("Grant admin access?")
            approve_button = st.form_submit_button("Approve User")

            if approve_button:
                if not new_user or not new_password:
                    st.warning("Username and password are required.")
                else:
                    try:
                        # Buscar linha correspondente
                        user_row = request_df[request_df["Username"] == new_user]
                        if user_row.empty:
                            st.warning("User not found in access requests.")
                        else:
                            row_index = user_row.index[0]
                            is_admin_str = "TRUE" if is_admin else "FALSE"

                            # Pega o E-mail associado do Access Requests
                            email = user_row.iloc[0]["E-mail"].strip()

                            # Atualizar Access Requests
                            requests_ws.update_cell(row_index + 2, request_df.columns.get_loc("Approved") + 1, "TRUE")
                            requests_ws.update_cell(row_index + 2, request_df.columns.get_loc("Is_Admin") + 1, is_admin_str)

                            # Verificar se já existe na aba Users
                            users_df = pd.DataFrame(users_ws.get_all_records())
                            if new_user not in users_df["Username"].values:
                                users_ws.append_row([
                                    new_user,
                                    new_password,
                                    email,
                                    is_admin_str,
                                    "TRUE"
                                ])

                            st.success(f"✅ {new_user} has been approved and added to the system.")
                            st.info("🔐 The user is now authorized to log into Aurum.")
                    except Exception as e:
                        st.error(f"❌ Failed to approve user: {e}")

# --- FORMULÁRIO ---
def get_worksheet(sheet_name="Aurum_data"):
    gc = gspread.authorize(credentials)
    sh = gc.open_by_key("1HVYbot3Z9OBccBw7jKNw5acodwiQpfXgavDTIptSKic")
    return sh.worksheet(sheet_name)

# --- Função para carregar dados de qualquer aba ---
def load_sheet_data(sheet_name, sheets):
    try:
        worksheet = sheets.worksheet(sheet_name)
        records = worksheet.get_all_records()
        return pd.DataFrame(records)
    except Exception as e:
        st.error(f"❌ Failed to load data from sheet '{sheet_name}': {e}")
        return pd.DataFrame()

# --- Função para submissão de alertas ---
ddef display_alert_submission_form(sheet_id):
    sheets = connect_to_sheets()
    if sheets is None:
        st.error("❌ Unable to connect to the database to submit alerts.")
        return

    field_keys = {
        "title": "alert_title_input",
        "description": "alert_description_input",
        "category": "alert_category_select",
        "risk_level": "alert_risk_select",
        "species": "alert_species_input",
        "country": "alert_country_input",
        "source_link": "alert_source_input",
        "author_choice": "alert_author_choice"
    }

    categories = ["Species", "Country", "Marketplace", "Operation", "Policy", "Other"]
    risk_levels = ["Low", "Medium", "High"]

    for key in ["title", "description", "species", "country", "source_link"]:
        st.session_state.setdefault(field_keys[key], "")

    if st.session_state.get(field_keys["category"]) not in categories:
        st.session_state[field_keys["category"]] = categories[0]
    if st.session_state.get(field_keys["risk_level"]) not in risk_levels:
        st.session_state[field_keys["risk_level"]] = risk_levels[0]

    st.session_state.setdefault(field_keys["author_choice"], "Show my username")

    with st.form("alert_form"):
        title = st.text_input("Alert Title", key=field_keys["title"])
        description = st.text_area("Alert Description", key=field_keys["description"])
        category = st.selectbox("Category", categories, key=field_keys["category"])
        risk_level = st.selectbox("Risk Level", risk_levels, key=field_keys["risk_level"])
        species = st.text_input("Species involved (optional)", key=field_keys["species"])
        country = st.text_input("Country or Region (optional)", key=field_keys["country"])
        source_link = st.text_input("Source Link (optional)", key=field_keys["source_link"])

        author_choice = st.radio(
            "Choose how to display your name:",
            ["Show my username", "Submit anonymously"],
            key=field_keys["author_choice"]
        )

        created_by = st.session_state["user_email"]
        display_as = st.session_state["user"] if author_choice == "Show my username" else "Anonymous"

        submitted = st.form_submit_button("📤 Submit Alert")

    if submitted:
        if not title or not description:
            st.warning("Title and Description are required.")
        else:
            alert_id = str(uuid4())
            created_at = datetime.now(brt).strftime("%Y-%m-%d %H:%M:%S (BRT)")
            public = True

            alert_row = [
                alert_id, created_at, created_by, display_as, title, description,
                category, species, country, risk_level, source_link,
                str(public)
            ]

            try:
                worksheet = sheets.worksheet("Alerts")
                worksheet.append_row(alert_row, value_input_option="USER_ENTERED")
                st.success("✅ Alert submitted successfully!")
                st.balloons()

                # Limpa os campos do formulário
                for k in field_keys.values():
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to submit alert: {e}")

def display_alert_update_timeline(sheet_id):
    df_alerts = load_alerts()
    df_updates = load_alert_updates()

    if df_alerts.empty:
        st.error("❌ Could not load alerts from the database.")
        return

    user = st.session_state["user"]

    # Alertas criados pelo usuário
    created_alerts = df_alerts[df_alerts["Created By"] == user]

    # Alertas atualizados pelo usuário
    relevant_updates = df_updates[
        (df_updates["User"] == user) | (df_updates["User"] == "Anonymous")
    ]
    updated_alert_ids = relevant_updates["Alert ID"].unique()
    updated_alerts = df_alerts[df_alerts["Alert ID"].isin(updated_alert_ids)]

    # Junta ambos
    df_user_alerts = pd.concat([created_alerts, updated_alerts]).drop_duplicates(subset="Alert ID")

    if df_user_alerts.empty:
        st.info("You haven't submitted or updated any alerts yet.")
        return

    selected_title = st.selectbox("Select an alert to update:", df_user_alerts["Title"].tolist())
    selected_row = df_user_alerts[df_user_alerts["Title"] == selected_title].iloc[0]
    alert_id = selected_row["Alert ID"]

    timeline = df_updates[df_updates["Alert ID"] == alert_id].sort_values("Timestamp")

    if not timeline.empty:
        st.markdown("### Update Timeline")
        for _, row in timeline.iterrows():
            st.markdown(f"**{row['Timestamp']}** – *{row['User']}*: {row['Update Text']}")
    else:
        st.info("This alert has no updates yet.")

    update_author_choice = st.radio(
        "Choose how to display your name in this update:",
        ["Show my username", "Submit anonymously"],
        key="update_author_choice"
    )
    update_user = user if update_author_choice == "Show my username" else "Anonymous"

    with st.form(f"update_form_{alert_id}"):
        st.markdown("**Add a new update to this alert:**")
        new_update = st.text_area("Update Description")
        submitted = st.form_submit_button("➕ Add Update")

    if submitted and new_update.strip():
        sheets = connect_to_sheets()
        if sheets is None:
            st.error("❌ Unable to connect to submit update.")
            return

        try:
            update_row = [alert_id, datetime.now(brt).strftime("%Y-%m-%d %H:%M:%S (BRT)"), update_user, new_update.strip()]
            try:
                update_ws = sheets.worksheet("Alert Updates")
            except gspread.exceptions.WorksheetNotFound:
                update_ws = sheets.add_worksheet(title="Alert Updates", rows="1000", cols="4")
                update_ws.append_row(["Alert ID", "Timestamp", "User", "Update Text"])

            update_ws.append_row(update_row)
            st.success("✅ Update added to alert!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to add update: {e}")

# --- Interface em colunas: Alertas (superior) e Casos (inferior) ---
if "user" in st.session_state:

    # --- Colunas superiores: Alertas ---
    st.markdown("### Alert Management")
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("**Submit New Alert**", expanded=False):
            display_alert_submission_form(SHEET_ID)

    with col2:
        with st.expander("**Update My Alerts**", expanded=False):
            display_alert_update_timeline(SHEET_ID)

    # --- Colunas inferiores: Casos ---
    st.markdown("### Case Management")
    col3, col4 = st.columns(2)

     with col3:
        with st.expander("**Submit New Case**", expanded=False):
            # Chaves dos campos para controlar o form
            field_keys = {
                "case_id": "case_id_input",
                "n_seized": "n_seized_input",
                "year": "year_input",
                "country": "country_input",
                "seizure_status": "seizure_status_input",
                "transit": "transit_input",
                "notes": "notes_input"
            }

            default_values = {
                "case_id": "",
                "n_seized": "",
                "year": 2024,
                "country": "",
                "seizure_status": "",
                "transit": "",
                "notes": ""
            }

            for key, default in default_values.items():
                st.session_state.setdefault(field_keys[key], default)

            with st.form("aurum_form"):
                case_id = st.text_input("Case #", key=field_keys["case_id"])
                seizure_country = st.text_input("Country of seizure or shipment")
                n_seized = st.text_input("N seized specimens (e.g. 2 lion + 1 chimpanze)", key=field_keys["n_seized"])
                year = st.number_input("Year", step=1, format="%d", min_value=1900, max_value=2100, key=field_keys["year"])
                country = st.text_input("Country of offenders", key=field_keys["country"])
                seizure_status = st.text_input("Seizure status", key=field_keys["seizure_status"])
                transit = st.text_input("Transit feature", key=field_keys["transit"])
                notes = st.text_area("Additional notes", key=field_keys["notes"])

                submitted = st.form_submit_button("Submit Case")

            if submitted:
                new_row = [
                    datetime.now(brt).strftime("%Y-%m-%d %H:%M:%S (BRT)"),
                    case_id,
                    seizure_country,
                    n_seized,
                    year,
                    country,
                    seizure_status,
                    transit,
                    notes,
                    st.session_state["user"]
                ]

                worksheet = get_worksheet()
                if worksheet:
                    worksheet.append_row(new_row)
                    st.success("✅ Case submitted to Aurum successfully!")

                    for k in field_keys.values():
                        if k in st.session_state:
                            del st.session_state[k]
                    st.rerun()
                else:
                    st.error("❌ Failed to connect to database. Please try again later.")

    with col4:
        with st.expander("**Edit My Cases**", expanded=False):
            worksheet = get_worksheet()
            if worksheet:
                try:
                    records = worksheet.get_all_records()
                    df_user = pd.DataFrame(records)
                    df_user = df_user[df_user["Author"] == st.session_state["user"]]

                    if df_user.empty:
                        st.info("You haven't submitted any cases yet.")
                    else:
                        selected_case = st.selectbox("Select a case to edit:", df_user["Case #"].unique())
                        if selected_case:
                            row_index = df_user[df_user["Case #"] == selected_case].index[0] + 2
                            current_row = df_user.loc[df_user["Case #"] == selected_case].iloc[0]

                            with st.form("edit_case_form"):
                                new_case_id = st.text_input("Case #", value=current_row["Case #"])
                                new_seizure_country = st.text_input("Country of seizure or shipment", value=current_row["Country of seizure or shipment"])
                                new_n_seized = st.text_input("N seized specimens", value=current_row["N seized specimens"])
                                new_year = st.number_input("Year", step=1, format="%d", value=int(current_row["Year"]))
                                new_country = st.text_input("Country of offenders", value=current_row["Country of offenders"])
                                new_status = st.text_input("Seizure status", value=current_row["Seizure status"])
                                new_transit = st.text_input("Transit feature", value=current_row["Transit feature"])
                                new_notes = st.text_area("Additional notes", value=current_row["Notes"])

                                submitted_edit = st.form_submit_button("Save Changes")

                            if submitted_edit:
                                updated_row = [
                                    current_row["Timestamp"],
                                    new_case_id,
                                    new_seizure_country,
                                    new_n_seized,
                                    new_year,
                                    new_country,
                                    new_status,
                                    new_transit,
                                    new_notes,
                                    st.session_state["user"]
                                ]
                                worksheet.update(f"A{row_index}:J{row_index}", [updated_row])
                                st.success("✅ Case updated successfully!")
                                st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to load or update your cases: {e}")
            else:
                st.error("❌ Unable to connect to load your cases.")

    st.subheader("Upload Multiple Cases (Batch Mode)")
    uploaded_file_batch = st.file_uploader(
        "Upload an Excel or CSV file with multiple cases",
        type=["xlsx", "csv"],
        key="uploaded_file_batch"
    )

    if uploaded_file_batch is not None:
        st.info("📄 File uploaded. Click the button below to confirm batch submission.")
        submit_batch = st.button("📥 **Submit Batch Upload**")

        if submit_batch:
            try:
                if uploaded_file_batch.name.endswith(".csv"):
                    batch_data = pd.read_csv(uploaded_file_batch)
                else:
                    batch_data = pd.read_excel(uploaded_file_batch)

                # Normaliza os nomes das colunas
                batch_data.columns = (
                    batch_data.columns
                    .str.normalize('NFKD')
                    .str.encode('ascii', errors='ignore')
                    .str.decode('utf-8')
                    .str.strip()
                    .str.lower()
                )

                required_cols_original = [
                    "Case #", "Country of seizure or shipment", "N seized specimens", "Year",
                    "Country of offenders", "Seizure status", "Transit feature", "Notes"
                ]
                required_cols_normalized = [col.lower() for col in required_cols_original]

                missing_cols = [
                    orig for orig, norm in zip(required_cols_original, required_cols_normalized)
                    if norm not in batch_data.columns
                ]

                if missing_cols:
                    st.error("🚫 Upload blocked: the uploaded file has incorrect formatting.")
                    st.markdown(f"""
                    The file must include the following required columns:

                    - Case #
                    - Country of seizure or shipment
                    - N seized specimens
                    - Year
                    - Country of offenders
                    - Seizure status
                    - Transit feature
                    - Notes

                    The following columns are missing:  
                    **{', '.join(missing_cols)}**

                    > 💡 Tip: You can download the correct template from the sidebar (“Download Template”) and fill it with your data.
                    """)
                else:
                    batch_data = batch_data.fillna("")
                    batch_data["Timestamp"] = datetime.now(brt).strftime("%Y-%m-%d %H:%M:%S (BRT)")
                    batch_data["Author"] = st.session_state["user"]

                    # Renomeia colunas normalizadas de volta para os nomes originais
                    rename_map = dict(zip(required_cols_normalized, required_cols_original))
                    batch_data.rename(columns=rename_map, inplace=True)

                    ordered_cols = [
                        "Timestamp", "Case #", "Country of seizure or shipment", "N seized specimens", "Year",
                        "Country of offenders", "Seizure status", "Transit feature",
                        "Notes", "Author"
                    ]
                    batch_data = batch_data[ordered_cols]

                    rows_to_append = batch_data.values.tolist()

                    worksheet = get_worksheet()
                    if worksheet:
                        worksheet.append_rows(rows_to_append, value_input_option="USER_ENTERED")
                        st.success("✅ Batch upload completed successfully!")

                        if "uploaded_file_batch" in st.session_state:
                            del st.session_state["uploaded_file_batch"]

                        st.rerun()
                    else:
                        st.error("❌ Unable to connect to the database. Please try again later.")

            except Exception as e:
                st.error(f"❌ Error during upload: {e}")

    st.markdown("## My Cases")
    worksheet = get_worksheet()
    if worksheet:
        try:
            records = worksheet.get_all_records()
            if not records:
                st.info("No data available at the moment.")
            else:
                data = pd.DataFrame(records)

                # Aplica filtro de autor se não for admin
                if not st.session_state.get("is_admin"):
                    data = data[data["Author"] == st.session_state["user"]]

                # Filtro por espécie com base em "N seized specimens"
                if "N seized specimens" in data.columns:
                    species_matches = data["N seized specimens"].str.extractall(r'\d+\s*([A-Z][a-z]+(?:_[a-z]+)+)')
                    species_list = sorted(species_matches[0].dropna().unique())

                    selected_species = st.multiselect("Filter by species:", species_list)

                    if selected_species:
                        data = data[data["N seized specimens"].str.contains("|".join(selected_species))]

                st.dataframe(data)

        except Exception as e:
            st.error(f"❌ Failed to load your cases: {e}")
    else:
        st.error("❌ Unable to connect to load your cases.")

# --- SUGGESTIONS AND COMMENTS (SIDEBAR) ---
if "show_sidebar_feedback" not in st.session_state:
    st.session_state["show_sidebar_feedback"] = False

# --- BOTÃO FIXO NA SIDEBAR ---
feedback_toggle = st.sidebar.button("💬 Suggestions and Comments")

# Alterna visibilidade do formulário
if feedback_toggle:
    st.session_state["show_sidebar_feedback"] = not st.session_state["show_sidebar_feedback"]

# Exibe o formulário se o botão estiver ativado
if st.session_state["show_sidebar_feedback"]:
    with st.sidebar.form("suggestion_form"):
        st.markdown("### 💬 Feedback Form")
        name = st.text_input("Name", key="suggestion_name")
        email = st.text_input("E-mail", key="suggestion_email")
        institution = st.text_input("Institution", key="suggestion_institution")
        message = st.text_area("Suggestions or comments", key="suggestion_message")
        submitted = st.form_submit_button("Submit")

        if submitted:
            if not name or not email or not institution or not message:
                st.warning("All fields are required.")
            else:
                timestamp = datetime.now(brt).strftime("%Y-%m-%d %H:%M:%S (BRT)")

                sheets = connect_to_sheets()
                if sheets:
                    try:
                        try:
                            suggestion_ws = sheets.worksheet("Suggestions")
                        except gspread.exceptions.WorksheetNotFound:
                            suggestion_ws = sheets.add_worksheet(title="Suggestions", rows="1000", cols="5")
                            suggestion_ws.append_row(["Timestamp", "Name", "Email", "Institution", "Message"])

                        suggestion_ws.append_row([
                            timestamp,
                            name,
                            email,
                            institution,
                            message.strip()
                        ])

                        st.success("✅ Thank you for your feedback!")
                        st.session_state["show_sidebar_feedback"] = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to submit feedback: {e}")
                else:
                    st.error("❌ Unable to connect to submit feedback.")

def display_suggestions_section(SHEET_ID):
    """Exibe sugestões enviadas pelo formulário (apenas admins)."""
    if not st.session_state.get("is_admin"):
        return

    st.markdown("## 💬 User Suggestions and Comments")

    sheets = connect_to_sheets()
    if sheets:
        try:
            suggestion_ws = sheets.worksheet("Suggestions")
            df = pd.DataFrame(suggestion_ws.get_all_records())
            df.columns = [col.strip() for col in df.columns]

            if df.empty:
                st.info("No feedback has been submitted yet.")
            else:
                st.dataframe(df.sort_values("Timestamp", ascending=False))

        except gspread.exceptions.WorksheetNotFound:
            st.warning("📭 The 'Suggestions' sheet was not found.")
        except Exception as e:
            st.error(f"❌ Failed to load suggestions: {e}")
    else:
        st.error("❌ Unable to connect to load suggestions.")

# CHAMADA DA VISUALIZAÇÃO (APENAS ADMIN)
display_suggestions_section(SHEET_ID)

st.sidebar.markdown("---")    
st.sidebar.markdown("**How to cite:** Carvalho, A. F. Aurum: A Platform for Criminal Intelligence in Wildlife Trafficking. Wildlife Conservation Society, 2025.")
