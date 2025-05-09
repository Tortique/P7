import json

import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(page_title="Prédiction de Solvabilité", layout="wide")

# Fonction pour récupérer les données du client avec mise en cache
@st.cache_data
def get_client_data(client_id):
    API_DATA_URL = f"https://p7-ywri.onrender.com/client/{client_id}"
    response = requests.get(API_DATA_URL)
    if response.status_code == 200:
        return response.json()
    else:
        return None

@st.cache_data
def get_column_data(column_name):
    url = f"https://p7-ywri.onrender.com/column/{column_name}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Erreur pour la colonne {column_name}")
        return None

variables_to_plot = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_CREDIT",
    "DAYS_BIRTH",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
    "DAYS_REGISTRATION",
    "AMT_INCOME_TOTAL",
    "REGION_POPULATION_RELATIVE",
    "NAME_CONTRACT_TYPE"
]

# Variables modifiables
modifiable_vars = [
    "AMT_CREDIT",
    "DAYS_BIRTH",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
    "DAYS_REGISTRATION",
    "AMT_INCOME_TOTAL"
]

@st.cache_data
def load_local_importances(path="all_local_importances.json"):
    with open(path, "r") as f:
        return json.load(f)

@st.cache_data
def get_prediction(score):
    seuil_bas = 0.7
    seuil_haut = 0.78
    seuil_max = 0.85
    if score < seuil_bas:
        st.success(f"🟢 Score : {score:.2f}")
        st.success("Client considéré comme **solvable**.")
    elif seuil_bas <= score < seuil_haut:
        st.warning(f"🟡 Score : {score:.2f}")
        st.warning("Client **solvable mais à vérifier**.")
    elif seuil_haut <= score < seuil_max:
        st.warning(f"🟠 Score : {score:.2f}")
        st.warning("Client **non solvable à vérifier**.")
    else:
        st.error(f"🔴 Score : {score:.2f}")
        st.error("Client considéré comme **non solvable**.")

# ---------------------- UI ----------------------

# Initialisation de la session state
if "afficher_donnees" not in st.session_state:
    st.session_state.afficher_donnees = False

# Initialisation du fichier importances locales
all_local_importances = load_local_importances()

# Champ d'identifiant client
client_id = st.sidebar.number_input("ID du client", value=100001)

# Bouton dans la sidebar
if st.sidebar.button("📥 Charger"):
    if client_id:
        st.session_state.afficher_donnees = True
        data = get_client_data(client_id)
        if data:
            st.session_state["data"] = data
        else:
            st.session_state.afficher_donnees = False
            st.error("❌️ ID Client introuvable.")
    else:
        st.warning("⚠️ Veuillez entrer un identifiant.")

if st.session_state.afficher_donnees and "data" in st.session_state:
    data = st.session_state.data

    st.header(f"Fiche du client #{client_id}")
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.subheader("📌 Informations personnelles")
        st.metric("Âge", f"{-int(data['DAYS_BIRTH'] / 365)} ans")
        st.metric("Revenu annuel", f"{int(data['AMT_INCOME_TOTAL']):,} €")
        st.metric("Nombre d’enfants", int(data["CNT_CHILDREN"]))
        st.metric("Situation", "Marié" if data.get("NAME_FAMILY_STATUS_Married") else "Autre")

    with col2:
        st.subheader("💳 Crédit demandé")
        st.metric("Montant crédit", f"{int(data['AMT_CREDIT']):,} €")
        st.metric("Annuité", f"{int(data['AMT_ANNUITY']):,} €")
        st.metric("Montant bien", f"{int(data['AMT_GOODS_PRICE']):,} €")

    with col3:
        st.subheader("🏠 Autres")
        st.metric("Voiture", "✅ Oui" if data["FLAG_OWN_CAR"] else "❌ Non")
        st.metric("Propriétaire immobilier", "✅ Oui" if data["FLAG_OWN_REALTY"] else "❌ Non")
        st.metric("Type logement", "Maison / Appartement" if data.get("NAME_HOUSING_TYPE_House / apartment") else "Autre")

    st.session_state["data"] = data

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        with st.expander("Comparaison par rapport à la population"):

            selected_var = st.selectbox("Choisissez une variable à analyser :", variables_to_plot,
                                        help="Variable pour laquelle on comparera la valeur du client à celle de la population")

            if selected_var and st.session_state.afficher_donnees:
                st.markdown(f"### Analyse de `{selected_var}`")

                client_value = data.get(selected_var, None)
                if client_value is not None:
                    st.write(f"Valeur pour le client : `{client_value}`")

                    # Récupération de la distribution
                    column_data = get_column_data(selected_var)

                    if column_data:
                        df = pd.DataFrame({selected_var: column_data})

                        # Créer le graphique avec Seaborn
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.histplot(df[selected_var], kde=False, bins=30, ax=ax, color="#1b9e77")
                        sns.set_palette("colorblind")
                        ax.axvline(client_value, color='#d95f02', linestyle='--', label='Client')
                        ax.set_title(f"Distribution de {selected_var}")
                        ax.legend()

                        st.pyplot(fig)
                        st.caption(f"Distribution de la variable {selected_var} dans la population. La ligne rouge "
                                   f"indique la valeur du client.")
                    else:
                        st.warning(f"Donnée indisponible pour la variable : `{selected_var}`")
                else:
                    st.warning(f"Valeur manquante pour la variable : `{selected_var}`")
    with col_exp2:
        matching = [item for item in all_local_importances if float(item["id"]) == float(client_id)]
        if matching:
            instance_data = matching[0]
            with st.expander("Voir les données les plus importantes"):
                local_df = pd.DataFrame(instance_data["features"])
                local_df = local_df.sort_values("importance", key=abs, ascending=False).head(15)

                st.subheader(f"Importance locale pour le client {client_id}")

                features = local_df["feature"]
                importances = local_df["importance"]

                colors = ['#b22222' if val > 0 else '#2a4d9b' for val in importances]  # rouge si positif, bleu si négatif

                fig = go.Figure(go.Bar(
                    x=importances,
                    y=features,
                    orientation='h',
                    marker_color=colors,
                ))

                fig.update_layout(
                    title="Contribution des variables à la décision",
                    xaxis_title="Importance (valeurs SHAP approximatives)",
                    yaxis_title="",
                    yaxis=dict(autorange="reversed"),
                    plot_bgcolor='white',
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Aucune Importance locale pour le client")
    # Bouton pour prédire la solvabilité
    if st.button("🔮 Prédire la solvabilité",
                 help="Score entre 0 et 1, plus le score s'approche de 1 plus le prêt est risqué"):
        if 'predition' not in st.session_state or st.session_state.client_id != client_id:
            API_PREDICT_URL = "https://p7-ywri.onrender.com/predict"  # URL de ton API
            response = requests.post(API_PREDICT_URL, json={'id_client': client_id})
            if response.status_code == 200:
                prediction = response.json()
                st.session_state.prediction = prediction
                st.session_state.client_id = client_id
                score = prediction["prediction"][0]
                get_prediction(score)
            else:
                st.error("🔴 Erreur lors de la prédiction")
        else:
            prediction = st.session_state.prediction
            st.write(f"Prédiction déjà effectuée : {prediction['prediction']}")
    with st.expander("Modifier plusieurs variables"):
        with st.form("modification_multiple"):
            new_values = {}
            for var in modifiable_vars:
                current_val = float(data.get(var, 0))
                new_values[var] = st.number_input(f"{var}", value=current_val)

            submitted = st.form_submit_button("🔄 Simuler avec nouvelles valeurs")
            if submitted:
                modifications = {
                    k: v for k, v in new_values.items() if v != float(data.get(k, 0))
                }

                if modifications:
                    body = {
                        "id_client": st.session_state["client_id"],
                        "modifications": modifications
                    }
                    response = requests.post("https://p7-ywri.onrender.com/predict_custom", json=body)
                    if response.status_code == 200:
                        prediction = response.json()
                        st.session_state.prediction = prediction
                        st.session_state.client_id = client_id
                        score = prediction["prediction"][0]
                        get_prediction(score)
    # Ajout d’un expander pour debug / données brutes
    with st.expander("Voir les données complètes"):
        st.json(data)