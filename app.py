import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de la page
st.set_page_config(page_title="Prédiction de Solvabilité", layout="wide")

# 🏦 Fonction pour récupérer les données du client avec mise en cache
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
        return pd.Series(response.json())
    else:
        st.error(f"Erreur pour la colonne {column_name}")
        return None

variables_to_plot = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3"
]

# 🆔 Champ d'identifiant client
client_id = st.sidebar.number_input("ID du client", value=100001)

if st.sidebar.button("📥 Charger"):
    if client_id:
        data = get_client_data(client_id)  # Utilisation du cache

        if data:
            st.markdown(f"## Fiche du client #{client_id}")
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.markdown("### 📌 Informations personnelles")
                st.metric("Âge", f"{-int(data['DAYS_BIRTH'] / 365)} ans")
                st.metric("Revenu annuel", f"{int(data['AMT_INCOME_TOTAL']):,} €")
                st.metric("Nombre d’enfants", int(data["CNT_CHILDREN"]))
                st.metric("Situation", "Marié" if data.get("NAME_FAMILY_STATUS_Married") else "Autre")

            with col2:
                st.markdown("### 💳 Crédit demandé")
                st.metric("Montant crédit", f"{int(data['AMT_CREDIT']):,} €")
                st.metric("Annuité", f"{int(data['AMT_ANNUITY']):,} €")
                st.metric("Montant bien", f"{int(data['AMT_GOODS_PRICE']):,} €")

            with col3:
                st.markdown("### 🏠 Autres")
                st.metric("Voiture", "✅" if data["FLAG_OWN_CAR"] else "❌")
                st.metric("Immobilier", "✅" if data["FLAG_OWN_REALTY"] else "❌")
                st.metric("Type logement", "Appartement" if data.get("NAME_HOUSING_TYPE_House / apartment") else "Autre")

            st.session_state["data"] = data

            # Ajout d’un expander pour debug / données brutes
            with st.expander("Voir les données complètes"):
                st.json(data)
        else:
            st.error("❌️ ID Client introuvable.")
    else:
        st.warning("⚠️ Veuillez entrer un identifiant.")

        st.subheader("Comparaison par rapport à la population")

    for var in variables_to_plot:
        col_data = get_column_data(var)

        if col_data is not None and var in data:
            fig, ax = plt.subplots()
            sns.histplot(col_data, bins=30, color="lightblue", kde=True, ax=ax)
            ax.axvline(data[var], color="red", linestyle="--", label=f"Client : {data[var]:,.2f}")
            ax.set_title(f"Distribution de {var}")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning(f"Donnée indisponible pour la variable : {var}")

        # Vérification si les données sont chargées
        if "data" in st.session_state:
            data = st.session_state["data"]

            # Bouton pour prédire la solvabilité
            if st.button("🔮 Prédire la solvabilité"):
                API_PREDICT_URL = "https://p7-ywri.onrender.com/predict"  # URL de ton API
                response = requests.post(API_PREDICT_URL, json={'id_client': client_id})

                if response.status_code == 200:
                    seuil = 0.78
                    prediction = response.json()
                    if prediction["prediction"][0] < seuil:
                        st.success(f"🟢 Résultat : {prediction['prediction']}")
                        st.success(f"Client considéré comme solvable")
                    else:
                        st.error(f"🔴 Résultat : {prediction['prediction']}")
                        st.error(f"Client considéré comme *non* solvable")
                else:
                    st.error("🔴 Erreur lors de la prédiction")