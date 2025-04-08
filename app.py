import pandas as pd
import streamlit as st
import requests

# Configuration de la page
st.set_page_config(page_title="Prédiction de Solvabilité", layout="wide")

# Titre de l'application
st.title("🔍 Prédiction de Solvabilité")

# 🏦 Fonction pour récupérer les données du client avec mise en cache
@st.cache_data(ttl=60)
def get_client_data(client_id):
    API_DATA_URL = f"http://127.0.0.1:8000/client/{client_id}"
    response = requests.get(API_DATA_URL)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# 🆔 Champ d'identifiant client
client_id = st.text_input("🆔 Entrez l'identifiant du client", "")

if st.button("📥 Charger les données du client"):
    if client_id:
        client_data = get_client_data(client_id)  # Utilisation du cache
        if client_data:
            df_client = pd.DataFrame(client_data.items(), columns=["Variable", "Valeur"])
            st.write("### 📊 Données du client")
            st.dataframe(df_client)
            st.session_state["client_data"] = client_data
        else:
            st.error("🔴 Client introuvable ! Vérifiez l'ID.")
    else:
        st.warning("⚠️ Veuillez entrer un identifiant.")

# Vérification si les données sont chargées
if "client_data" in st.session_state:
    client_data = st.session_state["client_data"]

    # Bouton pour prédire la solvabilité
    if st.button("🔮 Prédire la solvabilité"):
        API_PREDICT_URL = "http://127.0.0.1:8000/predict"  # URL de ton API
        response = requests.post(API_PREDICT_URL, json={'id_client': client_id})

        if response.status_code == 200:
            prediction = response.json()
            st.success(f"🟢 Résultat : {prediction['prediction']}")
        else:
            st.error("🔴 Erreur lors de la prédiction")