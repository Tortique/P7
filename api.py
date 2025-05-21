import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

app = FastAPI()

model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
df_test = joblib.load('df_test.pkl')

if not hasattr(model, "predict"):
    raise ValueError("❌ ERREUR : Le modèle chargé n'est pas entraîné ! Vérifie que tu l'as bien `fit()` avant de le sauvegarder.")

print(f"📊 Modèle sélectionné : {model}")
print(f"🔍 Type du modèle : {type(model)}")
print(f"📢 Le modèle est-il entraîné ? {hasattr(model, 'booster_')}")
class ClientRequest(BaseModel):
    id_client: int

class CustomClientRequest(BaseModel):
    id_client: int
    modifications: dict

@app.post("/predict")
async def predict(request: ClientRequest):
    client_id = request.id_client

    df_test["SK_ID_CURR"] = pd.to_numeric(df_test["SK_ID_CURR"], errors='coerce')

    client_data = df_test[df_test["SK_ID_CURR"] == client_id]

    if client_data.empty:
        raise HTTPException(status_code=404, detail="Client non trouvé")

    scaled_features = scaler.transform(client_data)

    prediction = model.predict_proba(scaled_features)[:, 1]

    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()  # Convertir en liste Python
    if isinstance(prediction, (np.int64, np.int32)):
        prediction = int(prediction)  # Convertir en int natif

    return {"id_client": client_id, "prediction": prediction}

@app.get("/get_first")
def get_first():
    df_test["SK_ID_CURR"] = pd.to_numeric(df_test["SK_ID_CURR"], errors='coerce')
    return df_test.head(10).to_dict(orient="records")

@app.get("/client/{client_id}")
def get_client(client_id: int):
    # Conversion de la colonne id_client en numérique
    df_test["SK_ID_CURR"] = pd.to_numeric(df_test["SK_ID_CURR"], errors='coerce')

    # Filtrage des données du client
    client_data = df_test[df_test["SK_ID_CURR"] == client_id]

    # Vérifier si le client existe
    if client_data.empty:
        raise HTTPException(status_code=404, detail="Client non trouvé")

    # Retourner les données sous forme de dictionnaire
    return client_data.to_dict(orient="records")[0]

@app.get("/column/{column_name}")
def get_column(column_name: str):
    if column_name not in df_test.columns:
        return {"error": "Colonne non trouvée"}

    # Récupérer la colonne (sans les valeurs manquantes)
    column_values = df_test[column_name].dropna().tolist()

    # Préparation des données pour la prédiction (toutes les lignes)
    df_test_numeric = df_test.copy()
    df_test_numeric["SK_ID_CURR"] = pd.to_numeric(df_test_numeric["SK_ID_CURR"], errors='coerce')

    # On applique le scaler sur les features (à adapter si besoin, ici on suppose que df_test est prêt)
    scaled_features = scaler.transform(df_test_numeric)

    # Prédiction pour toutes les lignes
    predictions = model.predict_proba(scaled_features)[:, 1]

    # Convertir les prédictions en liste Python simple
    predictions_list = predictions.tolist()

    return {
        "column_name": column_name,
        "column_values": column_values,
        "predictions": predictions_list
    }

@app.post("/predict_custom")
async def predict_custom(request: CustomClientRequest):
    client_id = request.id_client
    mods = request.modifications

    # Récupérer ligne client
    df_test["SK_ID_CURR"] = pd.to_numeric(df_test["SK_ID_CURR"], errors='coerce')
    client_data = df_test[df_test["SK_ID_CURR"] == client_id]

    if client_data.empty:
        raise HTTPException(status_code=404, detail="Client non trouvé")

    # Copier la ligne dans un DataFrame temporaire
    temp_df = client_data.copy()

    # Appliquer les modifications reçues
    for key, value in mods.items():
        if key in temp_df.columns:
            temp_df.iloc[0, temp_df.columns.get_loc(key)] = value

    # Scaler et prédiction
    scaled = scaler.transform(temp_df)
    prediction = model.predict_proba(scaled)[:, 1].tolist()

    return {
        "id_client": client_id,
        "prediction": prediction,
        "modifications_appliquées": mods
    }