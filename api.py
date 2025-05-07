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
    if column_name in df_test.columns:
        return df_test[column_name].dropna().tolist()
    else:
        return {"error": "Colonne non trouvée"}