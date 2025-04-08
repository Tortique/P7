import pandas as pd
import pytest
from api import app
from fastapi.testclient import TestClient
from api import df_test

client = TestClient(app)

def test_predict_valid_client():
    data = {"id_client": 100001}
    response = client.post("/predict", json=data)

    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_client_not_found():
    data = {"id_client": 1}  # Un ID qui n'existe sûrement pas

    response = client.post("/predict", json=data)

    assert response.status_code == 404  # Vérifie que l'API retourne bien 404
    assert response.json()["detail"] == "Client non trouvé"

def test_get_first():
    # Requête GET vers l'endpoint
    response = client.get("/get_first")
    assert response.status_code == 200  # Vérifie que la requête réussit

def test_get_client_found():
    client_id = 100001
    response = client.get(f"/client/{client_id}")

    assert response.status_code == 200
    json_data = response.json()
    print(json_data)
    assert json_data["0"] == 100001

def test_get_client_not_found():
    client_id = 999999  # ID inexistant
    response = client.get(f"/client/{client_id}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Client non trouvé"

def test_get_client_invalid_id():
    response = client.get("/client/abc")  # ID non numérique

    assert response.status_code == 422  # FastAPI doit renvoyer une erreur 422