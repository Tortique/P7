# Prédiction de Solvabilité

## Description

Cette application **Streamlit** permet de prédire la solvabilité d’un client à partir de ses données personnelles et financières.  
Elle s’appuie sur une API REST hébergée dans le cloud pour récupérer les données clients, les prédictions et les importances des variables.

---

## Fonctionnalités

- Chargement des données client via API distante  
- Visualisation des informations personnelles, crédit demandé, et autres caractéristiques  
- Analyse univariée et bivariée des variables par rapport à la population  
- Visualisation des importances locales et globales des variables  
- Simulation de prédiction avec modification de plusieurs variables  
- Affichage d’un gauge interactif représentant le score de risque  
- Debugging via affichage des données brutes JSON

---

## Installation

1. Cloner le dépôt :

    ```bash
    git clone https://github.com/ton-utilisateur/ton-projet.git
    cd ton-projet
    ```

2. Créer un environnement virtuel (optionnel mais recommandé) :

    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate     # Windows
    ```

3. Installer les dépendances :

    ```bash
    pip install -r requirements.txt
    ```

---

## Configuration

L’application utilise une API distante disponible à l’URL https://p7-ywri.onrender.com/
Et l'application est disponible à l'URL : https://p7-sl.onrender.com/

--- 

## Dépendances principales

 - streamlit

 - pandas

 - requests

 - matplotlib

 - seaborn

 - plotly

 - pyecharts



