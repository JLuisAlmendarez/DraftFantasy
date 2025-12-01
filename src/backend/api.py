from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN
mlflow.set_tracking_uri("databricks")

app = FastAPI(title="API de Predicción de Reseñas")

MODEL_NAME = "workspace.default.model_draft_fantasy_champion"

def get_best_model(model_name: str):
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    
    best_version = None
    best_rmse = float("inf")
    
    for v in versions:
        run_id = v.run_id
        run_data = client.get_run(run_id).data
        rmse = run_data.metrics.get("rmse")
        if rmse is not None and rmse < best_rmse:
            best_rmse = rmse
            best_version = v
    
    if best_version is None:
        raise ValueError(f"No se encontró ninguna versión de {model_name} con métrica RMSE")
    
    print(f"Mejor versión: {best_version.version} con RMSE={best_rmse}")
    
    model_uri = f"models:/{model_name}/{best_version.version}"
    model = mlflow.sklearn.load_model(model_uri)
    
    return model

model = get_best_model(MODEL_NAME)

class Item(BaseModel):
    None

@app.get("/")
def root():
    return {"message": "Bienvenido a la API de predicción"}

@app.get("/health")
def health():
    return {"status": "ok", "message": "API is running"}

@app.post("/predict")
def predict(item: Item):
    pred = model.predict(None)
    return {"prediction": int(pred[0])}
