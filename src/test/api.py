from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import mlflow.sklearn
import pandas as pd
import numpy as np
import pickle
import re
import os
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

load_dotenv()

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN
mlflow.set_tracking_uri("databricks")

app = FastAPI(title="API de Predicción Fantasy Football")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    
    print(f"✅ Mejor versión: {best_version.version} con RMSE={best_rmse}")
    
    model_uri = f"models:/{model_name}/{best_version.version}"
    model = mlflow.sklearn.load_model(model_uri)
    
    return model

# Cargar modelo
print("📦 Cargando modelo...")
model = get_best_model(MODEL_NAME)
print("✅ Modelo cargado exitosamente")

# Cargar preprocessor
PREPROCESSOR_PATH = "preprocessor.pkl"

def load_preprocessor():
    """Carga el preprocessor guardado"""
    if os.path.exists(PREPROCESSOR_PATH):
        with open(PREPROCESSOR_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(
            f"❌ Preprocessor no encontrado en '{PREPROCESSOR_PATH}'. "
            "Debes ejecutar save_preprocessor.py primero."
        )

try:
    print("📦 Cargando preprocessor...")
    preprocessor = load_preprocessor()
    print("✅ Preprocessor cargado exitosamente")
except FileNotFoundError as e:
    print(f"⚠️ {e}")
    preprocessor = None

def preprocess_dataframe(df_raw):
    """Aplica el mismo preprocesamiento que en el entrenamiento"""
    df = df_raw.copy()
    
    # Limpiar valores problemáticos
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Convertir columnas numéricas (igual que en entrenamiento)
    for col in df.columns:
        conv = pd.to_numeric(df[col], errors="coerce")
        if conv.notna().mean() > 0.5:
            df[col] = conv
    
    # Eliminar TARGET si existe
    TARGET = "FPTS"
    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])
    
    # Eliminar columnas que no se usan
    drop_cols = ['Player', 'Team']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    
    # Transformar con el preprocessor
    if preprocessor is None:
        raise HTTPException(status_code=500, detail="Preprocessor no disponible")
    
    try:
        X_transformed = preprocessor.transform(df)
        return X_transformed
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en preprocesamiento: {str(e)}")

class PredictRequest(BaseModel):
    players: List[dict]
    drafted_players: List[str] = []
    my_team_positions: List[str] = []

class PredictResponse(BaseModel):
    best_player: str
    position: str
    predicted_fpts: float
    alternative_picks: List[dict] = []

@app.get("/")
def root():
    return {
        "message": "Bienvenido a la API de predicción Fantasy Football",
        "endpoints": ["/health", "/predict"],
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "API is running",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }

@app.post("/predict", response_model=PredictResponse)
def predict_best_pick(request: PredictRequest):
    try:
        print(f"\n🔍 Recibiendo solicitud con {len(request.players)} jugadores")
        
        # Convertir a DataFrame
        df_raw = pd.DataFrame(request.players)
        print(f"📊 Columnas recibidas: {df_raw.columns.tolist()}")
        
        # Guardar info del jugador antes del preprocesamiento
        player_info = df_raw[['Player', 'Position']].copy()
        
        # Filtrar jugadores ya drafteados
        mask_available = ~df_raw['Player'].isin(request.drafted_players)
        df_available = df_raw[mask_available].reset_index(drop=True)
        player_info_available = player_info[mask_available].reset_index(drop=True)
        
        print(f"✅ Jugadores disponibles: {len(df_available)}")
        
        if df_available.empty:
            raise HTTPException(status_code=404, detail="No hay jugadores disponibles")
        
        # Preprocesar datos
        print("🔧 Preprocesando datos...")
        X_transformed = preprocess_dataframe(df_available)
        print(f"✅ Shape después de preprocesar: {X_transformed.shape}")
        
        # Hacer predicción
        print("🤖 Generando predicciones...")
        predictions = model.predict(X_transformed)
        print(f"✅ Predicciones generadas: {len(predictions)}")
        
        # Crear DataFrame con resultados
        results_df = player_info_available.copy()
        results_df['predicted_fpts'] = predictions
        
        # Ordenar por predicción
        results_df = results_df.sort_values('predicted_fpts', ascending=False)
        
        # Mejor pick
        best_pick = results_df.iloc[0]
        print(f"🏆 Mejor pick: {best_pick['Player']} - {best_pick['predicted_fpts']:.2f} pts")
        
        # Top 5 alternativas
        alternatives = []
        for _, player in results_df.iloc[1:6].iterrows():
            alternatives.append({
                'player': player['Player'],
                'position': player['Position'],
                'predicted_fpts': float(player['predicted_fpts'])
            })
        
        return PredictResponse(
            best_player=best_pick['Player'],
            position=best_pick['Position'],
            predicted_fpts=float(best_pick['predicted_fpts']),
            alternative_picks=alternatives
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.post("/predict_batch")
def predict_batch(request: PredictRequest):
    """Retorna predicciones para TODOS los jugadores disponibles"""
    try:
        df_raw = pd.DataFrame(request.players)
        player_info = df_raw[['Player', 'Position']].copy()
        
        # Filtrar drafteados
        mask_available = ~df_raw['Player'].isin(request.drafted_players)
        df_available = df_raw[mask_available].reset_index(drop=True)
        player_info_available = player_info[mask_available].reset_index(drop=True)
        
        if df_available.empty:
            return {"predictions": []}
        
        # Preprocesar y predecir
        X_transformed = preprocess_dataframe(df_available)
        predictions = model.predict(X_transformed)
        
        # Crear resultados
        results = []
        for idx, (_, player) in enumerate(player_info_available.iterrows()):
            results.append({
                'player': player['Player'],
                'position': player['Position'],
                'predicted_fpts': float(predictions[idx])
            })
        
        # Ordenar por predicción
        results.sort(key=lambda x: x['predicted_fpts'], reverse=True)
        
        return {"predictions": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)