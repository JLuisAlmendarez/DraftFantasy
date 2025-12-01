import os
import pickle
import re
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from prefect import task, flow
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import mlflow
from mlflow.models.signature import infer_signature
import optuna
from prefect.client.schemas.schedules import CronSchedule

load_dotenv()
#os.environ["PREFECT_API_URL"] = os.getenv("PREFECT_API_URL")

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

mlflow.set_tracking_uri("databricks")
EXPERIMENT_NAME = "/Users/almendarez1002@gmail.com/draft-fantasy"

experiment = mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

@task
def preprocess_data():

    df_DST = pd.read_csv("../data/DST.csv")
    df_K = pd.read_csv("../data/K.csv")
    df_QB = pd.read_csv("../data/QB.csv")
    df_RB = pd.read_csv("../data/RB.csv")
    df_TE = pd.read_csv("../data/TE.csv")
    df_WR = pd.read_csv("../data/WR.csv")

    df_DST['Position'] = 'DST'
    df_K['Position'] = 'K'
    df_QB['Position'] = 'QB'
    df_RB['Position'] = 'RB'
    df_TE['Position'] = 'TE'
    df_WR['Position'] = 'WR'

    df = pd.concat([df_DST, df_K, df_QB, df_RB, df_TE, df_WR], ignore_index=True)

    cols_convertidas = []

    for col in df.columns:
        conv = pd.to_numeric(df[col], errors="coerce")
        if conv.notna().mean() > 0.5:
            df[col] = conv
            cols_convertidas.append(col)

    TARGET = "FPTS"
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].notna()].reset_index(drop=True)

    id_like = ["Player", "Team", TARGET]  
    leak_like_patterns = [
        r"^FPTS\/G$",  
        r"\brank\b",
        r"\btier\b",
    ]
    leak_regex = re.compile("|".join(leak_like_patterns), flags=re.IGNORECASE)

    drop_cols = set(id_like)
    drop_cols.update([c for c in df.columns if leak_regex.search(str(c))])

    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns 
        if c not in drop_cols
    ]

    cat_cols = ["Position"]

    numeric_transformer = SimpleImputer(strategy="constant", fill_value=0)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop"
    )

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        df.drop(columns=[TARGET]),
        df[TARGET],
        test_size=0.2,
        random_state=42
    )

    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    return X_train, X_test, y_train, y_test

def objective(trial, X_train, y_train, X_test, y_test):
    mlflow.sklearn.autolog(disable=True)

    model_type = trial.suggest_categorical(
        "model",
        ["xgboost", "lightgbm", "random_forest"]
    )

    with mlflow.start_run(nested=True, run_name=f"trial_{model_type}"):


        if model_type == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True)
            }
            model = XGBRegressor(
                **params,
                random_state=42,
                tree_method="hist",
            )
        elif model_type == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "num_leaves": trial.suggest_int("num_leaves", 20, 120),
                "max_depth": trial.suggest_int("max_depth", -1, 20),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True)
            }
            model = LGBMRegressor(
                **params,
                random_state=42
            )
        elif model_type == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 40),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }
            model = RandomForestRegressor(
                **params,
                random_state=42,
                n_jobs=-1
            )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        return rmse
    
@task
def run_optuna(X_train, y_train, X_test, y_test):
    if mlflow.active_run():
        mlflow.end_run()

    study = optuna.create_study(
        study_name=EXPERIMENT_NAME,
        direction="minimize"
    )

    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=50)

    print("🏆 Mejor RMSE:", study.best_value)
    print("🏆 Mejores parámetros:", study.best_params)

    return study.best_params

@task
def train_and_register_champion(params, X_train, y_train, X_test, y_test):

    model_type = params["model"]

    with mlflow.start_run(nested=True, run_name="ChampionModel") as run:

        mlflow.sklearn.autolog(disable=True)
        if model_type == "xgboost":
            model = XGBRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                reg_lambda=params["reg_lambda"],
                reg_alpha=params["reg_alpha"],
                random_state=42,
                tree_method="hist",
            )
        elif model_type == "lightgbm":
            model = LGBMRegressor(
                n_estimators=params["n_estimators"],
                num_leaves=params["num_leaves"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                reg_lambda=params["reg_lambda"],
                reg_alpha=params["reg_alpha"],
                random_state=42
            )
        elif model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"❌ Modelo desconocido: {model_type}")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
            "mae": mean_absolute_error(y_test, preds),
            "r2": r2_score(y_test, preds)
        }

        print("\n📊 Métricas del Champion:", metrics)

        mlflow.log_metrics(metrics)
        mlflow.log_params(params)

        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train[:5]

        try:
            input_example = input_example.toarray()
        except:
            pass

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        model_uri = f"runs:/{run.info.run_id}/model"

        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name="workspace.default.model_draft_fantasy_champion"
        )

        print(f"\n✅ Modelo registrado en Unity Catalog: {registered_model.name}")
        print(f"   Versión: {registered_model.version}")
        print(f"   Run ID: {run.info.run_id}")

        return model
    
@flow
def tournament():
    if mlflow.active_run():
        mlflow.end_run()

    X_train, X_test, y_train, y_test = preprocess_data()

    with mlflow.start_run(run_name="Tournament"):
        
        best_params = run_optuna(X_train, y_train, X_test, y_test)
        
        train_and_register_champion(best_params, X_train, y_train, X_test, y_test)

        print("\n🎉 Pipeline completado exitosamente!")





if __name__ == "__main__":
    tournament()

#Esperar a ejecutar cron hasta la parte de docker...