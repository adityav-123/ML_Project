from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).parent
TRAINING_FILE = ROOT / "Training(disease).csv"
TESTING_FILE = ROOT / "Testing(disease).csv"
STATIC_DIR = ROOT / "static"

class PredictionRequest(BaseModel):
    symptoms: list[str] = Field(min_length=1, max_length=30)
    model: str = "Random Forest"

def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.loc[:, ~frame.columns.str.contains(r"^Unnamed")].copy()
    frame.columns = frame.columns.str.strip()
    return frame.drop_duplicates()

@lru_cache(maxsize=1)
def get_engine() -> dict[str, Any]:
    if not TRAINING_FILE.exists() or not TESTING_FILE.exists():
        raise FileNotFoundError("Training(disease).csv and Testing(disease).csv are required")
    train = clean_frame(pd.read_csv(TRAINING_FILE))
    test = clean_frame(pd.read_csv(TESTING_FILE))
    target = "prognosis"
    symptoms = [column for column in train.columns if column != target]
    x_train = train[symptoms].apply(pd.to_numeric, errors="coerce").fillna(0)
    x_test = test.reindex(columns=symptoms, fill_value=0)[symptoms].apply(pd.to_numeric, errors="coerce").fillna(0)
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train[target].astype(str))
    y_test = encoder.transform(test[target].astype(str))
    model = RandomForestClassifier(n_estimators=180, random_state=42, n_jobs=-1, class_weight="balanced")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, average="macro", zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, average="macro", zero_division=0)), 4),
    }
    importance = sorted(zip(symptoms, model.feature_importances_), key=lambda pair: pair[1], reverse=True)
    return {"model": model, "encoder": encoder, "symptoms": symptoms, "diseases": encoder.classes_.tolist(), "metrics": metrics, "importance": [{"name": name, "value": round(float(value), 4)} for name, value in importance[:8]], "records": int(len(train))}

def readable(value: str) -> str:
    return value.replace("_", " ").title()

app = FastAPI(title="CareLens API", version="2.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/health")
def health() -> dict[str, str]:
    get_engine()
    return {"status": "ok"}

@app.get("/api/overview")
def overview() -> dict[str, Any]:
    engine = get_engine()
    return {"records": engine["records"], "symptom_count": len(engine["symptoms"]), "disease_count": len(engine["diseases"]), "model": "Random Forest", "metrics": engine["metrics"], "top_features": [{"name": readable(item["name"]), "value": item["value"]} for item in engine["importance"]], "diseases": engine["diseases"]}

@app.get("/api/symptoms")
def symptom_list() -> dict[str, list[dict[str, str]]]:
    return {"symptoms": [{"id": name, "label": readable(name)} for name in get_engine()["symptoms"]]}

@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    engine = get_engine()
    unknown = [symptom for symptom in request.symptoms if symptom not in engine["symptoms"]]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown symptom: {unknown[0]}")
    values = pd.DataFrame(np.zeros((1, len(engine["symptoms"])), dtype=float), columns=engine["symptoms"])
    for symptom in set(request.symptoms):
        values.iloc[0, engine["symptoms"].index(symptom)] = 1
    prediction = engine["model"].predict(values)[0]
    probabilities = engine["model"].predict_proba(values)[0]
    top = np.argsort(probabilities)[::-1][:3]
    return {"disease": engine["encoder"].inverse_transform([prediction])[0], "confidence": round(float(probabilities[prediction]), 4), "alternatives": [{"disease": engine["encoder"].inverse_transform([int(index)])[0], "confidence": round(float(probabilities[index]), 4)} for index in top[1:]], "symptoms": [readable(symptom) for symptom in request.symptoms], "model": "Random Forest", "disclaimer": "Educational estimate only — not a medical diagnosis."}
