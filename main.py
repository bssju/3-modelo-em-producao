from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
import os

app = FastAPI(title="House Prices API")

model = None
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 
            'YearBuilt', 'YearRemodAdd', 'LotArea', 'Fireplaces', 
            'TotalBaths', 'TotalSF']

class PredictRequest(BaseModel):
    OverallQual: int
    GrLivArea: int
    GarageCars: int
    TotalBsmtSF: float
    YearBuilt: int
    YearRemodAdd: int
    LotArea: int
    Fireplaces: int
    TotalBaths: float
    TotalSF: float

@app.on_event("startup")
async def load_model():
    global model
    try:
        with open("models/house_prices_model.pkl", "rb") as f:
            model = pickle.load(f)
        print("✅ Modelo carregado com sucesso!")
    except Exception as e:
        print(f"❌ Erro: {e}")
        model = None

@app.get("/health")
async def health_check():
    return {"model_loaded": model is not None}

@app.post("/predict")
async def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    df = pd.DataFrame([request.dict()])
    df = df[features]
    pred_log = model.predict(df)[0]
    pred_price = np.exp(pred_log)
    return {"predicted_price": round(pred_price, 2)}

@app.get("/")
async def root():
    return {"message": "API rodando! Acesse /docs"}