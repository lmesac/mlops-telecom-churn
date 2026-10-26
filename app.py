from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

# Inicializar la aplicación
app = FastAPI(title="API de Predicción de Churn - MLOps", version="1.0")

# Cargar el modelo entrenado
model_path = Path(__file__).resolve().parent / "notebooks" / "model_churn.pkl"
model = joblib.load(model_path)

# Definir la estructura de los datos de entrada (asegúrate de incluir campos representativos de tu dataset)
class TelecomData(BaseModel):
    account_length: float
    data_usage: float
    customer_service_calls: float

@app.get("/")
def home():
    return {"mensaje": "¡Bienvenido a la API de Predicción de Churn de Telecomunicaciones!"}

@app.post("/predict")
def predict(data: TelecomData):
    # Convertir los datos de entrada a DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Realizar predicción
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    return {
        "churn_prediccion": int(prediction[0]),
        "probabilidad_fuga": float(probability[0][1])
     }