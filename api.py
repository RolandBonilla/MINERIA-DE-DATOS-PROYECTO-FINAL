import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Inicializar la app FastAPI (la variable DEBE llamarse 'app')
app = FastAPI(
    title="API de Predicción de Churn en Telecomunicaciones",
    description="Servicio web para evaluar la probabilidad de abandono de clientes mediante Regresión Logística.",
    version="1.0"
)

# Cargar el modelo serializado exportado
model_pipeline = joblib.load('modelo_churn.pkl')

# Definir el esquema JSON de entrada
class ClienteSchema(BaseModel):
    gender: str = "Male"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = 12
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "Yes"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "Yes"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 85.50
    TotalCharges: float = 1026.00

@app.get("/")
def inicio():
    return {"mensaje": "API de Predicción de Churn Activa y Operativa"}

@app.post("/predict")
def predecir_churn(cliente: ClienteSchema):
    # Compatibilidad con Pydantic v1 y v2
    data_dict = cliente.dict() if hasattr(cliente, 'dict') else cliente.model_dump()
    df_input = pd.DataFrame([data_dict])
    
    # Calcular las variables derivadas (Feature Engineering)
    df_input['Cargo_Promedio_Mensual'] = df_input.apply(
        lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else row['MonthlyCharges'], axis=1
    )
    
    servicios = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    n_serv = sum([1 for s in servicios if df_input[s].iloc[0] == 'Yes'])
    df_input['Indice_Fidelidad'] = (n_serv * (df_input['tenure'].iloc[0] + 1)) / (df_input['MonthlyCharges'].iloc[0] + 1)
    
    # Realizar la predicción
    probabilidad = float(model_pipeline.predict_proba(df_input)[0][1])
    prediccion = int(model_pipeline.predict(df_input)[0])
    
    return {
        "prediccion_churn": prediccion,
        "probabilidad_fuga": round(probabilidad, 4),
        "porcentaje_riesgo": f"{round(probabilidad * 100, 2)}%",
        "estado": "Riesgo de Fuga" if prediccion == 1 else "Cliente Seguro / Retenido",
        "recomendacion": "Ofrecer plan de permanencia anual con incentivo" if prediccion == 1 else "Mantener servicios actuales"
    }