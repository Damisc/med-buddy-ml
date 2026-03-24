from fastapi import FastAPI
from pydantic import BaseModel

from backend.predictor import predict


app = FastAPI(
    title="Heart Diseases Prediction API",
    version="1.0.0"
)

# input schema matching training features
class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.get("/health")
def health_check():
    return {"status": "ok"}


# Heart Disease Prediction endpoint
@app.post("/predict-heart-disease")
def predict_heart_disease(input_data: HeartDiseaseInput):
    input_data=input_data.model_dump()
    result = predict(input_data=input_data)
    return {
        "Prediction": result["Prediction"],
        "Probability": result["Probability"],
        "diagnosis": (
            "Heart Disease Detected"
            if result["Prediction"] == 1
            else "No Heart DIsease Detected"
        )
    }






# # {
#     "age": 52,
#     "sex": 1,
#     "cp": 0,
#     "trestbps": 125,
#     "chol": 212,
#     "fbs": 0,
#     "restecg": 1,
#     "thalach": 168,
#     "exang": 0,
#     "oldpeak": 1.0,
#     "slope": 2,
#     "ca": 0,
#     "thal": 2
# }