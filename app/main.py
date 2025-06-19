from fastapi import FastAPI
from app.schemas import IncidentReport, PredictionResponse
from app.predictor import BERTPredictor
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Food Incident Classifier")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = BERTPredictor()

@app.post("/predict", response_model=PredictionResponse)
def predict_incident(report: IncidentReport):
    # Match training preprocessing: concatenate title + text
    full_text = f"{report.title.strip()}. {report.text.strip()}"

    hazard, product = predictor.predict(full_text)
    return PredictionResponse(hazard_type=hazard, product_category=product)

# http POST http://127.0.0.1:8000/predict title="Salmonella outbreak in peanut butter" text="Multiple consumers were hospitalized after eating the product." country="USA" year:=2023 month:=8 day:=12 