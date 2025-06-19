from pydantic import BaseModel

class IncidentReport(BaseModel):
    title: str
    text: str
    country: str
    year: int
    month: int
    day: int

class PredictionResponse(BaseModel):
    hazard_type: str
    product_category: str
