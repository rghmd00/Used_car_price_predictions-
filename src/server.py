# server.py
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from src.predict import predict_price
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Car Price Prediction API")

API_KEY = "secret123"
api_key_header = APIKeyHeader(name="Token")

class CarInput(BaseModel):
    brand: str
    model_year: int
    engine: str
    fuel_type: str
    transmission: str
    accident: str

@app.post("/predict")
def predict_endpoint(
    input_data: CarInput,
    token: str = Security(api_key_header)
):
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    try:
        logger.info("Received input: %s", input_data)
        result = predict_price(input_data.dict())
        return {"predicted_price": result}
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
