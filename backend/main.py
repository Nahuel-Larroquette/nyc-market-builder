from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")

class HouseFeatures(BaseModel):
    beds: int
    bath: float
    sqft: float
    property_type: str
    borough: str

@app.get("/")
def read_root():
    return {"message": "NYC Market Builder API is running"}

@app.post("/predict")
def predict_price(house: HouseFeatures):
    input_data = pd.DataFrame([{
        "BEDS": house.beds,
        "BATH": house.bath,
        "PROPERTYSQFT": house.sqft,
        "SQFT_ESTIMATED": False,
        "TYPE_Co-op": house.property_type == "Co-op",
        "TYPE_Condo": house.property_type == "Condo",
        "TYPE_House": house.property_type == "House",
        "TYPE_Multi-family": house.property_type == "Multi-family",
        "TYPE_Townhouse": house.property_type == "Townhouse",
        "SUBLOCALITY_Brooklyn": house.borough == "Brooklyn",
        "SUBLOCALITY_Manhattan": house.borough == "Manhattan",
        "SUBLOCALITY_Queens": house.borough == "Queens",
        "SUBLOCALITY_Staten Island": house.borough == "Staten Island",
    }])

    log_price_pred = model.predict(input_data)[0]
    price_pred = np.exp(log_price_pred)

    return {"predicted_price": round(float(price_pred), 2)}