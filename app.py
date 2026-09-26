import os
import pathlib
import pickle
import sys
from typing import Annotated, Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel, Field

# Ensure src can be discovered for the transformer reference
ROOT_DIR = pathlib.Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import required so pickle unpickles the transformer safely
from src.features.build_features import InsuranceFeatureEngineer
# ---------------------------------------------------------------------------
# Load Saved Pipeline
# ---------------------------------------------------------------------------
model_path = ROOT_DIR / "models" / "model.pkl"

if not model_path.exists():
    raise FileNotFoundError(f"Model file not found at {model_path}. Run train_model.py first.")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# ---------------------------------------------------------------------------
# API Setup & Schema
# ---------------------------------------------------------------------------
app = FastAPI(title="Insurance Premium Predictor")

class UserInput(BaseModel):
    age: Annotated[int, Field(..., gt=0, lt=120, description="Age of user")]
    weight: Annotated[float, Field(..., gt=0, description="Weight in kg")]
    height: Annotated[float, Field(..., gt=0, lt=2.5, description="Height in meters")]
    income_lpa: Annotated[float, Field(..., gt=0, description="Annual income in LPA")]
    smoker: Annotated[bool, Field(..., description="Smoker true/false")]
    city: Annotated[str, Field(..., description="City name")]
    occupation: Annotated[
        Literal[
            "retired", "freelancer", "student", "government_job",
            "business_owner", "unemployed", "private_job"
        ],
        Field(..., description="Occupation"),
    ]

@app.post("/predict")
def predict_premium(data: UserInput):
    # Pass raw input directly to let pipeline handle calculations
    input_df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_df)[0]
    return JSONResponse(status_code=200, content={"predicted_category": str(prediction)})