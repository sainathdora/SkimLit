import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
import joblib
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow everything
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)
# Load model once at startup to save time
print("Loading Tribrid Model...")
model = joblib.load('Tribid_model.pkl')

@app.get("/")
def home():
    return {"status": "Model API is running"}

@app.post("/predict")
def predict(text: str):
    # Wrap your prediction logic here
    prediction = model.predict([text])
    return {"prediction": prediction.tolist()}