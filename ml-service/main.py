from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import json
from datetime import datetime

app = FastAPI(title="KPT Prediction Service")

# Robust path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

try:
    model_improved = joblib.load(os.path.join(MODELS_DIR, "model_improved.joblib"))
    model_baseline = joblib.load(os.path.join(MODELS_DIR, "model_baseline.joblib"))
    features = joblib.load(os.path.join(MODELS_DIR, "features.joblib"))
    
    with open(os.path.join(DATA_DIR, "merchant_stats.json"), "r") as f:
        merchant_stats = json.load(f)
    merchant_stats_df = pd.DataFrame(merchant_stats)
    
    with open(os.path.join(DATA_DIR, "metrics.json"), "r") as f:
        metrics = json.load(f)
        
except Exception as e:
    print(f"Error loading resources: {e}")
    # In case data isn't ready yet, we initialize empty
    model_improved = None
    merchant_stats_df = pd.DataFrame()
    metrics = []

class PredictionRequest(BaseModel):
    merchant_id: int
    item_count: int
    order_time: str # ISO format
    current_load_features: dict = {}

@app.get("/")
def read_root():
    return {"status": "up", "service": "KPT Prediction ML Service"}

@app.post("/api/predict")
def predict_kpt(req: PredictionRequest):
    if model_improved is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    # Extract merchant features
    m_data = merchant_stats_df[merchant_stats_df['merchant_id'] == req.merchant_id]
    
    # Cuisine Map (must match train_model.py)
    cuisine_map = {
        'Bakery': 0, 'Cafe': 1, 'QSR': 2, 'Casual Dining': 3, 'Fine Dining': 4
    }

    if m_data.empty:
        bias_score = 0.0
        seating_capacity = 40
        cuisine_type = 'QSR'
    else:
        m_row = m_data.iloc[0]
        bias_score = m_row['merchant_bias_score']
        seating_capacity = m_row.get('seating_capacity', 40)
        cuisine_type = m_row.get('cuisine_type', 'QSR')

    # Prepare features
    order_time_str = req.order_time.replace('Z', '+00:00')
    order_dt = datetime.fromisoformat(order_time_str)
    
    load_f = req.current_load_features
    kitchen_load = load_f.get('kitchen_load_estimate', 5.0)
    rush_ind = load_f.get('rush_indicator', 0)
    complexity = req.item_count * 1.2
    
    cuisine_enc = cuisine_map.get(cuisine_type, 2) # Default to QSR (2)

    input_df = pd.DataFrame([{
        'item_count': req.item_count,
        'kitchen_load_estimate': kitchen_load,
        'merchant_bias_score': bias_score,
        'rush_indicator': rush_ind,
        'order_complexity_score': complexity,
        'hour_of_day': order_dt.hour,
        'seating_capacity': seating_capacity,
        'cuisine_enc': cuisine_enc
    }])

    
    # Predict
    pred_kpt = float(model_improved.predict(input_df)[0])
    
    # Confidence score (simulated based on model bias score)
    confidence = max(0.6, 1.0 - abs(bias_score / 20.0))
    
    return {
        "predicted_kpt": round(pred_kpt, 2),
        "confidence_score": round(confidence, 2),
        "bias_correction_applied": round(bias_score, 2)
    }

@app.get("/api/merchant/{merchant_id}")
def get_merchant_data(merchant_id: int):
    m_data = merchant_stats_df[merchant_stats_df['merchant_id'] == merchant_id]
    if m_data.empty:
        raise HTTPException(status_code=404, detail="Merchant not found")
    
    return m_data.iloc[0].to_dict()

@app.get("/api/stats/overview")
def get_global_stats():
    return {
        "metrics": metrics,
        "total_merchants": len(merchant_stats_df)
    }

@app.get("/api/dataset/sample")
def get_dataset_sample(limit: int = 50, source: str = "orders"):
    try:
        filename = "orders.csv" if source == "orders" else "merchants.csv"
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Dataset not found at {file_path}")
        
        # Read a slightly larger chunk than needed to allow sampling
        chunk_size = limit * 4
        df_chunk = pd.read_csv(file_path, nrows=chunk_size)
        
        if df_chunk.empty:
            return []
            
        # Ensure we don't sample more than available
        sample_size = min(len(df_chunk), limit)
        df_sample = df_chunk.sample(n=sample_size)
        
        # Fill NaN to avoid JSON errors
        return df_sample.fillna("").to_dict(orient="records")

    except Exception as e:
        print(f"Dataset Sample Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
