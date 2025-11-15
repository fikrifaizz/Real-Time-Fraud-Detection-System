from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import pickle
import json
import uvicorn
from datetime import datetime
import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection with explainability",
    version="1.0.0"
)

@app.on_event("startup")
async def load_models():
    global lgb_model, xgb_model, cat_model, explainer, ensemble_config, feature_cols
    
    logger.info("Loading models...")
    
    try:
        # Load models
        with open('./models/lightgbm_model.pkl', 'rb') as f:
            lgb_model = pickle.load(f)
        
        with open('./models/xgboost_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        
        with open('./models/catboost_model.pkl', 'rb') as f:
            cat_model = pickle.load(f)
        
        with open('./models/shap_explainer.pkl', 'rb') as f:
            explainer = pickle.load(f)
        
        with open('./models/ensemble_config.json', 'r') as f:
            ensemble_config = json.load(f)
        
        with open('./models/top_features.json', 'r') as f:
            feature_data = json.load(f)
            feature_cols = feature_data['feature_names']
        
        logger.info(f"Models loaded successfully")
        logger.info(f"Feature count: {len(feature_cols)}")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

class TransactionInput(BaseModel):
    TransactionAmt: float = Field(..., gt=0, description="Transaction amount")
    ProductCD: str = Field(..., description="Product code")
    card1: int = Field(..., description="Card identifier")
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "TransactionAmt": 150.0,
                "ProductCD": "W",
                "card1": 13926,
                "card2": 150.0,
                "card4": "visa",
                "card6": "credit",
                "addr1": 315.0,
                "P_emaildomain": "gmail.com"
            }
        }

class RiskFactor(BaseModel):
    feature: str
    value: float
    impact: str
    shap_value: float

class PredictionResponse(BaseModel):
    transaction_id: str
    prediction: str
    fraud_probability: float
    risk_level: str
    threshold_used: float
    top_risk_factors: List[RiskFactor]
    timestamp: str
    recommended_action: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

def preprocess_transaction(transaction: dict) -> pd.DataFrame:
    # Create DataFrame
    df = pd.DataFrame([transaction])
    
    # Load sample data to get proper encoding
    try:
        sample_df = pd.read_csv('data/processed/train_set.csv', nrows=1)
        
        # Handle categorical encoding
        from sklearn.preprocessing import LabelEncoder
        
        # Encode categorical columns in transaction
        cat_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']
        for col in cat_cols:
            if col in df.columns and df[col].notna().any():
                # Simple encoding: convert to string hash or default to 0
                if df[col].dtype == 'object':
                    df[col] = hash(str(df[col].iloc[0])) % 1000
        
        # Fill missing features with 0
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Ensure correct order and types
        df = df[feature_cols]
        
        # Convert to numeric
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        # Fallback: simple preprocessing
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        df = df[feature_cols]
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        return df

def get_ensemble_prediction(df: pd.DataFrame) -> float:
    weights = ensemble_config['weights']
    
    # Individual predictions
    lgb_pred = lgb_model.predict_proba(df)[:, 1][0]
    xgb_pred = xgb_model.predict_proba(df)[:, 1][0]
    cat_pred = cat_model.predict_proba(df)[:, 1][0]
    
    logger.info(f"Individual predictions - LGB: {lgb_pred:.4f}, XGB: {xgb_pred:.4f}, CAT: {cat_pred:.4f}")
    
    # Weighted ensemble
    ensemble_pred = (
        weights['lightgbm'] * lgb_pred +
        weights['xgboost'] * xgb_pred +
        weights['catboost'] * cat_pred
    )
    
    return ensemble_pred

def explain_prediction(df: pd.DataFrame, fraud_proba: float, top_n: int = 5) -> List[Dict]:
    try:
        # Get SHAP values
        shap_vals = explainer.shap_values(df)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        
        # Get top contributing features
        feature_contrib = pd.DataFrame({
            'feature': df.columns,
            'value': df.iloc[0].values,
            'shap_value': shap_vals[0]
        }).sort_values('shap_value', key=abs, ascending=False).head(top_n)
        
        # Format explanation
        risk_factors = []
        for _, row in feature_contrib.iterrows():
            risk_factors.append({
                'feature': row['feature'],
                'value': float(row['value']),
                'impact': 'increase' if row['shap_value'] > 0 else 'decrease',
                'shap_value': float(row['shap_value'])
            })
        
        return risk_factors
        
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {str(e)}")
        # Return dummy explanation if SHAP fails
        return [{
            'feature': 'TransactionAmt',
            'value': float(df['TransactionAmt'].iloc[0]) if 'TransactionAmt' in df.columns else 0.0,
            'impact': 'increase' if fraud_proba > 0.5 else 'decrease',
            'shap_value': 0.0
        }]

def get_recommended_action(fraud_proba: float, threshold: float) -> str:
    if fraud_proba >= threshold:
        if fraud_proba >= 0.9:
            return "BLOCK - High confidence fraud. Block transaction immediately."
        elif fraud_proba >= 0.7:
            return "REVIEW - Likely fraud. Manual review required."
        else:
            return "CHALLENGE - Potential fraud. Request additional verification."
    else:
        return "APPROVE - Low fraud risk. Approve transaction."

@app.get("/", response_model=Dict)
async def root():
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": lgb_model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionInput):
    try:
        # Generate transaction ID
        transaction_id = f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        logger.info(f"Processing transaction: {transaction_id}")
        logger.info(f"Input: {transaction.dict()}")
        
        # Preprocess
        df = preprocess_transaction(transaction.dict())
        logger.info(f"Preprocessed shape: {df.shape}")
        
        # Get prediction
        fraud_proba = get_ensemble_prediction(df)
        logger.info(f"Ensemble probability: {fraud_proba:.4f}")
        
        # Get threshold
        threshold = ensemble_config['threshold']
        
        # Determine prediction
        prediction = "FRAUD" if fraud_proba >= threshold else "LEGITIMATE"
        
        # Determine risk level
        if fraud_proba >= 0.7:
            risk_level = "HIGH"
        elif fraud_proba >= 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Get explanation
        risk_factors = explain_prediction(df, fraud_proba)
        
        # Get recommended action
        recommended_action = get_recommended_action(fraud_proba, threshold)
        
        # Build response
        response = {
            "transaction_id": transaction_id,
            "prediction": prediction,
            "fraud_probability": round(fraud_proba, 4),
            "risk_level": risk_level,
            "threshold_used": threshold,
            "top_risk_factors": risk_factors,
            "timestamp": datetime.now().isoformat(),
            "recommended_action": recommended_action
        }
        
        logger.info(f"Prediction: {prediction}, Probability: {fraud_proba:.4f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing transaction: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    return {
        "model_performance": ensemble_config['performance'],
        "threshold": ensemble_config['threshold'],
        "ensemble_weights": ensemble_config['weights']
    }

@app.get("/debug/features")
async def debug_features():
    """Debug endpoint to check feature count"""
    return {
        "feature_count": len(feature_cols),
        "sample_features": feature_cols[:10]
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )