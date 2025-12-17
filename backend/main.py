"""
FastAPI Backend for B-Decide AI
Provides REST API endpoints for churn prediction and recommendations
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import pickle
import os
import sys
import io
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessor import ChurnDataPreprocessor
from decision_engine.recommender import CustomerRecommender
from nlp.explainer import ChurnExplainer

# Initialize FastAPI app
app = FastAPI(
    title="B-Decide AI API",
    description="Decision Intelligence SaaS platform for customer churn prediction and retention",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models and processors
model = None
preprocessor = None
recommender = None
explainer = None

# Pydantic models for request/response
class CustomerData(BaseModel):
    """Single customer data for prediction"""
    customerID: Optional[str] = Field(default="unknown", description="Customer ID")
    gender: Optional[str] = Field(default="Male", description="Customer gender")
    SeniorCitizen: Optional[int] = Field(default=0, description="Senior citizen flag (0/1)")
    Partner: Optional[str] = Field(default="No", description="Has partner (Yes/No)")
    Dependents: Optional[str] = Field(default="No", description="Has dependents (Yes/No)")
    tenure: float = Field(..., description="Tenure in months", ge=0)
    PhoneService: Optional[str] = Field(default="Yes", description="Has phone service")
    MultipleLines: Optional[str] = Field(default="No", description="Has multiple lines")
    InternetService: Optional[str] = Field(default="DSL", description="Internet service type")
    OnlineSecurity: Optional[str] = Field(default="No", description="Has online security")
    OnlineBackup: Optional[str] = Field(default="No", description="Has online backup")
    DeviceProtection: Optional[str] = Field(default="No", description="Has device protection")
    TechSupport: Optional[str] = Field(default="No", description="Has tech support")
    StreamingTV: Optional[str] = Field(default="No", description="Has streaming TV")
    StreamingMovies: Optional[str] = Field(default="No", description="Has streaming movies")
    Contract: Optional[str] = Field(default="Month-to-month", description="Contract type")
    PaperlessBilling: Optional[str] = Field(default="Yes", description="Has paperless billing")
    PaymentMethod: Optional[str] = Field(default="Electronic check", description="Payment method")
    MonthlyCharges: float = Field(..., description="Monthly charges", ge=0)
    TotalCharges: Optional[float] = Field(default=0, description="Total charges", ge=0)
    
    class Config:
        schema_extra = {
            "example": {
                "customerID": "7590-VHVEG",
                "tenure": 4,
                "MonthlyCharges": 75.5,
                "TotalCharges": 302.0,
                "Contract": "Month-to-month",
                "InternetService": "DSL"
            }
        }


class PredictionResponse(BaseModel):
    """Response for churn prediction"""
    customerID: str
    churn_probability: float
    risk_level: str
    prediction: str
    timestamp: str


class RecommendationResponse(BaseModel):
    """Response for recommendation"""
    customerID: str
    churn_probability: float
    risk_level: str
    recommended_action: str
    action_description: str
    confidence: float
    priority: int
    explanation: str
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    total_customers: int
    predictions: List[PredictionResponse]
    summary: Dict


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    timestamp: str


# Startup event to load models
@app.on_event("startup")
async def load_models():
    """Load models and processors on startup"""
    global model, preprocessor, recommender, explainer
    
    try:
        # Load ML model
        model_path = 'models/churn_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("✓ ML model loaded successfully")
        else:
            print("⚠ Warning: Model file not found. Please train the model first.")
        
        # Load preprocessor
        preprocessor = ChurnDataPreprocessor()
        preprocessor_path = 'models/preprocessor.pkl'
        if os.path.exists(preprocessor_path):
            preprocessor.load_preprocessor('models')
            print("✓ Preprocessor loaded successfully")
        else:
            print("⚠ Warning: Preprocessor not found. Please train the model first.")
        
        # Initialize recommender and explainer
        recommender = CustomerRecommender()
        explainer = ChurnExplainer()
        print("✓ Recommender and Explainer initialized")
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        # Don't fail startup, just log the error


# API Endpoints

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to B-Decide AI API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Single customer churn prediction",
            "/recommend": "Get recommendation for a customer",
            "/batch-predict": "Batch predictions from CSV upload"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model and preprocessor else "degraded",
        model_loaded=model is not None,
        preprocessor_loaded=preprocessor is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """
    Predict churn probability for a single customer
    
    Args:
        customer: Customer data
        
    Returns:
        Prediction response with churn probability and risk level
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Convert to DataFrame
        customer_dict = customer.dict()
        customer_id = customer_dict.get('customerID', 'unknown')
        df = pd.DataFrame([customer_dict])
        
        # Preprocess
        X = preprocessor.preprocess_new_data(df)
        
        # Predict
        churn_prob = float(model.predict_proba(X)[0, 1])
        
        # Determine risk level
        if churn_prob >= 0.70:
            risk_level = 'Critical'
        elif churn_prob >= 0.50:
            risk_level = 'High'
        elif churn_prob >= 0.30:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        prediction = 'Churn' if churn_prob >= 0.5 else 'No Churn'
        
        return PredictionResponse(
            customerID=customer_id,
            churn_probability=round(churn_prob, 4),
            risk_level=risk_level,
            prediction=prediction,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendation(customer: CustomerData):
    """
    Get personalized recommendation for a customer
    
    Args:
        customer: Customer data
        
    Returns:
        Recommendation with action, explanation, and confidence
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Get prediction first
        customer_dict = customer.dict()
        customer_id = customer_dict.get('customerID', 'unknown')
        df = pd.DataFrame([customer_dict])
        
        # Preprocess and predict
        X = preprocessor.preprocess_new_data(df)
        churn_prob = float(model.predict_proba(X)[0, 1])
        
        # Generate recommendation
        recommendation = recommender.generate_recommendation(customer_dict, churn_prob)
        
        # Generate explanation
        explanation = explainer.generate_full_explanation(
            churn_probability=churn_prob,
            recommendation=recommendation,
            customer_features=recommendation.get('customer_features')
        )
        
        return RecommendationResponse(
            customerID=customer_id,
            churn_probability=round(churn_prob, 4),
            risk_level=recommendation['risk_level'],
            recommended_action=recommendation['action'],
            action_description=recommendation['action_description'],
            confidence=round(recommendation['confidence'], 4),
            priority=recommendation['priority'],
            explanation=explanation,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")


@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(file: UploadFile = File(...)):
    """
    Predict churn for multiple customers from CSV file
    
    Args:
        file: CSV file with customer data
        
    Returns:
        Batch predictions with summary statistics
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Empty CSV file")
        
        # Store customer IDs if present
        customer_ids = df['customerID'].tolist() if 'customerID' in df.columns else [f"customer_{i}" for i in range(len(df))]
        
        # Preprocess
        X = preprocessor.preprocess_new_data(df)
        
        # Predict
        churn_probs = model.predict_proba(X)[:, 1]
        
        # Create predictions list
        predictions = []
        risk_counts = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}
        
        for i, (customer_id, prob) in enumerate(zip(customer_ids, churn_probs)):
            prob = float(prob)
            
            # Determine risk level
            if prob >= 0.70:
                risk_level = 'Critical'
            elif prob >= 0.50:
                risk_level = 'High'
            elif prob >= 0.30:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            risk_counts[risk_level] += 1
            
            predictions.append(PredictionResponse(
                customerID=customer_id,
                churn_probability=round(prob, 4),
                risk_level=risk_level,
                prediction='Churn' if prob >= 0.5 else 'No Churn',
                timestamp=datetime.now().isoformat()
            ))
        
        # Create summary
        summary = {
            'avg_churn_probability': round(float(np.mean(churn_probs)), 4),
            'risk_distribution': risk_counts,
            'high_risk_count': risk_counts['Critical'] + risk_counts['High'],
            'churn_predictions': sum(1 for p in predictions if p.prediction == 'Churn')
        }
        
        return BatchPredictionResponse(
            total_customers=len(predictions),
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        return {"status": "No model loaded"}
    
    info = {
        "model_type": type(model).__name__,
        "model_loaded": True,
        "preprocessor_loaded": preprocessor is not None,
        "feature_count": len(preprocessor.feature_columns) if preprocessor and preprocessor.feature_columns else None
    }
    
    return info


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("Starting B-Decide AI API Server")
    print("="*60 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

