"""
Enhanced FastAPI Backend for B-Decide AI
Includes batch recommendations with explanations and dynamic rules support
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
from decision_engine.dynamic_fuzzy_rules import DynamicFuzzyRulesEngine
from nlp.explainer import ChurnExplainer
from nlp.batch_explainer import BatchChurnExplainer

# Initialize FastAPI app
app = FastAPI(
    title="B-Decide AI API (Enhanced)",
    description="Enhanced Decision Intelligence SaaS platform with batch recommendations and dynamic rules",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models and processors
model = None
preprocessor = None
recommender = None
explainer = None
batch_explainer = None
dynamic_rules_engine = None

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


class BatchRecommendationItem(BaseModel):
    """Single item in batch recommendation response"""
    customer_id: str
    churn_probability: float
    risk_level: str
    urgency: str
    recommended_action: str
    action_code: str
    confidence: float
    priority: int
    risk_assessment: str
    action_rationale: str
    full_explanation: str
    estimated_cost: Optional[float] = None
    expected_retention_lift: Optional[float] = None
    action_category: Optional[str] = None


class BatchRecommendationResponse(BaseModel):
    """Response for batch recommendations"""
    total_customers: int
    recommendations: List[BatchRecommendationItem]
    executive_summary: str
    summary_statistics: Dict
    timestamp: str


class RulesReloadResponse(BaseModel):
    """Response for rules reload"""
    success: bool
    message: str
    rules_loaded: int
    config_version: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    dynamic_rules_enabled: bool
    timestamp: str


# Startup event to load models
@app.on_event("startup")
async def load_models():
    """Load models and processors on startup"""
    global model, preprocessor, recommender, explainer, batch_explainer, dynamic_rules_engine
    
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
        
        # Initialize recommender and explainers
        recommender = CustomerRecommender()
        explainer = ChurnExplainer()
        batch_explainer = BatchChurnExplainer(max_workers=4)
        print("✓ Recommender and Explainers initialized")
        
        # Initialize dynamic rules engine
        try:
            dynamic_rules_engine = DynamicFuzzyRulesEngine('decision_engine/rules_config.json')
            print("✓ Dynamic Fuzzy Rules Engine loaded")
        except Exception as e:
            print(f"⚠ Warning: Could not load dynamic rules engine: {e}")
            dynamic_rules_engine = None
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")


# API Endpoints

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to B-Decide AI API (Enhanced)",
        "version": "2.0.0",
        "new_features": [
            "Batch recommendations with explanations",
            "Dynamic fuzzy logic rules (hot-reload)",
            "Enhanced analytics and reporting"
        ],
        "endpoints": {
            "/health": "Health check",
            "/predict": "Single customer churn prediction",
            "/recommend": "Get recommendation for a customer",
            "/batch-predict": "Batch predictions from CSV upload",
            "/batch-recommend": "Batch recommendations with full explanations (NEW)",
            "/reload-rules": "Reload fuzzy logic rules from config file (NEW)",
            "/rules-stats": "Get fuzzy logic rules statistics (NEW)"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model and preprocessor else "degraded",
        model_loaded=model is not None,
        preprocessor_loaded=preprocessor is not None,
        dynamic_rules_enabled=dynamic_rules_engine is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """Predict churn probability for a single customer"""
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
    """Get personalized recommendation for a customer"""
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
        
        # Generate recommendation (use dynamic rules if available)
        if dynamic_rules_engine:
            features = {
                'churn_probability': churn_prob,
                'tenure_months': customer_dict.get('tenure', 0),
                'monthly_charges': customer_dict.get('MonthlyCharges', 0),
                'total_charges': customer_dict.get('TotalCharges', 0)
            }
            recommendation = dynamic_rules_engine.get_recommendation(features)
            recommendation['risk_level'] = batch_explainer._determine_risk_level(churn_prob)
            recommendation['customer_features'] = features
        else:
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


@app.post("/batch-recommend", response_model=BatchRecommendationResponse)
async def batch_recommend(file: UploadFile = File(...)):
    """
    Generate batch recommendations with full explanations for CSV upload.
    Returns comprehensive analysis with predictions, recommendations, and NLP explanations.
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
        
        # Store customer IDs
        customer_ids = df['customerID'].tolist() if 'customerID' in df.columns else [f"customer_{i}" for i in range(len(df))]
        
        # Preprocess
        X = preprocessor.preprocess_new_data(df)
        
        # Predict churn probabilities
        churn_probs = model.predict_proba(X)[:, 1]
        
        # Generate recommendations using dynamic rules if available
        recommendations = []
        for i, prob in enumerate(churn_probs):
            if dynamic_rules_engine:
                customer_row = df.iloc[i]
                features = {
                    'churn_probability': float(prob),
                    'tenure_months': float(customer_row.get('tenure', 0)),
                    'monthly_charges': float(customer_row.get('MonthlyCharges', 0)),
                    'total_charges': float(customer_row.get('TotalCharges', 0))
                }
                rec = dynamic_rules_engine.get_recommendation(features)
                rec['risk_level'] = batch_explainer._determine_risk_level(float(prob))
                recommendations.append(rec)
            else:
                customer_dict = df.iloc[i].to_dict()
                rec = recommender.generate_recommendation(customer_dict, float(prob))
                recommendations.append(rec)
        
        # Extract customer features for explanations
        customer_features = []
        for i in range(len(df)):
            features = {
                'churn_probability': float(churn_probs[i]),
                'tenure_months': float(df.iloc[i].get('tenure', 0)),
                'monthly_charges': float(df.iloc[i].get('MonthlyCharges', 0)),
                'total_charges': float(df.iloc[i].get('TotalCharges', 0))
            }
            customer_features.append(features)
        
        # Generate batch explanations
        explanations = batch_explainer.generate_batch_explanations(
            churn_probs,
            recommendations,
            customer_features
        )
        
        # Create comprehensive report DataFrame
        report_df = batch_explainer.create_batch_report_dataframe(
            customer_ids,
            churn_probs,
            recommendations,
            explanations
        )
        
        # Generate executive summary
        executive_summary = batch_explainer.generate_executive_summary(report_df)
        
        # Convert DataFrame to list of dicts for response
        recommendations_list = report_df.to_dict('records')
        
        # Summary statistics
        summary_stats = {
            'avg_churn_probability': float(report_df['churn_probability'].mean()),
            'risk_distribution': report_df['risk_level'].value_counts().to_dict(),
            'urgency_distribution': report_df['urgency'].value_counts().to_dict(),
            'top_actions': report_df['recommended_action'].value_counts().head(5).to_dict(),
            'total_estimated_cost': float(report_df['estimated_cost'].sum()) if 'estimated_cost' in report_df.columns else 0,
            'avg_confidence': float(report_df['confidence'].mean())
        }
        
        return BatchRecommendationResponse(
            total_customers=len(report_df),
            recommendations=[BatchRecommendationItem(**rec) for rec in recommendations_list],
            executive_summary=executive_summary,
            summary_statistics=summary_stats,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch recommendation error: {str(e)}")


@app.post("/reload-rules", response_model=RulesReloadResponse)
async def reload_rules():
    """
    Reload fuzzy logic rules from configuration file (hot-reload).
    Allows updating business rules without restarting the server.
    """
    global dynamic_rules_engine
    
    if dynamic_rules_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Dynamic rules engine not initialized"
        )
    
    try:
        success = dynamic_rules_engine.reload_rules()
        stats = dynamic_rules_engine.get_rule_statistics()
        
        return RulesReloadResponse(
            success=success,
            message="Rules reloaded successfully" if success else "Failed to reload rules",
            rules_loaded=stats['enabled_rules'],
            config_version=stats['config_version'],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rules reload error: {str(e)}")


@app.get("/rules-stats")
async def get_rules_statistics():
    """Get statistics about loaded fuzzy logic rules"""
    if dynamic_rules_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Dynamic rules engine not initialized"
        )
    
    try:
        stats = dynamic_rules_engine.get_rule_statistics()
        return {
            "statistics": stats,
            "rules_file": dynamic_rules_engine.rules_file,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting rules stats: {str(e)}")


@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        return {"status": "No model loaded"}
    
    info = {
        "model_type": type(model).__name__,
        "model_loaded": True,
        "preprocessor_loaded": preprocessor is not None,
        "feature_count": len(preprocessor.feature_columns) if preprocessor and preprocessor.feature_columns else None,
        "dynamic_rules_enabled": dynamic_rules_engine is not None,
        "rules_count": dynamic_rules_engine.get_rule_statistics()['enabled_rules'] if dynamic_rules_engine else 0
    }
    
    return info


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("Starting B-Decide AI Enhanced API Server")
    print("Version 2.0.0 - Now with Batch Recommendations & Dynamic Rules")
    print("="*70 + "\n")
    
    uvicorn.run(
        "main_enhanced:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

