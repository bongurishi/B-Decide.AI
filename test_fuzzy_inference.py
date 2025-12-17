"""
Test script for advanced fuzzy inference engine
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_engine.fuzzy_inference_engine import FuzzyInferenceEngine
from decision_engine.fuzzy_recommender import FuzzyRecommendationEngine

def test_fuzzy_inference():
    """Test the fuzzy inference engine with the provided configuration."""
    
    print("="*70)
    print("TESTING ADVANCED FUZZY INFERENCE ENGINE")
    print("="*70 + "\n")
    
    # Test configuration from user
    config = {
        "fuzzyVariables": {
            "input": [
                {
                    "name": "user_engagement",
                    "type": "linguistic",
                    "terms": [
                        {"label": "low", "membership": "trapezoid", "params": [0, 0, 20, 40]},
                        {"label": "medium", "membership": "triangle", "params": [30, 50, 70]},
                        {"label": "high", "membership": "trapezoid", "params": [60, 80, 100, 100]}
                    ]
                },
                {
                    "name": "purchase_history",
                    "type": "linguistic",
                    "terms": [
                        {"label": "rare", "membership": "trapezoid", "params": [0, 0, 1, 3]},
                        {"label": "occasional", "membership": "triangle", "params": [2, 5, 8]},
                        {"label": "frequent", "membership": "trapezoid", "params": [7, 10, 20, 20]}
                    ]
                }
            ],
            "output": [
                {
                    "name": "recommendation_strength",
                    "type": "linguistic",
                    "terms": [
                        {"label": "weak", "membership": "trapezoid", "params": [0, 0, 20, 40]},
                        {"label": "moderate", "membership": "triangle", "params": [30, 50, 70]},
                        {"label": "strong", "membership": "trapezoid", "params": [60, 80, 100, 100]}
                    ]
                }
            ]
        },
        "rules": [
            {
                "id": 1,
                "description": "Low engagement and rare purchases → weak recommendation",
                "if": [
                    {"variable": "user_engagement", "is": "low"},
                    {"variable": "purchase_history", "is": "rare"}
                ],
                "then": {"variable": "recommendation_strength", "is": "weak"},
                "weight": 1.0
            },
            {
                "id": 2,
                "description": "Medium engagement and occasional purchases → moderate recommendation",
                "if": [
                    {"variable": "user_engagement", "is": "medium"},
                    {"variable": "purchase_history", "is": "occasional"}
                ],
                "then": {"variable": "recommendation_strength", "is": "moderate"},
                "weight": 1.0
            },
            {
                "id": 3,
                "description": "High engagement or frequent purchases → strong recommendation",
                "if": [
                    {"variable": "user_engagement", "is": "high"},
                    {"variable": "purchase_history", "is": "frequent"}
                ],
                "then": {"variable": "recommendation_strength", "is": "strong"},
                "weight": 1.0
            }
        ],
        "aggregation": "max",
        "defuzzification": "centroid"
    }
    
    # Initialize engine
    print("1. Initializing fuzzy inference engine...")
    engine = FuzzyInferenceEngine(config_dict=config)
    print("   [OK] Engine initialized\n")
    
    # Test cases
    test_cases = [
        {"user_engagement": 10, "purchase_history": 1},   # Low, rare → weak
        {"user_engagement": 50, "purchase_history": 5},    # Medium, occasional → moderate
        {"user_engagement": 85, "purchase_history": 12},   # High, frequent → strong
        {"user_engagement": 25, "purchase_history": 4},    # Mixed case
    ]
    
    print("2. Testing inference with various inputs:\n")
    for i, inputs in enumerate(test_cases, 1):
        print(f"   Test Case {i}:")
        print(f"     Input: {inputs}")
        
        outputs = engine.infer(inputs)
        strength = outputs['recommendation_strength']
        
        print(f"     Output: recommendation_strength = {strength:.2f}")
        
        # Interpret
        if strength < 30:
            interpretation = "Weak"
        elif strength < 70:
            interpretation = "Moderate"
        else:
            interpretation = "Strong"
        
        print(f"     Interpretation: {interpretation} recommendation\n")
    
    # Test fuzzy recommender
    print("3. Testing Fuzzy Recommendation Engine for churn prediction:\n")
    fuzzy_recommender = FuzzyRecommendationEngine()
    
    test_customers = [
        {
            'churn_probability': 0.82,
            'tenure_months': 4,
            'monthly_charges': 75.5,
            'total_charges': 302.0
        },
        {
            'churn_probability': 0.45,
            'tenure_months': 24,
            'monthly_charges': 65.0,
            'total_charges': 1560.0
        }
    ]
    
    for i, customer in enumerate(test_customers, 1):
        print(f"   Customer {i}:")
        print(f"     Churn: {customer['churn_probability']:.2%}, Tenure: {customer['tenure_months']} months")
        
        recommendation = fuzzy_recommender.generate_recommendation(customer)
        
        print(f"     Recommendation: {recommendation['action_description']}")
        print(f"     Strength: {recommendation['fuzzy_strength']:.1f}/100")
        print(f"     Priority: {recommendation['priority']}")
        print()
    
    print("="*70)
    print("[OK] ALL TESTS COMPLETED")
    print("="*70)


if __name__ == "__main__":
    test_fuzzy_inference()

