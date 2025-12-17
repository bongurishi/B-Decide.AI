"""
Recommendation Engine for B-Decide AI
Generates personalized customer retention recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decision_engine.fuzzy_rules import FuzzyRulesEngine


class CustomerRecommender:
    """
    Generates personalized recommendations for customer retention
    based on churn predictions and customer attributes.
    """
    
    def __init__(self):
        """Initialize the recommender with fuzzy rules engine."""
        self.fuzzy_engine = FuzzyRulesEngine()
        
    def extract_customer_features(self, customer_data: Dict, 
                                  churn_probability: float) -> Dict[str, float]:
        """
        Extract relevant features for recommendation engine.
        
        Args:
            customer_data: Dictionary with customer attributes
            churn_probability: Predicted churn probability
            
        Returns:
            Dictionary with extracted features
        """
        features = {
            'churn_probability': churn_probability
        }
        
        # Map common customer attributes
        feature_mapping = {
            'tenure': 'tenure_months',
            'tenure_months': 'tenure_months',
            'MonthlyCharges': 'monthly_charges',
            'monthly_charges': 'monthly_charges',
            'TotalCharges': 'total_charges',
            'total_charges': 'total_charges'
        }
        
        for key, mapped_key in feature_mapping.items():
            if key in customer_data:
                value = customer_data[key]
                # Handle string values
                if isinstance(value, str):
                    try:
                        value = float(value)
                    except ValueError:
                        value = 0.0
                features[mapped_key] = float(value)
        
        return features
    
    def generate_recommendation(self, customer_data: Dict, 
                               churn_probability: float) -> Dict:
        """
        Generate a personalized recommendation for a customer.
        
        Args:
            customer_data: Dictionary with customer attributes
            churn_probability: Predicted churn probability (0-1)
            
        Returns:
            Dictionary with recommendation details
        """
        # Extract features for fuzzy engine
        features = self.extract_customer_features(customer_data, churn_probability)
        
        # Get recommendation from fuzzy engine
        recommendation = self.fuzzy_engine.get_recommendation(features)
        
        # Add action description
        recommendation['action_description'] = self.fuzzy_engine.get_action_description(
            recommendation['action']
        )
        
        # Add risk level
        recommendation['risk_level'] = self._get_risk_level(churn_probability)
        
        # Add customer features used
        recommendation['customer_features'] = features
        
        return recommendation
    
    def generate_batch_recommendations(self, customers_df: pd.DataFrame, 
                                      churn_probabilities: np.ndarray) -> pd.DataFrame:
        """
        Generate recommendations for multiple customers.
        
        Args:
            customers_df: DataFrame with customer data
            churn_probabilities: Array of churn probabilities
            
        Returns:
            DataFrame with recommendations for each customer
        """
        recommendations = []
        
        for idx, (_, customer_row) in enumerate(customers_df.iterrows()):
            customer_dict = customer_row.to_dict()
            churn_prob = float(churn_probabilities[idx])
            
            recommendation = self.generate_recommendation(customer_dict, churn_prob)
            
            recommendations.append({
                'customer_id': customer_dict.get('customerID', f'customer_{idx}'),
                'churn_probability': churn_prob,
                'risk_level': recommendation['risk_level'],
                'recommended_action': recommendation['action'],
                'action_description': recommendation['action_description'],
                'confidence': recommendation['confidence'],
                'priority': recommendation['priority']
            })
        
        return pd.DataFrame(recommendations)
    
    def _get_risk_level(self, churn_probability: float) -> str:
        """
        Categorize churn probability into risk levels.
        
        Args:
            churn_probability: Churn probability (0-1)
            
        Returns:
            Risk level string
        """
        if churn_probability >= 0.70:
            return 'Critical'
        elif churn_probability >= 0.50:
            return 'High'
        elif churn_probability >= 0.30:
            return 'Medium'
        else:
            return 'Low'
    
    def get_recommendation_summary(self, recommendations_df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for a batch of recommendations.
        
        Args:
            recommendations_df: DataFrame with recommendations
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_customers': len(recommendations_df),
            'risk_distribution': recommendations_df['risk_level'].value_counts().to_dict(),
            'avg_churn_probability': recommendations_df['churn_probability'].mean(),
            'action_distribution': recommendations_df['recommended_action'].value_counts().to_dict(),
            'high_priority_customers': len(recommendations_df[recommendations_df['priority'] <= 2])
        }
        
        return summary
    
    def filter_high_priority(self, recommendations_df: pd.DataFrame, 
                           priority_threshold: int = 2) -> pd.DataFrame:
        """
        Filter recommendations to show only high priority customers.
        
        Args:
            recommendations_df: DataFrame with recommendations
            priority_threshold: Maximum priority level to include
            
        Returns:
            Filtered DataFrame
        """
        return recommendations_df[
            recommendations_df['priority'] <= priority_threshold
        ].sort_values('churn_probability', ascending=False)
    
    def export_action_plan(self, recommendations_df: pd.DataFrame, 
                          output_path: str):
        """
        Export recommendations as an action plan CSV.
        
        Args:
            recommendations_df: DataFrame with recommendations
            output_path: Path to save the action plan
        """
        # Sort by priority and churn probability
        action_plan = recommendations_df.sort_values(
            ['priority', 'churn_probability'], 
            ascending=[True, False]
        )
        
        # Add urgency flag
        action_plan['urgent'] = action_plan['priority'] <= 2
        
        # Save to CSV
        action_plan.to_csv(output_path, index=False)
        print(f"✓ Action plan exported to {output_path}")
    
    def get_action_statistics(self, recommendations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get statistics grouped by recommended action.
        
        Args:
            recommendations_df: DataFrame with recommendations
            
        Returns:
            DataFrame with action statistics
        """
        stats = recommendations_df.groupby('recommended_action').agg({
            'customer_id': 'count',
            'churn_probability': ['mean', 'min', 'max'],
            'confidence': 'mean'
        }).round(3)
        
        stats.columns = ['count', 'avg_churn_prob', 'min_churn_prob', 
                        'max_churn_prob', 'avg_confidence']
        
        return stats.sort_values('count', ascending=False)


if __name__ == "__main__":
    # Example usage
    recommender = CustomerRecommender()
    
    # Test single customer recommendation
    test_customer = {
        'customerID': 'CUST_12345',
        'tenure': 4,
        'MonthlyCharges': 75.5,
        'TotalCharges': 302.0
    }
    
    churn_prob = 0.82
    
    print("=== Customer Recommender Test ===\n")
    print(f"Customer ID: {test_customer['customerID']}")
    print(f"Churn Probability: {churn_prob:.2%}\n")
    
    recommendation = recommender.generate_recommendation(test_customer, churn_prob)
    
    print(f"Risk Level: {recommendation['risk_level']}")
    print(f"Recommended Action: {recommendation['action_description']}")
    print(f"Confidence: {recommendation['confidence']:.2f}")
    print(f"Priority: {recommendation['priority']}")

