"""
Enhanced Fuzzy Recommendation Engine for B-Decide AI
Integrates advanced fuzzy inference engine with linguistic variables and IF-THEN rules
"""

import sys
import os
from typing import Dict, Optional
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decision_engine.fuzzy_inference_engine import FuzzyInferenceEngine


class FuzzyRecommendationEngine:
    """
    Enhanced recommendation engine using advanced fuzzy inference.
    Supports linguistic variables, membership functions, and IF-THEN rules.
    """
    
    def __init__(self, config_path: str = 'decision_engine/fuzzy_recommendation_config.json'):
        """
        Initialize fuzzy recommendation engine.
        
        Args:
            config_path: Path to fuzzy inference configuration JSON
        """
        self.config_path = config_path
        self.fuzzy_engine = None
        self.action_mapping = {}
        self.load_config()
    
    def load_config(self):
        """Load fuzzy inference configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Initialize fuzzy inference engine
                self.fuzzy_engine = FuzzyInferenceEngine(config_dict=config)
                
                # Load action mapping
                self.action_mapping = config.get('action_mapping', {})
                
                print(f"[OK] Fuzzy recommendation engine loaded from {self.config_path}")
            else:
                print(f"[WARNING] Config file not found at {self.config_path}")
                print("Creating default configuration...")
                self._create_default_config()
                
        except Exception as e:
            print(f"[ERROR] Error loading fuzzy config: {str(e)}")
            self.fuzzy_engine = None
    
    def _create_default_config(self):
        """Create default configuration if file doesn't exist."""
        default_config = {
            "fuzzyVariables": {
                "input": [
                    {
                        "name": "churn_probability",
                        "type": "linguistic",
                        "terms": [
                            {"label": "low", "membership": "trapezoid", "params": [0, 0, 0.2, 0.4]},
                            {"label": "medium", "membership": "triangle", "params": [0.3, 0.5, 0.7]},
                            {"label": "high", "membership": "trapezoid", "params": [0.6, 0.8, 1.0, 1.0]}
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
                    "description": "High churn → strong recommendation",
                    "if": [{"variable": "churn_probability", "is": "high"}],
                    "then": {"variable": "recommendation_strength", "is": "strong"},
                    "weight": 1.0
                }
            ],
            "aggregation": "max",
            "defuzzification": "centroid",
            "action_mapping": {}
        }
        
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        self.fuzzy_engine = FuzzyInferenceEngine(config_dict=default_config)
    
    def _normalize_inputs(self, customer_features: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize customer features to match fuzzy variable ranges.
        
        Args:
            customer_features: Raw customer features
            
        Returns:
            Normalized features for fuzzy inference
        """
        normalized = {}
        
        # Normalize churn probability (0-1 to 0-100)
        if 'churn_probability' in customer_features:
            normalized['churn_probability'] = customer_features['churn_probability'] * 100
        
        # Normalize tenure (months, already in good range)
        if 'tenure_months' in customer_features:
            normalized['tenure_months'] = min(customer_features['tenure_months'], 72)
        elif 'tenure' in customer_features:
            normalized['tenure_months'] = min(customer_features['tenure'], 72)
        
        # Normalize monthly charges (dollars, already in good range)
        if 'monthly_charges' in customer_features:
            normalized['monthly_charges'] = min(customer_features['monthly_charges'], 150)
        elif 'MonthlyCharges' in customer_features:
            normalized['monthly_charges'] = min(customer_features['MonthlyCharges'], 150)
        
        # Normalize total charges (dollars, scale down if needed)
        if 'total_charges' in customer_features:
            normalized['total_charges'] = min(customer_features['total_charges'], 10000)
        elif 'TotalCharges' in customer_features:
            normalized['total_charges'] = min(customer_features['TotalCharges'], 10000)
        
        return normalized
    
    def _map_strength_to_action(self, strength: float, priority: Optional[float] = None) -> Dict:
        """
        Map recommendation strength to specific action.
        
        Args:
            strength: Recommendation strength (0-100)
            priority: Action priority (optional)
            
        Returns:
            Dictionary with action details
        """
        # Determine strength category
        if strength >= 60:
            category = "strong"
        elif strength >= 30:
            category = "moderate"
        else:
            category = "weak"
        
        # Get action from mapping
        if category in self.action_mapping:
            actions = self.action_mapping[category].get('actions', [])
            default_action = self.action_mapping[category].get('default', 'continue_standard_service')
            
            # Select action (could be enhanced with more logic)
            selected_action = actions[0] if actions else default_action
        else:
            # Fallback actions
            if category == "strong":
                selected_action = "offer_20_percent_discount_and_premium_support"
            elif category == "moderate":
                selected_action = "offer_15_percent_discount"
            else:
                selected_action = "continue_standard_service"
        
        # Map priority
        if priority is None:
            if strength >= 60:
                priority = 1
            elif strength >= 40:
                priority = 2
            elif strength >= 30:
                priority = 3
            else:
                priority = 5
        else:
            # Round priority to integer
            priority = int(round(priority))
        
        return {
            'action': selected_action,
            'action_description': self._get_action_description(selected_action),
            'confidence': strength / 100.0,  # Convert to 0-1 range
            'priority': priority,
            'strength_category': category
        }
    
    def _get_action_description(self, action: str) -> str:
        """Get human-readable action description."""
        descriptions = {
            'offer_20_percent_discount_and_premium_support': 
                'Offer 20% discount for 6 months plus premium support upgrade',
            'offer_vip_retention_package': 
                'Provide VIP retention package with dedicated account manager',
            'offer_15_percent_discount': 
                'Offer 15% discount for 3 months',
            'offer_loyalty_rewards_program': 
                'Enroll in loyalty rewards program with points and perks',
            'offer_10_percent_discount_or_upgrade': 
                'Offer 10% discount or free service upgrade',
            'schedule_account_review_call': 
                'Schedule personalized account review call',
            'send_appreciation_email_with_benefits': 
                'Send appreciation email highlighting exclusive benefits',
            'continue_standard_service': 
                'Continue standard service with regular monitoring'
        }
        return descriptions.get(action, 'Custom retention action')
    
    def generate_recommendation(self, customer_features: Dict[str, float]) -> Dict:
        """
        Generate recommendation using fuzzy inference.
        
        Args:
            customer_features: Dictionary with customer attributes
                              (churn_probability, tenure_months, monthly_charges, total_charges)
            
        Returns:
            Dictionary with recommendation details
        """
        if self.fuzzy_engine is None:
            return self._get_default_recommendation()
        
        try:
            # Normalize inputs
            normalized_inputs = self._normalize_inputs(customer_features)
            
            # Perform fuzzy inference
            fuzzy_outputs = self.fuzzy_engine.infer(normalized_inputs)
            
            # Extract outputs
            strength = fuzzy_outputs.get('recommendation_strength', 50.0)
            priority = fuzzy_outputs.get('action_priority', None)
            
            # Map to action
            recommendation = self._map_strength_to_action(strength, priority)
            
            # Add risk level
            churn_prob = customer_features.get('churn_probability', 0.5)
            if churn_prob >= 0.70:
                recommendation['risk_level'] = 'Critical'
            elif churn_prob >= 0.50:
                recommendation['risk_level'] = 'High'
            elif churn_prob >= 0.30:
                recommendation['risk_level'] = 'Medium'
            else:
                recommendation['risk_level'] = 'Low'
            
            # Add customer features for reference
            recommendation['customer_features'] = customer_features
            recommendation['fuzzy_strength'] = strength
            recommendation['fuzzy_priority'] = priority if priority else None
            
            return recommendation
            
        except Exception as e:
            print(f"Error in fuzzy recommendation: {str(e)}")
            return self._get_default_recommendation()
    
    def _get_default_recommendation(self) -> Dict:
        """Get default recommendation if fuzzy engine fails."""
        return {
            'action': 'continue_standard_service',
            'action_description': 'Continue standard service with regular monitoring',
            'confidence': 0.5,
            'priority': 5,
            'risk_level': 'Medium',
            'strength_category': 'weak',
            'customer_features': {}
        }
    
    def reload_config(self) -> bool:
        """
        Reload configuration from file (hot-reload).
        
        Returns:
            True if reload successful
        """
        try:
            self.load_config()
            return True
        except Exception as e:
            print(f"Error reloading config: {str(e)}")
            return False
    
    def get_engine_info(self) -> Dict:
        """Get information about the fuzzy inference engine."""
        if self.fuzzy_engine is None:
            return {'status': 'not_loaded'}
        
        info = self.fuzzy_engine.get_variable_info()
        info['config_path'] = self.config_path
        info['action_mapping_available'] = len(self.action_mapping) > 0
        
        return info


if __name__ == "__main__":
    # Test fuzzy recommendation engine
    engine = FuzzyRecommendationEngine()
    
    print("\n=== Fuzzy Recommendation Engine Test ===\n")
    
    # Test cases
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
        },
        {
            'churn_probability': 0.15,
            'tenure_months': 48,
            'monthly_charges': 45.0,
            'total_charges': 2160.0
        }
    ]
    
    for i, customer in enumerate(test_customers, 1):
        print(f"Test Customer {i}:")
        print(f"  Churn Probability: {customer['churn_probability']:.2%}")
        print(f"  Tenure: {customer['tenure_months']} months")
        print(f"  Monthly Charges: ${customer['monthly_charges']:.2f}")
        
        recommendation = engine.generate_recommendation(customer)
        
        print(f"\n  Recommendation:")
        print(f"    Action: {recommendation['action_description']}")
        print(f"    Strength: {recommendation['fuzzy_strength']:.1f}/100")
        print(f"    Confidence: {recommendation['confidence']:.2%}")
        print(f"    Priority: {recommendation['priority']}")
        print(f"    Risk Level: {recommendation['risk_level']}")
        print()
    
    # Get engine info
    print("\n=== Engine Information ===")
    info = engine.get_engine_info()
    print(f"Input Variables: {list(info['input_variables'].keys())}")
    print(f"Output Variables: {list(info['output_variables'].keys())}")
    print(f"Rules Count: {info['rules_count']}")

