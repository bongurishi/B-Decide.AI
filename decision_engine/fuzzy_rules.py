"""
Fuzzy Logic Rules Engine for B-Decide AI
Defines business rules for customer retention recommendations
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class FuzzyRule:
    """
    Represents a fuzzy rule for decision making.
    """
    name: str
    conditions: Dict[str, Tuple[float, float]]  # {parameter: (min, max)}
    action: str
    priority: int
    confidence: float


class FuzzyRulesEngine:
    """
    Fuzzy logic-based decision engine for customer retention strategies.
    Uses membership functions to evaluate rules and recommend actions.
    """
    
    def __init__(self):
        """Initialize the fuzzy rules engine with predefined rules."""
        self.rules = self._define_rules()
        
    def _define_rules(self) -> List[FuzzyRule]:
        """
        Define fuzzy rules for customer retention recommendations.
        
        Returns:
            List of FuzzyRule objects
        """
        rules = [
            # Critical churn risk rules
            FuzzyRule(
                name="critical_new_customer",
                conditions={
                    'churn_probability': (0.70, 1.00),
                    'tenure_months': (0, 6)
                },
                action="offer_20_percent_discount_and_premium_support",
                priority=1,
                confidence=0.95
            ),
            
            FuzzyRule(
                name="critical_high_value",
                conditions={
                    'churn_probability': (0.75, 1.00),
                    'monthly_charges': (70, 150)
                },
                action="offer_vip_retention_package",
                priority=1,
                confidence=0.93
            ),
            
            # High churn risk rules
            FuzzyRule(
                name="high_risk_short_tenure",
                conditions={
                    'churn_probability': (0.60, 0.80),
                    'tenure_months': (0, 12)
                },
                action="offer_15_percent_discount",
                priority=2,
                confidence=0.88
            ),
            
            FuzzyRule(
                name="high_risk_low_engagement",
                conditions={
                    'churn_probability': (0.65, 0.85),
                    'total_charges': (0, 1000)
                },
                action="offer_loyalty_rewards_program",
                priority=2,
                confidence=0.85
            ),
            
            # Medium churn risk rules
            FuzzyRule(
                name="medium_risk_price_sensitive",
                conditions={
                    'churn_probability': (0.45, 0.65),
                    'monthly_charges': (60, 100)
                },
                action="offer_10_percent_discount_or_upgrade",
                priority=3,
                confidence=0.78
            ),
            
            FuzzyRule(
                name="medium_risk_service_issues",
                conditions={
                    'churn_probability': (0.50, 0.70),
                    'tenure_months': (12, 36)
                },
                action="schedule_account_review_call",
                priority=3,
                confidence=0.75
            ),
            
            # Low churn risk - proactive engagement
            FuzzyRule(
                name="low_risk_long_tenure",
                conditions={
                    'churn_probability': (0.20, 0.45),
                    'tenure_months': (36, 100)
                },
                action="send_appreciation_email_with_benefits",
                priority=4,
                confidence=0.70
            ),
            
            FuzzyRule(
                name="low_risk_loyal_customer",
                conditions={
                    'churn_probability': (0.0, 0.30),
                    'total_charges': (3000, 10000)
                },
                action="invite_to_referral_program",
                priority=5,
                confidence=0.65
            ),
            
            # Default fallback rule
            FuzzyRule(
                name="monitor_only",
                conditions={
                    'churn_probability': (0.0, 0.20)
                },
                action="continue_standard_service",
                priority=6,
                confidence=0.60
            )
        ]
        
        return rules
    
    def _triangular_membership(self, x: float, a: float, b: float, c: float) -> float:
        """
        Calculate triangular membership function value.
        
        Args:
            x: Input value
            a: Left boundary
            b: Peak
            c: Right boundary
            
        Returns:
            Membership degree (0 to 1)
        """
        if x <= a or x >= c:
            return 0.0
        elif x == b:
            return 1.0
        elif a < x < b:
            return (x - a) / (b - a)
        else:  # b < x < c
            return (c - x) / (c - b)
    
    def _trapezoidal_membership(self, x: float, a: float, b: float, 
                                c: float, d: float) -> float:
        """
        Calculate trapezoidal membership function value.
        
        Args:
            x: Input value
            a: Left boundary
            b: Left peak
            c: Right peak
            d: Right boundary
            
        Returns:
            Membership degree (0 to 1)
        """
        if x <= a or x >= d:
            return 0.0
        elif b <= x <= c:
            return 1.0
        elif a < x < b:
            return (x - a) / (b - a)
        else:  # c < x < d
            return (d - x) / (d - c)
    
    def _evaluate_condition(self, value: float, min_val: float, max_val: float) -> float:
        """
        Evaluate membership degree for a condition.
        
        Args:
            value: Actual value
            min_val: Minimum threshold
            max_val: Maximum threshold
            
        Returns:
            Membership degree (0 to 1)
        """
        # Use trapezoidal membership with 10% buffer zones
        buffer = (max_val - min_val) * 0.1
        a = max(0, min_val - buffer)
        b = min_val
        c = max_val
        d = max_val + buffer
        
        return self._trapezoidal_membership(value, a, b, c, d)
    
    def evaluate_rules(self, customer_data: Dict[str, float]) -> List[Dict]:
        """
        Evaluate all rules for a given customer.
        
        Args:
            customer_data: Dictionary with customer attributes
                          (e.g., churn_probability, tenure_months, monthly_charges)
            
        Returns:
            List of matched rules with their confidence scores
        """
        matched_rules = []
        
        for rule in self.rules:
            # Calculate membership degrees for all conditions
            memberships = []
            
            for param, (min_val, max_val) in rule.conditions.items():
                if param in customer_data:
                    value = customer_data[param]
                    membership = self._evaluate_condition(value, min_val, max_val)
                    memberships.append(membership)
            
            # Use minimum membership (AND operation in fuzzy logic)
            if memberships:
                rule_strength = min(memberships)
                
                # Only consider rules with significant strength (>0.3)
                if rule_strength > 0.3:
                    matched_rules.append({
                        'rule_name': rule.name,
                        'action': rule.action,
                        'priority': rule.priority,
                        'strength': rule_strength,
                        'confidence': rule.confidence * rule_strength
                    })
        
        # Sort by priority and confidence
        matched_rules.sort(key=lambda x: (x['priority'], -x['confidence']))
        
        return matched_rules
    
    def get_recommendation(self, customer_data: Dict[str, float]) -> Dict:
        """
        Get the best recommendation for a customer.
        
        Args:
            customer_data: Dictionary with customer attributes
            
        Returns:
            Dictionary with recommendation details
        """
        matched_rules = self.evaluate_rules(customer_data)
        
        if not matched_rules:
            # Default recommendation if no rules match
            return {
                'action': 'continue_standard_service',
                'confidence': 0.5,
                'priority': 6,
                'rule_name': 'default',
                'strength': 0.5
            }
        
        # Return the highest priority and confidence rule
        return matched_rules[0]
    
    def get_action_description(self, action: str) -> str:
        """
        Get human-readable description of an action.
        
        Args:
            action: Action code
            
        Returns:
            Human-readable action description
        """
        action_descriptions = {
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
            
            'invite_to_referral_program': 
                'Invite to referral program with rewards',
            
            'continue_standard_service': 
                'Continue standard service with regular monitoring',
            
            'default': 
                'Monitor customer behavior and provide standard service'
        }
        
        return action_descriptions.get(action, 'Custom retention action required')
    
    def add_custom_rule(self, rule: FuzzyRule):
        """
        Add a custom rule to the engine.
        
        Args:
            rule: FuzzyRule object to add
        """
        self.rules.append(rule)
        # Re-sort rules by priority
        self.rules.sort(key=lambda x: x.priority)


if __name__ == "__main__":
    # Example usage
    engine = FuzzyRulesEngine()
    
    # Test customer with high churn risk
    test_customer = {
        'churn_probability': 0.82,
        'tenure_months': 4,
        'monthly_charges': 75.5,
        'total_charges': 302.0
    }
    
    print("=== Fuzzy Rules Engine Test ===\n")
    print(f"Customer Data: {test_customer}\n")
    
    # Get all matched rules
    matched_rules = engine.evaluate_rules(test_customer)
    print(f"Matched Rules: {len(matched_rules)}\n")
    
    for rule in matched_rules[:3]:  # Show top 3
        print(f"Rule: {rule['rule_name']}")
        print(f"  Action: {engine.get_action_description(rule['action'])}")
        print(f"  Confidence: {rule['confidence']:.2f}")
        print(f"  Priority: {rule['priority']}")
        print()
    
    # Get best recommendation
    recommendation = engine.get_recommendation(test_customer)
    print(f"Best Recommendation: {engine.get_action_description(recommendation['action'])}")
    print(f"Confidence: {recommendation['confidence']:.2f}")

