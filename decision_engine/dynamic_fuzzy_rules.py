"""
Dynamic Fuzzy Logic Rules Engine for B-Decide AI
Loads rules from JSON configuration file for easy updates without code changes
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class DynamicFuzzyRule:
    """Represents a dynamically loaded fuzzy rule."""
    id: str
    name: str
    enabled: bool
    priority: int
    confidence: float
    conditions: Dict[str, Dict[str, float]]
    action: str
    action_description: str


class DynamicFuzzyRulesEngine:
    """
    Dynamic fuzzy logic engine that loads rules from JSON configuration.
    Supports hot-reloading and rule management.
    """
    
    def __init__(self, rules_file: str = 'decision_engine/rules_config.json'):
        """
        Initialize the dynamic fuzzy rules engine.
        
        Args:
            rules_file: Path to the JSON rules configuration file
        """
        self.rules_file = rules_file
        self.rules: List[DynamicFuzzyRule] = []
        self.action_catalog: Dict = {}
        self.config_version: str = ""
        self.last_loaded: Optional[datetime] = None
        self.load_rules()
    
    def load_rules(self) -> bool:
        """
        Load rules from JSON configuration file.
        
        Returns:
            True if rules loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.rules_file):
                print(f"⚠️ Warning: Rules file not found at {self.rules_file}")
                print("Creating default rules file...")
                self._create_default_rules_file()
            
            with open(self.rules_file, 'r') as f:
                config = json.load(f)
            
            # Parse rules
            self.rules = []
            for rule_data in config.get('rules', []):
                if rule_data.get('enabled', True):
                    rule = DynamicFuzzyRule(
                        id=rule_data['id'],
                        name=rule_data['name'],
                        enabled=rule_data['enabled'],
                        priority=rule_data['priority'],
                        confidence=rule_data['confidence'],
                        conditions=rule_data['conditions'],
                        action=rule_data['action'],
                        action_description=rule_data['action_description']
                    )
                    self.rules.append(rule)
            
            # Load action catalog
            self.action_catalog = config.get('action_catalog', {})
            self.config_version = config.get('version', 'unknown')
            self.last_loaded = datetime.now()
            
            # Sort rules by priority
            self.rules.sort(key=lambda x: x.priority)
            
            print(f"✓ Loaded {len(self.rules)} rules from {self.rules_file}")
            print(f"  Version: {self.config_version}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading rules: {str(e)}")
            return False
    
    def reload_rules(self) -> bool:
        """
        Reload rules from file (hot-reload capability).
        
        Returns:
            True if reload successful
        """
        print("🔄 Reloading fuzzy logic rules...")
        return self.load_rules()
    
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
        Evaluate membership degree for a condition using trapezoidal function.
        
        Args:
            value: Actual value
            min_val: Minimum threshold
            max_val: Maximum threshold
            
        Returns:
            Membership degree (0 to 1)
        """
        # Use trapezoidal membership with 10% buffer zones
        range_width = max_val - min_val
        buffer = range_width * 0.1 if range_width > 0 else 0.05
        
        a = max(0, min_val - buffer)
        b = min_val
        c = max_val
        d = max_val + buffer
        
        return self._trapezoidal_membership(value, a, b, c, d)
    
    def evaluate_rules(self, customer_data: Dict[str, float]) -> List[Dict]:
        """
        Evaluate all enabled rules for a given customer.
        
        Args:
            customer_data: Dictionary with customer attributes
            
        Returns:
            List of matched rules with their confidence scores
        """
        matched_rules = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            # Calculate membership degrees for all conditions
            memberships = []
            
            for param, constraints in rule.conditions.items():
                if param in customer_data:
                    value = customer_data[param]
                    min_val = constraints['min']
                    max_val = constraints['max']
                    membership = self._evaluate_condition(value, min_val, max_val)
                    memberships.append(membership)
            
            # Use minimum membership (AND operation in fuzzy logic)
            if memberships:
                rule_strength = min(memberships)
                
                # Only consider rules with significant strength (>0.3)
                if rule_strength > 0.3:
                    matched_rules.append({
                        'rule_id': rule.id,
                        'rule_name': rule.name,
                        'action': rule.action,
                        'action_description': rule.action_description,
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
                'action_description': 'Continue standard service with regular monitoring',
                'confidence': 0.5,
                'priority': 6,
                'rule_id': 'default',
                'rule_name': 'default',
                'strength': 0.5
            }
        
        # Return the highest priority and confidence rule
        best_match = matched_rules[0]
        
        # Add action metadata if available
        if best_match['action'] in self.action_catalog:
            best_match['action_metadata'] = self.action_catalog[best_match['action']]
        
        return best_match
    
    def get_rule_statistics(self) -> Dict:
        """
        Get statistics about loaded rules.
        
        Returns:
            Dictionary with rule statistics
        """
        return {
            'total_rules': len(self.rules),
            'enabled_rules': sum(1 for r in self.rules if r.enabled),
            'priority_distribution': {
                f'priority_{i}': sum(1 for r in self.rules if r.priority == i)
                for i in range(1, 7)
            },
            'config_version': self.config_version,
            'last_loaded': self.last_loaded.isoformat() if self.last_loaded else None
        }
    
    def _create_default_rules_file(self):
        """Create a default rules configuration file if none exists."""
        default_config = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "description": "Default fuzzy logic rules for B-Decide AI",
            "rules": [
                {
                    "id": "rule_001",
                    "name": "critical_new_customer",
                    "enabled": True,
                    "priority": 1,
                    "confidence": 0.95,
                    "conditions": {
                        "churn_probability": {"min": 0.70, "max": 1.00},
                        "tenure_months": {"min": 0, "max": 6}
                    },
                    "action": "offer_20_percent_discount_and_premium_support",
                    "action_description": "Offer 20% discount for 6 months plus premium support upgrade"
                }
            ],
            "action_catalog": {}
        }
        
        os.makedirs(os.path.dirname(self.rules_file), exist_ok=True)
        with open(self.rules_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"✓ Created default rules file at {self.rules_file}")


if __name__ == "__main__":
    # Test dynamic fuzzy rules engine
    engine = DynamicFuzzyRulesEngine()
    
    # Test customer
    test_customer = {
        'churn_probability': 0.82,
        'tenure_months': 4,
        'monthly_charges': 75.5,
        'total_charges': 302.0
    }
    
    print("\n=== Dynamic Fuzzy Rules Engine Test ===\n")
    print(f"Customer Data: {test_customer}\n")
    
    # Get recommendation
    recommendation = engine.get_recommendation(test_customer)
    print(f"Recommended Action: {recommendation['action_description']}")
    print(f"Confidence: {recommendation['confidence']:.2f}")
    print(f"Priority: {recommendation['priority']}")
    
    # Get statistics
    print("\n=== Rule Statistics ===")
    stats = engine.get_rule_statistics()
    print(f"Total Rules: {stats['total_rules']}")
    print(f"Enabled Rules: {stats['enabled_rules']}")
    print(f"Version: {stats['config_version']}")

