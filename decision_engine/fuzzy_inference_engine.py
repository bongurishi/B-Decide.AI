"""
Advanced Fuzzy Inference Engine for B-Decide AI
Implements Mamdani-style fuzzy logic with linguistic variables, membership functions,
and IF-THEN rules with aggregation and defuzzification
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import os


@dataclass
class FuzzyTerm:
    """Represents a linguistic term with membership function."""
    label: str
    membership_type: str  # "trapezoid" or "triangle"
    params: List[float]   # Parameters for membership function


@dataclass
class FuzzyVariable:
    """Represents a fuzzy input or output variable."""
    name: str
    type: str  # "linguistic"
    terms: List[FuzzyTerm]
    min_value: float = 0.0
    max_value: float = 100.0


@dataclass
class FuzzyRule:
    """Represents a fuzzy IF-THEN rule."""
    id: int
    description: str
    if_conditions: List[Dict[str, str]]  # [{"variable": "x", "is": "low"}]
    then_conclusion: Any  # Can be Dict[str, str] or List[Dict[str, str]] for multiple outputs
    weight: float = 1.0


class FuzzyInferenceEngine:
    """
    Advanced fuzzy inference engine implementing Mamdani-style fuzzy logic.
    Supports linguistic variables, multiple membership functions, and defuzzification.
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        """
        Initialize fuzzy inference engine.
        
        Args:
            config_path: Path to JSON configuration file
            config_dict: Configuration dictionary (alternative to file)
        """
        self.input_variables: Dict[str, FuzzyVariable] = {}
        self.output_variables: Dict[str, FuzzyVariable] = {}
        self.rules: List[FuzzyRule] = []
        self.aggregation_method: str = "max"
        self.defuzzification_method: str = "centroid"
        
        if config_dict:
            self.load_from_dict(config_dict)
        elif config_path:
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.load_from_dict(config)
    
    def load_from_dict(self, config: Dict):
        """Load configuration from dictionary."""
        # Load fuzzy variables
        fuzzy_vars = config.get('fuzzyVariables', {})
        
        # Input variables
        for var_data in fuzzy_vars.get('input', []):
            terms = []
            for term_data in var_data['terms']:
                term = FuzzyTerm(
                    label=term_data['label'],
                    membership_type=term_data['membership'],
                    params=term_data['params']
                )
                terms.append(term)
            
            var = FuzzyVariable(
                name=var_data['name'],
                type=var_data['type'],
                terms=terms
            )
            self.input_variables[var.name] = var
        
        # Output variables
        for var_data in fuzzy_vars.get('output', []):
            terms = []
            for term_data in var_data['terms']:
                term = FuzzyTerm(
                    label=term_data['label'],
                    membership_type=term_data['membership'],
                    params=term_data['params']
                )
                terms.append(term)
            
            var = FuzzyVariable(
                name=var_data['name'],
                type=var_data['type'],
                terms=terms
            )
            self.output_variables[var.name] = var
        
        # Load rules
        for rule_data in config.get('rules', []):
            # Handle both single and multiple outputs in THEN clause
            then_clause = rule_data['then']
            if isinstance(then_clause, list):
                # Multiple outputs
                then_conclusion = then_clause
            else:
                # Single output (convert to list for consistency)
                then_conclusion = [then_clause]
            
            rule = FuzzyRule(
                id=rule_data['id'],
                description=rule_data.get('description', ''),
                if_conditions=rule_data['if'],
                then_conclusion=then_conclusion,
                weight=rule_data.get('weight', 1.0)
            )
            self.rules.append(rule)
        
        # Load aggregation and defuzzification methods
        self.aggregation_method = config.get('aggregation', 'max')
        self.defuzzification_method = config.get('defuzzification', 'centroid')
    
    def _trapezoid_membership(self, x: float, params: List[float]) -> float:
        """
        Calculate trapezoidal membership function.
        
        Args:
            x: Input value
            params: [a, b, c, d] where a-b is left slope, b-c is plateau, c-d is right slope
            
        Returns:
            Membership degree (0 to 1)
        """
        if len(params) != 4:
            raise ValueError("Trapezoid requires 4 parameters")
        
        a, b, c, d = params
        
        if x <= a or x >= d:
            return 0.0
        elif b <= x <= c:
            return 1.0
        elif a < x < b:
            return (x - a) / (b - a) if (b - a) > 0 else 0.0
        else:  # c < x < d
            return (d - x) / (d - c) if (d - c) > 0 else 0.0
    
    def _triangle_membership(self, x: float, params: List[float]) -> float:
        """
        Calculate triangular membership function.
        
        Args:
            x: Input value
            params: [a, b, c] where a is left, b is peak, c is right
            
        Returns:
            Membership degree (0 to 1)
        """
        if len(params) != 3:
            raise ValueError("Triangle requires 3 parameters")
        
        a, b, c = params
        
        if x <= a or x >= c:
            return 0.0
        elif x == b:
            return 1.0
        elif a < x < b:
            return (x - a) / (b - a) if (b - a) > 0 else 0.0
        else:  # b < x < c
            return (c - x) / (c - b) if (c - b) > 0 else 0.0
    
    def _calculate_membership(self, x: float, term: FuzzyTerm) -> float:
        """
        Calculate membership degree for a value and term.
        
        Args:
            x: Input value
            term: Fuzzy term with membership function
            
        Returns:
            Membership degree (0 to 1)
        """
        if isinstance(term, FuzzyTerm):
            if term.membership_type == "trapezoid":
                return self._trapezoid_membership(x, term.params)
            elif term.membership_type == "triangle":
                return self._triangle_membership(x, term.params)
            else:
                raise ValueError(f"Unknown membership type: {term.membership_type}")
        else:
            raise ValueError(f"Invalid term type: {type(term)}")
    
    def _fuzzify(self, variable_name: str, value: float) -> Dict[str, float]:
        """
        Fuzzify a crisp input value into membership degrees for all terms.
        
        Args:
            variable_name: Name of the fuzzy variable
            value: Crisp input value
            
        Returns:
            Dictionary mapping term labels to membership degrees
        """
        if variable_name not in self.input_variables:
            raise ValueError(f"Unknown input variable: {variable_name}")
        
        variable = self.input_variables[variable_name]
        memberships = {}
        
        for term in variable.terms:
            memberships[term.label] = self._calculate_membership(value, term)
        
        return memberships
    
    def _evaluate_rule(self, rule: FuzzyRule, input_values: Dict[str, float]) -> Tuple[float, Dict[str, str]]:
        """
        Evaluate a fuzzy rule and return the firing strength and output terms.
        
        Args:
            rule: Fuzzy rule to evaluate
            input_values: Dictionary of input variable values
            
        Returns:
            Tuple of (firing_strength, output_terms_dict)
        """
        # Evaluate all IF conditions (AND operation)
        firing_strengths = []
        
        for condition in rule.if_conditions:
            var_name = condition['variable']
            term_label = condition['is']
            
            if var_name not in input_values:
                return (0.0, {})  # Missing input
            
            # Fuzzify the input
            memberships = self._fuzzify(var_name, input_values[var_name])
            
            # Get membership for the specified term
            if term_label in memberships:
                firing_strengths.append(memberships[term_label])
            else:
                return (0.0, {})  # Term not found
        
        # AND operation: take minimum
        if firing_strengths:
            firing_strength = min(firing_strengths) * rule.weight
        else:
            return (0.0, {})
        
        # Extract output terms from THEN clause
        output_terms = {}
        if isinstance(rule.then_conclusion, list):
            # Multiple outputs
            for output in rule.then_conclusion:
                output_terms[output['variable']] = output['is']
        else:
            # Single output
            output_terms[rule.then_conclusion['variable']] = rule.then_conclusion['is']
        
        return (firing_strength, output_terms)
    
    def _aggregate(self, rule_outputs: List[Tuple[float, Dict[str, str]]]) -> Dict[str, np.ndarray]:
        """
        Aggregate rule outputs using specified method.
        
        Args:
            rule_outputs: List of (firing_strength, output_terms) tuples
                         where output_terms maps variable names to term labels
            
        Returns:
            Dictionary mapping output variable names to aggregated membership functions
        """
        aggregated = {}
        
        for var_name in self.output_variables:
            var = self.output_variables[var_name]
            # Create universe of discourse (0 to 100 with 0.1 resolution)
            universe = np.arange(0, 101, 0.1)
            aggregated_membership = np.zeros_like(universe)
            
            for firing_strength, output_terms in rule_outputs:
                # Check if this rule has output for this variable
                if var_name in output_terms:
                    term_label = output_terms[var_name]
                    
                    # Find the term in the variable's terms list
                    term = None
                    for t in var.terms:
                        if isinstance(t, FuzzyTerm) and t.label == term_label:
                            term = t
                            break
                    
                    if term is not None:
                        # Calculate membership function for this term
                        term_membership = np.array([
                            self._calculate_membership(x, term)
                            for x in universe
                        ])
                        
                        # Apply firing strength (implication: min)
                        clipped = np.minimum(term_membership, firing_strength)
                        
                        # Aggregate using max (OR operation)
                        aggregated_membership = np.maximum(aggregated_membership, clipped)
            
            aggregated[var_name] = {
                'universe': universe,
                'membership': aggregated_membership
            }
        
        return aggregated
    
    def _defuzzify_centroid(self, universe: np.ndarray, membership: np.ndarray) -> float:
        """
        Defuzzify using centroid method.
        
        Args:
            universe: Universe of discourse values
            membership: Membership function values
            
        Returns:
            Crisp output value
        """
        # Avoid division by zero
        if np.sum(membership) == 0:
            return np.mean(universe)
        
        centroid = np.sum(universe * membership) / np.sum(membership)
        return float(centroid)
    
    def _defuzzify(self, aggregated: Dict[str, Dict]) -> Dict[str, float]:
        """
        Defuzzify aggregated membership functions to crisp values.
        
        Args:
            aggregated: Aggregated membership functions
            
        Returns:
            Dictionary mapping output variable names to crisp values
        """
        results = {}
        
        for var_name, data in aggregated.items():
            universe = data['universe']
            membership = data['membership']
            
            if self.defuzzification_method == "centroid":
                results[var_name] = self._defuzzify_centroid(universe, membership)
            else:
                # Default to centroid
                results[var_name] = self._defuzzify_centroid(universe, membership)
        
        return results
    
    def infer(self, input_values: Dict[str, float]) -> Dict[str, float]:
        """
        Perform fuzzy inference on input values.
        
        Args:
            input_values: Dictionary mapping input variable names to crisp values
            
        Returns:
            Dictionary mapping output variable names to crisp values
        """
        # Step 1: Evaluate all rules
        rule_outputs = []
        
        for rule in self.rules:
            firing_strength, output_terms = self._evaluate_rule(rule, input_values)
            
            if firing_strength > 0 and output_terms:
                rule_outputs.append((firing_strength, output_terms))
        
        # Step 2: Aggregate rule outputs
        if not rule_outputs:
            # No rules fired, return default values
            return {var_name: 50.0 for var_name in self.output_variables}
        
        aggregated = self._aggregate(rule_outputs)
        
        # Step 3: Defuzzify
        crisp_outputs = self._defuzzify(aggregated)
        
        return crisp_outputs
    
    def get_variable_info(self) -> Dict:
        """Get information about all variables."""
        return {
            'input_variables': {
                name: {
                    'name': var.name,
                    'type': var.type,
                    'terms': [t.label for t in var.terms]
                }
                for name, var in self.input_variables.items()
            },
            'output_variables': {
                name: {
                    'name': var.name,
                    'type': var.type,
                    'terms': [t.label for t in var.terms]
                }
                for name, var in self.output_variables.items()
            },
            'rules_count': len(self.rules),
            'aggregation': self.aggregation_method,
            'defuzzification': self.defuzzification_method
        }


if __name__ == "__main__":
    # Test fuzzy inference engine
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
    engine = FuzzyInferenceEngine(config_dict=config)
    
    print("=== Fuzzy Inference Engine Test ===\n")
    
    # Test cases
    test_cases = [
        {"user_engagement": 10, "purchase_history": 1},   # Low, rare
        {"user_engagement": 50, "purchase_history": 5},    # Medium, occasional
        {"user_engagement": 85, "purchase_history": 12},   # High, frequent
        {"user_engagement": 25, "purchase_history": 4},    # Mixed
    ]
    
    for i, inputs in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"  Inputs: {inputs}")
        
        # Perform inference
        outputs = engine.infer(inputs)
        
        print(f"  Output: recommendation_strength = {outputs['recommendation_strength']:.2f}")
        
        # Interpret result
        strength = outputs['recommendation_strength']
        if strength < 30:
            interpretation = "Weak"
        elif strength < 70:
            interpretation = "Moderate"
        else:
            interpretation = "Strong"
        
        print(f"  Interpretation: {interpretation} recommendation\n")
    
    # Get variable info
    print("\n=== Engine Information ===")
    info = engine.get_variable_info()
    print(f"Input Variables: {list(info['input_variables'].keys())}")
    print(f"Output Variables: {list(info['output_variables'].keys())}")
    print(f"Rules Count: {info['rules_count']}")

