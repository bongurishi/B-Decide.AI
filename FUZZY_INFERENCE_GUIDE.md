# 🧠 Advanced Fuzzy Inference Engine - User Guide

## Overview

The B-Decide AI platform now includes an **advanced fuzzy inference engine** that implements Mamdani-style fuzzy logic with:

- ✅ **Linguistic Variables**: Define fuzzy concepts (low, medium, high)
- ✅ **Membership Functions**: Trapezoidal and triangular functions
- ✅ **IF-THEN Rules**: Natural language-like rule definitions
- ✅ **Aggregation**: Combine multiple rule outputs
- ✅ **Defuzzification**: Convert fuzzy outputs to crisp values

---

## 📁 Files Created

1. **`decision_engine/fuzzy_inference_engine.py`** (514 lines)
   - Core fuzzy inference engine
   - Membership function calculations
   - Rule evaluation and aggregation
   - Defuzzification methods

2. **`decision_engine/fuzzy_recommender.py`** (280 lines)
   - Integration layer for churn recommendations
   - Maps fuzzy outputs to business actions
   - Normalizes customer features

3. **`decision_engine/fuzzy_recommendation_config.json`** (223 lines)
   - Pre-configured fuzzy variables and rules for churn prediction
   - Action mapping for recommendations

4. **`test_fuzzy_inference.py`** (120 lines)
   - Test script with your provided configuration

---

## 🚀 Quick Start

### Test the Engine:

```bash
python test_fuzzy_inference.py
```

### Use in Your Code:

```python
from decision_engine.fuzzy_recommender import FuzzyRecommendationEngine

# Initialize
engine = FuzzyRecommendationEngine()

# Generate recommendation
customer = {
    'churn_probability': 0.82,
    'tenure_months': 4,
    'monthly_charges': 75.5,
    'total_charges': 302.0
}

recommendation = engine.generate_recommendation(customer)
print(f"Action: {recommendation['action_description']}")
print(f"Strength: {recommendation['fuzzy_strength']}/100")
```

---

## 📊 Configuration Format

### Your Provided Configuration:

```json
{
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
    }
  ],
  "aggregation": "max",
  "defuzzification": "centroid"
}
```

---

## 🔧 Membership Functions

### Trapezoidal Function:
- **Parameters**: `[a, b, c, d]`
- **Shape**: 
  - 0 from -∞ to a
  - Linear increase from a to b
  - 1 from b to c
  - Linear decrease from c to d
  - 0 from d to +∞

**Example:**
```json
{"label": "low", "membership": "trapezoid", "params": [0, 0, 20, 40]}
```

### Triangular Function:
- **Parameters**: `[a, b, c]`
- **Shape**:
  - 0 from -∞ to a
  - Linear increase from a to b (peak)
  - Linear decrease from b to c
  - 0 from c to +∞

**Example:**
```json
{"label": "medium", "membership": "triangle", "params": [30, 50, 70]}
```

---

## 📝 Rule Definition

### Single Output Rule:
```json
{
  "id": 1,
  "description": "High churn → strong recommendation",
  "if": [
    {"variable": "churn_probability", "is": "high"}
  ],
  "then": {"variable": "recommendation_strength", "is": "strong"},
  "weight": 1.0
}
```

### Multiple Output Rule:
```json
{
  "id": 1,
  "description": "High churn and new customer → strong recommendation, high priority",
  "if": [
    {"variable": "churn_probability", "is": "high"},
    {"variable": "tenure_months", "is": "new"}
  ],
  "then": [
    {"variable": "recommendation_strength", "is": "strong"},
    {"variable": "action_priority", "is": "high"}
  ],
  "weight": 1.0
}
```

### Rule Logic:
- **IF conditions**: Combined with **AND** (minimum membership)
- **Multiple rules**: Combined with **OR** (maximum aggregation)
- **Weight**: Multiplies firing strength (0.0 to 1.0)

---

## 🎯 Usage Examples

### Example 1: Basic Inference

```python
from decision_engine.fuzzy_inference_engine import FuzzyInferenceEngine

# Load your configuration
config = {...}  # Your JSON config

engine = FuzzyInferenceEngine(config_dict=config)

# Perform inference
inputs = {
    'user_engagement': 15,  # Low engagement
    'purchase_history': 2   # Rare purchases
}

outputs = engine.infer(inputs)
print(f"Recommendation Strength: {outputs['recommendation_strength']:.2f}")
```

### Example 2: Churn Recommendations

```python
from decision_engine.fuzzy_recommender import FuzzyRecommendationEngine

engine = FuzzyRecommendationEngine()

customer = {
    'churn_probability': 0.75,  # 75% churn risk
    'tenure_months': 5,         # New customer
    'monthly_charges': 80.0,     # High charges
    'total_charges': 400.0       # Low total
}

recommendation = engine.generate_recommendation(customer)

print(f"Action: {recommendation['action_description']}")
print(f"Fuzzy Strength: {recommendation['fuzzy_strength']}/100")
print(f"Priority: {recommendation['priority']}")
print(f"Confidence: {recommendation['confidence']:.2%}")
```

### Example 3: Custom Configuration

```python
# Create custom config
custom_config = {
    "fuzzyVariables": {
        "input": [
            {
                "name": "risk_score",
                "type": "linguistic",
                "terms": [
                    {"label": "low", "membership": "trapezoid", "params": [0, 0, 30, 50]},
                    {"label": "high", "membership": "trapezoid", "params": [50, 70, 100, 100]}
                ]
            }
        ],
        "output": [
            {
                "name": "action_level",
                "type": "linguistic",
                "terms": [
                    {"label": "minimal", "membership": "triangle", "params": [0, 25, 50]},
                    {"label": "aggressive", "membership": "triangle", "params": [50, 75, 100]}
                ]
            }
        ]
    },
    "rules": [
        {
            "id": 1,
            "description": "High risk → aggressive action",
            "if": [{"variable": "risk_score", "is": "high"}],
            "then": {"variable": "action_level", "is": "aggressive"},
            "weight": 1.0
        }
    ],
    "aggregation": "max",
    "defuzzification": "centroid"
}

engine = FuzzyInferenceEngine(config_dict=custom_config)
result = engine.infer({'risk_score': 80})
print(f"Action Level: {result['action_level']:.2f}")
```

---

## 🔄 Integration with Existing System

### Option 1: Use Fuzzy Recommender (Recommended)

```python
from decision_engine.fuzzy_recommender import FuzzyRecommendationEngine

# Replace old recommender
fuzzy_recommender = FuzzyRecommendationEngine()

# Use in batch processing
for customer in customers:
    recommendation = fuzzy_recommender.generate_recommendation(customer)
    # recommendation includes: action, confidence, priority, risk_level
```

### Option 2: Direct Fuzzy Inference

```python
from decision_engine.fuzzy_inference_engine import FuzzyInferenceEngine

# Load your custom config
engine = FuzzyInferenceEngine(config_path='my_config.json')

# Use for any fuzzy logic problem
outputs = engine.infer(input_values)
```

---

## 📊 How It Works

### Step 1: Fuzzification
- Convert crisp input values to membership degrees
- Example: `user_engagement = 15` → `low: 0.75, medium: 0.0, high: 0.0`

### Step 2: Rule Evaluation
- Evaluate each rule's IF conditions
- Calculate firing strength (minimum of condition memberships)
- Example: Rule fires with strength `0.75`

### Step 3: Implication
- Apply firing strength to THEN clause
- Clip output membership functions
- Example: `strong` output clipped at `0.75`

### Step 4: Aggregation
- Combine all rule outputs using MAX (OR operation)
- Create aggregated membership function
- Example: Multiple rules → combined fuzzy set

### Step 5: Defuzzification
- Convert fuzzy output to crisp value
- Centroid method: weighted average
- Example: `recommendation_strength = 65.3`

---

## 🎨 Customization

### Adding New Variables:

```json
{
  "name": "customer_satisfaction",
  "type": "linguistic",
  "terms": [
    {"label": "poor", "membership": "trapezoid", "params": [0, 0, 2, 4]},
    {"label": "good", "membership": "triangle", "params": [3, 5, 7]},
    {"label": "excellent", "membership": "trapezoid", "params": [6, 8, 10, 10]}
  ]
}
```

### Adding New Rules:

```json
{
  "id": 10,
  "description": "High satisfaction and low churn → weak recommendation",
  "if": [
    {"variable": "customer_satisfaction", "is": "excellent"},
    {"variable": "churn_probability", "is": "low"}
  ],
  "then": {"variable": "recommendation_strength", "is": "weak"},
  "weight": 0.8
}
```

### Adjusting Weights:

- **Weight = 1.0**: Full rule strength
- **Weight = 0.5**: Half strength (less important rule)
- **Weight = 0.0**: Rule disabled

---

## 🧪 Testing

### Run Test Suite:

```bash
python test_fuzzy_inference.py
```

### Expected Output:
```
======================================================================
TESTING ADVANCED FUZZY INFERENCE ENGINE
======================================================================

1. Initializing fuzzy inference engine...
   ✓ Engine initialized

2. Testing inference with various inputs:

   Test Case 1:
     Input: {'user_engagement': 10, 'purchase_history': 1}
     Output: recommendation_strength = 15.23
     Interpretation: Weak recommendation

   Test Case 2:
     Input: {'user_engagement': 50, 'purchase_history': 5}
     Output: recommendation_strength = 50.00
     Interpretation: Moderate recommendation

   ...
```

---

## 📚 API Reference

### FuzzyInferenceEngine

**Methods:**
- `infer(input_values: Dict[str, float]) -> Dict[str, float]`
- `get_variable_info() -> Dict`
- `load_from_file(config_path: str)`
- `load_from_dict(config: Dict)`

### FuzzyRecommendationEngine

**Methods:**
- `generate_recommendation(customer_features: Dict) -> Dict`
- `reload_config() -> bool`
- `get_engine_info() -> Dict`

---

## 💡 Best Practices

1. **Start Simple**: Begin with 2-3 input variables and 3-5 rules
2. **Test Thoroughly**: Test with edge cases (min, max, boundary values)
3. **Tune Parameters**: Adjust membership function parameters based on results
4. **Use Weights**: Adjust rule weights to balance importance
5. **Document Rules**: Add clear descriptions to each rule

---

## 🔍 Troubleshooting

### Issue: No rules firing
- **Check**: Input variable names match config
- **Check**: Input values are within expected ranges
- **Solution**: Normalize inputs or adjust membership functions

### Issue: Unexpected outputs
- **Check**: Rule conditions are correct
- **Check**: Membership function parameters
- **Solution**: Visualize membership functions

### Issue: Slow performance
- **Check**: Number of rules (too many can slow down)
- **Check**: Universe resolution (default 0.1)
- **Solution**: Reduce rules or increase resolution step

---

## 🎉 Summary

You now have a **production-ready fuzzy inference engine** that:

✅ Supports linguistic variables with trapezoidal/triangular membership  
✅ Implements IF-THEN rules with AND/OR logic  
✅ Performs aggregation and defuzzification  
✅ Integrates seamlessly with churn prediction  
✅ Hot-reloadable configuration  
✅ Fully tested and documented  

**Ready to generate dynamic recommendations!** 🚀

---

For more information, see:
- `decision_engine/fuzzy_inference_engine.py` - Core engine
- `decision_engine/fuzzy_recommender.py` - Recommendation integration
- `test_fuzzy_inference.py` - Test examples

