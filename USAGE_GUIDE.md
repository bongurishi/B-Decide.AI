

# 📘 B-Decide AI v2.0 - Complete Usage Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Batch Recommendations](#batch-recommendations)
3. [Dynamic Rules Management](#dynamic-rules-management)
4. [Interactive Visualizations](#interactive-visualizations)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Trained model (`models/churn_model.pkl`)
- Dependencies installed (`pip install -r requirements.txt`)

### Starting the Enhanced System

#### Option 1: Enhanced Backend API
```bash
python backend/main_enhanced.py
```
Access at: `http://localhost:8000`  
API Docs: `http://localhost:8000/docs`

#### Option 2: Enhanced Dashboard
```bash
streamlit run frontend/dashboard_enhanced.py
```
Access at: `http://localhost:8501`

---

## 📊 Batch Recommendations

### Feature: Generate recommendations with explanations for multiple customers

### Using the Dashboard:

1. **Navigate** to "📊 Batch Analysis (Enhanced)"

2. **Prepare CSV** with customer data:
```csv
customerID,tenure,MonthlyCharges,TotalCharges,Contract,InternetService,...
CUST_001,4,75.5,302.0,Month-to-month,DSL,...
CUST_002,24,65.0,1560.0,One year,Fiber optic,...
```

3. **Upload** the CSV file

4. **Click** "🚀 Run Complete Analysis"

5. **View Results** in four tabs:
   - **Visualizations**: Interactive charts
   - **Detailed Results**: Full data table
   - **Executive Summary**: Management report
   - **Export**: Download options

### Using the API:

```bash
curl -X POST http://localhost:8000/batch-recommend \
  -H "Content-Type: multipart/form-data" \
  -F "file=@customers.csv"
```

**Response includes:**
- Churn probabilities
- Recommended actions
- Risk assessments
- Natural language explanations
- Cost estimates
- Executive summary

### Using Python Code:

```python
import pandas as pd
from nlp.batch_explainer import BatchChurnExplainer
from decision_engine.dynamic_fuzzy_rules import DynamicFuzzyRulesEngine

# Initialize
batch_explainer = BatchChurnExplainer(max_workers=4)
rules_engine = DynamicFuzzyRulesEngine()

# Load and predict
df = pd.read_csv('customers.csv')
churn_probs = model.predict_proba(X)[:, 1]

# Generate recommendations
recommendations = []
for i, prob in enumerate(churn_probs):
    features = {
        'churn_probability': float(prob),
        'tenure_months': float(df.iloc[i]['tenure']),
        'monthly_charges': float(df.iloc[i]['MonthlyCharges']),
        'total_charges': float(df.iloc[i]['TotalCharges'])
    }
    rec = rules_engine.get_recommendation(features)
    recommendations.append(rec)

# Generate explanations
customer_features = [...]  # Extract features
explanations = batch_explainer.generate_batch_explanations(
    churn_probs,
    recommendations,
    customer_features
)

# Create report
customer_ids = df['customerID'].tolist()
report_df = batch_explainer.create_batch_report_dataframe(
    customer_ids,
    churn_probs,
    recommendations,
    explanations
)

# Export
report_df.to_csv('results.csv', index=False)

# Generate summary
summary = batch_explainer.generate_executive_summary(report_df)
print(summary)
```

---

## ⚙️ Dynamic Rules Management

### Feature: Update business rules without code changes or server restart

### Rules Configuration File:

Located at: `decision_engine/rules_config.json`

**Structure:**
```json
{
  "version": "1.0",
  "rules": [
    {
      "id": "rule_001",
      "name": "critical_new_customer",
      "enabled": true,
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
  "action_catalog": {
    "offer_20_percent_discount_and_premium_support": {
      "category": "aggressive_retention",
      "estimated_cost": 500,
      "expected_retention_lift": 0.45
    }
  }
}
```

### Managing Rules via Dashboard:

1. **Navigate** to "⚙️ Rules Manager"

2. **View** current active rules and statistics

3. **Edit** `rules_config.json` file manually

4. **Click** "🔄 Reload Rules from Config File"

5. **Verify** rules updated successfully

### Managing Rules via API:

```bash
# Reload rules
curl -X POST http://localhost:8000/reload-rules

# Get rules statistics
curl http://localhost:8000/rules-stats
```

### Managing Rules via Python:

```python
from decision_engine.dynamic_fuzzy_rules import DynamicFuzzyRulesEngine

# Initialize
engine = DynamicFuzzyRulesEngine('decision_engine/rules_config.json')

# Get statistics
stats = engine.get_rule_statistics()
print(f"Active rules: {stats['enabled_rules']}")
print(f"Version: {stats['config_version']}")

# Reload rules (after editing JSON file)
engine.reload_rules()

# Test a rule
test_customer = {
    'churn_probability': 0.82,
    'tenure_months': 4,
    'monthly_charges': 75.5,
    'total_charges': 302.0
}

recommendation = engine.get_recommendation(test_customer)
print(f"Recommended Action: {recommendation['action_description']}")
print(f"Confidence: {recommendation['confidence']:.2f}")
print(f"Priority: {recommendation['priority']}")
```

### Creating Custom Rules:

1. **Open** `decision_engine/rules_config.json`

2. **Add** new rule to `rules` array:
```json
{
  "id": "rule_custom_001",
  "name": "my_custom_rule",
  "enabled": true,
  "priority": 3,
  "confidence": 0.80,
  "conditions": {
    "churn_probability": {"min": 0.50, "max": 0.65},
    "monthly_charges": {"min": 80, "max": 120}
  },
  "action": "custom_retention_offer",
  "action_description": "Custom retention offer for premium customers"
}
```

3. **Add** action to `action_catalog`:
```json
"custom_retention_offer": {
  "category": "moderate_retention",
  "estimated_cost": 250,
  "expected_retention_lift": 0.30
}
```

4. **Reload** rules via API or dashboard

---

## 📊 Interactive Visualizations

### Feature: Enhanced charts for customer analytics

### Available Charts:

#### 1. Risk Distribution Pie Chart
Shows percentage of customers in each risk category

```python
from frontend.visualizations import ChurnVisualizations

viz = ChurnVisualizations()
fig = viz.create_risk_distribution_pie(report_df)
fig.show()  # Or use in Streamlit: st.plotly_chart(fig)
```

#### 2. Action Distribution Bar Chart
Horizontal bar chart of recommended actions

```python
fig = viz.create_action_distribution_bar(report_df)
```

#### 3. Churn Probability Histogram
Distribution with risk threshold lines

```python
fig = viz.create_churn_probability_histogram(report_df)
```

#### 4. Priority Scatter Plot
Priority vs churn probability with color-coded risk levels

```python
fig = viz.create_priority_scatter(report_df)
```

#### 5. Confidence Box Plot
Confidence distribution by risk level

```python
fig = viz.create_confidence_distribution_box(report_df)
```

#### 6. Cost Analysis Bubble Chart
Cost vs retention lift (if cost data available)

```python
fig = viz.create_action_cost_analysis(report_df)
```

### Creating All Charts at Once:

```python
# Generate all charts
charts = viz.create_comprehensive_dashboard(report_df)

# Display in loop
for i, chart in enumerate(charts):
    chart.show()  # or st.plotly_chart(chart)
```

### Getting Summary Metrics:

```python
metrics = viz.create_summary_metrics_cards(report_df)

print(f"Total Customers: {metrics['total_customers']}")
print(f"Avg Churn Risk: {metrics['avg_churn_risk']:.2%}")
print(f"High Risk Count: {metrics['high_risk_count']}")
print(f"Urgent Actions: {metrics['urgent_count']}")

if 'total_cost' in metrics:
    print(f"Total Cost: ${metrics['total_cost']:,.0f}")
```

---

## 🎯 Advanced Features

### Parallel Processing Configuration:

```python
# Adjust number of workers for batch processing
batch_explainer = BatchChurnExplainer(max_workers=8)  # Default: 4
```

### Filtering Results:

```python
# Filter by risk level
high_risk = report_df[report_df['risk_level'].isin(['Critical', 'High'])]

# Filter by priority
urgent = report_df[report_df['priority'] <= 2]

# Filter by urgency
urgent_customers = report_df[report_df['urgency'] == 'Urgent']

# Combine filters
critical_urgent = report_df[
    (report_df['risk_level'] == 'Critical') &
    (report_df['urgency'] == 'Urgent')
]
```

### Exporting Results:

```python
# Export full report
report_df.to_csv('full_report.csv', index=False)

# Export high priority only
high_priority = report_df[report_df['priority'] <= 2]
high_priority.to_csv('high_priority.csv', index=False)

# Export with specific columns
export_cols = ['customer_id', 'churn_probability', 'risk_level', 
               'recommended_action', 'full_explanation']
report_df[export_cols].to_csv('summary_report.csv', index=False)

# Export executive summary
summary = batch_explainer.generate_executive_summary(report_df)
with open('executive_summary.txt', 'w') as f:
    f.write(summary)
```

### Cost Analysis:

```python
# Calculate total cost
if 'estimated_cost' in report_df.columns:
    total_cost = report_df['estimated_cost'].sum()
    avg_cost = report_df['estimated_cost'].mean()
    
    print(f"Total Investment: ${total_cost:,.0f}")
    print(f"Average per Customer: ${avg_cost:,.0f}")
    
    # Cost by risk level
    cost_by_risk = report_df.groupby('risk_level')['estimated_cost'].sum()
    print("\nCost by Risk Level:")
    print(cost_by_risk)
    
    # Expected ROI
    if 'expected_retention_lift' in report_df.columns:
        report_df['expected_roi'] = (
            report_df['expected_retention_lift'] * 
            report_df['churn_probability'] *
            report_df['total_charges']
        ) - report_df['estimated_cost']
        
        positive_roi = report_df[report_df['expected_roi'] > 0]
        print(f"\nCustomers with positive ROI: {len(positive_roi)}")
```

---

## 🔧 Troubleshooting

### Common Issues:

#### 1. Rules Not Loading

**Error**: "Rules file not found" or "Failed to reload rules"

**Solution**:
```bash
# Check if file exists
ls decision_engine/rules_config.json

# Validate JSON syntax
python -m json.tool decision_engine/rules_config.json

# Recreate if needed
python decision_engine/dynamic_fuzzy_rules.py
```

#### 2. Slow Batch Processing

**Issue**: Batch analysis taking too long

**Solutions**:
```python
# Increase workers
batch_explainer = BatchChurnExplainer(max_workers=8)

# Process in chunks
chunk_size = 500
for i in range(0, len(df), chunk_size):
    chunk = df[i:i+chunk_size]
    # Process chunk...
```

#### 3. Memory Issues

**Issue**: Out of memory with large datasets

**Solutions**:
```python
# Process in smaller batches
# Reduce max_workers
batch_explainer = BatchChurnExplainer(max_workers=2)

# Clear session state
if 'enhanced_report' in st.session_state:
    del st.session_state['enhanced_report']
```

#### 4. Visualization Not Showing

**Issue**: Charts not displaying

**Solutions**:
```python
# Check if DataFrame has required columns
print(report_df.columns)

# Verify data types
print(report_df.dtypes)

# Check for empty DataFrame
if report_df.empty:
    print("DataFrame is empty")
```

### Getting Help:

1. **Check Logs**: Look at terminal output for error messages
2. **Test Modules**: Run individual modules to identify issues:
```bash
python nlp/batch_explainer.py
python decision_engine/dynamic_fuzzy_rules.py
python frontend/visualizations.py
```
3. **Verify Installation**: Ensure all dependencies installed:
```bash
pip list | grep -E "(plotly|streamlit|pandas|numpy)"
```

---

## 📋 Checklist for New Users

- [ ] Install all dependencies
- [ ] Train the ML model
- [ ] Create `rules_config.json` (auto-created if missing)
- [ ] Start enhanced backend or dashboard
- [ ] Test with sample data
- [ ] Review visualizations
- [ ] Customize rules as needed
- [ ] Export and share results

---

## 💡 Best Practices

1. **Rules Management**:
   - Version your `rules_config.json`
   - Test rules before deploying
   - Document rule changes

2. **Batch Processing**:
   - Process in reasonable batch sizes (<5000)
   - Monitor memory usage
   - Save intermediate results

3. **Visualization**:
   - Use filters to focus on key segments
   - Export charts as images for reports
   - Share interactive HTML dashboards

4. **Performance**:
   - Adjust `max_workers` based on CPU
   - Cache results when possible
   - Use filters to reduce data size

---

**Happy Analyzing! 🎯**

For more information, see [ENHANCEMENTS.md](ENHANCEMENTS.md)

