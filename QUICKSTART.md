# 🚀 B-Decide AI - Quick Start Guide

This guide will get you up and running with B-Decide AI in 5 minutes!

---

## ⚡ Super Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate Sample Data & Train Model
```bash
# Generate sample data (if you don't have the IBM Telco dataset)
python data/sample_data_generator.py

# Train the model
python train.py
```

### Step 3: Launch the Application
```bash
# Option A: Run Frontend Dashboard (Recommended for beginners)
streamlit run frontend/dashboard.py

# Option B: Run Backend API
python backend/main.py
# Then visit http://localhost:8000/docs for API documentation
```

**That's it!** 🎉 The dashboard will open in your browser.

---

## 🐳 Docker Quick Start (Even Easier!)

If you have Docker installed:

```bash
# Build and start everything
cd docker
docker-compose up -d

# Access the application
# - Dashboard: http://localhost:8501
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

---

## 📝 Detailed Setup Instructions

### 1. Prerequisites Check

```bash
# Check Python version (need 3.11+)
python --version

# Check pip
pip --version

# (Optional) Check Docker
docker --version
```

### 2. Project Setup

```bash
# Clone repository
git clone https://github.com/yourusername/B-Decide.AI.git
cd B-Decide.AI

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Data Preparation

**Option A: Use Sample Data** (Quickest)
```bash
python data/sample_data_generator.py
```

**Option B: Use Real IBM Telco Dataset**
1. Download from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
2. Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in `data/raw/` folder
3. Rename to `telco_churn.csv`

### 4. Train the Model

```bash
python train.py
```

Expected output:
```
✓ Data loaded successfully
✓ Model training complete!
✓ Accuracy: 0.8534
✓ ROC-AUC: 0.8912
```

### 5. Launch Applications

**Start the Dashboard:**
```bash
streamlit run frontend/dashboard.py
```
Opens at `http://localhost:8501`

**Start the API (separate terminal):**
```bash
python backend/main.py
```
Available at `http://localhost:8000`
API docs at `http://localhost:8000/docs`

---

## 🎯 First Predictions

### Using the Dashboard

1. Open dashboard at `http://localhost:8501`
2. Navigate to "📈 Single Prediction"
3. Enter customer details
4. Click "🔮 Predict Churn"
5. View results and recommendations!

### Using the API

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customerID": "TEST_001",
    "tenure": 4,
    "MonthlyCharges": 75.5,
    "TotalCharges": 302.0,
    "Contract": "Month-to-month",
    "InternetService": "DSL"
  }'
```

### Using Python Code

```python
from data.preprocessor import ChurnDataPreprocessor
import pickle
import pandas as pd

# Load model
with open('models/churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load preprocessor
preprocessor = ChurnDataPreprocessor()
preprocessor.load_preprocessor('models')

# Customer data
customer = {
    'customerID': 'TEST_001',
    'tenure': 4,
    'MonthlyCharges': 75.5,
    'TotalCharges': 302.0,
    'Contract': 'Month-to-month',
    # ... add other required fields
}

# Predict
df = pd.DataFrame([customer])
X = preprocessor.preprocess_new_data(df)
churn_prob = model.predict_proba(X)[0, 1]

print(f"Churn Probability: {churn_prob:.2%}")
```

---

## 📊 Test with Sample Customers

Upload this CSV to the dashboard's batch analysis:

```csv
customerID,tenure,MonthlyCharges,TotalCharges,Contract,InternetService
CUST_001,4,75.5,302.0,Month-to-month,DSL
CUST_002,24,65.0,1560.0,One year,Fiber optic
CUST_003,48,45.0,2160.0,Two year,DSL
```

Save as `test_customers.csv` and upload via dashboard.

---

## 🔧 Troubleshooting

### Model Not Found Error
```bash
# Retrain the model
python train.py
```

### Port Already in Use
```bash
# Change port in backend/main.py or:
uvicorn backend.main:app --port 8001

# For Streamlit:
streamlit run frontend/dashboard.py --server.port 8502
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Docker Issues
```bash
# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

---

## 📚 Next Steps

After getting started:

1. **Explore the Dashboard** - Try all features
2. **Test the API** - Check out `/docs` endpoint
3. **Read the README** - Full documentation
4. **Customize Rules** - Modify fuzzy logic rules in `decision_engine/fuzzy_rules.py`
5. **Deploy** - Use Docker for production deployment

---

## 💡 Tips

- **Start Simple:** Use the dashboard first before diving into API
- **Check Logs:** Look at terminal output for debugging
- **Model Updates:** Retrain periodically with new data
- **API Testing:** Use Swagger UI at `/docs` for interactive testing
- **Batch Processing:** Process multiple customers via CSV upload

---

## 🆘 Getting Help

- **Documentation:** Read `README.md` for detailed info
- **Issues:** Check terminal output for error messages
- **Samples:** Look at example usage in module `__main__` blocks
- **API Docs:** Visit `http://localhost:8000/docs`

---

## ✅ Verification Checklist

Before reporting issues, verify:

- [ ] Python 3.11+ installed
- [ ] All dependencies installed (`pip list`)
- [ ] Model trained (check `models/churn_model.pkl` exists)
- [ ] Data file present (`data/raw/telco_churn.csv`)
- [ ] Correct working directory
- [ ] Ports 8000 and 8501 available
- [ ] Virtual environment activated (if using venv)

---

**Happy Predicting! 🎯**

For more details, see the full [README.md](README.md)

