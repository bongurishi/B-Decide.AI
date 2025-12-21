### B-Decide AI — AI-Powered Decision Intelligence Platform

My Blood • My Legacy • My Brand

🔗 Live Demo: https://b-decide-ai.streamlit.app/


# 🎯 B-Decide AI

**Decision Intelligence SaaS Platform for Customer Churn Prediction & Retention**

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-production--ready-success)

B-Decide AI is a comprehensive, production-ready Decision Intelligence platform that combines Machine Learning, Fuzzy Logic, and Natural Language Processing to predict customer churn and recommend personalized retention strategies.

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Docker Deployment](#-docker-deployment)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### 🤖 Machine Learning
- **XGBoost-based churn prediction** with 85-90% accuracy
- Comprehensive feature engineering and preprocessing
- Model persistence and versioning
- Real-time prediction capabilities

### 🧠 Fuzzy Logic Decision Engine
- **9 intelligent recommendation rules** based on customer attributes
- Priority-based action recommendations
- Confidence scoring for each recommendation
- Customizable fuzzy membership functions

### 💬 Natural Language Explanations
- **Human-readable explanations** for predictions
- Context-aware insights generation
- Batch analysis summaries
- Multi-level risk categorization

### 🌐 REST API (FastAPI)
- **High-performance async API** endpoints
- Swagger/OpenAPI documentation
- Single and batch prediction support
- CORS-enabled for web integration

### 📊 Interactive Dashboard (Streamlit)
- **Beautiful, responsive UI** with modern design
- Real-time predictions and visualizations
- CSV upload for batch analysis
- Downloadable action plans and reports

### 🐳 Production-Ready Deployment
- **Docker containerization** with multi-stage builds
- Docker Compose for orchestration
- Health checks and auto-restart
- Scalable architecture

---

## 🏗️ Architecture

```
B-Decide.AI/
│
├── data/                      # Data processing module
│   ├── raw/                   # Raw dataset storage
│   ├── processed/             # Processed data
│   └── preprocessor.py        # Data preprocessing pipeline
│
├── models/                    # ML models and artifacts
│   ├── train_model.py         # Model training script
│   ├── churn_model.pkl        # Trained XGBoost model
│   ├── preprocessor.pkl       # Fitted preprocessor
│   └── feature_importance.csv # Feature importance data
│
├── decision_engine/           # Fuzzy logic recommendation system
│   ├── fuzzy_rules.py         # Fuzzy rule definitions
│   └── recommender.py         # Recommendation generator
│
├── nlp/                       # Natural language processing
│   └── explainer.py           # Explanation generator
│
├── backend/                   # FastAPI REST API
│   └── main.py                # API endpoints and logic
│
├── frontend/                  # Streamlit dashboard
│   └── dashboard.py           # Interactive web interface
│
├── docker/                    # Docker configuration
│   ├── Dockerfile             # Container definition
│   ├── docker-compose.yml     # Multi-container orchestration
│   └── .dockerignore          # Docker ignore patterns
│
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.11+**
- **pip** (Python package manager)
- **Docker** (optional, for containerized deployment)
- **Git**

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/B-Decide.AI.git
cd B-Decide.AI
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download sample dataset** (IBM Telco Customer Churn)
- Download from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
- Place CSV file in `data/raw/telco_churn.csv`

---

## 🎯 Quick Start

### Step 1: Train the Model

```python
# Train the churn prediction model
python -c "
from models.train_model import train_and_save_model
predictor, metrics = train_and_save_model('data/raw/telco_churn.csv')
print(f'Model trained with ROC-AUC: {metrics[\"roc_auc\"]:.4f}')
"
```

### Step 2: Start the Backend API

```bash
# Start FastAPI server
cd backend
python main.py

# API will be available at: http://localhost:8000
# Swagger docs at: http://localhost:8000/docs
```

### Step 3: Launch the Dashboard

```bash
# Start Streamlit dashboard (in a new terminal)
streamlit run frontend/dashboard.py

# Dashboard will open at: http://localhost:8501
```

---

## 📖 Usage

### Single Customer Prediction (Python)

```python
from data.preprocessor import ChurnDataPreprocessor
from decision_engine.recommender import CustomerRecommender
from nlp.explainer import ChurnExplainer
import pickle

# Load model
with open('models/churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load preprocessor
preprocessor = ChurnDataPreprocessor()
preprocessor.load_preprocessor('models')

# Customer data
customer = {
    'customerID': 'CUST_001',
    'tenure': 4,
    'MonthlyCharges': 75.5,
    'TotalCharges': 302.0,
    'Contract': 'Month-to-month',
    'InternetService': 'DSL',
    # ... other features
}

# Preprocess
import pandas as pd
df = pd.DataFrame([customer])
X = preprocessor.preprocess_new_data(df)

# Predict
churn_prob = model.predict_proba(X)[0, 1]
print(f"Churn Probability: {churn_prob:.2%}")

# Get recommendation
recommender = CustomerRecommender()
recommendation = recommender.generate_recommendation(customer, churn_prob)
print(f"Recommended Action: {recommendation['action_description']}")

# Generate explanation
explainer = ChurnExplainer()
explanation = explainer.generate_full_explanation(
    churn_prob, recommendation, recommendation['customer_features']
)
print(explanation)
```

### Batch Analysis

```python
# Process multiple customers from CSV
df = pd.read_csv('data/raw/customers.csv')
X = preprocessor.preprocess_new_data(df)
churn_probs = model.predict_proba(X)[:, 1]

# Generate batch recommendations
recommendations_df = recommender.generate_batch_recommendations(df, churn_probs)

# Export action plan
recommender.export_action_plan(recommendations_df, 'action_plan.csv')
```

---

## 🔌 API Documentation

### Endpoints

#### `GET /` - Root
Returns API information and available endpoints.

#### `GET /health` - Health Check
Check API and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "timestamp": "2024-01-01T12:00:00"
}
```

#### `POST /predict` - Single Prediction
Predict churn for a single customer.

**Request Body:**
```json
{
  "customerID": "CUST_001",
  "tenure": 4,
  "MonthlyCharges": 75.5,
  "TotalCharges": 302.0,
  "Contract": "Month-to-month",
  "InternetService": "DSL"
}
```

**Response:**
```json
{
  "customerID": "CUST_001",
  "churn_probability": 0.8234,
  "risk_level": "Critical",
  "prediction": "Churn",
  "timestamp": "2024-01-01T12:00:00"
}
```

#### `POST /recommend` - Get Recommendation
Get personalized recommendation with explanation.

**Response:**
```json
{
  "customerID": "CUST_001",
  "churn_probability": 0.8234,
  "risk_level": "Critical",
  "recommended_action": "offer_20_percent_discount_and_premium_support",
  "action_description": "Offer 20% discount for 6 months plus premium support upgrade",
  "confidence": 0.89,
  "priority": 1,
  "explanation": "This customer is at critical risk...",
  "timestamp": "2024-01-01T12:00:00"
}
```

#### `POST /batch-predict` - Batch Predictions
Upload CSV for batch predictions.

**Request:** Multipart form data with CSV file

**Response:**
```json
{
  "total_customers": 100,
  "predictions": [...],
  "summary": {
    "avg_churn_probability": 0.4523,
    "risk_distribution": {
      "Critical": 15,
      "High": 25,
      "Medium": 35,
      "Low": 25
    },
    "high_risk_count": 40,
    "churn_predictions": 48
  }
}
```

### Testing API with cURL

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 4,
    "MonthlyCharges": 75.5,
    "TotalCharges": 302.0
  }'

# Batch prediction
curl -X POST http://localhost:8000/batch-predict \
  -F "file=@customers.csv"
```

---

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start all services
cd docker
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Services:**
- **Backend API:** http://localhost:8000
- **Frontend Dashboard:** http://localhost:8501

### Using Dockerfile Only

```bash
# Build image
docker build -t bdecide-ai -f docker/Dockerfile .

# Run container
docker run -p 8000:8000 -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  bdecide-ai
```

---

## 📊 Model Performance

### Metrics (on IBM Telco Dataset)

| Metric | Score |
|--------|-------|
| **Accuracy** | 85-90% |
| **Precision** | 80-85% |
| **Recall** | 75-80% |
| **F1 Score** | 77-82% |
| **ROC-AUC** | 88-92% |

### Top Features

1. **Tenure** - Customer relationship duration
2. **Monthly Charges** - Recurring payment amount
3. **Total Charges** - Lifetime customer value
4. **Contract Type** - Contract duration commitment
5. **Internet Service** - Service tier and type

---

## 🎓 Decision Rules

### Fuzzy Logic Rules Summary

| Risk Level | Churn Prob | Action Priority | Recommended Actions |
|------------|-----------|-----------------|---------------------|
| **Critical** | 70-100% | 1 (Urgent) | 20% discount + premium support, VIP package |
| **High** | 50-70% | 2 (Important) | 15% discount, loyalty rewards program |
| **Medium** | 30-50% | 3 (Standard) | 10% discount, account review call |
| **Low** | 0-30% | 4-5 (Monitor) | Appreciation email, referral program |

---

## 🧪 Testing

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
```

### Test Individual Modules

```python
# Test preprocessor
python data/preprocessor.py

# Test fuzzy rules
python decision_engine/fuzzy_rules.py

# Test explainer
python nlp/explainer.py
```

---

## 📈 Scaling and Production

### Performance Optimization

1. **Model Optimization**
   - Use model quantization for smaller size
   - Implement model caching
   - Batch predictions for efficiency

2. **API Optimization**
   - Enable Redis caching
   - Use load balancer (Nginx)
   - Implement rate limiting

3. **Database Integration**
   - Store predictions in PostgreSQL/MongoDB
   - Implement audit logging
   - Track model performance over time

### Monitoring

- Add Prometheus metrics
- Implement structured logging
- Set up alerting for model drift

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Add type hints to functions
- Write docstrings for all modules and functions
- Add unit tests for new features

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **IBM Telco Dataset** - Sample dataset for churn prediction
- **XGBoost Team** - High-performance gradient boosting library
- **FastAPI** - Modern web framework for APIs
- **Streamlit** - Rapid dashboard development framework

---

## 📞 Support

For questions, issues, or suggestions:

- **Issues:** [GitHub Issues](https://github.com/yourusername/B-Decide.AI/issues)
- **Email:** support@bdecide.ai
- **Documentation:** [Full Docs](https://docs.bdecide.ai)

---

## 🗺️ Roadmap

### Version 1.1 (Planned)
- [ ] PostgreSQL integration for prediction history
- [ ] A/B testing framework for recommendations
- [ ] Email notification system
- [ ] Advanced dashboards with more charts

### Version 2.0 (Future)
- [ ] Multi-model ensemble predictions
- [ ] AutoML capabilities
- [ ] Real-time streaming predictions
- [ ] Multi-tenant support

---
### Screenshots
<img width="1782" height="861" alt="image" src="https://github.com/user-attachments/assets/3e7a4790-1874-4299-85dc-e972bbb4ac0f" />
<img width="1919" height="881" alt="image" src="https://github.com/user-attachments/assets/3aad950a-2469-4ca5-8970-34b0efe24d5f" />
<img width="1630" height="882" alt="image" src="https://github.com/user-attachments/assets/0874e6c5-0d91-4c63-9810-7ddc7c27cb85" />
<img width="505" height="742" alt="image" src="https://github.com/user-attachments/assets/3d728c08-ffd7-4665-abdc-f3f697c0a066" />
<img width="1844" height="786" alt="image" src="https://github.com/user-attachments/assets/05329767-90e6-45e7-97fa-680c1743afaf" />
<img width="1919" height="903" alt="image" src="https://github.com/user-attachments/assets/2b5cd513-80a7-4c07-ac28-f2710f327e6c" />
<img width="1919" height="885" alt="image" src="https://github.com/user-attachments/assets/22033b2b-e749-4424-b118-e03725679ca1" />
<img width="1919" height="913" alt="image" src="https://github.com/user-attachments/assets/d4dcba70-7998-45cb-be9d-e043cbc04ec1" />
<img width="1919" height="841" alt="image" src="https://github.com/user-attachments/assets/f3ceb37a-3a6b-4ee8-9086-b590b7fd3f70" />

-------

##  Live Demo (4-Minutes Walkthrough)

▶️ Watch the full working demo of B-Decide.AI:
[Click to view demo video](https://docs.google.com/videos/d/1h_ET4tEAACzyGXqCG61UxxEVG1O4w3xU-MOxXw6BO2s/edit?usp=sharing)

**Made with ❤️ for better business decisions**

⭐ **Star this repo** if you find it useful!


Author

Bongu Rishi
AI  Decision Intelligence Engineer
Brand: B My Blood • My Legacy • My Brand
bogurishi07@gmail.com




