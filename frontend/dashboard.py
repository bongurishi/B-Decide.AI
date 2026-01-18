"""
Streamlit Dashboard for B-Decide AI
Interactive web interface for churn prediction and recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import pickle
from typing import Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessor import ChurnDataPreprocessor
from decision_engine.recommender import CustomerRecommender
from nlp.explainer import ChurnExplainer
from frontend.intro import render_intro

# Page configuration
st.set_page_config(
    page_title="B-Decide AI - Customer Churn Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .brand-meaning {
        font-size: 1.05rem;
        font-weight: 600;
        color: #424242;
        text-align: center;
        margin-top: -0.5rem;
        margin-bottom: 0.25rem;
    }
    .brand-tagline {
        font-size: 1.6rem;
        font-weight: 800;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.75rem;
        letter-spacing: 0.2px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-critical {
        color: #D32F2F;
        font-weight: bold;
    }
    .risk-high {
        color: #F57C00;
        font-weight: bold;
    }
    .risk-medium {
        color: #FBC02D;
        font-weight: bold;
    }
    .risk-low {
        color: #388E3C;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class ChurnDashboard:
    """Main dashboard class for B-Decide AI"""
    
    def __init__(self):
        """Initialize dashboard with models and processors"""
        self.model = None
        self.preprocessor = None
        self.recommender = CustomerRecommender()
        self.explainer = ChurnExplainer()
        self.load_models()
    
    def load_models(self):
        """Load ML model and preprocessor"""
        try:
            # Load model
            model_path = 'models/churn_model.pkl'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                st.session_state['model_loaded'] = True
            else:
                st.session_state['model_loaded'] = False
                return
            
            # Load preprocessor
            self.preprocessor = ChurnDataPreprocessor()
            preprocessor_path = 'models/preprocessor.pkl'
            if os.path.exists(preprocessor_path):
                self.preprocessor.load_preprocessor('models')
                st.session_state['preprocessor_loaded'] = True
            else:
                st.session_state['preprocessor_loaded'] = False
                
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.session_state['model_loaded'] = False
            st.session_state['preprocessor_loaded'] = False
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<div class="main-header"> B-Decide AI</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="brand-meaning">B = MyBlood, MyBrand, MyLegacy</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="brand-tagline">From Data → Decisions → Impact</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="sub-header">Decision Intelligence Platform for Customer Retention</div>',
            unsafe_allow_html=True
        )
        
        # Model status indicator
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.session_state.get('model_loaded', False):
                st.success(" Models Loaded Successfully")
            else:
                st.error(" Models Not Loaded - Please train the model first")
    
    def render_sidebar(self):
        """Render sidebar with navigation and settings"""
        st.sidebar.title(" Navigation")
        
        page = st.sidebar.radio(
            "Select Page",
            [" Home", " Single Prediction", " Batch Analysis", " About"]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚙️ Settings")
        
        # Threshold settings
        st.sidebar.number_input(
            "Churn Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key='churn_threshold',
            help="Probability threshold for churn classification"
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("###  Quick Guide")
        st.sidebar.info("""
        **How to Use:**
        1. Train model or load existing
        2. Upload CSV or enter data
        3. View predictions & recommendations
        4. Export action plans
        """)
        
        return page
    
    def render_home(self):
        """Render home page with overview"""
        st.header(" Welcome to B-Decide AI")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(" What We Do")
            st.markdown("""
            B-Decide AI is an advanced **Decision Intelligence Platform** that helps 
            businesses predict customer churn and take proactive retention actions.
            
            **Key Features:**
            -  **ML-Powered Predictions**: XGBoost-based churn prediction
            -  **Fuzzy Logic Engine**: Smart recommendation system
            -  **Natural Language Explanations**: Clear, actionable insights
            -  **Interactive Dashboard**: Beautiful visualizations
            -  **Production Ready**: REST API with Docker support
            """)
        
        with col2:
            st.subheader(" Platform Capabilities")
            
            # Create metrics cards
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Prediction Accuracy", "~85-90%", "High Confidence")
                st.metric("Risk Categories", "4 Levels", "Critical to Low")
            
            with metric_col2:
                st.metric("Recommendation Rules", "9 Actions", "Fuzzy Logic")
                st.metric("Processing Speed", "<1 sec", "Per Customer")
        
        st.markdown("---")
        
        # Quick stats
        st.subheader(" How It Works")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 1️ Predict")
            st.info("Upload customer data and get churn probability predictions using our trained ML model.")
        
        with col2:
            st.markdown("### 2️ Recommend")
            st.info("Get personalized retention strategies based on fuzzy logic rules and customer attributes.")
        
        with col3:
            st.markdown("### 3️ Explain")
            st.info("Receive clear, natural language explanations for all predictions and recommendations.")
    
    def render_single_prediction(self):
        """Render single customer prediction page"""
        st.header(" Single Customer Prediction")
        
        if not st.session_state.get('model_loaded', False):
            st.warning(" Please train the model first before making predictions.")
            return
        
        st.markdown("Enter customer information to get churn prediction and personalized recommendations.")
        
        # Input form
        with st.form("customer_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                customer_id = st.text_input("Customer ID", "CUST_001")
                tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0)
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=780.0)
            
            with col2:
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                payment_method = st.selectbox("Payment Method", 
                    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
            
            with col3:
                online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            
            submitted = st.form_submit_button(" Predict Churn")
        
        if submitted:
            # Create customer data
            customer_data = {
                'customerID': customer_id,
                'tenure': tenure,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges,
                'Contract': contract,
                'InternetService': internet_service,
                'PaymentMethod': payment_method,
                'OnlineSecurity': online_security,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                # Add default values for other features
                'gender': 'Male',
                'SeniorCitizen': 0,
                'Partner': 'No',
                'Dependents': 'No',
                'PhoneService': 'Yes',
                'MultipleLines': 'No',
                'OnlineBackup': 'No',
                'DeviceProtection': 'No',
                'StreamingMovies': 'No',
                'PaperlessBilling': 'Yes'
            }
            
            try:
                # Preprocess and predict
                df = pd.DataFrame([customer_data])
                X = self.preprocessor.preprocess_new_data(df)
                churn_prob = float(self.model.predict_proba(X)[0, 1])
                
                # Get recommendation
                recommendation = self.recommender.generate_recommendation(customer_data, churn_prob)
                
                # Generate explanation
                explanation = self.explainer.generate_full_explanation(
                    churn_prob, recommendation, recommendation.get('customer_features')
                )
                
                # Display results
                st.markdown("---")
                st.subheader(" Prediction Results")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Churn Probability", f"{churn_prob:.1%}")
                
                with col2:
                    risk_level = recommendation['risk_level']
                    risk_class = f"risk-{risk_level.lower()}"
                    st.markdown(f'<div class="{risk_class}">Risk Level: {risk_level}</div>', 
                               unsafe_allow_html=True)
                
                with col3:
                    st.metric("Confidence", f"{recommendation['confidence']:.1%}")
                
                with col4:
                    priority_emoji = "" if recommendation['priority'] <= 2 else "" if recommendation['priority'] <= 4 else ""
                    st.metric("Priority", f"{priority_emoji} {recommendation['priority']}")
                
                # Gauge chart for churn probability
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=churn_prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Churn Risk Score"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 50], 'color': "yellow"},
                            {'range': [50, 70], 'color': "orange"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendation
                st.subheader(" Recommended Action")
                st.success(recommendation['action_description'])
                
                # Explanation
                st.subheader(" Detailed Explanation")
                st.text_area("Analysis", explanation, height=300)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    def render_batch_analysis(self):
        """Render batch analysis page"""
        st.header(" Batch Customer Analysis")
        
        if not st.session_state.get('model_loaded', False):
            st.warning(" Please train the model first before making predictions.")
            return
        
        st.markdown("Upload a CSV file with customer data for batch predictions and analysis.")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV file should contain customer attributes"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                
                st.success(f" File uploaded successfully: {len(df)} customers")
                
                # Show data preview
                with st.expander(" Data Preview"):
                    st.dataframe(df.head(10))
                
                # Process button
                if st.button(" Analyze All Customers"):
                    with st.spinner("Analyzing customers..."):
                        # Store customer IDs
                        customer_ids = df['customerID'].tolist() if 'customerID' in df.columns else [f"customer_{i}" for i in range(len(df))]
                        
                        # Preprocess and predict
                        X = self.preprocessor.preprocess_new_data(df)
                        churn_probs = self.model.predict_proba(X)[:, 1]
                        
                        # Generate recommendations
                        recommendations_df = self.recommender.generate_batch_recommendations(df, churn_probs)
                        
                        # Store in session state
                        st.session_state['batch_results'] = recommendations_df
                        st.session_state['churn_probs'] = churn_probs
                
                # Display results if available
                if 'batch_results' in st.session_state:
                    recommendations_df = st.session_state['batch_results']
                    churn_probs = st.session_state['churn_probs']
                    
                    st.markdown("---")
                    st.subheader(" Analysis Results")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Customers", len(recommendations_df))
                    
                    with col2:
                        avg_risk = np.mean(churn_probs)
                        st.metric("Avg Churn Risk", f"{avg_risk:.1%}")
                    
                    with col3:
                        high_risk = len(recommendations_df[recommendations_df['risk_level'].isin(['Critical', 'High'])])
                        st.metric("High Risk Customers", high_risk)
                    
                    with col4:
                        urgent = len(recommendations_df[recommendations_df['priority'] <= 2])
                        st.metric("Urgent Actions", urgent)
                    
                    # Charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Risk distribution
                        risk_counts = recommendations_df['risk_level'].value_counts()
                        fig1 = px.pie(
                            values=risk_counts.values,
                            names=risk_counts.index,
                            title="Risk Level Distribution",
                            color=risk_counts.index,
                            color_discrete_map={
                                'Critical': '#D32F2F',
                                'High': '#F57C00',
                                'Medium': '#FBC02D',
                                'Low': '#388E3C'
                            }
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        # Churn probability distribution
                        fig2 = px.histogram(
                            recommendations_df,
                            x='churn_probability',
                            nbins=20,
                            title="Churn Probability Distribution",
                            labels={'churn_probability': 'Churn Probability'},
                            color_discrete_sequence=['#1E88E5']
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Action distribution
                    st.subheader("Recommended Actions")
                    action_counts = recommendations_df['action_description'].value_counts()
                    fig3 = px.bar(
                        x=action_counts.values,
                        y=action_counts.index,
                        orientation='h',
                        title="Distribution of Recommended Actions",
                        labels={'x': 'Number of Customers', 'y': 'Action'},
                        color=action_counts.values,
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # Results table
                    st.subheader(" Detailed Results")
                    
                    # Filter options
                    filter_col1, filter_col2 = st.columns(2)
                    with filter_col1:
                        risk_filter = st.multiselect(
                            "Filter by Risk Level",
                            options=['Critical', 'High', 'Medium', 'Low'],
                            default=['Critical', 'High', 'Medium', 'Low']
                        )
                    
                    with filter_col2:
                        priority_filter = st.slider(
                            "Max Priority Level",
                            min_value=1,
                            max_value=6,
                            value=6
                        )
                    
                    # Apply filters
                    filtered_df = recommendations_df[
                        (recommendations_df['risk_level'].isin(risk_filter)) &
                        (recommendations_df['priority'] <= priority_filter)
                    ].sort_values('churn_probability', ascending=False)
                    
                    st.dataframe(filtered_df, use_container_width=True)
                    
                    # Export button
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label=" Download Results (CSV)",
                        data=csv,
                        file_name="churn_analysis_results.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    def render_about(self):
        """Render about page"""
        st.header(" About B-Decide AI")
        
        st.markdown("""
        ##  B-Decide AI Platform
        
        **Version:** 1.0.0  
        **Built with:** Python 3.11+
        
        ###  Technology Stack
        
        - **Machine Learning:** XGBoost, Scikit-learn
        - **Decision Engine:** Fuzzy Logic Rules
        - **NLP:** Natural Language Explanations
        - **Backend:** FastAPI
        - **Frontend:** Streamlit
        - **Deployment:** Docker
        
        ###  Architecture
        
        ```
        ├── data/              # Data preprocessing
        ├── models/            # ML models
        ├── decision_engine/   # Fuzzy logic & recommendations
        ├── nlp/               # Natural language explanations
        ├── backend/           # FastAPI REST API
        └── frontend/          # Streamlit dashboard
        ```
        
        ###  Model Performance
        
        The XGBoost model achieves:
        - **Accuracy:** ~85-90%
        - **Precision:** ~80-85%
        - **Recall:** ~75-80%
        - **ROC-AUC:** ~88-92%
        
        ###  Data Privacy
        
        All data processing is done locally. No customer data is sent to external services.
        
        ###  Documentation
        
        For detailed documentation, please refer to the README.md file in the project repository.
        
        ###  Development
        
        This platform is production-ready and can be deployed using Docker or traditional methods.
        
        ---
        
        **Made with ❤️ for better business decisions**
        """)
    
    def run(self):
        """Main dashboard execution"""
        # Show splash intro first (one time per session)
        render_intro(
            title="B-Decide.AI",
            subheading="B = MyBlood, MyBrand, MyLegacy",
            tagline="From Data → Decisions → Impact",
            started_state_key="bdecide_started",
        )

        self.render_header()
        page = self.render_sidebar()
        
        if " Home" in page:
            self.render_home()
        elif " Single Prediction" in page:
            self.render_single_prediction()
        elif " Batch Analysis" in page:
            self.render_batch_analysis()
        elif " About" in page:
            self.render_about()


if __name__ == "__main__":
    dashboard = ChurnDashboard()
    dashboard.run()


