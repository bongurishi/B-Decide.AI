"""
Enhanced Streamlit Dashboard for B-Decide AI
Includes batch recommendations with explanations and interactive analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import pickle
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessor import ChurnDataPreprocessor
from decision_engine.dynamic_fuzzy_rules import DynamicFuzzyRulesEngine
from nlp.batch_explainer import BatchChurnExplainer
from frontend.visualizations import ChurnVisualizations
from frontend.intro import render_intro

# Page configuration
st.set_page_config(
    page_title="B-Decide AI - Enhanced Dashboard",
    page_icon="🎯",
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
        font-size: 1.05rem;
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


class EnhancedChurnDashboard:
    """Enhanced dashboard with batch recommendations and analytics"""
    
    def __init__(self):
        """Initialize dashboard with models"""
        self.model = None
        self.preprocessor = None
        self.dynamic_rules = None
        self.batch_explainer = BatchChurnExplainer(max_workers=4)
        self.viz = ChurnVisualizations()
        self.load_models()
    
    def load_models(self):
        """Load ML model, preprocessor, and dynamic rules"""
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
                return
            
            # Load dynamic rules
            try:
                self.dynamic_rules = DynamicFuzzyRulesEngine('decision_engine/rules_config.json')
                st.session_state['dynamic_rules_loaded'] = True
            except Exception as e:
                print(f"Could not load dynamic rules: {e}")
                st.session_state['dynamic_rules_loaded'] = False
                
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.session_state['model_loaded'] = False
            st.session_state['preprocessor_loaded'] = False
            st.session_state['dynamic_rules_loaded'] = False
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<div class="main-header">🎯 B-Decide AI</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="brand-meaning">B = MyBlood, MyBrand, MyLegacy</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="brand-tagline">From Data → Decisions → Impact</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="sub-header">Enhanced Dashboard for Customer Retention Analytics</div>',
            unsafe_allow_html=True
        )
        
        # Status indicators
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.session_state.get('model_loaded', False):
                st.success("✅ ML Model Ready")
            else:
                st.error("❌ Model Not Loaded")
        
        with col2:
            if st.session_state.get('dynamic_rules_loaded', False):
                st.success("✅ Dynamic Rules Active")
            else:
                st.warning("⚠️ Using Default Rules")
        
        with col3:
            st.info("🚀 Enhanced Mode v2.0")
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        st.sidebar.title("📊 Navigation")
        
        page = st.sidebar.radio(
            "Select Page",
            [
                "🏠 Home",
                "📈 Single Prediction",
                "📊 Batch Analysis (Enhanced)",
                "📋 Action Analytics",
                "⚙️ Rules Manager",
                "ℹ️ About"
            ]
        )
        
        st.sidebar.markdown("---")
        
        # Settings
        st.sidebar.markdown("### ⚙️ Settings")
        st.sidebar.number_input(
            "Churn Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key='churn_threshold'
        )
        
        st.sidebar.checkbox(
            "Show Explanations",
            value=True,
            key='show_explanations'
        )
        
        st.sidebar.markdown("---")
        
        # Quick stats
        if st.session_state.get('dynamic_rules_loaded', False):
            st.sidebar.markdown("### 📈 Rules Info")
            stats = self.dynamic_rules.get_rule_statistics()
            st.sidebar.metric("Active Rules", stats['enabled_rules'])
            st.sidebar.metric("Config Version", stats['config_version'])
        
        return page
    
    def render_batch_analysis_enhanced(self):
        """Enhanced batch analysis with full recommendations and explanations"""
        st.header("📊 Enhanced Batch Analysis")
        
        if not st.session_state.get('model_loaded', False):
            st.warning("⚠️ Please train the model first.")
            return
        
        st.markdown("""
        Upload a CSV file to get:
        - ✅ Churn predictions for all customers
        - ✅ Personalized recommendations with explanations
        - ✅ Interactive analytics and visualizations
        - ✅ Executive summary report
        - ✅ Downloadable results with full details
        """)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose CSV File",
            type=['csv'],
            help="Upload customer data CSV"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} customers")
                
                # Preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(df.head(10))
                
                # Analysis button
                if st.button("🚀 Run Complete Analysis", type="primary"):
                    with st.spinner("Analyzing customers..."):
                        # Get customer IDs
                        customer_ids = df['customerID'].tolist() if 'customerID' in df.columns else [f"customer_{i}" for i in range(len(df))]
                        
                        # Preprocess
                        X = self.preprocessor.preprocess_new_data(df)
                        
                        # Predict
                        churn_probs = self.model.predict_proba(X)[:, 1]
                        
                        # Generate recommendations
                        recommendations = []
                        for i, prob in enumerate(churn_probs):
                            if self.dynamic_rules:
                                customer_row = df.iloc[i]
                                features = {
                                    'churn_probability': float(prob),
                                    'tenure_months': float(customer_row.get('tenure', 0)),
                                    'monthly_charges': float(customer_row.get('MonthlyCharges', 0)),
                                    'total_charges': float(customer_row.get('TotalCharges', 0))
                                }
                                rec = self.dynamic_rules.get_recommendation(features)
                                rec['risk_level'] = self.batch_explainer._determine_risk_level(float(prob))
                                recommendations.append(rec)
                            else:
                                st.warning("Using default recommender")
                                break
                        
                        # Generate customer features
                        customer_features = []
                        for i in range(len(df)):
                            features = {
                                'churn_probability': float(churn_probs[i]),
                                'tenure_months': float(df.iloc[i].get('tenure', 0)),
                                'monthly_charges': float(df.iloc[i].get('MonthlyCharges', 0)),
                                'total_charges': float(df.iloc[i].get('TotalCharges', 0))
                            }
                            customer_features.append(features)
                        
                        # Generate explanations
                        explanations = self.batch_explainer.generate_batch_explanations(
                            churn_probs,
                            recommendations,
                            customer_features
                        )
                        
                        # Create report
                        report_df = self.batch_explainer.create_batch_report_dataframe(
                            customer_ids,
                            churn_probs,
                            recommendations,
                            explanations
                        )
                        
                        # Store in session
                        st.session_state['enhanced_report'] = report_df
                        st.session_state['churn_probs'] = churn_probs
                        
                        st.success("✅ Analysis complete!")
                
                # Display results
                if 'enhanced_report' in st.session_state:
                    report_df = st.session_state['enhanced_report']
                    
                    st.markdown("---")
                    st.subheader("📈 Analysis Results")
                    
                    # Summary metrics
                    metrics = self.viz.create_summary_metrics_cards(report_df)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Customers", metrics['total_customers'])
                    with col2:
                        st.metric("Avg Churn Risk", f"{metrics['avg_churn_risk']:.1%}")
                    with col3:
                        st.metric("High Risk", metrics['high_risk_count'])
                    with col4:
                        st.metric("Urgent Actions", metrics['urgent_count'])
                    
                    if 'total_cost' in metrics:
                        col5, col6 = st.columns(2)
                        with col5:
                            st.metric("Total Est. Cost", f"${metrics['total_cost']:,.0f}")
                        with col6:
                            st.metric("Avg Cost/Customer", f"${metrics['avg_cost_per_customer']:,.0f}")
                    
                    st.markdown("---")
                    
                    # Tabs for different views
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📊 Visualizations",
                        "📋 Detailed Results",
                        "📝 Executive Summary",
                        "💾 Export"
                    ])
                    
                    with tab1:
                        st.subheader("Interactive Analytics")
                        
                        # Row 1: Pie and Bar
                        col1, col2 = st.columns(2)
                        with col1:
                            fig1 = self.viz.create_risk_distribution_pie(report_df)
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with col2:
                            fig2 = self.viz.create_churn_probability_histogram(report_df)
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        # Row 2: Action distribution
                        fig3 = self.viz.create_action_distribution_bar(report_df)
                        st.plotly_chart(fig3, use_container_width=True)
                        
                        # Row 3: Scatter and Box
                        col1, col2 = st.columns(2)
                        with col1:
                            fig4 = self.viz.create_priority_scatter(report_df)
                            st.plotly_chart(fig4, use_container_width=True)
                        
                        with col2:
                            if 'confidence' in report_df.columns:
                                fig5 = self.viz.create_confidence_distribution_box(report_df)
                                st.plotly_chart(fig5, use_container_width=True)
                        
                        # Cost analysis if available
                        if 'estimated_cost' in report_df.columns:
                            fig6 = self.viz.create_action_cost_analysis(report_df)
                            st.plotly_chart(fig6, use_container_width=True)
                    
                    with tab2:
                        st.subheader("Complete Analysis Table")
                        
                        # Filters
                        filter_col1, filter_col2 = st.columns(2)
                        with filter_col1:
                            risk_filter = st.multiselect(
                                "Filter by Risk Level",
                                options=['Critical', 'High', 'Medium', 'Low'],
                                default=['Critical', 'High', 'Medium', 'Low']
                            )
                        
                        with filter_col2:
                            priority_filter = st.slider(
                                "Max Priority",
                                min_value=1,
                                max_value=6,
                                value=6
                            )
                        
                        # Apply filters
                        filtered_df = report_df[
                            (report_df['risk_level'].isin(risk_filter)) &
                            (report_df['priority'] <= priority_filter)
                        ]
                        
                        # Display options
                        show_explanations = st.checkbox("Show Full Explanations", value=False)
                        
                        if show_explanations:
                            st.dataframe(filtered_df, use_container_width=True, height=400)
                        else:
                            display_cols = ['customer_id', 'churn_probability', 'risk_level', 
                                          'recommended_action', 'confidence', 'priority']
                            st.dataframe(filtered_df[display_cols], use_container_width=True, height=400)
                    
                    with tab3:
                        st.subheader("Executive Summary")
                        
                        # Generate and display summary
                        summary = self.batch_explainer.generate_executive_summary(report_df)
                        st.code(summary, language=None)
                        
                        # Top priority customers
                        st.subheader("🔴 Top Priority Customers (Immediate Action Required)")
                        urgent_df = report_df[report_df['urgency'] == 'Urgent'].head(10)
                        if len(urgent_df) > 0:
                            for idx, row in urgent_df.iterrows():
                                with st.expander(f"Customer: {row['customer_id']} (Risk: {row['churn_probability']:.1%})"):
                                    st.markdown(f"**Risk Level:** {row['risk_level']}")
                                    st.markdown(f"**Recommended Action:** {row['recommended_action']}")
                                    st.markdown(f"**Explanation:** {row['full_explanation']}")
                        else:
                            st.info("No urgent priority customers found")
                    
                    with tab4:
                        st.subheader("Export Options")
                        
                        # Full export
                        csv_full = report_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Complete Report (CSV)",
                            data=csv_full,
                            file_name="bdecide_complete_analysis.csv",
                            mime="text/csv"
                        )
                        
                        # High priority only
                        high_priority_df = report_df[report_df['priority'] <= 2]
                        csv_priority = high_priority_df.to_csv(index=False)
                        st.download_button(
                            label="🔴 Download High Priority Only (CSV)",
                            data=csv_priority,
                            file_name="bdecide_high_priority.csv",
                            mime="text/csv"
                        )
                        
                        # Summary text
                        summary = self.batch_explainer.generate_executive_summary(report_df)
                        st.download_button(
                            label="📄 Download Executive Summary (TXT)",
                            data=summary,
                            file_name="bdecide_executive_summary.txt",
                            mime="text/plain"
                        )
                        
                        st.success("✅ All exports ready for download")
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    def render_rules_manager(self):
        """Render rules management interface"""
        st.header("⚙️ Fuzzy Logic Rules Manager")
        
        if not st.session_state.get('dynamic_rules_loaded', False):
            st.warning("⚠️ Dynamic rules engine not loaded")
            return
        
        st.markdown("""
        Manage fuzzy logic rules for customer retention recommendations.
        Rules can be updated in the `rules_config.json` file.
        """)
        
        # Current rules statistics
        stats = self.dynamic_rules.get_rule_statistics()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rules", stats['total_rules'])
        with col2:
            st.metric("Enabled Rules", stats['enabled_rules'])
        with col3:
            st.metric("Config Version", stats['config_version'])
        
        # Reload button
        if st.button("🔄 Reload Rules from Config File"):
            with st.spinner("Reloading rules..."):
                success = self.dynamic_rules.reload_rules()
                if success:
                    st.success("✅ Rules reloaded successfully!")
                else:
                    st.error("❌ Failed to reload rules")
        
        # Display current rules
        st.subheader("📋 Active Rules")
        
        if self.dynamic_rules.rules:
            for rule in self.dynamic_rules.rules:
                with st.expander(f"Rule: {rule.name} (Priority: {rule.priority})"):
                    st.markdown(f"**ID:** `{rule.id}`")
                    st.markdown(f"**Enabled:** {'✅ Yes' if rule.enabled else '❌ No'}")
                    st.markdown(f"**Confidence:** {rule.confidence:.2f}")
                    st.markdown(f"**Action:** {rule.action_description}")
                    st.markdown("**Conditions:**")
                    for param, constraints in rule.conditions.items():
                        st.markdown(f"  • {param}: {constraints['min']} - {constraints['max']}")
        
        st.markdown("---")
        st.info("💡 Tip: Edit `decision_engine/rules_config.json` to modify rules, then click Reload.")
    
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
        
        if "🏠 Home" in page:
            from frontend.dashboard import ChurnDashboard
            temp_dash = ChurnDashboard()
            temp_dash.render_home()
        elif "📈 Single Prediction" in page:
            from frontend.dashboard import ChurnDashboard
            temp_dash = ChurnDashboard()
            temp_dash.render_single_prediction()
        elif "📊 Batch Analysis" in page:
            self.render_batch_analysis_enhanced()
        elif "📋 Action Analytics" in page:
            st.header("📋 Action Analytics")
            st.info("Upload data in Batch Analysis to see detailed action analytics")
        elif "⚙️ Rules Manager" in page:
            self.render_rules_manager()
        elif "ℹ️ About" in page:
            from frontend.dashboard import ChurnDashboard
            temp_dash = ChurnDashboard()
            temp_dash.render_about()


if __name__ == "__main__":
    dashboard = EnhancedChurnDashboard()
    dashboard.run()

