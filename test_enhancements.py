"""
Test Script for B-Decide AI Enhancements
Tests all new features: batch recommendations, dynamic rules, and visualizations
"""

import pandas as pd
import numpy as np
import os
import sys

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*70}")
    print(f"{text:^70}")
    print(f"{'='*70}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def print_info(text):
    print(f"{YELLOW}→ {text}{RESET}")


# Test 1: Dynamic Fuzzy Rules Engine
def test_dynamic_rules():
    print_header("TEST 1: Dynamic Fuzzy Rules Engine")
    
    try:
        from decision_engine.dynamic_fuzzy_rules import DynamicFuzzyRulesEngine
        
        # Initialize engine
        print_info("Initializing dynamic rules engine...")
        engine = DynamicFuzzyRulesEngine('decision_engine/rules_config.json')
        print_success("Engine initialized")
        
        # Get statistics
        stats = engine.get_rule_statistics()
        print_success(f"Loaded {stats['enabled_rules']} rules")
        print_success(f"Config version: {stats['config_version']}")
        
        # Test recommendation
        print_info("Testing recommendation generation...")
        test_customer = {
            'churn_probability': 0.82,
            'tenure_months': 4,
            'monthly_charges': 75.5,
            'total_charges': 302.0
        }
        
        recommendation = engine.get_recommendation(test_customer)
        print_success(f"Generated recommendation: {recommendation['action']}")
        print_success(f"Confidence: {recommendation['confidence']:.2f}")
        print_success(f"Priority: {recommendation['priority']}")
        
        # Test reload
        print_info("Testing rules reload...")
        success = engine.reload_rules()
        if success:
            print_success("Rules reloaded successfully")
        else:
            print_error("Rules reload failed")
        
        return True
        
    except Exception as e:
        print_error(f"Dynamic rules test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# Test 2: Batch Explainer
def test_batch_explainer():
    print_header("TEST 2: Batch NLP Explainer")
    
    try:
        from nlp.batch_explainer import BatchChurnExplainer
        
        # Initialize
        print_info("Initializing batch explainer...")
        batch_explainer = BatchChurnExplainer(max_workers=2)
        print_success("Batch explainer initialized")
        
        # Create sample data
        n = 10
        customer_ids = [f'CUST_{i:03d}' for i in range(n)]
        churn_probs = np.random.beta(2, 5, n)
        
        recommendations = []
        customer_features = []
        
        for i in range(n):
            rec = {
                'action': 'offer_15_percent_discount',
                'action_description': 'Offer 15% discount for 3 months',
                'confidence': np.random.beta(8, 2),
                'priority': np.random.choice([1, 2, 3, 4, 5]),
                'risk_level': np.random.choice(['Critical', 'High', 'Medium', 'Low'])
            }
            recommendations.append(rec)
            
            features = {
                'churn_probability': churn_probs[i],
                'tenure_months': np.random.uniform(1, 72),
                'monthly_charges': np.random.uniform(20, 120),
                'total_charges': np.random.uniform(100, 5000)
            }
            customer_features.append(features)
        
        # Generate explanations
        print_info(f"Generating explanations for {n} customers...")
        explanations = batch_explainer.generate_batch_explanations(
            churn_probs,
            recommendations,
            customer_features
        )
        print_success(f"Generated {len(explanations)} explanations")
        
        # Create report DataFrame
        print_info("Creating report DataFrame...")
        report_df = batch_explainer.create_batch_report_dataframe(
            customer_ids,
            churn_probs,
            recommendations,
            explanations
        )
        print_success(f"Created report with {len(report_df)} rows and {len(report_df.columns)} columns")
        
        # Generate executive summary
        print_info("Generating executive summary...")
        summary = batch_explainer.generate_executive_summary(report_df)
        print_success("Executive summary generated")
        print(f"\n{summary}\n")
        
        # Test export
        output_file = 'test_batch_report.csv'
        report_df.to_csv(output_file, index=False)
        print_success(f"Report exported to {output_file}")
        
        # Cleanup
        if os.path.exists(output_file):
            os.remove(output_file)
            print_info("Cleaned up test file")
        
        return True
        
    except Exception as e:
        print_error(f"Batch explainer test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# Test 3: Visualizations
def test_visualizations():
    print_header("TEST 3: Interactive Visualizations")
    
    try:
        from frontend.visualizations import ChurnVisualizations
        
        # Initialize
        print_info("Initializing visualizations...")
        viz = ChurnVisualizations()
        print_success("Visualizations initialized")
        
        # Create sample data
        n = 100
        sample_data = pd.DataFrame({
            'customer_id': [f'CUST_{i:03d}' for i in range(n)],
            'churn_probability': np.random.beta(2, 5, n),
            'risk_level': np.random.choice(['Critical', 'High', 'Medium', 'Low'], n, p=[0.15, 0.25, 0.35, 0.25]),
            'priority': np.random.choice([1, 2, 3, 4, 5], n),
            'confidence': np.random.beta(8, 2, n),
            'recommended_action': np.random.choice([
                'Offer 20% discount',
                'VIP package',
                'Loyalty rewards',
                'Account review',
                'Standard service'
            ], n),
            'action_description': np.random.choice([
                'Offer 20% discount',
                'VIP package',
                'Loyalty rewards',
                'Account review',
                'Standard service'
            ], n),
            'estimated_cost': np.random.uniform(50, 500, n),
            'expected_retention_lift': np.random.uniform(0.1, 0.5, n)
        })
        
        print_info(f"Created sample data with {len(sample_data)} customers")
        
        # Test individual charts
        print_info("Testing risk distribution pie chart...")
        fig1 = viz.create_risk_distribution_pie(sample_data)
        print_success("Risk pie chart created")
        
        print_info("Testing action distribution bar chart...")
        fig2 = viz.create_action_distribution_bar(sample_data)
        print_success("Action bar chart created")
        
        print_info("Testing churn probability histogram...")
        fig3 = viz.create_churn_probability_histogram(sample_data)
        print_success("Churn histogram created")
        
        print_info("Testing priority scatter plot...")
        fig4 = viz.create_priority_scatter(sample_data)
        print_success("Priority scatter plot created")
        
        print_info("Testing confidence box plot...")
        fig5 = viz.create_confidence_distribution_box(sample_data)
        print_success("Confidence box plot created")
        
        print_info("Testing cost analysis chart...")
        fig6 = viz.create_action_cost_analysis(sample_data)
        print_success("Cost analysis chart created")
        
        # Test comprehensive dashboard
        print_info("Creating comprehensive dashboard...")
        charts = viz.create_comprehensive_dashboard(sample_data)
        print_success(f"Created {len(charts)} charts")
        
        # Test summary metrics
        print_info("Calculating summary metrics...")
        metrics = viz.create_summary_metrics_cards(sample_data)
        print_success("Summary metrics calculated:")
        print(f"  • Total Customers: {metrics['total_customers']}")
        print(f"  • Avg Churn Risk: {metrics['avg_churn_risk']:.2%}")
        print(f"  • High Risk Count: {metrics['high_risk_count']}")
        print(f"  • Urgent Count: {metrics['urgent_count']}")
        
        if 'total_cost' in metrics:
            print(f"  • Total Cost: ${metrics['total_cost']:,.0f}")
            print(f"  • Avg Cost/Customer: ${metrics['avg_cost_per_customer']:,.0f}")
        
        return True
        
    except Exception as e:
        print_error(f"Visualizations test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# Test 4: Integration Test
def test_integration():
    print_header("TEST 4: Integration Test")
    
    try:
        from decision_engine.dynamic_fuzzy_rules import DynamicFuzzyRulesEngine
        from nlp.batch_explainer import BatchChurnExplainer
        from frontend.visualizations import ChurnVisualizations
        
        print_info("Running end-to-end integration test...")
        
        # Initialize all components
        rules_engine = DynamicFuzzyRulesEngine()
        batch_explainer = BatchChurnExplainer(max_workers=2)
        viz = ChurnVisualizations()
        
        # Create sample customers
        n = 50
        customer_ids = [f'CUST_{i:03d}' for i in range(n)]
        churn_probs = np.random.beta(2, 5, n)
        
        # Generate recommendations using dynamic rules
        print_info(f"Generating recommendations for {n} customers...")
        recommendations = []
        customer_features = []
        
        for i in range(n):
            prob = float(churn_probs[i])
            features = {
                'churn_probability': prob,
                'tenure_months': float(np.random.uniform(1, 72)),
                'monthly_charges': float(np.random.uniform(20, 120)),
                'total_charges': float(np.random.uniform(100, 5000))
            }
            
            rec = rules_engine.get_recommendation(features)
            recommendations.append(rec)
            customer_features.append(features)
        
        print_success("Recommendations generated")
        
        # Generate explanations
        print_info("Generating batch explanations...")
        explanations = batch_explainer.generate_batch_explanations(
            churn_probs,
            recommendations,
            customer_features
        )
        print_success("Explanations generated")
        
        # Create report
        print_info("Creating comprehensive report...")
        report_df = batch_explainer.create_batch_report_dataframe(
            customer_ids,
            churn_probs,
            recommendations,
            explanations
        )
        print_success("Report created")
        
        # Create visualizations
        print_info("Creating visualizations...")
        charts = viz.create_comprehensive_dashboard(report_df)
        print_success(f"Created {len(charts)} interactive charts")
        
        # Calculate metrics
        metrics = viz.create_summary_metrics_cards(report_df)
        print_success("Metrics calculated")
        
        # Generate summary
        summary = batch_explainer.generate_executive_summary(report_df)
        print_success("Executive summary generated")
        
        print(f"\n{GREEN}{'='*70}")
        print(f"{'INTEGRATION TEST RESULTS':^70}")
        print(f"{'='*70}{RESET}")
        print(f"  ✓ Processed {n} customers")
        print(f"  ✓ Generated {len(recommendations)} recommendations")
        print(f"  ✓ Created {len(explanations)} explanations")
        print(f"  ✓ Built {len(charts)} visualizations")
        print(f"  ✓ Average churn risk: {metrics['avg_churn_risk']:.2%}")
        print(f"  ✓ High risk customers: {metrics['high_risk_count']}")
        
        return True
        
    except Exception as e:
        print_error(f"Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print_header("B-DECIDE AI ENHANCEMENTS TEST SUITE")
    print_info("Testing all new features...")
    
    results = {
        'Dynamic Rules': test_dynamic_rules(),
        'Batch Explainer': test_batch_explainer(),
        'Visualizations': test_visualizations(),
        'Integration': test_integration()
    }
    
    print_header("TEST SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = f"{GREEN}PASSED{RESET}" if result else f"{RED}FAILED{RESET}"
        print(f"  {test_name:20} : {status}")
    
    print(f"\n{BLUE}{'='*70}")
    if passed == total:
        print(f"{GREEN}ALL TESTS PASSED ({passed}/{total}){RESET}")
    else:
        print(f"{YELLOW}SOME TESTS FAILED ({passed}/{total}){RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

