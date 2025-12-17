"""
Batch NLP Explainer for B-Decide AI
Efficiently generates explanations for multiple customers
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlp.explainer import ChurnExplainer


class BatchChurnExplainer:
    """
    Optimized batch explanation generator for multiple customers.
    Uses parallel processing for efficiency.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize batch explainer.
        
        Args:
            max_workers: Maximum number of parallel workers
        """
        self.explainer = ChurnExplainer()
        self.max_workers = max_workers
    
    def generate_batch_explanations(
        self,
        churn_probabilities: np.ndarray,
        recommendations: List[Dict],
        customer_features: List[Dict]
    ) -> List[Dict]:
        """
        Generate explanations for multiple customers in parallel.
        
        Args:
            churn_probabilities: Array of churn probabilities
            recommendations: List of recommendation dictionaries
            customer_features: List of customer feature dictionaries
            
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        
        # Process in parallel for speed
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for i in range(len(churn_probabilities)):
                future = executor.submit(
                    self._generate_single_explanation,
                    float(churn_probabilities[i]),
                    recommendations[i],
                    customer_features[i] if i < len(customer_features) else None
                )
                futures.append((i, future))
            
            # Collect results in order
            results = [None] * len(churn_probabilities)
            for idx, future in futures:
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Error generating explanation for customer {idx}: {e}")
                    results[idx] = self._get_default_explanation()
            
            explanations = results
        
        return explanations
    
    def _generate_single_explanation(
        self,
        churn_probability: float,
        recommendation: Dict,
        customer_features: Dict = None
    ) -> Dict:
        """
        Generate explanation for a single customer.
        
        Args:
            churn_probability: Churn probability
            recommendation: Recommendation dictionary
            customer_features: Customer features dictionary
            
        Returns:
            Explanation dictionary
        """
        try:
            # Get risk level
            risk_level = recommendation.get('risk_level', self._determine_risk_level(churn_probability))
            
            # Generate short and full explanations
            short_explanation = self.explainer.explain_prediction(
                churn_probability,
                risk_level,
                customer_features
            )
            
            action_explanation = self.explainer.explain_recommendation(recommendation)
            
            # Combine into concise format
            full_explanation = f"{short_explanation} {action_explanation}"
            
            return {
                'risk_assessment': short_explanation,
                'action_rationale': action_explanation,
                'full_explanation': full_explanation,
                'risk_level': risk_level,
                'urgency': self._get_urgency(recommendation.get('priority', 5))
            }
            
        except Exception as e:
            print(f"Error in explanation generation: {e}")
            return self._get_default_explanation()
    
    def _determine_risk_level(self, churn_probability: float) -> str:
        """Determine risk level from churn probability."""
        if churn_probability >= 0.70:
            return 'Critical'
        elif churn_probability >= 0.50:
            return 'High'
        elif churn_probability >= 0.30:
            return 'Medium'
        else:
            return 'Low'
    
    def _get_urgency(self, priority: int) -> str:
        """Get urgency level from priority."""
        if priority <= 2:
            return 'Urgent'
        elif priority <= 4:
            return 'Medium'
        else:
            return 'Low'
    
    def _get_default_explanation(self) -> Dict:
        """Get default explanation for error cases."""
        return {
            'risk_assessment': 'Customer analysis in progress.',
            'action_rationale': 'Standard service recommended.',
            'full_explanation': 'Customer analysis in progress. Standard service recommended.',
            'risk_level': 'Medium',
            'urgency': 'Medium'
        }
    
    def create_batch_report_dataframe(
        self,
        customer_ids: List[str],
        churn_probabilities: np.ndarray,
        recommendations: List[Dict],
        explanations: List[Dict]
    ) -> pd.DataFrame:
        """
        Create a comprehensive DataFrame with all batch results.
        
        Args:
            customer_ids: List of customer IDs
            churn_probabilities: Array of churn probabilities
            recommendations: List of recommendations
            explanations: List of explanations
            
        Returns:
            DataFrame with complete batch analysis
        """
        records = []
        
        for i in range(len(customer_ids)):
            record = {
                'customer_id': customer_ids[i],
                'churn_probability': float(churn_probabilities[i]),
                'risk_level': explanations[i]['risk_level'],
                'urgency': explanations[i]['urgency'],
                'recommended_action': recommendations[i].get('action_description', 'N/A'),
                'action_code': recommendations[i].get('action', 'continue_standard_service'),
                'confidence': recommendations[i].get('confidence', 0.5),
                'priority': recommendations[i].get('priority', 5),
                'risk_assessment': explanations[i]['risk_assessment'],
                'action_rationale': explanations[i]['action_rationale'],
                'full_explanation': explanations[i]['full_explanation']
            }
            
            # Add action metadata if available
            if 'action_metadata' in recommendations[i]:
                metadata = recommendations[i]['action_metadata']
                record['estimated_cost'] = metadata.get('estimated_cost', 0)
                record['expected_retention_lift'] = metadata.get('expected_retention_lift', 0)
                record['action_category'] = metadata.get('category', 'unknown')
            
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Sort by priority and churn probability
        df = df.sort_values(['priority', 'churn_probability'], ascending=[True, False])
        
        return df
    
    def generate_executive_summary(self, batch_df: pd.DataFrame) -> str:
        """
        Generate an executive summary for batch analysis.
        
        Args:
            batch_df: DataFrame with batch analysis results
            
        Returns:
            Executive summary string
        """
        total_customers = len(batch_df)
        avg_churn_prob = batch_df['churn_probability'].mean()
        
        risk_counts = batch_df['risk_level'].value_counts().to_dict()
        urgent_count = len(batch_df[batch_df['urgency'] == 'Urgent'])
        
        summary = f"""
╔════════════════════════════════════════════════════════════════╗
║          B-DECIDE AI - BATCH ANALYSIS EXECUTIVE SUMMARY        ║
╚════════════════════════════════════════════════════════════════╝

📊 OVERVIEW
  • Total Customers Analyzed: {total_customers:,}
  • Average Churn Risk: {avg_churn_prob:.1%}
  • Customers Requiring Urgent Action: {urgent_count}

🎯 RISK DISTRIBUTION
  • Critical Risk: {risk_counts.get('Critical', 0)} customers ({risk_counts.get('Critical', 0)/total_customers*100:.1f}%)
  • High Risk: {risk_counts.get('High', 0)} customers ({risk_counts.get('High', 0)/total_customers*100:.1f}%)
  • Medium Risk: {risk_counts.get('Medium', 0)} customers ({risk_counts.get('Medium', 0)/total_customers*100:.1f}%)
  • Low Risk: {risk_counts.get('Low', 0)} customers ({risk_counts.get('Low', 0)/total_customers*100:.1f}%)

💡 TOP RECOMMENDATIONS
"""
        
        # Top actions
        top_actions = batch_df['recommended_action'].value_counts().head(3)
        for idx, (action, count) in enumerate(top_actions.items(), 1):
            summary += f"  {idx}. {action}: {count} customers\n"
        
        # Cost analysis if available
        if 'estimated_cost' in batch_df.columns:
            total_cost = batch_df['estimated_cost'].sum()
            summary += f"\n💰 ESTIMATED RETENTION COST\n"
            summary += f"  • Total Investment: ${total_cost:,.0f}\n"
            summary += f"  • Average per Customer: ${total_cost/total_customers:,.0f}\n"
        
        summary += "\n" + "="*65
        
        return summary


if __name__ == "__main__":
    # Test batch explainer
    batch_explainer = BatchChurnExplainer()
    
    # Sample data
    customer_ids = ['CUST_001', 'CUST_002', 'CUST_003']
    churn_probs = np.array([0.82, 0.45, 0.15])
    
    recommendations = [
        {'action': 'offer_20_percent_discount_and_premium_support', 
         'action_description': 'Offer 20% discount + support',
         'confidence': 0.89, 'priority': 1, 'risk_level': 'Critical'},
        {'action': 'offer_10_percent_discount_or_upgrade',
         'action_description': 'Offer 10% discount',
         'confidence': 0.75, 'priority': 3, 'risk_level': 'Medium'},
        {'action': 'continue_standard_service',
         'action_description': 'Continue standard service',
         'confidence': 0.65, 'priority': 5, 'risk_level': 'Low'}
    ]
    
    customer_features = [
        {'tenure_months': 4, 'monthly_charges': 75.5},
        {'tenure_months': 24, 'monthly_charges': 65.0},
        {'tenure_months': 48, 'monthly_charges': 45.0}
    ]
    
    # Generate explanations
    explanations = batch_explainer.generate_batch_explanations(
        churn_probs, recommendations, customer_features
    )
    
    # Create report
    report_df = batch_explainer.create_batch_report_dataframe(
        customer_ids, churn_probs, recommendations, explanations
    )
    
    print("\n=== Batch Explainer Test ===\n")
    print(report_df[['customer_id', 'risk_level', 'recommended_action']].to_string())
    
    # Executive summary
    print(batch_explainer.generate_executive_summary(report_df))

