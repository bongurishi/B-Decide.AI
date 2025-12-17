"""
NLP Explainer Module for B-Decide AI
Converts predictions and recommendations into human-readable explanations
"""

from typing import Dict, List, Optional
import random


class ChurnExplainer:
    """
    Generates natural language explanations for churn predictions
    and business recommendations.
    """
    
    def __init__(self):
        """Initialize the explainer with templates and phrase banks."""
        self.risk_descriptors = self._init_risk_descriptors()
        self.action_templates = self._init_action_templates()
        
    def _init_risk_descriptors(self) -> Dict:
        """
        Initialize risk level descriptors.
        
        Returns:
            Dictionary mapping risk levels to descriptive phrases
        """
        return {
            'Critical': {
                'intro': [
                    'This customer is at critical risk of churning.',
                    'Immediate attention required - customer shows very high churn risk.',
                    'Alert: This customer has an extremely high probability of leaving.'
                ],
                'urgency': [
                    'Urgent action needed within 48 hours.',
                    'Immediate intervention required.',
                    'Time-sensitive situation - act now.'
                ]
            },
            'High': {
                'intro': [
                    'This customer shows high churn risk.',
                    'Customer exhibits elevated churn probability.',
                    'Warning: This customer is likely to churn.'
                ],
                'urgency': [
                    'Action recommended within 1 week.',
                    'Proactive outreach advised.',
                    'Timely intervention will improve retention.'
                ]
            },
            'Medium': {
                'intro': [
                    'This customer has moderate churn risk.',
                    'Customer shows some indicators of potential churn.',
                    'Moderate risk level detected.'
                ],
                'urgency': [
                    'Consider proactive engagement.',
                    'Monitor closely and engage within 2 weeks.',
                    'Strategic touchpoint recommended.'
                ]
            },
            'Low': {
                'intro': [
                    'This customer has low churn risk.',
                    'Customer shows strong retention indicators.',
                    'Minimal churn risk detected.'
                ],
                'urgency': [
                    'Continue standard engagement.',
                    'Maintain regular touchpoints.',
                    'Focus on loyalty building.'
                ]
            }
        }
    
    def _init_action_templates(self) -> Dict:
        """
        Initialize action explanation templates.
        
        Returns:
            Dictionary mapping actions to explanation templates
        """
        return {
            'offer_20_percent_discount_and_premium_support': {
                'action': 'Offer 20% discount for 6 months plus premium support upgrade',
                'rationale': 'to address high churn risk and improve customer satisfaction',
                'benefit': 'This aggressive retention offer provides immediate value and enhanced service quality.'
            },
            'offer_vip_retention_package': {
                'action': 'Provide VIP retention package with dedicated account manager',
                'rationale': 'to retain high-value customer with personalized service',
                'benefit': 'Premium treatment will strengthen relationship and demonstrate commitment.'
            },
            'offer_15_percent_discount': {
                'action': 'Offer 15% discount for 3 months',
                'rationale': 'to reduce price sensitivity and improve value perception',
                'benefit': 'Temporary price reduction will demonstrate flexibility and encourage continued subscription.'
            },
            'offer_loyalty_rewards_program': {
                'action': 'Enroll in loyalty rewards program with points and perks',
                'rationale': 'to increase engagement and long-term commitment',
                'benefit': 'Rewards program creates ongoing incentives for continued patronage.'
            },
            'offer_10_percent_discount_or_upgrade': {
                'action': 'Offer 10% discount or free service upgrade',
                'rationale': 'to enhance value proposition and address satisfaction concerns',
                'benefit': 'Flexible options allow customer to choose their preferred benefit.'
            },
            'schedule_account_review_call': {
                'action': 'Schedule personalized account review call',
                'rationale': 'to understand concerns and optimize service delivery',
                'benefit': 'Direct communication helps identify issues and strengthen relationship.'
            },
            'send_appreciation_email_with_benefits': {
                'action': 'Send appreciation email highlighting exclusive benefits',
                'rationale': 'to reinforce positive relationship and educate on value',
                'benefit': 'Recognition and education improve satisfaction and loyalty.'
            },
            'invite_to_referral_program': {
                'action': 'Invite to referral program with rewards',
                'rationale': 'to leverage satisfaction and create advocacy',
                'benefit': 'Referral program turns loyal customers into brand ambassadors.'
            },
            'continue_standard_service': {
                'action': 'Continue standard service with regular monitoring',
                'rationale': 'as customer shows strong retention indicators',
                'benefit': 'Maintain consistent quality service and monitor for any changes.'
            }
        }
    
    def _get_feature_insights(self, customer_features: Dict[str, float]) -> List[str]:
        """
        Generate insights based on customer features.
        
        Args:
            customer_features: Dictionary with customer attributes
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Tenure insights
        if 'tenure_months' in customer_features:
            tenure = customer_features['tenure_months']
            if tenure < 6:
                insights.append('very short tenure (under 6 months)')
            elif tenure < 12:
                insights.append('relatively new customer (under 1 year)')
            elif tenure < 36:
                insights.append('established customer (1-3 years)')
            else:
                insights.append('long-term loyal customer (3+ years)')
        
        # Monthly charges insights
        if 'monthly_charges' in customer_features:
            charges = customer_features['monthly_charges']
            if charges < 30:
                insights.append('low-tier service plan')
            elif charges < 70:
                insights.append('mid-tier service plan')
            else:
                insights.append('premium service plan')
        
        # Total charges insights
        if 'total_charges' in customer_features:
            total = customer_features['total_charges']
            if total < 500:
                insights.append('limited historical spend')
            elif total < 2000:
                insights.append('moderate lifetime value')
            else:
                insights.append('high lifetime value')
        
        return insights
    
    def explain_prediction(self, churn_probability: float, 
                          risk_level: str,
                          customer_features: Optional[Dict[str, float]] = None) -> str:
        """
        Generate explanation for churn prediction.
        
        Args:
            churn_probability: Predicted churn probability (0-1)
            risk_level: Risk level (Critical/High/Medium/Low)
            customer_features: Optional customer features for context
            
        Returns:
            Human-readable explanation string
        """
        # Get risk descriptors
        descriptors = self.risk_descriptors.get(risk_level, self.risk_descriptors['Medium'])
        
        # Build explanation
        explanation = f"{random.choice(descriptors['intro'])} "
        explanation += f"The churn probability is {churn_probability:.1%}, "
        
        # Add feature insights if available
        if customer_features:
            insights = self._get_feature_insights(customer_features)
            if insights:
                explanation += f"based on factors including {', '.join(insights[:2])}. "
        else:
            explanation += "indicating significant concern. "
        
        # Add urgency
        explanation += f"{random.choice(descriptors['urgency'])}"
        
        return explanation
    
    def explain_recommendation(self, recommendation: Dict) -> str:
        """
        Generate explanation for business recommendation.
        
        Args:
            recommendation: Dictionary with recommendation details
            
        Returns:
            Human-readable recommendation explanation
        """
        action_key = recommendation.get('action', 'continue_standard_service')
        action_info = self.action_templates.get(
            action_key, 
            self.action_templates['continue_standard_service']
        )
        
        explanation = f"Recommended Action: {action_info['action']}. "
        explanation += f"This action is recommended {action_info['rationale']}. "
        explanation += f"{action_info['benefit']} "
        
        # Add confidence information
        confidence = recommendation.get('confidence', 0.5)
        if confidence > 0.85:
            explanation += f"We have high confidence ({confidence:.1%}) in this recommendation."
        elif confidence > 0.70:
            explanation += f"This recommendation has good confidence ({confidence:.1%})."
        else:
            explanation += f"Consider this recommendation (confidence: {confidence:.1%}) along with other factors."
        
        return explanation
    
    def generate_full_explanation(self, churn_probability: float,
                                  recommendation: Dict,
                                  customer_features: Optional[Dict[str, float]] = None) -> str:
        """
        Generate complete explanation combining prediction and recommendation.
        
        Args:
            churn_probability: Predicted churn probability
            recommendation: Recommendation dictionary
            customer_features: Optional customer features
            
        Returns:
            Complete explanation string
        """
        risk_level = recommendation.get('risk_level', self._determine_risk_level(churn_probability))
        
        # Build complete explanation
        full_explanation = "=== Customer Churn Analysis ===\n\n"
        
        # Prediction explanation
        full_explanation += "🔍 Risk Assessment:\n"
        prediction_text = self.explain_prediction(
            churn_probability, risk_level, customer_features
        )
        full_explanation += f"{prediction_text}\n\n"
        
        # Recommendation explanation
        full_explanation += "💡 Recommended Action:\n"
        recommendation_text = self.explain_recommendation(recommendation)
        full_explanation += f"{recommendation_text}\n\n"
        
        # Add priority information
        priority = recommendation.get('priority', 5)
        if priority <= 2:
            full_explanation += "⚠️  Priority: HIGH - Immediate action required\n"
        elif priority <= 4:
            full_explanation += "📋 Priority: MEDIUM - Action recommended soon\n"
        else:
            full_explanation += "✓ Priority: STANDARD - Continue monitoring\n"
        
        return full_explanation
    
    def _determine_risk_level(self, churn_probability: float) -> str:
        """
        Determine risk level from churn probability.
        
        Args:
            churn_probability: Churn probability (0-1)
            
        Returns:
            Risk level string
        """
        if churn_probability >= 0.70:
            return 'Critical'
        elif churn_probability >= 0.50:
            return 'High'
        elif churn_probability >= 0.30:
            return 'Medium'
        else:
            return 'Low'
    
    def generate_batch_summary(self, total_customers: int,
                              risk_distribution: Dict[str, int],
                              avg_churn_prob: float) -> str:
        """
        Generate summary explanation for batch analysis.
        
        Args:
            total_customers: Total number of customers analyzed
            risk_distribution: Count of customers per risk level
            avg_churn_prob: Average churn probability
            
        Returns:
            Summary explanation string
        """
        summary = "=== Batch Analysis Summary ===\n\n"
        summary += f"📊 Total Customers Analyzed: {total_customers}\n"
        summary += f"📈 Average Churn Risk: {avg_churn_prob:.1%}\n\n"
        
        summary += "Risk Distribution:\n"
        for risk_level in ['Critical', 'High', 'Medium', 'Low']:
            count = risk_distribution.get(risk_level, 0)
            percentage = (count / total_customers * 100) if total_customers > 0 else 0
            summary += f"  • {risk_level}: {count} customers ({percentage:.1f}%)\n"
        
        # Add insights
        summary += "\n💭 Key Insights:\n"
        critical_count = risk_distribution.get('Critical', 0)
        high_count = risk_distribution.get('High', 0)
        
        if critical_count > 0:
            summary += f"  ⚠️  {critical_count} customers require immediate attention\n"
        if high_count > 0:
            summary += f"  📢 {high_count} customers should be contacted within 1 week\n"
        
        if avg_churn_prob > 0.60:
            summary += "  🔴 Overall churn risk is HIGH - consider company-wide retention initiatives\n"
        elif avg_churn_prob > 0.40:
            summary += "  🟡 Overall churn risk is MODERATE - focus on high-risk segments\n"
        else:
            summary += "  🟢 Overall churn risk is LOW - maintain current service quality\n"
        
        return summary


if __name__ == "__main__":
    # Example usage
    explainer = ChurnExplainer()
    
    # Test prediction explanation
    print("=== NLP Explainer Test ===\n")
    
    customer_features = {
        'churn_probability': 0.82,
        'tenure_months': 4,
        'monthly_charges': 75.5,
        'total_charges': 302.0
    }
    
    recommendation = {
        'action': 'offer_20_percent_discount_and_premium_support',
        'confidence': 0.89,
        'priority': 1,
        'risk_level': 'Critical'
    }
    
    # Generate full explanation
    full_explanation = explainer.generate_full_explanation(
        churn_probability=0.82,
        recommendation=recommendation,
        customer_features=customer_features
    )
    
    print(full_explanation)
    
    # Test batch summary
    print("\n" + "="*50 + "\n")
    
    batch_summary = explainer.generate_batch_summary(
        total_customers=100,
        risk_distribution={'Critical': 15, 'High': 25, 'Medium': 35, 'Low': 25},
        avg_churn_prob=0.45
    )
    
    print(batch_summary)

