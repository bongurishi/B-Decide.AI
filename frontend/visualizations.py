"""
Interactive Visualization Components for B-Decide AI Dashboard
Enhanced charts for action outcomes and analytics
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class ChurnVisualizations:
    """
    Collection of interactive visualization functions for churn analysis.
    """
    
    @staticmethod
    def create_risk_distribution_pie(
        recommendations_df: pd.DataFrame,
        title: str = "Customer Risk Distribution"
    ) -> go.Figure:
        """
        Create an interactive pie chart showing risk level distribution.
        
        Args:
            recommendations_df: DataFrame with recommendations
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        risk_counts = recommendations_df['risk_level'].value_counts()
        
        colors = {
            'Critical': '#D32F2F',
            'High': '#F57C00',
            'Medium': '#FBC02D',
            'Low': '#388E3C'
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.4,
            marker=dict(
                colors=[colors.get(level, '#999') for level in risk_counts.index]
            ),
            textposition='inside',
            textinfo='label+percent+value',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=18, color='#1E88E5')),
            showlegend=True,
            height=400,
            margin=dict(t=80, b=40, l=40, r=40)
        )
        
        return fig
    
    @staticmethod
    def create_action_distribution_bar(
        recommendations_df: pd.DataFrame,
        title: str = "Recommended Actions Distribution"
    ) -> go.Figure:
        """
        Create horizontal bar chart showing distribution of recommended actions.
        
        Args:
            recommendations_df: DataFrame with recommendations
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        if 'action_description' in recommendations_df.columns:
            action_col = 'action_description'
        else:
            action_col = 'recommended_action'
        
        action_counts = recommendations_df[action_col].value_counts().head(10)
        
        fig = go.Figure(data=[go.Bar(
            x=action_counts.values,
            y=action_counts.index,
            orientation='h',
            marker=dict(
                color=action_counts.values,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Count")
            ),
            hovertemplate='<b>%{y}</b><br>Customers: %{x}<extra></extra>'
        )])
        
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=18, color='#1E88E5')),
            xaxis_title="Number of Customers",
            yaxis_title="",
            height=max(400, len(action_counts) * 40),
            margin=dict(t=80, b=60, l=20, r=40),
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_churn_probability_histogram(
        recommendations_df: pd.DataFrame,
        title: str = "Churn Probability Distribution"
    ) -> go.Figure:
        """
        Create histogram showing distribution of churn probabilities.
        
        Args:
            recommendations_df: DataFrame with recommendations
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure(data=[go.Histogram(
            x=recommendations_df['churn_probability'],
            nbinsx=20,
            marker=dict(
                color='#1E88E5',
                line=dict(color='white', width=1)
            ),
            hovertemplate='Probability Range: %{x}<br>Count: %{y}<extra></extra>'
        )])
        
        # Add risk level lines
        fig.add_vline(x=0.3, line_dash="dash", line_color="yellow", 
                     annotation_text="Medium Risk", annotation_position="top")
        fig.add_vline(x=0.5, line_dash="dash", line_color="orange",
                     annotation_text="High Risk", annotation_position="top")
        fig.add_vline(x=0.7, line_dash="dash", line_color="red",
                     annotation_text="Critical Risk", annotation_position="top")
        
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=18, color='#1E88E5')),
            xaxis_title="Churn Probability",
            yaxis_title="Number of Customers",
            height=400,
            margin=dict(t=80, b=60, l=60, r=40),
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_priority_scatter(
        recommendations_df: pd.DataFrame,
        title: str = "Priority vs Churn Probability"
    ) -> go.Figure:
        """
        Create scatter plot showing priority vs churn probability.
        
        Args:
            recommendations_df: DataFrame with recommendations
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        fig = px.scatter(
            recommendations_df,
            x='churn_probability',
            y='priority',
            color='risk_level',
            size='confidence' if 'confidence' in recommendations_df.columns else None,
            hover_data=['customer_id'] if 'customer_id' in recommendations_df.columns else None,
            color_discrete_map={
                'Critical': '#D32F2F',
                'High': '#F57C00',
                'Medium': '#FBC02D',
                'Low': '#388E3C'
            },
            title=title
        )
        
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=18, color='#1E88E5')),
            xaxis_title="Churn Probability",
            yaxis_title="Priority Level",
            height=450,
            margin=dict(t=80, b=60, l=60, r=40)
        )
        
        fig.update_yaxis(autorange="reversed")  # Lower priority number = higher priority
        
        return fig
    
    @staticmethod
    def create_confidence_distribution_box(
        recommendations_df: pd.DataFrame,
        title: str = "Confidence Distribution by Risk Level"
    ) -> go.Figure:
        """
        Create box plot showing confidence distribution across risk levels.
        
        Args:
            recommendations_df: DataFrame with recommendations
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        if 'confidence' not in recommendations_df.columns:
            return go.Figure()
        
        fig = go.Figure()
        
        colors = {
            'Critical': '#D32F2F',
            'High': '#F57C00',
            'Medium': '#FBC02D',
            'Low': '#388E3C'
        }
        
        for risk_level in ['Critical', 'High', 'Medium', 'Low']:
            data = recommendations_df[recommendations_df['risk_level'] == risk_level]['confidence']
            if len(data) > 0:
                fig.add_trace(go.Box(
                    y=data,
                    name=risk_level,
                    marker_color=colors[risk_level],
                    boxmean='sd'
                ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=18, color='#1E88E5')),
            yaxis_title="Confidence Score",
            xaxis_title="Risk Level",
            height=400,
            margin=dict(t=80, b=60, l=60, r=40),
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_action_cost_analysis(
        recommendations_df: pd.DataFrame,
        title: str = "Cost vs Expected Retention Lift"
    ) -> go.Figure:
        """
        Create bubble chart showing cost vs retention lift for actions.
        
        Args:
            recommendations_df: DataFrame with recommendations
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        if 'estimated_cost' not in recommendations_df.columns or 'expected_retention_lift' not in recommendations_df.columns:
            return go.Figure()
        
        # Aggregate by action
        action_analysis = recommendations_df.groupby('recommended_action').agg({
            'estimated_cost': 'first',
            'expected_retention_lift': 'first',
            'customer_id': 'count'
        }).reset_index()
        action_analysis.rename(columns={'customer_id': 'customer_count'}, inplace=True)
        
        fig = px.scatter(
            action_analysis,
            x='estimated_cost',
            y='expected_retention_lift',
            size='customer_count',
            hover_data=['recommended_action'],
            text='recommended_action',
            title=title
        )
        
        fig.update_traces(
            textposition='top center',
            marker=dict(
                color='#1E88E5',
                line=dict(width=2, color='white')
            )
        )
        
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=18, color='#1E88E5')),
            xaxis_title="Estimated Cost ($)",
            yaxis_title="Expected Retention Lift",
            height=500,
            margin=dict(t=80, b=60, l=60, r=40)
        )
        
        return fig
    
    @staticmethod
    def create_summary_metrics_cards(
        recommendations_df: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Calculate summary metrics for display cards.
        
        Args:
            recommendations_df: DataFrame with recommendations
            
        Returns:
            Dictionary with metric values
        """
        metrics = {
            'total_customers': len(recommendations_df),
            'avg_churn_risk': recommendations_df['churn_probability'].mean(),
            'high_risk_count': len(recommendations_df[
                recommendations_df['risk_level'].isin(['Critical', 'High'])
            ]),
            'urgent_count': len(recommendations_df[recommendations_df['priority'] <= 2])
        }
        
        if 'confidence' in recommendations_df.columns:
            metrics['avg_confidence'] = recommendations_df['confidence'].mean()
        
        if 'estimated_cost' in recommendations_df.columns:
            metrics['total_cost'] = recommendations_df['estimated_cost'].sum()
            metrics['avg_cost_per_customer'] = recommendations_df['estimated_cost'].mean()
        
        if 'expected_retention_lift' in recommendations_df.columns:
            metrics['avg_expected_lift'] = recommendations_df['expected_retention_lift'].mean()
        
        return metrics
    
    @staticmethod
    def create_comprehensive_dashboard(
        recommendations_df: pd.DataFrame
    ) -> List[go.Figure]:
        """
        Create all visualization charts for comprehensive dashboard.
        
        Args:
            recommendations_df: DataFrame with recommendations
            
        Returns:
            List of plotly figures
        """
        charts = []
        
        # 1. Risk Distribution Pie
        charts.append(ChurnVisualizations.create_risk_distribution_pie(recommendations_df))
        
        # 2. Action Distribution Bar
        charts.append(ChurnVisualizations.create_action_distribution_bar(recommendations_df))
        
        # 3. Churn Probability Histogram
        charts.append(ChurnVisualizations.create_churn_probability_histogram(recommendations_df))
        
        # 4. Priority Scatter
        charts.append(ChurnVisualizations.create_priority_scatter(recommendations_df))
        
        # 5. Confidence Box Plot
        if 'confidence' in recommendations_df.columns:
            charts.append(ChurnVisualizations.create_confidence_distribution_box(recommendations_df))
        
        # 6. Cost Analysis (if available)
        if 'estimated_cost' in recommendations_df.columns:
            charts.append(ChurnVisualizations.create_action_cost_analysis(recommendations_df))
        
        return charts


if __name__ == "__main__":
    # Test visualizations with sample data
    print("Creating sample data for visualization testing...")
    
    np.random.seed(42)
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
        'estimated_cost': np.random.uniform(50, 500, n),
        'expected_retention_lift': np.random.uniform(0.1, 0.5, n)
    })
    
    # Create visualizations
    viz = ChurnVisualizations()
    
    print("\n✓ Creating visualizations...")
    charts = viz.create_comprehensive_dashboard(sample_data)
    print(f"✓ Created {len(charts)} interactive charts")
    
    # Calculate metrics
    metrics = viz.create_summary_metrics_cards(sample_data)
    print(f"\n✓ Summary Metrics:")
    for key, value in metrics.items():
        print(f"  • {key}: {value}")

