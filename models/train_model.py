"""
ML Model Training Module for B-Decide AI
XGBoost-based churn prediction with comprehensive metrics
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    roc_auc_score, f1_score, confusion_matrix, classification_report
)
import pickle
import os
import sys
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessor import ChurnDataPreprocessor


class ChurnPredictor:
    """
    XGBoost-based churn prediction model with training and evaluation capabilities.
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize the churn predictor.
        
        Args:
            params: XGBoost hyperparameters (optional, uses defaults if None)
        """
        # Default XGBoost parameters optimized for churn prediction
        self.default_params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'scale_pos_weight': 1,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        self.params = params if params else self.default_params
        self.model = None
        self.feature_importance = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              verbose: bool = True) -> 'ChurnPredictor':
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            verbose: Whether to print training progress
            
        Returns:
            self (for method chaining)
        """
        print("\n=== Training XGBoost Churn Prediction Model ===\n")
        
        # Initialize model
        self.model = xgb.XGBClassifier(**self.params)
        
        # Prepare evaluation set if validation data provided
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=verbose
        )
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n✓ Model training complete!")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict churn labels (0 or 1).
        
        Args:
            X: Features DataFrame
            
        Returns:
            Array of predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict churn probabilities.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Array of predicted probabilities for the positive class (churn)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance with comprehensive metrics.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print("\n=== Model Evaluation ===\n")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Print metrics
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}")
        print(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get top N most important features.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.feature_importance.head(top_n)
    
    def plot_feature_importance(self, top_n: int = 15, save_path: Optional[str] = None):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to plot
            save_path: Path to save the plot (optional)
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance', fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Feature importance plot saved to {save_path}")
        else:
            plt.show()
    
    def save_model(self, save_dir: str = 'models'):
        """
        Save the trained model to disk.
        
        Args:
            save_dir: Directory to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, 'churn_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save feature importance
        if self.feature_importance is not None:
            importance_path = os.path.join(save_dir, 'feature_importance.csv')
            self.feature_importance.to_csv(importance_path, index=False)
        
        print(f"✓ Model saved to {model_path}")
    
    def load_model(self, save_dir: str = 'models'):
        """
        Load a trained model from disk.
        
        Args:
            save_dir: Directory containing the saved model
        """
        model_path = os.path.join(save_dir, 'churn_model.pkl')
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load feature importance if available
        importance_path = os.path.join(save_dir, 'feature_importance.csv')
        if os.path.exists(importance_path):
            self.feature_importance = pd.read_csv(importance_path)
        
        print(f"✓ Model loaded from {model_path}")


def train_and_save_model(data_path: str, 
                         save_dir: str = 'models',
                         test_size: float = 0.2,
                         random_state: int = 42) -> Tuple[ChurnPredictor, Dict[str, float]]:
    """
    Complete pipeline to train and save a churn prediction model.
    
    Args:
        data_path: Path to the raw data CSV file
        save_dir: Directory to save the model and preprocessor
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (trained predictor, evaluation metrics)
    """
    print("="*60)
    print("B-DECIDE AI - CHURN PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Step 1: Preprocess data
    preprocessor = ChurnDataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        data_path, test_size=test_size, random_state=random_state
    )
    
    # Save preprocessor
    preprocessor.save_preprocessor(save_dir)
    
    # Step 2: Train model
    predictor = ChurnPredictor()
    predictor.train(X_train, y_train, X_test, y_test)
    
    # Step 3: Evaluate model
    metrics = predictor.evaluate(X_test, y_test)
    
    # Step 4: Display feature importance
    print("\n=== Top 10 Most Important Features ===\n")
    print(predictor.get_feature_importance(top_n=10))
    
    # Step 5: Save model
    predictor.save_model(save_dir)
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE!")
    print("="*60)
    
    return predictor, metrics


if __name__ == "__main__":
    # Example usage
    # Uncomment the following lines to train a model
    # Make sure to place your dataset at data/raw/telco_churn.csv
    
    # predictor, metrics = train_and_save_model(
    #     data_path='data/raw/telco_churn.csv',
    #     save_dir='models'
    # )
    
    print("\nML Training module ready!")
    print("\nTo train a model, use:")
    print("  python models/train_model.py")
    print("\nMake sure your dataset is at: data/raw/telco_churn.csv")

