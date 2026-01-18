"""
Main Training Script for B-Decide AI
Trains the churn prediction model and saves it for production use
"""

import os
import sys
from models.train_model import train_and_save_model


def main():
    """
    Main training function.
    Trains XGBoost model on telco churn data and saves to models/ directory.
    """
    print("\n" + "="*70)
    print("B-DECIDE AI - MODEL TRAINING SCRIPT")
    print("="*70 + "\n")
    
    # Check if data file exists
    data_path = 'data/raw/telco_churn.csv'
    
    if not os.path.exists(data_path):
        print(" Error: Dataset not found!")
        print(f"\nPlease download the IBM Telco Customer Churn dataset and place it at:")
        print(f"  {data_path}")
        print("\nDataset available at:")
        print("  https://www.kaggle.com/blastchar/telco-customer-churn")
        print("\nOr create a sample dataset with the required columns:")
        print("  customerID, tenure, MonthlyCharges, TotalCharges, Churn, etc.")
        sys.exit(1)
    
    try:
        # Train model
        print(f" Loading data from: {data_path}")
        predictor, metrics = train_and_save_model(
            data_path=data_path,
            save_dir='models',
            test_size=0.2,
            random_state=42
        )
        
        # Print summary
        print("\n" + "="*70)
        print(" TRAINING COMPLETE!")
        print("="*70)
        print("\nModel Performance Metrics:")
        print(f"  • Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  • Precision: {metrics['precision']:.4f}")
        print(f"  • Recall:    {metrics['recall']:.4f}")
        print(f"  • F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  • ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print("\n Files saved:")
        print("  ✓ models/churn_model.pkl")
        print("  ✓ models/preprocessor.pkl")
        print("  ✓ models/feature_importance.csv")
        
        print("\n Next Steps:")
        print("  1. Start the backend:  python backend/main.py")
        print("  2. Start the frontend: streamlit run frontend/dashboard.py")
        print("  3. Or use Docker:      docker-compose up")
        
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


