"""
Data Preprocessing Module for B-Decide AI
Handles data loading, cleaning, encoding, and scaling for churn prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional
import pickle
import os


class ChurnDataPreprocessor:
    """
    Preprocessor for customer churn data.
    Handles missing values, categorical encoding, feature scaling, and train-test split.
    """
    
    def __init__(self):
        """Initialize the preprocessor with encoders and scaler."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.categorical_columns = []
        self.numerical_columns = []
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load CSV data from file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame containing the loaded data
        """
        try:
            df = pd.read_csv(file_path)
            print(f"✓ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        # Replace spaces with NaN
        df.replace(' ', np.nan, inplace=True)
        
        # For numerical columns, fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        print(f"✓ Missing values handled")
        return df
    
    def encode_categorical_variables(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables using Label Encoding.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit encoders (True for training, False for prediction)
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df = df.copy()
        
        # Identify categorical columns (excluding the target if present)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column from encoding if present
        if 'Churn' in categorical_cols:
            categorical_cols.remove('Churn')
        
        self.categorical_columns = categorical_cols
        
        for col in categorical_cols:
            if fit:
                # Create and fit new encoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                # Use existing encoder
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unknown categories
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    df[col] = le.transform(df[col].astype(str))
        
        print(f"✓ Categorical variables encoded: {len(categorical_cols)} columns")
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True, 
                       target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit scaler (True for training, False for prediction)
            target_col: Name of target column to exclude from scaling
            
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        # Get columns to scale (exclude target and customer ID if present)
        exclude_cols = []
        if target_col and target_col in df.columns:
            exclude_cols.append(target_col)
        if 'customerID' in df.columns:
            exclude_cols.append('customerID')
        
        cols_to_scale = [col for col in df.columns if col not in exclude_cols]
        self.numerical_columns = cols_to_scale
        
        if fit:
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        else:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        
        print(f"✓ Features scaled: {len(cols_to_scale)} columns")
        return df
    
    def prepare_features_target(self, df: pd.DataFrame, 
                               target_col: str = 'Churn') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Handle target encoding if it's categorical
        if df[target_col].dtype == 'object':
            df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})
        
        # Drop customerID if present
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        self.feature_columns = X.columns.tolist()
        
        print(f"✓ Features and target prepared: {X.shape[1]} features")
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, 
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"✓ Data split: Train={len(X_train)}, Test={len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, file_path: str, 
                           target_col: str = 'Churn',
                           test_size: float = 0.2,
                           random_state: int = 42) -> Tuple:
        """
        Complete preprocessing pipeline for training data.
        
        Args:
            file_path: Path to the CSV file
            target_col: Name of the target column
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("\n=== Starting Data Preprocessing Pipeline ===\n")
        
        # Load data
        df = self.load_data(file_path)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Encode categorical variables
        df = self.encode_categorical_variables(df, fit=True)
        
        # Prepare features and target
        X, y = self.prepare_features_target(df, target_col)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size, random_state)
        
        # Scale features
        X_train = self.scale_features(pd.DataFrame(X_train, columns=self.feature_columns), 
                                      fit=True, target_col=None)
        X_test = self.scale_features(pd.DataFrame(X_test, columns=self.feature_columns), 
                                     fit=False, target_col=None)
        
        print("\n=== Preprocessing Complete ===\n")
        return X_train, X_test, y_train, y_test
    
    def preprocess_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess new data for prediction using fitted transformers.
        
        Args:
            df: DataFrame containing new data
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        df = df.copy()
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Encode categorical variables
        df = self.encode_categorical_variables(df, fit=False)
        
        # Drop customerID if present
        if 'customerID' in df.columns:
            customer_ids = df['customerID']
            df = df.drop('customerID', axis=1)
        
        # Drop target if present
        if 'Churn' in df.columns:
            df = df.drop('Churn', axis=1)
        
        # Ensure same columns as training
        if self.feature_columns:
            missing_cols = set(self.feature_columns) - set(df.columns)
            for col in missing_cols:
                df[col] = 0
            df = df[self.feature_columns]
        
        # Scale features
        df = self.scale_features(df, fit=False)
        
        return df
    
    def save_preprocessor(self, save_dir: str = 'models'):
        """
        Save the preprocessor (scaler and encoders) to disk.
        
        Args:
            save_dir: Directory to save the preprocessor
        """
        os.makedirs(save_dir, exist_ok=True)
        
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns
        }
        
        with open(os.path.join(save_dir, 'preprocessor.pkl'), 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        print(f"✓ Preprocessor saved to {save_dir}/preprocessor.pkl")
    
    def load_preprocessor(self, save_dir: str = 'models'):
        """
        Load the preprocessor from disk.
        
        Args:
            save_dir: Directory containing the saved preprocessor
        """
        with open(os.path.join(save_dir, 'preprocessor.pkl'), 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.scaler = preprocessor_data['scaler']
        self.label_encoders = preprocessor_data['label_encoders']
        self.feature_columns = preprocessor_data['feature_columns']
        self.categorical_columns = preprocessor_data['categorical_columns']
        self.numerical_columns = preprocessor_data['numerical_columns']
        
        print(f"✓ Preprocessor loaded from {save_dir}/preprocessor.pkl")


if __name__ == "__main__":
    # Example usage
    preprocessor = ChurnDataPreprocessor()
    
    # For training
    # X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
    #     'data/raw/telco_churn.csv'
    # )
    # preprocessor.save_preprocessor()
    
    print("Data Preprocessor module ready!")


