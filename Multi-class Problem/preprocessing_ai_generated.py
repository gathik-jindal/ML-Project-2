
"""
Comprehensive Preprocessing Code for Travel Behavior Insights Dataset
Author: Generated for Kaggle Competition
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_data(train_path, test_path):
    """Load train and test datasets"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    return train_df, test_df


# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================

def perform_eda(df):
    """Perform basic exploratory data analysis"""
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)

    # Dataset Info
    print("\nDataset Info:")
    print(df.info())

    # Shape
    print(f"\nDataset Shape: {df.shape}")

    # First few rows
    print("\nFirst 5 rows:")
    print(df.head())

    # Statistical summary
    print("\nStatistical Summary:")
    print(df.describe())

    # Missing values
    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False))

    # Data types
    print("\nData Types:")
    print(df.dtypes.value_counts())

    # Duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates}")

    # Unique values for categorical columns
    print("\nUnique Values in Categorical Columns:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"{col}: {df[col].nunique()} unique values")

    return df


# ============================================================================
# 3. MISSING VALUE HANDLING
# ============================================================================

def handle_missing_values(df, strategy='auto'):
    """
    Handle missing values using different strategies

    Parameters:
    - strategy: 'auto', 'mean', 'median', 'mode', 'drop', or custom dict
    """
    print("\n" + "="*50)
    print("HANDLING MISSING VALUES")
    print("="*50)

    df_copy = df.copy()

    # Identify numerical and categorical columns
    numerical_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()

    if strategy == 'auto':
        # For numerical: use median
        if numerical_cols:
            num_imputer = SimpleImputer(strategy='median')
            df_copy[numerical_cols] = num_imputer.fit_transform(df_copy[numerical_cols])
            print(f"Imputed {len(numerical_cols)} numerical columns with median")

        # For categorical: use mode
        if categorical_cols:
            for col in categorical_cols:
                if df_copy[col].isnull().sum() > 0:
                    mode_value = df_copy[col].mode()[0] if len(df_copy[col].mode()) > 0 else 'Unknown'
                    df_copy[col].fillna(mode_value, inplace=True)
            print(f"Imputed {len(categorical_cols)} categorical columns with mode")

    elif strategy == 'mean':
        num_imputer = SimpleImputer(strategy='mean')
        df_copy[numerical_cols] = num_imputer.fit_transform(df_copy[numerical_cols])

    elif strategy == 'median':
        num_imputer = SimpleImputer(strategy='median')
        df_copy[numerical_cols] = num_imputer.fit_transform(df_copy[numerical_cols])

    elif strategy == 'drop':
        df_copy = df_copy.dropna()
        print(f"Dropped rows with missing values. New shape: {df_copy.shape}")

    print(f"\nMissing values after imputation: {df_copy.isnull().sum().sum()}")

    return df_copy


# ============================================================================
# 4. OUTLIER DETECTION AND HANDLING
# ============================================================================

def detect_outliers_iqr(df, columns=None, threshold=1.5):
    """Detect outliers using IQR method"""
    print("\n" + "="*50)
    print("OUTLIER DETECTION")
    print("="*50)

    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns

    outlier_indices = []

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outlier_indices.extend(outliers)

        print(f"{col}: {len(outliers)} outliers detected")

    return list(set(outlier_indices))


def handle_outliers(df, method='clip', columns=None, threshold=1.5):
    """
    Handle outliers using various methods

    Parameters:
    - method: 'clip', 'remove', 'log', 'cap'
    """
    df_copy = df.copy()

    if columns is None:
        columns = df_copy.select_dtypes(include=['int64', 'float64']).columns

    for col in columns:
        if method == 'clip':
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)

        elif method == 'log':
            if (df_copy[col] > 0).all():
                df_copy[col] = np.log1p(df_copy[col])

    print(f"Outliers handled using method: {method}")
    return df_copy


# ============================================================================
# 5. FEATURE ENGINEERING
# ============================================================================

def create_travel_features(df):
    """Create new features specific to travel behavior"""
    print("\n" + "="*50)
    print("FEATURE ENGINEERING")
    print("="*50)

    df_copy = df.copy()

    # Date-time features (if applicable)
    date_columns = df_copy.select_dtypes(include=['datetime64']).columns
    for col in date_columns:
        df_copy[f'{col}_year'] = df_copy[col].dt.year
        df_copy[f'{col}_month'] = df_copy[col].dt.month
        df_copy[f'{col}_day'] = df_copy[col].dt.day
        df_copy[f'{col}_dayofweek'] = df_copy[col].dt.dayofweek
        df_copy[f'{col}_quarter'] = df_copy[col].dt.quarter
        df_copy[f'{col}_is_weekend'] = df_copy[col].dt.dayofweek.isin([5, 6]).astype(int)
        print(f"Created datetime features from {col}")

    # Example: Duration features (if start and end dates exist)
    # if 'start_date' in df_copy.columns and 'end_date' in df_copy.columns:
    #     df_copy['trip_duration'] = (df_copy['end_date'] - df_copy['start_date']).dt.days

    # Example: Age groups
    # if 'age' in df_copy.columns:
    #     df_copy['age_group'] = pd.cut(df_copy['age'], 
    #                                    bins=[0, 18, 30, 45, 60, 100], 
    #                                    labels=['<18', '18-30', '30-45', '45-60', '60+'])

    # Example: Budget categories
    # if 'budget' in df_copy.columns:
    #     df_copy['budget_category'] = pd.cut(df_copy['budget'], 
    #                                          bins=[0, 1000, 5000, 10000, np.inf],
    #                                          labels=['Low', 'Medium', 'High', 'Luxury'])

    print("Feature engineering completed")
    return df_copy


# ============================================================================
# 6. ENCODING CATEGORICAL VARIABLES
# ============================================================================

def encode_categorical(df, method='label', target_col=None):
    """
    Encode categorical variables

    Parameters:
    - method: 'label', 'onehot', 'target', 'frequency'
    """
    print("\n" + "="*50)
    print("ENCODING CATEGORICAL VARIABLES")
    print("="*50)

    df_copy = df.copy()
    categorical_cols = df_copy.select_dtypes(include=['object']).columns.tolist()

    # Remove target column if present
    if target_col and target_col in categorical_cols:
        categorical_cols.remove(target_col)

    encoders = {}

    if method == 'label':
        for col in categorical_cols:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
            encoders[col] = le
            print(f"Label encoded: {col}")

    elif method == 'onehot':
        df_copy = pd.get_dummies(df_copy, columns=categorical_cols, drop_first=True)
        print(f"One-hot encoded {len(categorical_cols)} columns")

    elif method == 'frequency':
        for col in categorical_cols:
            freq_map = df_copy[col].value_counts(normalize=True).to_dict()
            df_copy[f'{col}_freq'] = df_copy[col].map(freq_map)
            print(f"Frequency encoded: {col}")

    print(f"\nShape after encoding: {df_copy.shape}")
    return df_copy, encoders


# ============================================================================
# 7. FEATURE SCALING
# ============================================================================

def scale_features(df, method='standard', columns=None):
    """
    Scale numerical features

    Parameters:
    - method: 'standard', 'minmax', 'robust'
    """
    print("\n" + "="*50)
    print("FEATURE SCALING")
    print("="*50)

    df_copy = df.copy()

    if columns is None:
        columns = df_copy.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()

    df_copy[columns] = scaler.fit_transform(df_copy[columns])

    print(f"Scaled {len(columns)} columns using {method} scaling")
    return df_copy, scaler


# ============================================================================
# 8. FEATURE SELECTION
# ============================================================================

def select_features(df, target_col, method='correlation', threshold=0.1):
    """Select important features"""
    print("\n" + "="*50)
    print("FEATURE SELECTION")
    print("="*50)

    df_copy = df.copy()

    if method == 'correlation':
        numerical_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns
        if target_col in numerical_cols:
            correlations = df_copy[numerical_cols].corr()[target_col].abs()
            selected_features = correlations[correlations > threshold].index.tolist()
            print(f"Selected {len(selected_features)} features with correlation > {threshold}")
            return selected_features

    return df_copy.columns.tolist()


# ============================================================================
# 9. DATA VALIDATION
# ============================================================================

def validate_data(df):
    """Perform data validation checks"""
    print("\n" + "="*50)
    print("DATA VALIDATION")
    print("="*50)

    # Check for infinite values
    inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
    if inf_counts.sum() > 0:
        print(f"Warning: Found {inf_counts.sum()} infinite values")

    # Check for negative values where they shouldn't exist
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        if (df[col] < 0).any():
            print(f"Warning: {col} contains negative values")

    # Check value ranges
    print("\nValue Ranges:")
    print(df.describe())

    print("\nData validation completed")


# ============================================================================
# 10. MAIN PREPROCESSING PIPELINE
# ============================================================================

def preprocess_pipeline(train_df, test_df, target_col=None, config=None):
    """
    Complete preprocessing pipeline

    Parameters:
    - train_df: Training dataset
    - test_df: Test dataset
    - target_col: Target column name
    - config: Dictionary with preprocessing configurations
    """
    print("\n" + "="*80)
    print(" "*20 + "PREPROCESSING PIPELINE")
    print("="*80)

    # Default configuration
    if config is None:
        config = {
            'missing_strategy': 'auto',
            'outlier_method': 'clip',
            'encoding_method': 'label',
            'scaling_method': 'standard',
            'feature_engineering': True
        }

    # Separate target from features if present in train
    if target_col and target_col in train_df.columns:
        y_train = train_df[target_col]
        X_train = train_df.drop(columns=[target_col])
    else:
        X_train = train_df.copy()
        y_train = None

    X_test = test_df.copy()

    # 1. EDA
    print("\n[Step 1/7] Exploratory Data Analysis")
    perform_eda(X_train)

    # 2. Handle missing values
    print("\n[Step 2/7] Handling Missing Values")
    X_train = handle_missing_values(X_train, strategy=config['missing_strategy'])
    X_test = handle_missing_values(X_test, strategy=config['missing_strategy'])

    # 3. Feature engineering
    if config['feature_engineering']:
        print("\n[Step 3/7] Feature Engineering")
        X_train = create_travel_features(X_train)
        X_test = create_travel_features(X_test)

    # 4. Handle outliers
    print("\n[Step 4/7] Handling Outliers")
    outlier_indices = detect_outliers_iqr(X_train)
    X_train = handle_outliers(X_train, method=config['outlier_method'])

    # 5. Encode categorical variables
    print("\n[Step 5/7] Encoding Categorical Variables")
    X_train, encoders = encode_categorical(X_train, method=config['encoding_method'])
    X_test, _ = encode_categorical(X_test, method=config['encoding_method'])

    # Align columns between train and test
    # Get common columns
    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    # 6. Feature scaling
    print("\n[Step 6/7] Feature Scaling")
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    X_train, scaler = scale_features(X_train, method=config['scaling_method'], columns=numerical_cols)
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # 7. Validation
    print("\n[Step 7/7] Data Validation")
    validate_data(X_train)

    print("\n" + "="*80)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nFinal Training Shape: {X_train.shape}")
    print(f"Final Test Shape: {X_test.shape}")

    return X_train, X_test, y_train, scaler, encoders


# ============================================================================
# 11. USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    """
    # Load data
    train_df, test_df = load_data('train.csv', 'test.csv')

    # Configure preprocessing
    config = {
        'missing_strategy': 'auto',
        'outlier_method': 'clip',
        'encoding_method': 'label',
        'scaling_method': 'standard',
        'feature_engineering': True
    }

    # Run preprocessing pipeline
    X_train, X_test, y_train, scaler, encoders = preprocess_pipeline(
        train_df, 
        test_df, 
        target_col='target_column_name',  # Replace with actual target column
        config=config
    )

    # Save preprocessed data
    X_train.to_csv('X_train_preprocessed.csv', index=False)
    X_test.to_csv('X_test_preprocessed.csv', index=False)
    if y_train is not None:
        y_train.to_csv('y_train.csv', index=False)

    print("\nPreprocessed data saved successfully!")
    """
    pass
