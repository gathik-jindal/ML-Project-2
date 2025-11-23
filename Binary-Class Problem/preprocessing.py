"""
Financial Risk Profiling - Comprehensive Preprocessing Script
This script handles data loading, cleaning, feature engineering, and preparation
for machine learning models in financial risk assessment.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_data(train_path, test_path=None):
    """
    Load training and test datasets.

    Parameters:
    -----------
    train_path : str
        Path to training CSV file
    test_path : str, optional
        Path to test CSV file

    Returns:
    --------
    tuple : (train_df, test_df)
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path else None

    print(f"Training data shape: {train_df.shape}")
    if test_df is not None:
        print(f"Test data shape: {test_df.shape}")

    return train_df, test_df


# ============================================================================
# 2. EXPLORATORY ANALYSIS
# ============================================================================

def analyze_missing_values(df, name="Dataset"):
    """
    Analyze and visualize missing values.
    """
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100

    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Percentage': missing_percent
    }).sort_values('Missing_Count', ascending=False)

    print(f"\n=== Missing Values in {name} ===")
    print(missing_df[missing_df['Missing_Count'] > 0])

    return missing_df


def identify_dtypes(df):
    """
    Identify numerical and categorical columns.
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    print(f"\nNumerical columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"Datetime columns ({len(datetime_cols)}): {datetime_cols}")

    return numerical_cols, categorical_cols, datetime_cols


def detect_outliers_iqr(df, numerical_cols, threshold=1.5):
    """
    Detect outliers using Interquartile Range (IQR) method.
    """
    outlier_indices = {}

    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        if len(outliers) > 0:
            outlier_indices[col] = len(outliers)

    print(f"\n=== Outliers Detected (IQR method) ===")
    for col, count in sorted(outlier_indices.items(), key=lambda x: x[1], reverse=True):
        print(f"{col}: {count} outliers")

    return outlier_indices


# ============================================================================
# 3. MISSING VALUE IMPUTATION
# ============================================================================

def impute_missing_values(df, numerical_cols, categorical_cols, 
                          numerical_method='median', categorical_method='mode'):
    """
    Impute missing values using specified methods.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    numerical_cols : list
        List of numerical column names
    categorical_cols : list
        List of categorical column names
    numerical_method : str
        Method for numerical imputation ('mean', 'median', 'knn')
    categorical_method : str
        Method for categorical imputation ('mode', 'constant')

    Returns:
    --------
    pd.DataFrame : DataFrame with imputed values
    """
    df_imputed = df.copy()

    # Numerical columns
    if numerical_method == 'median':
        imputer_num = SimpleImputer(strategy='median')
    elif numerical_method == 'mean':
        imputer_num = SimpleImputer(strategy='mean')
    elif numerical_method == 'knn':
        imputer_num = KNNImputer(n_neighbors=5)
    else:
        imputer_num = SimpleImputer(strategy='median')

    if numerical_cols:
        df_imputed[numerical_cols] = imputer_num.fit_transform(df_imputed[numerical_cols])

    # Categorical columns
    if categorical_method == 'mode':
        imputer_cat = SimpleImputer(strategy='most_frequent')
    else:
        imputer_cat = SimpleImputer(strategy='constant', fill_value='Unknown')

    if categorical_cols:
        df_imputed[categorical_cols] = imputer_cat.fit_transform(df_imputed[categorical_cols])

    print(f"\nMissing values imputed using {numerical_method} for numerical and {categorical_method} for categorical")

    return df_imputed


# ============================================================================
# 4. OUTLIER HANDLING
# ============================================================================

def handle_outliers(df, numerical_cols, method='clip', threshold=1.5):
    """
    Handle outliers using specified method.

    Parameters:
    -----------
    method : str
        'clip' : Clip values to bounds
        'remove' : Remove rows with outliers
        'transform' : Log/sqrt transformation
    """
    df_cleaned = df.copy()

    if method == 'clip':
        for col in numerical_cols:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)

    elif method == 'remove':
        initial_size = len(df_cleaned)
        for col in numerical_cols:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & 
                                   (df_cleaned[col] <= upper_bound)]

        print(f"Rows removed: {initial_size - len(df_cleaned)} ({100*(initial_size - len(df_cleaned))/initial_size:.2f}%)")

    elif method == 'transform':
        for col in numerical_cols:
            if (df_cleaned[col] > 0).all():
                df_cleaned[col] = np.log1p(df_cleaned[col])

    return df_cleaned


# ============================================================================
# 5. FEATURE ENGINEERING
# ============================================================================

def create_financial_features(df):
    """
    Create domain-specific financial features.
    Customize based on your actual dataset columns.
    """
    df_engineered = df.copy()

    # Example financial ratios (adapt column names to your dataset)
    financial_feature_examples = {
        # 'debt_to_income': lambda x: x.get('total_debt', 0) / (x.get('annual_income', 1) + 1e-8),
        # 'savings_rate': lambda x: x.get('savings', 0) / (x.get('annual_income', 1) + 1e-8),
        # 'credit_utilization': lambda x: x.get('credit_used', 0) / (x.get('credit_limit', 1) + 1e-8),
        # 'loan_to_income': lambda x: x.get('total_loans', 0) / (x.get('annual_income', 1) + 1e-8),
        # 'age_income_interaction': lambda x: x.get('age', 0) * x.get('annual_income', 0),
    }

    # Add your features here based on your dataset
    # for feature_name, feature_func in financial_feature_examples.items():
    #     df_engineered[feature_name] = df_engineered.apply(feature_func, axis=1)

    print("\nFinancial features engineered (customize based on your columns)")
    return df_engineered


def create_polynomial_features(df, numerical_cols, degree=2, include_interaction=False):
    """
    Create polynomial features.
    """
    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree=degree, include_bias=False, 
                             interaction_only=include_interaction)
    poly_features = poly.fit_transform(df[numerical_cols])
    poly_feature_names = poly.get_feature_names_out(numerical_cols)

    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)

    print(f"Polynomial features created: {poly_df.shape[1]} new features")

    return pd.concat([df, poly_df], axis=1)


# ============================================================================
# 6. ENCODING
# ============================================================================

def encode_categorical(df, categorical_cols, method='onehot', rare_threshold=0.01):
    """
    Encode categorical variables.

    Parameters:
    -----------
    method : str
        'onehot' : One-hot encoding
        'label' : Label encoding
        'target' : Target encoding (requires target variable)
    """
    df_encoded = df.copy()

    if method == 'onehot':
        # Handle rare categories
        for col in categorical_cols:
            # Calculate category frequencies
            category_freq = df_encoded[col].value_counts(normalize=True)
            rare_categories = category_freq[category_freq < rare_threshold].index

            # Group rare categories
            df_encoded[col] = df_encoded[col].apply(
                lambda x: 'Rare' if x in rare_categories else x
            )

        # One-hot encode
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, 
                                   drop_first=True, dummy_na=False)
        print(f"\nOne-hot encoding applied. New shape: {df_encoded.shape}")

    elif method == 'label':
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le

        print(f"\nLabel encoding applied to {len(categorical_cols)} columns")

    return df_encoded


# ============================================================================
# 7. SCALING
# ============================================================================

def scale_features(df, numerical_cols, method='standard', fit_scaler=None):
    """
    Scale numerical features.

    Parameters:
    -----------
    method : str
        'standard' : StandardScaler (mean=0, std=1)
        'minmax' : MinMaxScaler (0-1 range)
    fit_scaler : scaler object, optional
        Pre-fitted scaler for test data

    Returns:
    --------
    tuple : (scaled_df, scaler)
    """
    df_scaled = df.copy()

    if method == 'standard':
        if fit_scaler is None:
            scaler = StandardScaler()
            df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
        else:
            scaler = fit_scaler
            df_scaled[numerical_cols] = scaler.transform(df_scaled[numerical_cols])

    elif method == 'minmax':
        if fit_scaler is None:
            scaler = MinMaxScaler()
            df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
        else:
            scaler = fit_scaler
            df_scaled[numerical_cols] = scaler.transform(df_scaled[numerical_cols])

    print(f"\n{method.upper()} scaling applied to {len(numerical_cols)} features")

    return df_scaled, scaler


# ============================================================================
# 8. MAIN PREPROCESSING PIPELINE
# ============================================================================

def preprocess_pipeline(train_path, test_path=None, target_col='target'):
    """
    Complete preprocessing pipeline.

    Parameters:
    -----------
    train_path : str
        Path to training data
    test_path : str
        Path to test data
    target_col : str
        Name of target column

    Returns:
    --------
    dict : Preprocessed datasets and objects
    """

    print("="*70)
    print("FINANCIAL RISK PROFILING - PREPROCESSING PIPELINE")
    print("="*70)

    # 1. Load data
    print("\n[STEP 1] Loading data...")
    train_df, test_df = load_data(train_path, test_path)

    # 2. Identify data types
    print("\n[STEP 2] Identifying data types...")
    numerical_cols, categorical_cols, datetime_cols = identify_dtypes(train_df)

    # 3. Analyze missing values
    print("\n[STEP 3] Analyzing missing values...")
    analyze_missing_values(train_df, "Training Data")
    if test_df is not None:
        analyze_missing_values(test_df, "Test Data")

    # 4. Detect outliers
    print("\n[STEP 4] Detecting outliers...")
    detect_outliers_iqr(train_df, numerical_cols)

    # 5. Separate target variable
    print("\n[STEP 5] Separating target variable...")
    if target_col in train_df.columns:
        y_train = train_df[target_col].copy()
        X_train = train_df.drop(columns=[target_col])
        print(f"Target variable shape: {y_train.shape}")
        print(f"Features shape: {X_train.shape}")
    else:
        X_train = train_df.copy()
        y_train = None
        print("No target column found in training data")

    # Update numerical and categorical columns (excluding target)
    numerical_cols = [col for col in numerical_cols if col != target_col]
    categorical_cols = [col for col in categorical_cols if col != target_col]

    # 6. Impute missing values
    print("\n[STEP 6] Imputing missing values...")
    X_train = impute_missing_values(X_train, numerical_cols, categorical_cols,
                                    numerical_method='median',
                                    categorical_method='mode')

    if test_df is not None:
        test_cols = [col for col in test_df.columns if col != target_col]
        test_numerical = [col for col in numerical_cols if col in test_cols]
        test_categorical = [col for col in categorical_cols if col in test_cols]
        test_df = impute_missing_values(test_df, test_numerical, test_categorical,
                                        numerical_method='median',
                                        categorical_method='mode')

    # 7. Handle outliers
    print("\n[STEP 7] Handling outliers...")
    X_train = handle_outliers(X_train, numerical_cols, method='clip', threshold=1.5)
    if test_df is not None:
        test_df = handle_outliers(test_df, numerical_cols, method='clip', threshold=1.5)

    # 8. Feature engineering
    print("\n[STEP 8] Engineering features...")
    X_train = create_financial_features(X_train)
    if test_df is not None:
        test_df = create_financial_features(test_df)

    # 9. Encode categorical variables
    print("\n[STEP 9] Encoding categorical variables...")
    X_train = encode_categorical(X_train, categorical_cols, method='onehot')

    # Update numerical cols after encoding
    all_numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    if test_df is not None:
        test_df = encode_categorical(test_df, categorical_cols, method='onehot')

    # 10. Scale features
    print("\n[STEP 10] Scaling features...")
    X_train_scaled, scaler = scale_features(X_train, all_numerical_cols, method='standard')

    if test_df is not None:
        test_df_scaled, _ = scale_features(test_df, all_numerical_cols, 
                                          method='standard', fit_scaler=scaler)
    else:
        test_df_scaled = None

    # Summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETED")
    print("="*70)
    print(f"Final training data shape: {X_train_scaled.shape}")
    if test_df_scaled is not None:
        print(f"Final test data shape: {test_df_scaled.shape}")
    if y_train is not None:
        print(f"Target variable shape: {y_train.shape}")

    return {
        'X_train': X_train_scaled,
        'y_train': y_train,
        'X_test': test_df_scaled,
        'scaler': scaler,
        'numerical_cols': all_numerical_cols,
        'categorical_cols': categorical_cols,
        'feature_names': X_train_scaled.columns.tolist()
    }


# ============================================================================
# 9. USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage (uncomment and modify paths)

    # preprocessed_data = preprocess_pipeline(
    #     train_path='train.csv',
    #     test_path='test.csv',
    #     target_col='financial_risk'  # Modify based on your target column name
    # )

    # # Save preprocessed data
    # preprocessed_data['X_train'].to_csv('X_train_preprocessed.csv', index=False)
    # preprocessed_data['X_test'].to_csv('X_test_preprocessed.csv', index=False)
    # if preprocessed_data['y_train'] is not None:
    #     pd.DataFrame(preprocessed_data['y_train']).to_csv('y_train.csv', index=False)

    print("Preprocessing script ready! Uncomment the main block and add your file paths.")
