
# Travel Behavior Insights - Preprocessing Notebook
# Kaggle Competition: https://www.kaggle.com/competitions/travel-behavior-insights/data

## Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
```

## 1. Load Data

```python
# Load training and test data
train_df = pd.read_csv('/kaggle/input/travel-behavior-insights/train.csv')
test_df = pd.read_csv('/kaggle/input/travel-behavior-insights/test.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Display first few rows
train_df.head()
```

## 2. Initial Data Exploration

```python
# Dataset information
print("=" * 80)
print("DATASET INFORMATION")
print("=" * 80)
train_df.info()

print("\n" + "=" * 80)
print("STATISTICAL SUMMARY")
print("=" * 80)
print(train_df.describe())

print("\n" + "=" * 80)
print("DATA TYPES")
print("=" * 80)
print(train_df.dtypes)
```

## 3. Missing Values Analysis

```python
# Check missing values
def analyze_missing_values(df, dataset_name='Dataset'):
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Percentage': missing_pct.values
    })

    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

    print(f"\n{dataset_name} - Missing Values Summary:")
    print(missing_df)

    # Visualize missing values
    if len(missing_df) > 0:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=missing_df, x='Column', y='Percentage', palette='viridis')
        plt.xticks(rotation=45, ha='right')
        plt.title(f'{dataset_name} - Missing Values Percentage')
        plt.ylabel('Percentage (%)')
        plt.tight_layout()
        plt.show()

    return missing_df

# Analyze missing values in train and test
train_missing = analyze_missing_values(train_df, 'Training Data')
test_missing = analyze_missing_values(test_df, 'Test Data')
```

## 4. Handle Missing Values

```python
def impute_missing_values(train, test):
    """Impute missing values for train and test datasets"""
    train_copy = train.copy()
    test_copy = test.copy()

    # Identify numerical and categorical columns
    numerical_cols = train_copy.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = train_copy.select_dtypes(include=['object', 'category']).columns.tolist()

    # Remove ID columns if present
    id_cols = [col for col in numerical_cols if 'id' in col.lower() or 'ID' in col]
    numerical_cols = [col for col in numerical_cols if col not in id_cols]

    # Impute numerical columns with median
    if numerical_cols:
        num_imputer = SimpleImputer(strategy='median')
        train_copy[numerical_cols] = num_imputer.fit_transform(train_copy[numerical_cols])
        test_copy[numerical_cols] = num_imputer.transform(test_copy[numerical_cols])
        print(f"Imputed {len(numerical_cols)} numerical columns with median")

    # Impute categorical columns with mode
    if categorical_cols:
        for col in categorical_cols:
            if train_copy[col].isnull().sum() > 0:
                mode_value = train_copy[col].mode()[0] if len(train_copy[col].mode()) > 0 else 'Unknown'
                train_copy[col].fillna(mode_value, inplace=True)
                test_copy[col].fillna(mode_value, inplace=True)
        print(f"Imputed {len(categorical_cols)} categorical columns with mode")

    print(f"\nMissing values after imputation:")
    print(f"Train: {train_copy.isnull().sum().sum()}")
    print(f"Test: {test_copy.isnull().sum().sum()}")

    return train_copy, test_copy

# Apply imputation
train_df, test_df = impute_missing_values(train_df, test_df)
```

## 5. Duplicate Detection

```python
# Check for duplicates
train_duplicates = train_df.duplicated().sum()
test_duplicates = test_df.duplicated().sum()

print(f"Duplicate rows in training data: {train_duplicates}")
print(f"Duplicate rows in test data: {test_duplicates}")

# Remove duplicates if any
if train_duplicates > 0:
    train_df = train_df.drop_duplicates()
    print(f"Removed {train_duplicates} duplicate rows from training data")
```

## 6. Outlier Detection and Visualization

```python
def detect_and_visualize_outliers(df, columns=None, n_std=3):
    """Detect outliers using standard deviation method"""
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # Remove ID columns
        columns = [col for col in columns if 'id' not in col.lower()]

    outlier_summary = []

    fig, axes = plt.subplots(len(columns), 2, figsize=(15, 5*len(columns)))
    if len(columns) == 1:
        axes = axes.reshape(1, -1)

    for idx, col in enumerate(columns):
        # Calculate outliers using IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_pct = (len(outliers) / len(df)) * 100

        outlier_summary.append({
            'Column': col,
            'Outlier_Count': len(outliers),
            'Percentage': outlier_pct,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound
        })

        # Box plot
        axes[idx, 0].boxplot(df[col].dropna())
        axes[idx, 0].set_title(f'{col} - Box Plot')
        axes[idx, 0].set_ylabel(col)

        # Histogram
        axes[idx, 1].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[idx, 1].axvline(lower_bound, color='r', linestyle='--', label='Lower Bound')
        axes[idx, 1].axvline(upper_bound, color='r', linestyle='--', label='Upper Bound')
        axes[idx, 1].set_title(f'{col} - Distribution')
        axes[idx, 1].set_xlabel(col)
        axes[idx, 1].legend()

    plt.tight_layout()
    plt.show()

    outlier_df = pd.DataFrame(outlier_summary)
    print("\nOutlier Summary:")
    print(outlier_df)

    return outlier_df

# Detect outliers
numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols = [col for col in numerical_cols if 'id' not in col.lower()][:5]  # Limit to first 5 for visualization

if numerical_cols:
    outlier_summary = detect_and_visualize_outliers(train_df, numerical_cols)
```

## 7. Handle Outliers

```python
def handle_outliers_iqr(train, test, columns=None, method='clip'):
    """
    Handle outliers using IQR method

    Parameters:
    - method: 'clip' (cap at bounds) or 'remove' (remove outliers)
    """
    train_copy = train.copy()
    test_copy = test.copy()

    if columns is None:
        columns = train_copy.select_dtypes(include=['int64', 'float64']).columns.tolist()
        columns = [col for col in columns if 'id' not in col.lower()]

    for col in columns:
        Q1 = train_copy[col].quantile(0.25)
        Q3 = train_copy[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if method == 'clip':
            train_copy[col] = train_copy[col].clip(lower=lower_bound, upper=upper_bound)
            test_copy[col] = test_copy[col].clip(lower=lower_bound, upper=upper_bound)

        print(f"{col}: Clipped to [{lower_bound:.2f}, {upper_bound:.2f}]")

    return train_copy, test_copy

# Apply outlier handling (optional - comment out if you don't want to handle outliers)
# train_df, test_df = handle_outliers_iqr(train_df, test_df, method='clip')
```

## 8. Feature Engineering

```python
def engineer_features(df):
    """Create new features for travel behavior data"""
    df_copy = df.copy()

    # Example 1: Convert date columns to datetime and extract features
    date_columns = [col for col in df_copy.columns if 'date' in col.lower()]
    for col in date_columns:
        if df_copy[col].dtype == 'object':
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            df_copy[f'{col}_year'] = df_copy[col].dt.year
            df_copy[f'{col}_month'] = df_copy[col].dt.month
            df_copy[f'{col}_day'] = df_copy[col].dt.day
            df_copy[f'{col}_dayofweek'] = df_copy[col].dt.dayofweek
            df_copy[f'{col}_is_weekend'] = df_copy[col].dt.dayofweek.isin([5, 6]).astype(int)
            df_copy[f'{col}_quarter'] = df_copy[col].dt.quarter

    # Example 2: Create interaction features (customize based on your data)
    # if 'distance' in df_copy.columns and 'duration' in df_copy.columns:
    #     df_copy['speed'] = df_copy['distance'] / (df_copy['duration'] + 1e-6)

    # Example 3: Create binned features
    # if 'age' in df_copy.columns:
    #     df_copy['age_group'] = pd.cut(df_copy['age'], 
    #                                    bins=[0, 18, 30, 45, 60, 100],
    #                                    labels=['Youth', 'Young_Adult', 'Middle_Age', 'Senior', 'Elderly'])

    # Example 4: Log transformation for skewed features
    skewed_cols = []
    for col in df_copy.select_dtypes(include=['int64', 'float64']).columns:
        if col not in date_columns and 'id' not in col.lower():
            skewness = df_copy[col].skew()
            if abs(skewness) > 1:  # Highly skewed
                skewed_cols.append(col)
                if (df_copy[col] > 0).all():
                    df_copy[f'{col}_log'] = np.log1p(df_copy[col])

    if skewed_cols:
        print(f"Created log transformations for skewed columns: {skewed_cols}")

    print(f"\nFeatures after engineering: {df_copy.shape[1]}")
    return df_copy

# Apply feature engineering
train_df = engineer_features(train_df)
test_df = engineer_features(test_df)
```

## 9. Encode Categorical Variables

```python
def encode_categorical_features(train, test, method='label', high_cardinality_threshold=50):
    """
    Encode categorical variables

    Parameters:
    - method: 'label', 'onehot', or 'frequency'
    - high_cardinality_threshold: For frequency encoding high-cardinality features
    """
    train_copy = train.copy()
    test_copy = test.copy()

    categorical_cols = train_copy.select_dtypes(include=['object', 'category']).columns.tolist()

    label_encoders = {}

    for col in categorical_cols:
        n_unique = train_copy[col].nunique()

        if n_unique > high_cardinality_threshold:
            # Frequency encoding for high cardinality
            freq_map = train_copy[col].value_counts(normalize=True).to_dict()
            train_copy[f'{col}_freq'] = train_copy[col].map(freq_map)
            test_copy[f'{col}_freq'] = test_copy[col].map(freq_map).fillna(0)
            print(f"{col}: Frequency encoded ({n_unique} unique values)")

        if method == 'label':
            # Label encoding
            le = LabelEncoder()
            train_copy[col] = le.fit_transform(train_copy[col].astype(str))
            test_copy[col] = test_copy[col].astype(str).map(
                dict(zip(le.classes_, le.transform(le.classes_)))
            ).fillna(-1).astype(int)
            label_encoders[col] = le
            print(f"{col}: Label encoded ({n_unique} unique values)")

        elif method == 'onehot' and n_unique <= 10:
            # One-hot encoding for low cardinality
            train_dummies = pd.get_dummies(train_copy[col], prefix=col, drop_first=True)
            test_dummies = pd.get_dummies(test_copy[col], prefix=col, drop_first=True)

            # Align columns
            for col_name in train_dummies.columns:
                if col_name not in test_dummies.columns:
                    test_dummies[col_name] = 0
            for col_name in test_dummies.columns:
                if col_name not in train_dummies.columns:
                    train_dummies[col_name] = 0

            train_copy = pd.concat([train_copy, train_dummies], axis=1)
            test_copy = pd.concat([test_copy, test_dummies], axis=1)
            train_copy = train_copy.drop(columns=[col])
            test_copy = test_copy.drop(columns=[col])
            print(f"{col}: One-hot encoded ({n_unique} unique values)")

    return train_copy, test_copy, label_encoders

# Apply encoding
train_df, test_df, encoders = encode_categorical_features(train_df, test_df, method='label')
```

## 10. Align Train and Test Columns

```python
# Ensure train and test have the same columns
def align_columns(train, test, target_col=None):
    """Align columns between train and test datasets"""
    # Identify columns in train but not in test
    train_only = set(train.columns) - set(test.columns)
    # Identify columns in test but not in train
    test_only = set(test.columns) - set(train.columns)

    print(f"Columns only in train: {train_only}")
    print(f"Columns only in test: {test_only}")

    # Get common columns
    if target_col:
        common_cols = list(set(train.columns) & set(test.columns))
        if target_col in train.columns:
            common_cols.append(target_col)
    else:
        common_cols = list(set(train.columns) & set(test.columns))

    train_aligned = train[common_cols]
    test_aligned = test[[col for col in common_cols if col != target_col and col in test.columns]]

    print(f"\nAligned columns: {len(common_cols)}")
    return train_aligned, test_aligned

# Note: Replace 'target' with your actual target column name
# train_df, test_df = align_columns(train_df, test_df, target_col='target')
```

## 11. Feature Scaling

```python
def scale_features(train, test, method='standard', exclude_cols=None):
    """
    Scale numerical features

    Parameters:
    - method: 'standard', 'minmax', or 'robust'
    - exclude_cols: List of columns to exclude from scaling (e.g., ID, target)
    """
    train_copy = train.copy()
    test_copy = test.copy()

    # Identify numerical columns
    numerical_cols = train_copy.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Exclude specific columns
    if exclude_cols:
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

    # Remove ID columns
    numerical_cols = [col for col in numerical_cols if 'id' not in col.lower()]

    # Select scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        scaler = RobustScaler()

    # Fit on train, transform both
    train_copy[numerical_cols] = scaler.fit_transform(train_copy[numerical_cols])
    test_copy[numerical_cols] = scaler.transform(test_copy[numerical_cols])

    print(f"Scaled {len(numerical_cols)} columns using {method} scaling")
    print(f"Scaled columns: {numerical_cols}")

    return train_copy, test_copy, scaler

# Apply scaling
# train_df, test_df, scaler = scale_features(train_df, test_df, method='standard')
```

## 12. Final Data Validation

```python
def validate_preprocessed_data(train, test):
    """Validate the preprocessed data"""
    print("=" * 80)
    print("FINAL DATA VALIDATION")
    print("=" * 80)

    print(f"\nTraining data shape: {train.shape}")
    print(f"Test data shape: {test.shape}")

    print(f"\nMissing values in train: {train.isnull().sum().sum()}")
    print(f"Missing values in test: {test.isnull().sum().sum()}")

    print(f"\nDuplicates in train: {train.duplicated().sum()}")
    print(f"Duplicates in test: {test.duplicated().sum()}")

    # Check for infinite values
    inf_train = np.isinf(train.select_dtypes(include=[np.number])).sum().sum()
    inf_test = np.isinf(test.select_dtypes(include=[np.number])).sum().sum()
    print(f"\nInfinite values in train: {inf_train}")
    print(f"Infinite values in test: {inf_test}")

    # Data types
    print(f"\nData types in train:")
    print(train.dtypes.value_counts())

    print("\nValidation completed!")

# Validate
validate_preprocessed_data(train_df, test_df)
```

## 13. Save Preprocessed Data

```python
# Save preprocessed data
train_df.to_csv('train_preprocessed.csv', index=False)
test_df.to_csv('test_preprocessed.csv', index=False)

print("Preprocessed data saved successfully!")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
```

## 14. Correlation Analysis (Optional)

```python
# Correlation heatmap
def plot_correlation_matrix(df, figsize=(12, 10)):
    """Plot correlation matrix for numerical features"""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if 'id' not in col.lower()]

    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()

        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True if len(numerical_cols) <= 15 else False, 
                    cmap='coolwarm', center=0, fmt='.2f', 
                    square=True, linewidths=0.5)
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.show()

        return corr_matrix
    else:
        print("Not enough numerical columns for correlation analysis")
        return None

# Plot correlation matrix
# corr_matrix = plot_correlation_matrix(train_df)
```

## Notes:
- Adjust preprocessing steps based on your specific dataset features
- Comment/uncomment sections as needed
- Replace 'target' with your actual target column name
- Test different encoding and scaling methods for best results
