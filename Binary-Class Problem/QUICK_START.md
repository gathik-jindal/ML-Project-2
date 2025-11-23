# QUICK START GUIDE FOR FINANCIAL RISK PROFILING PREPROCESSING

## Step 1: Install Required Libraries
```bash
pip install pandas numpy scikit-learn scipy
```

## Step 2: Import and Run the Pipeline

```python
from financial_risk_preprocessing import preprocess_pipeline

# Run complete preprocessing pipeline
preprocessed = preprocess_pipeline(
    train_path='train.csv',
    test_path='test.csv',
    target_col='target'  # Change to your actual target column name
)

# Access preprocessed data
X_train = preprocessed['X_train']
y_train = preprocessed['y_train']
X_test = preprocessed['X_test']
scaler = preprocessed['scaler']
```

## Step 3: Individual Function Usage

### Just Impute Missing Values
```python
import pandas as pd
from financial_risk_preprocessing import impute_missing_values, identify_dtypes

df = pd.read_csv('train.csv')
numerical_cols, categorical_cols, _ = identify_dtypes(df)

df_imputed = impute_missing_values(
    df, numerical_cols, categorical_cols,
    numerical_method='median',
    categorical_method='mode'
)
```

### Detect and Handle Outliers
```python
from financial_risk_preprocessing import detect_outliers_iqr, handle_outliers

# Detect outliers
outliers = detect_outliers_iqr(df, numerical_cols)

# Handle outliers by clipping
df_cleaned = handle_outliers(df, numerical_cols, method='clip', threshold=1.5)

# Or remove rows with outliers
df_cleaned = handle_outliers(df, numerical_cols, method='remove', threshold=1.5)
```

### Encode Categorical Variables
```python
from financial_risk_preprocessing import encode_categorical

# One-hot encoding
df_encoded = encode_categorical(df, categorical_cols, method='onehot')

# Label encoding
df_encoded = encode_categorical(df, categorical_cols, method='label')
```

### Scale Features
```python
from financial_risk_preprocessing import scale_features

# Scale training data
X_train_scaled, scaler = scale_features(df, numerical_cols, method='standard')

# Scale test data using fitted scaler
X_test_scaled, _ = scale_features(test_df, numerical_cols, 
                                  method='standard', fit_scaler=scaler)
```

## Step 4: Save Preprocessed Data

```python
# Save to CSV
X_train.to_csv('X_train_preprocessed.csv', index=False)
X_test.to_csv('X_test_preprocessed.csv', index=False)
pd.DataFrame(y_train).to_csv('y_train.csv', index=False)

# Save scaler for future use
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

## Step 5: Load and Use Saved Scaler

```python
import pickle

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Use for new predictions
new_data_scaled = scaler.transform(new_data)
```

## Customization Tips

### Add Custom Financial Features
Modify the `create_financial_features()` function:

```python
def create_financial_features(df):
    df_engineered = df.copy()

    # Example: Create debt-to-income ratio
    if 'total_debt' in df.columns and 'annual_income' in df.columns:
        df_engineered['debt_to_income'] = \
            df_engineered['total_debt'] / (df_engineered['annual_income'] + 1e-8)

    # Example: Create credit utilization
    if 'credit_used' in df.columns and 'credit_limit' in df.columns:
        df_engineered['credit_utilization'] = \
            df_engineered['credit_used'] / (df_engineered['credit_limit'] + 1e-8)

    return df_engineered
```

### Adjust Imputation Strategy
```python
# For numerical columns: 'mean', 'median', 'knn'
# For categorical columns: 'mode', 'constant'

df_imputed = impute_missing_values(
    df, numerical_cols, categorical_cols,
    numerical_method='knn',        # Use KNN imputation
    categorical_method='constant'  # Use constant (Unknown)
)
```

### Different Scaling Methods
```python
# Standard scaling (z-score normalization)
X_scaled, scaler = scale_features(df, numerical_cols, method='standard')

# Min-Max scaling (0-1 range)
X_scaled, scaler = scale_features(df, numerical_cols, method='minmax')
```

## Common Issues & Solutions

### Issue: Target column not found
**Solution:** Make sure the target column name matches exactly
```python
print(df.columns)  # Check available columns
preprocessed = preprocess_pipeline(..., target_col='actual_column_name')
```

### Issue: Shape mismatch between train and test
**Solution:** Ensure both files have similar column structure
```python
print(train_df.columns)
print(test_df.columns)
# Use only common columns for preprocessing
```

### Issue: Scaler fitted on train but used on test
**Solution:** Use the same scaler object for both
```python
X_train_scaled, scaler = scale_features(X_train, numerical_cols)
X_test_scaled, _ = scale_features(X_test, numerical_cols, fit_scaler=scaler)
# Not recommended: creating new scaler for test data
```

## Performance Tips

1. **For Large Datasets (>1M rows):**
   - Use `numerical_method='median'` (faster than KNN)
   - Consider sampling for outlier detection
   - Use `method='clip'` instead of removing rows

2. **For Small Datasets (<10K rows):**
   - Use `numerical_method='knn'` (better imputation)
   - Create polynomial features for better model fit
   - Use `method='remove'` for outliers (cleaner data)

3. **For Imbalanced Datasets:**
   - After preprocessing, use stratified train-test split
   - Consider SMOTE or class weights in your model

## Next Steps

After preprocessing:
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Train-test split
X_t, X_v, y_t, y_v = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_t, y_t)

# Evaluate
score = model.score(X_v, y_v)
print(f"Validation Accuracy: {score:.4f}")
```
