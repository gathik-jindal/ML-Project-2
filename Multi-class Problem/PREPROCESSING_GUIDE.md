# Travel Behavior Insights - Preprocessing Quick Start Guide

## Overview
This guide provides preprocessing code for the Kaggle Travel Behavior Insights competition.

## File Structure
```
├── travel_behavior_preprocessing.py    # Complete preprocessing pipeline (modular)
├── preprocessing_notebook.md           # Jupyter notebook style code with visualizations
└── README.md                          # This file
```

## Quick Start

### Option 1: Using the Complete Pipeline (Recommended for Speed)

```python
from travel_behavior_preprocessing import preprocess_pipeline, load_data

# Load data
train_df, test_df = load_data('train.csv', 'test.csv')

# Configure preprocessing
config = {
    'missing_strategy': 'auto',      # 'auto', 'mean', 'median', 'mode', 'drop'
    'outlier_method': 'clip',        # 'clip', 'remove', 'log'
    'encoding_method': 'label',      # 'label', 'onehot', 'frequency'
    'scaling_method': 'standard',    # 'standard', 'minmax', 'robust'
    'feature_engineering': True      # True or False
}

# Run preprocessing
X_train, X_test, y_train, scaler, encoders = preprocess_pipeline(
    train_df, 
    test_df, 
    target_col='target',  # Replace with actual target column
    config=config
)

# Save preprocessed data
X_train.to_csv('X_train_preprocessed.csv', index=False)
X_test.to_csv('X_test_preprocessed.csv', index=False)
```

### Option 2: Step-by-Step Notebook Approach (Recommended for Exploration)

Copy the code from `preprocessing_notebook.md` into a Jupyter notebook and run cell by cell.
This approach allows you to:
- Visualize data at each step
- Customize preprocessing for your specific needs
- Debug and understand the data better

## Preprocessing Steps Included

### 1. Data Loading
- Load train and test CSV files
- Initial shape inspection

### 2. Exploratory Data Analysis (EDA)
- Dataset information and statistics
- Data type identification
- Initial quality checks

### 3. Missing Value Handling
- Detection and visualization of missing values
- Imputation strategies:
  - Numerical: median (default) or mean
  - Categorical: mode or 'Unknown'
- Separate handling for train and test

### 4. Duplicate Detection & Removal
- Identify duplicate rows
- Remove duplicates from training data

### 5. Outlier Detection & Handling
- IQR method for outlier detection
- Multiple handling strategies:
  - Clipping (capping values at bounds)
  - Removal (dropping outlier rows)
  - Log transformation (for skewed data)

### 6. Feature Engineering
- **DateTime features**: Extract year, month, day, day of week, quarter, weekend indicator
- **Interaction features**: Create new features from combinations
- **Binning**: Create categorical groups from continuous variables
- **Log transformation**: Handle skewed distributions
- **Aggregation features**: Group statistics

### 7. Categorical Encoding
- **Label Encoding**: For ordinal and tree-based models
- **One-Hot Encoding**: For low-cardinality features
- **Frequency Encoding**: For high-cardinality features
- Automatic handling of high-cardinality features

### 8. Feature Alignment
- Ensure train and test have matching columns
- Handle missing columns in test set

### 9. Feature Scaling
- **StandardScaler**: Zero mean, unit variance (default)
- **MinMaxScaler**: Scale to [0, 1] range
- **RobustScaler**: Robust to outliers
- Fit on train, transform both train and test

### 10. Data Validation
- Check for infinite values
- Verify no missing values remain
- Confirm data types are correct
- Shape verification

## Common Travel Behavior Features

Typical features you might encounter in this dataset:

### Demographic Features
- Age, gender, income level
- Occupation, education
- Household size, marital status

### Travel Behavior Features
- Travel frequency, duration
- Purpose of travel (business, leisure, commute)
- Mode of transportation
- Distance traveled

### Temporal Features
- Date/time of travel
- Season, day of week
- Holiday indicator

### Geographic Features
- Origin, destination
- Region, city
- Urban/rural indicator

### Preference Features
- Preferred transportation mode
- Accommodation type
- Budget category

## Customization Tips

### For Different Data Types

**Text Data:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=100)
text_features = tfidf.fit_transform(df['text_column'])
```

**Date Ranges:**
```python
# Calculate trip duration
df['duration_days'] = (df['end_date'] - df['start_date']).dt.days
```

**Geographic Data:**
```python
# Calculate distance (if lat/lon available)
from geopy.distance import geodesic

df['distance_km'] = df.apply(lambda row: 
    geodesic((row['origin_lat'], row['origin_lon']), 
             (row['dest_lat'], row['dest_lon'])).km, axis=1)
```

### For Different Target Types

**Binary Classification:**
```python
# Ensure target is 0/1
y_train = (y_train > 0).astype(int)
```

**Multi-class Classification:**
```python
# Label encode target
le = LabelEncoder()
y_train = le.fit_transform(y_train)
```

**Regression:**
```python
# Log transform target if skewed
y_train = np.log1p(y_train)
```

## Best Practices

### 1. Handle Missing Values Before Encoding
Always impute missing values before encoding categorical variables.

### 2. Fit Only on Training Data
```python
# ✓ CORRECT
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✗ WRONG
scaler.fit(pd.concat([X_train, X_test]))  # Data leakage!
```

### 3. Save Preprocessing Objects
```python
import pickle

# Save scaler and encoders
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
```

### 4. Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=20)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
```

### 5. Cross-Validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5, 
                         scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

## Troubleshooting

### Issue: ValueError - Number of features mismatch
**Solution:** Ensure train and test have the same columns after encoding
```python
# Get common columns
common_cols = list(set(X_train.columns) & set(X_test.columns))
X_train = X_train[common_cols]
X_test = X_test[common_cols]
```

### Issue: KeyError during encoding
**Solution:** Handle unseen categories in test set
```python
# For label encoding
test_df[col] = test_df[col].map(
    dict(zip(le.classes_, le.transform(le.classes_)))
).fillna(-1)
```

### Issue: Memory error with large datasets
**Solution:** Process in chunks or use sparse matrices
```python
# Process in chunks
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)
```

### Issue: High cardinality categorical features
**Solution:** Use frequency encoding or target encoding
```python
# Frequency encoding
freq_map = train_df[col].value_counts(normalize=True).to_dict()
train_df[f'{col}_freq'] = train_df[col].map(freq_map)
```

## Model Training Example

After preprocessing, you can train models:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# For competition submission
submission = pd.DataFrame({
    'id': test_df['id'],  # Replace with actual ID column
    'target': y_pred       # Replace with actual target column name
})
submission.to_csv('submission.csv', index=False)
```

## Advanced Techniques

### 1. Feature Importance Analysis
```python
# Get feature importances
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20))
```

### 2. Ensemble Preprocessing
```python
# Try multiple preprocessing pipelines
configs = [
    {'scaling_method': 'standard', 'encoding_method': 'label'},
    {'scaling_method': 'minmax', 'encoding_method': 'onehot'},
    {'scaling_method': 'robust', 'encoding_method': 'frequency'}
]

for config in configs:
    X_train, X_test, _, _, _ = preprocess_pipeline(
        train_df, test_df, config=config
    )
    # Train and evaluate model
```

### 3. Automated Feature Engineering
```python
# Use libraries like featuretools
import featuretools as ft

es = ft.EntitySet(id='travel_data')
es = es.add_dataframe(dataframe_name='travel', 
                      dataframe=train_df, 
                      index='id')

feature_matrix, feature_defs = ft.dfs(entityset=es, 
                                      target_dataframe_name='travel',
                                      max_depth=2)
```

## Resources

- **Competition Page:** https://www.kaggle.com/competitions/travel-behavior-insights/
- **Scikit-learn Documentation:** https://scikit-learn.org/
- **Pandas Documentation:** https://pandas.pydata.org/
- **Feature Engineering Guide:** https://www.kaggle.com/learn/feature-engineering

## Contact & Support

For issues or questions:
1. Check the Kaggle competition discussion forum
2. Review common preprocessing patterns for similar competitions
3. Test different configurations systematically

## Version History

- v1.0 (Nov 2025): Initial preprocessing pipeline release
  - Complete modular pipeline
  - Notebook-style exploration code
  - Comprehensive documentation

---

**Good luck with the competition!** 🚀
