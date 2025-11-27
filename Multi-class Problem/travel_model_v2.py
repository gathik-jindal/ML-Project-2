import pandas as pd
import numpy as np
import time
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
# Models
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD DATA
# ==========================================
print("\n" + "="*40)
print(" STEP 1: LOAD & CLEAN DATA")
print("="*40)
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
except:
    print("Error: Files not found.")
    exit()

# Save IDs
test_ids = test_df['trip_id']
train_df.drop(columns=['trip_id'], inplace=True)
test_df.drop(columns=['trip_id'], inplace=True)

# Remove rows with missing target
train_df = train_df.dropna(subset=['spend_category'])

# ==========================================
# 2. ADVANCED FEATURE ENGINEERING
# ==========================================
print("\n" + "="*40)
print(" STEP 2: FEATURE ENGINEERING (Grouping Rare Categories)")
print("="*40)

def process_features(df, top_countries=None):
    # 1. Group Rare Countries (Crucial for preventing overfitting)
    if top_countries is None:
        # Get top 15 countries from training data
        top_countries = df['country'].value_counts().nlargest(15).index.tolist()
    
    # Apply grouping
    df['country_grouped'] = df['country'].apply(lambda x: x if x in top_countries else 'Other')
    df.drop(columns=['country'], inplace=True) # Drop original
    
    # 2. Create Numeric Features
    df['total_people'] = df['num_females'].fillna(0) + df['num_males'].fillna(0)
    df['total_nights'] = df['mainland_stay_nights'].fillna(0) + df['island_stay_nights'].fillna(0)
    df['is_alone'] = (df['total_people'] == 1).astype(int)
    
    # 3. Clean Missing Categoricals
    # Fill specific columns with 'Unknown' where missingness might be informative
    cols_to_fill_unknown = ['arrival_weather', 'travel_companions', 'main_activity']
    for c in cols_to_fill_unknown:
        if c in df.columns:
            df[c] = df[c].fillna('Unknown')
            
    return df, top_countries

# Process Train first to learn top countries
train_df, top_15_countries = process_features(train_df)
# Process Test using the SAME top countries
test_df, _ = process_features(test_df, top_countries=top_15_countries)

print(f"Top Countries kept: {top_15_countries}")
print("Rare countries grouped into 'Other'.")

# ==========================================
# 3. PREPROCESSING PIPELINE
# ==========================================
X = train_df.drop(columns=['spend_category'])
y = train_df['spend_category'].astype(int)

numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# RobustScaler handles outliers (like very long trips) better than StandardScaler
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

# Use 'constant' strategy to treat remaining NaNs as a category
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, numeric_cols),
    ('cat', cat_pipe, categorical_cols)
])

# ==========================================
# 4. MODEL TUNING (GridSearch)
# ==========================================
print("\n" + "="*40)
print(" STEP 3: TUNING MODELS")
print("="*40)

# Preprocess once to speed up GridSearch
X_proc = preprocessor.fit_transform(X)

# Feature Selection inside GridSearch is better
# We will test Logistic Regression thoroughly as it was your best model
model = LogisticRegression(multi_class='multinomial', max_iter=2000, random_state=42)

# Grid: Test C (Regularization) and Solver
# Smaller C = Stronger Regularization (Prevents Overfitting)
param_grid = {
    'C': [0.01, 0.1, 0.5, 1, 5], 
    'solver': ['lbfgs', 'newton-cg']
}

grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_proc, y)

print(f"Best Logistic Params: {grid.best_params_}")
print(f"Best CV Accuracy: {grid.best_score_:.4f}")
best_model = grid.best_estimator_

# Also check KNN quickly
knn = KNeighborsClassifier()
knn_grid = GridSearchCV(knn, {'n_neighbors': [10, 15, 20, 30]}, cv=5)
knn_grid.fit(X_proc, y)
print(f"Best KNN Accuracy: {knn_grid.best_score_:.4f} (k={knn_grid.best_params_['n_neighbors']})")

if knn_grid.best_score_ > grid.best_score_:
    print("Switching to KNN as it performed better.")
    best_model = knn_grid.best_estimator_

# ==========================================
# 5. SUBMISSION
# ==========================================
print("\n" + "="*40)
print(" STEP 4: GENERATING SUBMISSION")
print("="*40)

# Prepare Test Data
X_test_proc = preprocessor.transform(test_df)

# Predict
preds = best_model.predict(X_test_proc)

# Save
sub = pd.DataFrame({'trip_id': test_ids, 'spend_category': preds})
sub.to_csv('submission_travel_v2.csv', index=False)
print("Saved 'submission_travel_v2.csv'. Upload this one!")
