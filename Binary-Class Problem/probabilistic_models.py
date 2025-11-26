import pandas as pd
import numpy as np
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
# Models (Restricted to requested list)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD & PREPARE DATA
# ==========================================
print("\n" + "="*40)
print(" STEP 1: LOADING DATA")
print("="*40)

try:
    train_df = pd.read_csv('train_updated.csv')
    test_df = pd.read_csv('test_updated.csv')
    print(f"Train Shape: {train_df.shape}")
    print(f"Test Shape:  {test_df.shape}")
except FileNotFoundError:
    print("Error: Files not found. Ensure train_updated.csv and test_updated.csv are in the folder.")
    exit()

# Save Test IDs for submission
test_ids = test_df['ProfileID']

# Drop ID columns
train_df = train_df.drop(columns=['ProfileID'], errors='ignore')
test_df = test_df.drop(columns=['ProfileID'], errors='ignore')

# Feature Engineering
print("Feature Engineering: Adding 'IncomeToLoanRatio'...")
train_df['IncomeToLoanRatio'] = train_df['AnnualEarnings'] / (train_df['RequestedSum'] + 1)
test_df['IncomeToLoanRatio'] = test_df['AnnualEarnings'] / (test_df['RequestedSum'] + 1)

# Define Binary Target
X = train_df.drop(columns=['RiskFlag', 'FundUseCase'], errors='ignore')
y = train_df['RiskFlag']

# ==========================================
# 2. PREPROCESSING PIPELINE
# ==========================================
# Identify columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Pipeline Setup
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ]
)

# Feature Selector
selector = SelectKBest(f_classif, k=20)

# ==========================================
# 3. RUN MODELS & COMPARE
# ==========================================
print("\n" + "="*40)
print(" STEP 2: RUNNING MODEL COMPARISON")
print("="*40)

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Preprocessor on Train
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)

# Fit Selector on Train
X_train_sel = selector.fit_transform(X_train_proc, y_train)
X_val_sel = selector.transform(X_val_proc)

models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

best_score = 0
best_model_name = ""

for name, model in models.items():
    start = time.time()
    
    # KNN Optimization for speed
    if name == "K-Nearest Neighbors" and len(X_train_sel) > 20000:
        model.fit(X_train_sel[:20000], y_train.iloc[:20000])
    else:
        model.fit(X_train_sel, y_train)
        
    acc = model.score(X_val_sel, y_val)
    print(f"{name:20}: Accuracy = {acc:.4f} ({time.time()-start:.2f}s)")
    
    if acc > best_score:
        best_score = acc
        best_model_name = name

print(f"\nWinner: {best_model_name} ({best_score:.4f})")

# ==========================================
# 4. GENERATE SUBMISSION (FIXED)
# ==========================================
print("\n" + "="*40)
print(f" STEP 3: GENERATING SUBMISSION ({best_model_name})")
print("="*40)

# 1. Preprocess Full Training Data and Test Data
print("Preprocessing full datasets...")
X_full_proc = preprocessor.fit_transform(X)
X_test_proc = preprocessor.transform(test_df)

# 2. Select Features
print("Selecting features...")
X_full_sel = selector.fit_transform(X_full_proc, y)
X_test_sel = selector.transform(X_test_proc)

# 3. Retrain Best Model on Full Data
print(f"Retraining {best_model_name} on 100% of data...")
final_model = models[best_model_name]
final_model.fit(X_full_sel, y)

# 4. Predict CLASSES (0 or 1) instead of Probabilities
print("Predicting Classes (0/1) on Test data...")
preds = final_model.predict(X_test_sel)

# 5. Save
submission = pd.DataFrame({
    'ProfileID': test_ids,
    'RiskFlag': preds
})
submission.to_csv('submission.csv', index=False)
print("Success! 'submission.csv' has been saved with binary labels.")
print(submission.head())
