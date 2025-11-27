import pandas as pd
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.impute import SimpleImputer # Added for robustness against missing numeric data
from xgboost import XGBClassifier


def run_multi_class_xgboost():
    # ---------------------------------------------------------
    # 1. PATH CONFIGURATION
    # ---------------------------------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Assuming these are the filenames for your multi-class problem
    train_path = os.path.join(current_dir, 'train.csv')
    test_path = os.path.join(current_dir, 'test.csv')

    print(f"Loading data from: {current_dir}")

    # ---------------------------------------------------------
    # 2. LOAD DATA
    # ---------------------------------------------------------
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. {e}")
        return

    # ---------------------------------------------------------
    # 3. PREPARE FEATURES AND TARGET
    # ---------------------------------------------------------
    # Adjust these based on your multi-class dataset
    target_col = 'spend_category' # Your multi-class target
    id_col = 'trip_id'            # Your ID column

    if target_col not in train_df.columns:
        print(f"Error: Target column '{target_col}' not found.")
        return

    # Separate features and target
    X = train_df.drop(columns=[target_col])
    y_raw = train_df[target_col].copy() # Keep a copy of raw labels

    # Drop ID column from features
    if id_col in X.columns:
        X = X.drop(columns=[id_col])

    # Prepare test data
    test_ids = test_df[id_col] if id_col in test_df.columns else test_df.index
    X_test_submit = test_df.drop(columns=[id_col], errors='ignore')

    # Ensure same columns as train (Handles columns missing in test)
    for col in X.columns:
        if col not in X_test_submit.columns:
            X_test_submit[col] = 0
    X_test_submit = X_test_submit[X.columns]
    
    # ---------------------------------------------------------
    # 4. TARGET ENCODING (MANDATORY FOR MULTI-CLASS XGBOOST)
    # ---------------------------------------------------------
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    class_labels = list(le.classes_)

    print(f"\nDetected {n_classes} classes: {class_labels}")

    # ---------------------------------------------------------
    # 5. TRAIN-VALIDATION SPLIT
    # ---------------------------------------------------------
    X_train, X_val, y_train_enc, y_val_enc = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    # Get raw validation labels for reporting
    y_val_raw = le.inverse_transform(y_val_enc)

    print(f"\nTraining shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")

    # ---------------------------------------------------------
    # 6. DETECT COLUMN TYPES & PREPROCESSING SETUP
    # ---------------------------------------------------------
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()

    print("\nCategorical columns:", categorical_cols)
    print("Numeric columns:", numeric_cols)

    # Preprocessing: OHE for categorical, Imputer for numeric (handles NaNs)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", SimpleImputer(strategy='median'), numeric_cols), # Impute NaNs in numeric columns
        ],
        remainder='passthrough'
    )

    # ---------------------------------------------------------
    # 7. MODEL DEFINITION AND TRAINING
    # ---------------------------------------------------------
    print("\nTraining Multi-Class XGBoost Model...")

    # XGBoost configuration for multi-class problem
    xgb_clf = XGBClassifier(
        objective="multi:softprob", # Use multi:softprob for multi-class probability output
        num_class=n_classes,
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss", # Multi-class logloss
        use_label_encoder=False, # Use False for modern sklearn compliance
        tree_method='hist' # Faster training
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("xgb", xgb_clf),
        ]
    )

    # Fit the model using the integer-encoded target labels
    model.fit(X_train, y_train_enc)

    # ---------------------------------------------------------
    # 8. EVALUATION
    # ---------------------------------------------------------
    print("\nEvaluating Model on Validation Set (Multi-Class)...")

    # Predict integer-encoded labels
    val_preds_enc = model.predict(X_val)
    # Inverse transform to get human-readable labels
    val_preds_raw = le.inverse_transform(val_preds_enc)
    
    # Predict probabilities (needed for ROC-AUC)
    val_proba = model.predict_proba(X_val)

    print(f"Accuracy: {accuracy_score(y_val_raw, val_preds_raw):.4f}")
    
    # ROC-AUC calculation for multi-class (One vs Rest, Macro average)
    try:
        # Binarize validation labels for ROC-AUC calculation
        y_val_bin = label_binarize(y_val_raw, classes=class_labels) 
        auc = roc_auc_score(y_val_bin, val_proba, average="macro", multi_class="ovr")
        print(f"Macro ROC-AUC (OvR): {auc:.4f}")
    except Exception as e:
        print(f"ROC-AUC could not be computed: {e}")

    print("\nClassification Report:")
    # Use raw labels and targets for the report
    print(classification_report(y_val_raw, val_preds_raw, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_val_raw, val_preds_raw))

    # ---------------------------------------------------------
    # 9. PREDICT TEST SET
    # ---------------------------------------------------------
    print("\nGenerating predictions for Test set...")

    # Predict integer-encoded labels on the test set
    test_preds_enc = model.predict(X_test_submit)
    
    # Inverse transform predictions back to original class names
    test_preds_raw = le.inverse_transform(test_preds_enc)

    submission = pd.DataFrame({
        id_col: test_ids,
        target_col: test_preds_raw
    })

    output_file = os.path.join(current_dir, 'submission_xgboost.csv')
    submission.to_csv(output_file, index=False)

    print(f"Submission file saved to: {output_file}")
    print("Done.")


if __name__ == "__main__":
    run_multi_class_xgboost()