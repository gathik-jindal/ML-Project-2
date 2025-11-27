import pandas as pd
import os
import sys

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
# Note: SimpleImputer is good practice for robust pipelines, though not strictly
# needed if you are certain your numeric data has no NaNs.
from sklearn.impute import SimpleImputer 


def run_multi_class_lightgbm():
    """
    Runs a multi-class classification model using a LightGBM Classifier
    with a preprocessing pipeline and LabelEncoder for the target.
    """
    # ---------------------------------------------------------
    # 1. PATH CONFIGURATION
    # ---------------------------------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Assuming 'train.csv' and 'test.csv' are in the same directory
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
    # IMPORTANT: Adjust these to match your multi-class column names
    target_col = 'spend_category' 
    id_col = 'trip_id'            

    if target_col not in train_df.columns:
        print(f"Error: Target column '{target_col}' not found in training data.")
        return

    # Raw Target (y_raw) and Features (X)
    y_raw = train_df[target_col]
    X = train_df.drop(columns=[target_col, id_col], errors='ignore')
    
    # Prepare Test data
    test_ids = test_df[id_col] if id_col in test_df.columns else test_df.index
    X_test_submit = test_df.drop(columns=[id_col], errors='ignore')

    # Ensure test columns match training columns (handles columns missing in test)
    for col in X.columns:
        if col not in X_test_submit.columns:
            X_test_submit[col] = 0
    X_test_submit = X_test_submit[X.columns]

    # ---------------------------------------------------------
    # 4. TARGET ENCODING (MANDATORY FOR MULTI-CLASS LIGHTGBM)
    # ---------------------------------------------------------
    le = LabelEncoder()
    # Fit on ALL raw labels and transform to integer-encoded labels
    y_encoded = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    class_labels = list(le.classes_)
    
    print(f"\nDetected {n_classes} classes: {class_labels}")

    # ---------------------------------------------------------
    # 5. DATA SPLIT (Train/Validation)
    # ---------------------------------------------------------
    X_train, X_val, y_train_enc, y_val_enc = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    # Get raw validation labels for human-readable reporting
    y_val_raw = le.inverse_transform(y_val_enc) 

    print(f"Train set size: {X_train.shape[0]} samples")
    print(f"Validation set size: {X_val.shape[0]} samples")

    # ---------------------------------------------------------
    # 6. PREPROCESSING & MODEL PIPELINE
    # ---------------------------------------------------------
    
    # Detect categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    print(f"\nCategorical features: {categorical_cols}")
    print(f"Numeric features: {numeric_cols}")

    # Define the Preprocessor (OHE for categorical, Imputer for numeric, passthrough)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", SimpleImputer(strategy='median'), numeric_cols),
        ],
        remainder='drop'
    )

    # Define the LightGBM Model for Multi-Class
    lgbm_model = LGBMClassifier(
        objective='multiclass',    # Set objective to multi-class
        num_class=n_classes,       # Specify the number of classes
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )

    # Final model pipeline
    lgbm_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("lgbm", lgbm_model),
        ]
    )
    

    print("\nStarting model training...")
    # Fit with the integer-encoded target labels
    lgbm_pipeline.fit(X_train, y_train_enc) 
    print("Training complete.")

    # ---------------------------------------------------------
    # 7. EVALUATION
    # ---------------------------------------------------------
    print("\nEvaluating on Validation Set (Multi-Class):")
    
    # Predict integer-encoded labels
    val_preds_enc = lgbm_pipeline.predict(X_val) 
    # Inverse transform to get human-readable labels for reporting
    val_preds_raw = le.inverse_transform(val_preds_enc) 

    acc = accuracy_score(y_val_raw, val_preds_raw)
    print(f"Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    # Use raw labels for the report
    print(classification_report(y_val_raw, val_preds_raw, zero_division=0)) 
    
    print("Confusion Matrix:")
    # Use raw labels for the confusion matrix
    print(confusion_matrix(y_val_raw, val_preds_raw)) 

    # ---------------------------------------------------------
    # 8. GENERATE SUBMISSION
    # ---------------------------------------------------------
    print("\nGenerating predictions for Test set...")

    # Predict integer-encoded labels
    test_preds_enc = lgbm_pipeline.predict(X_test_submit)
    
    # Inverse transform predictions back to original class names
    test_preds_raw = le.inverse_transform(test_preds_enc)

    submission = pd.DataFrame({
        id_col: test_ids,
        target_col: test_preds_raw
    })

    # Save the submission file
    output_file = os.path.join(current_dir, 'submission_lightgbm.csv')
    submission.to_csv(output_file, index=False)

    print(f"Submission file saved to: {output_file}")


if __name__ == '__main__':
    try:
        run_multi_class_lightgbm()
    except ImportError:
        print("\nERROR: LightGBM library is not installed. Please install it using: pip install lightgbm")
        sys.exit(1)