import pandas as pd
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from xgboost import XGBClassifier


def run_model():
    # ---------------------------------------------------------
    # 1. PATH CONFIGURATION
    # ---------------------------------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))

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
    target_col = 'RiskFlag'
    id_col = 'ProfileID'

    if target_col not in train_df.columns:
        print(f"Error: Target column '{target_col}' not found.")
        return

    # Separate features and target
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    # Drop ID column
    if id_col in X.columns:
        X = X.drop(columns=[id_col])

    # Prepare test data
    test_ids = test_df[id_col] if id_col in test_df.columns else test_df.index
    X_test_submit = test_df.drop(columns=[id_col], errors='ignore')

    # Ensure same columns as train
    X_test_submit = X_test_submit[X.columns]

    # ---------------------------------------------------------
    # 4. TRAIN-VALIDATION SPLIT
    # ---------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")

    # ---------------------------------------------------------
    # 5. DETECT COLUMN TYPES
    # ---------------------------------------------------------
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()

    print("\nCategorical columns:", categorical_cols)
    print("Numeric columns:", numeric_cols)

    # ---------------------------------------------------------
    # 6. PREPROCESSING + MODEL
    # ---------------------------------------------------------
    print("\nTraining XGBoost Model...")

    # Preprocessing: OHE for categorical, passthrough for numeric
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    # XGBoost Model
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("xgb", XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
            )),
        ]
    )

    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 7. EVALUATION
    # ---------------------------------------------------------
    print("\nEvaluating Model on Validation Set...")

    val_preds = model.predict(X_val)
    val_proba = model.predict_proba(X_val)[:, 1]

    print(f"Accuracy: {accuracy_score(y_val, val_preds):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_val, val_proba):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_val, val_preds))

    print("Confusion Matrix:")
    print(confusion_matrix(y_val, val_preds))

    # ---------------------------------------------------------
    # 8. PREDICT TEST SET
    # ---------------------------------------------------------
    print("\nGenerating predictions for Test set...")

    test_preds = model.predict(X_test_submit)

    submission = pd.DataFrame({
        id_col: test_ids,
        target_col: test_preds
    })

    output_file = os.path.join(current_dir, 'submission_xgboost.csv')
    submission.to_csv(output_file, index=False)

    print(f"Submission file saved to: {output_file}")
    print("Done.")


if __name__ == "__main__":
    run_model()
