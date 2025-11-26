import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier


def train_xgboost_multiclass(
    train_path,
    test_path,
    target_col=None,
    id_col=None,
    n_estimators=150,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    n_jobs=4
):

    print("\n=== Loading Data ===")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Train Shape:", train_df.shape)
    print("Test Shape:", test_df.shape)

    # Auto-detect target column if not given
    if target_col is None:
        target_col = train_df.columns[-1]
    print("Target Column:", target_col)

    # Auto-detect ID column if not given
    if id_col is None:
        for cand in ["id", "ID", "trip_id", "ProfileID"]:
            if cand in train_df.columns:
                id_col = cand
                break
    print("ID Column:", id_col if id_col else "None")

    # Split X and y
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col].copy()

    if id_col and id_col in X.columns:
        X = X.drop(columns=[id_col])

    # Test data
    if id_col and id_col in test_df.columns:
        test_ids = test_df[id_col]
        X_test = test_df.drop(columns=[id_col])
    else:
        test_ids = test_df.index
        X_test = test_df.copy()

    # Fix missing columns in test
    for col in X.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[X.columns]

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Detect categorical and numeric columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()

    print("\nCategorical Columns:", categorical_cols)
    print("Numeric Columns:", numeric_cols)

    # Encode target
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    n_classes = len(le.classes_)

    print("\nDetected Classes:", list(le.classes_))

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    # XGBoost model
    xgb = XGBClassifier(
        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        num_class=n_classes if n_classes > 2 else None,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        use_label_encoder=True
    )

    model = Pipeline([
        ("preprocess", preprocessor),
        ("model", xgb)
    ])

    print("\n=== Training XGBoost Model ===")
    model.fit(X_train, y_train_enc)

    print("\n=== Validating Model ===")
    val_preds_enc = model.predict(X_val)
    val_preds = le.inverse_transform(val_preds_enc)

    print("\nAccuracy:", accuracy_score(y_val, val_preds))
    print("\nClassification Report:\n", classification_report(y_val, val_preds))
    print("\nConfusion Matrix:\n", confusion_matrix(y_val, val_preds))

    # ROC-AUC for multiclass
    try:
        y_val_bin = label_binarize(y_val, classes=le.classes_)
        val_proba = model.predict_proba(X_val)
        auc = roc_auc_score(y_val_bin, val_proba, average="macro", multi_class="ovr")
        print("\nMacro ROC-AUC:", auc)
    except Exception as e:
        print("ROC-AUC could not be computed:", e)

    print("\n=== Predicting Test Data ===")
    test_preds_enc = model.predict(X_test)
    test_preds = le.inverse_transform(test_preds_enc)

    # Create submission
    submission = pd.DataFrame({
        id_col if id_col else "id": test_ids,
        target_col: test_preds
    })

    output_path = os.path.join(os.path.dirname(train_path), "submission_xgboost.csv")
    submission.to_csv(output_path, index=False)

    print("\nSaved Submission File:", output_path)


    return model, le


if __name__ == "__main__":
    # Change to your CSV filenames
    train_xgboost_multiclass(
        train_path="train.csv",
        test_path="test.csv",
        target_col=None,   # autodetects last column
        id_col=None,       # autodetects if present
        n_estimators=150,
        n_jobs=4
    )
