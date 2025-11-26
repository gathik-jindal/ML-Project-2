import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier


def train_lightgbm_multiclass(
    train_path,
    test_path,
    target_col=None,
    id_col=None,
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=300,
    random_state=42,
    n_jobs=-1
):

    print("\n=== Loading Data ===")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Train Shape:", train_df.shape)
    print("Test Shape:", test_df.shape)

    # Auto-detect target column
    if target_col is None:
        target_col = train_df.columns[-1]

    print("Target Column:", target_col)

    # Auto-detect ID column
    if id_col is None:
        for cand in ["id", "ID", "ProfileID", "trip_id"]:
            if cand in train_df.columns:
                id_col = cand
                break

    print("ID Column:", id_col if id_col else "None")

    # X and y
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    if id_col and id_col in X.columns:
        X = X.drop(columns=[id_col])

    # Test set
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

    # Train/Val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Categorical / numeric
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
            ("num", "passthrough", numeric_cols),
        ]
    )

    # LightGBM Model
    model_lgb = LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs
    )

    model = Pipeline([
        ("preprocess", preprocessor),
        ("model", model_lgb)
    ])

    print("\n=== Training LightGBM ===")
    model.fit(X_train, y_train_enc)

    print("\n=== Validating ===")
    val_preds_enc = model.predict(X_val)
    val_preds = le.inverse_transform(val_preds_enc)

    print("\nAccuracy:", accuracy_score(y_val, val_preds))
    print("\nClassification Report:\n", classification_report(y_val, val_preds))
    print("\nConfusion Matrix:\n", confusion_matrix(y_val, val_preds))

    print("\n=== Predicting Test ===")
    test_preds_enc = model.predict(X_test)
    test_preds = le.inverse_transform(test_preds_enc)

    # Submission file
    submission = pd.DataFrame({
        id_col if id_col else "id": test_ids,
        target_col: test_preds
    })

    output_path = os.path.join(os.path.dirname(train_path), "submission_lightgbm.csv")
    submission.to_csv(output_path, index=False)
    print("\nSaved Submission:", output_path)

    return model, le


if __name__ == "__main__":
    train_lightgbm_multiclass(
        train_path="train.csv",
        test_path="test.csv",
        target_col=None,
        id_col=None,
        n_estimators=300,
        learning_rate=0.05,
        n_jobs=-1
    )
