import pandas as pd
import os
import sys
# Import RandomForestClassifier instead of Decision Tree or XGBoost
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


def run_random_forest():
    """
    Runs a binary classification model using a Random Forest Classifier
    with a preprocessing pipeline.
    """
    # ---------------------------------------------------------
    # 1. PATH CONFIGURATION
    # ---------------------------------------------------------
    # Get the directory of the current script to find data files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Assuming 'train.csv' and 'test.csv' are in the same directory
    train_path = os.path.join(current_dir, 'train.csv')
    test_path = os.path.join(current_dir, 'test.csv')

    print(f"Loading data from: {current_dir}")

    # ---------------------------------------------------------
    # 2. LOAD DATA
    # ---------------------------------------------------------
    try:
        # NOTE: For this code to run, you must ensure 'train.csv' and 'test.csv'
        # are present in the same directory as this script.
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. Please ensure 'train.csv' and 'test.csv' are present. {e}")
        return

    # ---------------------------------------------------------
    # 3. PREPARE FEATURES AND TARGET
    # ---------------------------------------------------------
    target_col = 'RiskFlag'
    id_col = 'ProfileID'

    if target_col not in train_df.columns:
        print(f"Error: Target column '{target_col}' not found in training data.")
        return

    # Store test IDs and remove them from the feature set
    test_ids = test_df[id_col]
    
    # Features (X) and Target (y)
    X = train_df.drop(columns=[target_col, id_col])
    y = train_df[target_col]
    X_test_submit = test_df.drop(columns=[id_col])

    # Infer data types for preprocessing
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical features: {categorical_cols}")
    print(f"Numeric features: {numeric_cols}")


    # ---------------------------------------------------------
    # 4. DATA SPLIT (Train/Validation)
    # ---------------------------------------------------------
    # Split the training data to have a held-out validation set for evaluation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train set size: {X_train.shape[0]} samples")
    print(f"Validation set size: {X_val.shape[0]} samples")

    # ---------------------------------------------------------
    # 5. PREPROCESSING & MODEL PIPELINE
    # ---------------------------------------------------------

    # 5a. Define the Preprocessor (One-Hot Encoding for categorical, pass-through for numeric)
    preprocessor = ColumnTransformer(
        transformers=[
            # 'handle_unknown="ignore"' is crucial for test/validation sets 
            # that might have categories not seen in the training set
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder='drop' # Drop any other columns not explicitly listed
    )

    # 5b. Define the Random Forest Model
    # Random Forests are generally less sensitive to hyperparameters than GBMs.
    rf_model = RandomForestClassifier(
        n_estimators=100,      # Number of trees in the forest
        max_depth=10,          # Maximum depth of the trees
        min_samples_leaf=5,    # Minimum number of samples required at a leaf node
        random_state=42,
        n_jobs=-1              # Use all available CPU cores for speed
    )

    # 5c. Final model pipeline
    rf_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("rf", rf_model),
        ]
    )

    print("\nStarting model training...")
    rf_pipeline.fit(X_train, y_train)
    print("Training complete.")

    # ---------------------------------------------------------
    # 6. EVALUATION
    # ---------------------------------------------------------
    print("\nEvaluating on Validation Set:")
    val_preds = rf_pipeline.predict(X_val)

    acc = accuracy_score(y_val, val_preds)
    print(f"Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_val, val_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, val_preds))

    # ---------------------------------------------------------
    # 7. GENERATE SUBMISSION
    # ---------------------------------------------------------
    print("\nGenerating predictions for Test set...")

    test_preds = rf_pipeline.predict(X_test_submit)

    submission = pd.DataFrame({
        id_col: test_ids,
        target_col: test_preds
    })

    # Save the submission file
    output_file = os.path.join(current_dir, 'submission_random_forest.csv')
    submission.to_csv(output_file, index=False)

    print(f"Submission file saved to: {output_file}")


if __name__ == '__main__':
    # We don't need a specific try-except for the library here since
    # RandomForestClassifier is part of the standard scikit-learn installation
    # which we assume is available if DecisionTreeClassifier was used.
    run_random_forest()