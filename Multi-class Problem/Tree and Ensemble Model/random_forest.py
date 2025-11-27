import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def run_random_forest_tuning():
    # ---------------------------------------------------------
    # 1. PATH CONFIGURATION
    # ---------------------------------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir, 'train.csv')
    test_path = os.path.join(current_dir, 'test.csv')

    target_col = 'spend_category'
    id_col = 'trip_id'

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
    # Ensure target is integer
    y = train_df[target_col].astype(int)

    # Features (drop ID and target from train, drop ID from test)
    X = train_df.drop(columns=[target_col, id_col], errors='ignore')
    X_test_submit = test_df.drop(columns=[id_col], errors='ignore')

    # Extract test IDs
    test_ids = test_df[id_col] if id_col in test_df.columns else test_df.index
    
    # ---------------------------------------------------------
    # 4. TRAIN/VALIDATION SPLIT (Optional, but good for quick testing)
    # ---------------------------------------------------------
    # Using all data for GridSearchCV in the original script's spirit,
    # but still splitting to show evaluation against the best model.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Total Training shape for Grid Search: {X.shape}")
    print(f"Validation shape: {X_val.shape}")

    # ---------------------------------------------------------
    # 5. PREPROCESSING + MODEL PIPELINE
    # ---------------------------------------------------------
    print("\nSetting up Preprocessing Pipeline...")

    # Detect categorical + numeric columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    # Preprocessing pipeline (Imputing NaNs and One-Hot Encoding)
    preprocessor = ColumnTransformer(
        transformers=[
            # Categorical: One-Hot Encode (handling unknown categories)
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            # Numeric: Impute missing values with the median
            ("num", SimpleImputer(strategy='median'), numeric_cols),
        ],
        remainder='passthrough'
    )

    # Final model pipeline: Preprocessing -> Random Forest Classifier
    rf_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42))
        ]
    )
    # 

    # ---------------------------------------------------------
    # 6. HYPERPARAMETER TUNING (GridSearchCV)
    # ---------------------------------------------------------
    print("Starting GridSearchCV for Hyperparameter Tuning (this may take time)...")

    # Define the parameter grid for tuning.
    # Note: Parameters in a pipeline are named 'stepname__parameter',
    # e.g., 'classifier__n_estimators'
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None], # Smaller grid for faster execution
        'classifier__min_samples_split': [2, 5]
    }

    # Use X and y (the full training set) for the cross-validation
    grid_search = GridSearchCV(
        estimator=rf_pipeline,  # Use the pipeline as the estimator
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1
    )

    # Fit the grid search on the training data.
    # The pipeline handles all preprocessing internally.
    grid_search.fit(X, y)

    best_rf_model = grid_search.best_estimator_
    print(f"\nBest parameters found: {grid_search.best_params_}")

    # ---------------------------------------------------------
    # 7. EVALUATION
    # ---------------------------------------------------------
    print("\nEvaluating Best Model on Validation Set (20% of original training data):")
    # We use the X_val split from section 4 for a quick performance check
    val_preds = best_rf_model.predict(X_val)

    acc = accuracy_score(y_val, val_preds)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, val_preds, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, val_preds))

    # ---------------------------------------------------------
    # 8. GENERATE SUBMISSION
    # ---------------------------------------------------------
    print("\nGenerating predictions for Test set...")

    # The best_rf_model (which is a Pipeline) is used to predict.
    # It automatically applies the best preprocessing steps.
    test_predictions = best_rf_model.predict(X_test_submit)

    submission = pd.DataFrame({
        id_col: test_ids,
        target_col: test_predictions.astype(int)
    })

    output_file = os.path.join(current_dir, 'submission_random_forest.csv')
    submission.to_csv(output_file, index=False)

    print(f"Submission file saved to: {output_file}")


if __name__ == "__main__":
    run_random_forest_tuning()