import pandas as pd
import os
import sys  # Added for path management print (optional)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # Added for handling potential missing values

def run_multi_class_decision_tree():
    # ---------------------------------------------------------
    # 1. PATH CONFIGURATION
    # ---------------------------------------------------------
    # Assuming the script and data files are in the same directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # NOTE: You'll need to update 'train.csv' and 'test.csv' to the actual
    # filenames used in your multi-class problem if they are different.
    train_path = os.path.join(current_dir, 'train.csv')
    test_path = os.path.join(current_dir, 'test.csv')
    
    # Define column names from your original multi-class script
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
    
    if target_col not in train_df.columns:
        print(f"Error: Target column '{target_col}' not found in training data.")
        return
    
    # Separate target and features
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col].astype(int) # Ensure target is integer

    # Drop ID from features
    if id_col in X.columns:
        X = X.drop(columns=[id_col])

    # Prepare test data for final submission
    test_ids = test_df[id_col] if id_col in test_df.columns else test_df.index
    X_test_submit = test_df.drop(columns=[id_col], errors='ignore')

    # Ensure the test columns match training
    # This is a safe check, but the preprocessing pipeline handles alignment
    # based on the columns it sees in the fit step.
    X_test_submit = X_test_submit[X.columns]

    # ---------------------------------------------------------
    # 4. TRAIN/VALIDATION SPLIT
    # ---------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # stratify=y is good practice for classification tasks, especially multi-class

    print(f"Training shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")

    # ---------------------------------------------------------
    # 5. PREPROCESSING + MODEL
    # ---------------------------------------------------------
    print("\nTraining Multi-Class Decision Tree...")

    # Detect categorical + numeric columns
    # We detect all columns first, then ensure they are separated
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()

    # Preprocessing pipeline
    # NOTE: Added SimpleImputer for numeric columns to handle potential NaNs
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            # Impute missing values in numeric features with the median
            ("num", SimpleImputer(strategy='median'), numeric_cols), 
        ],
        # Ensure all columns are accounted for; drop='passthrough' is also an option
        remainder='passthrough' 
    )

    # Final model pipeline
    dt_model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            # You can tune max_depth or other parameters here
            ("tree", DecisionTreeClassifier(max_depth=10, random_state=42)),
        ]
    )
    
    # 

    dt_model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 6. EVALUATION
    # ---------------------------------------------------------
    print("\nEvaluating on Validation Set:")
    val_preds = dt_model.predict(X_val)

    acc = accuracy_score(y_val, val_preds)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    # Use target_names if your target categories are meaningful labels
    print(classification_report(y_val, val_preds, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, val_preds))

    # ---------------------------------------------------------
    # 7. GENERATE SUBMISSION
    # ---------------------------------------------------------
    print("\nGenerating predictions for Test set...")

    test_preds = dt_model.predict(X_test_submit)

    submission = pd.DataFrame({
        id_col: test_ids,
        target_col: test_preds.astype(int)
    })

    output_file = os.path.join(current_dir, 'submission_decision_tree.csv')
    submission.to_csv(output_file, index=False)

    print(f"Submission file saved to: {output_file}")


if __name__ == "__main__":
    run_multi_class_decision_tree()