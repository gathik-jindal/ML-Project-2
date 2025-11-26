import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# --- Load Data and Preprocessing (same as before) ---
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df['spend_category'] = train_df['spend_category'].astype(int)
test_trip_ids = test_df['trip_id']
y_train = train_df['spend_category']
X_train = train_df.drop(['trip_id', 'spend_category'], axis=1)
X_test = test_df.drop('trip_id', axis=1)

combined_df = pd.concat([X_train, X_test], ignore_index=True)
categorical_cols = combined_df.select_dtypes(include=['object']).columns
combined_df_encoded = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=False)
X_train_encoded = combined_df_encoded.iloc[:len(X_train)]
X_test_encoded = combined_df_encoded.iloc[len(X_train):]
# ---------------------------------------------------

# --- Random Forest Model with Hyperparameter Tuning ---
rf = RandomForestClassifier(random_state=42)

# Define the parameter grid for tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# Use GridSearchCV for cross-validated hyperparameter tuning
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train_encoded, y_train)

# Get the best model
best_rf_model = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")

# Make predictions on the test data using the best model
test_predictions = best_rf_model.predict(X_test_encoded)

# --- Create Submission File ---
submission_df = pd.DataFrame({
    'trip_id': test_trip_ids,
    'spend_category': test_predictions.astype(int)
})

# Save the submission file
submission_df.to_csv('submission_random_forest.csv', index=False)