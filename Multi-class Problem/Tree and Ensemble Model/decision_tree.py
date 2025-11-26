import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- 1. Load Data ---
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# --- 2. Preprocessing and Feature Engineering ---
# Convert target variable to integer
train_df['spend_category'] = train_df['spend_category'].astype(int)

# Separate IDs and target
test_trip_ids = test_df['trip_id']
y_train = train_df['spend_category']

# Features to drop before combining (trip_id and the target)
X_train = train_df.drop(['trip_id', 'spend_category'], axis=1)
X_test = test_df.drop('trip_id', axis=1)

# Combine for consistent One-Hot Encoding
combined_df = pd.concat([X_train, X_test], ignore_index=True)

# Apply One-Hot Encoding to all categorical ('object') columns
categorical_cols = combined_df.select_dtypes(include=['object']).columns
combined_df_encoded = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=False)

# Split data back into training and testing sets
X_train_encoded = combined_df_encoded.iloc[:len(X_train)]
X_test_encoded = combined_df_encoded.iloc[len(X_train):]

# --- 3. Model Training ---
# Initialize and train the Decision Tree Classifier
# A simple, un-tuned model is used for this example.
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_encoded, y_train)

# --- 4. Prediction ---
test_predictions = dt_model.predict(X_test_encoded)

# --- 5. Create Submission File ---
submission_df = pd.DataFrame({
    'trip_id': test_trip_ids,
    'spend_category': test_predictions.astype(int)
})

# Save the submission
submission_df.to_csv('submission_decision_tree.csv', index=False)