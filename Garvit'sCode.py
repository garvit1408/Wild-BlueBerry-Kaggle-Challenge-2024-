import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load the dataset
train_file_path = 'train.csv'
test_file = 'test.csv'  
test_data = pd.read_csv(test_file)
data = pd.read_csv(train_file_path)

# Prepare the data by removing unnecessary columns
X = data.drop(columns=['id', 'Row#', 'yield'])  # Features
y = data['yield']  # Target variable

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Step 4: Initialize Gradient Boosting Regressor
gbr = GradientBoostingRegressor(
    n_estimators=604,
    learning_rate=0.0321,
    max_depth=6,
    min_samples_split=4,
    min_samples_leaf=5,
    subsample=0.8,
    loss='absolute_error',
    random_state=42
)

# Step 5: Fit the model on the scaled training data
gbr.fit(X_train, y_train)

# Step 6: Make predictions on the validation set
gbr_pred_val = gbr.predict(X_val)

# Step 7: Calculate and print the Mean Absolute Error (MAE)
mae_gbr = mean_absolute_error(y_val, gbr_pred_val)
print(f"Mean Absolute Error with Gradient Boosting: {mae_gbr}")

# Apply same preprocessing to test data
X_test = test_data.drop(columns=['id', 'Row#'])  # Adjust based on the actual columns in the test file
X_test = X_test[X.columns]  # Ensure feature consistency with training data
X_test = scaler.transform(X_test)  # Scale test data using the fitted scaler

# Predict on the test data using the Gradient Boosting model
gbr_pred_test = gbr.predict(X_test)

# Save predictions to a CSV file in the required format
submission = pd.DataFrame({
    'id': test_data['id'],
    'yield': gbr_pred_test
})
submission.to_csv('submission_1.csv', index=False)
print("Predictions saved to submission.csv")