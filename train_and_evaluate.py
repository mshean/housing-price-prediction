import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data(filepath):
    """Load the dataset and display basic info."""
    data = pd.read_csv(filepath)
    print("First 5 rows:")
    print(data.head())
    print("\nDataset info:")
    print(data.info())
    print("\nSummary statistics:")
    print(data.describe())
    return data

def prepare_data(data):
    """Prepare data for model training."""
    # Separate features and target
    X = data.drop('median_house_value', axis=1)
    y = data['median_house_value']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler to disk
    joblib.dump(scaler, "scaler.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train a model and evaluate its performance."""
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation:")
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R²):", r2)

    # Save model to disk
    joblib.dump(model, "linear_regression_model.pkl")

def main():
    # Step 1: Load data
    filepath = os.path.join(os.path.expanduser("~"), "Downloads", "cleaned_housing.csv")
    data = load_data(filepath)

    # Step 2: Prepare data for training
    X_train, X_test, y_train, y_test = prepare_data(data)

    # Step 3: Train and evaluate model
    train_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()