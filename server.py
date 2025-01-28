from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the model and scaler
model = joblib.load('linear_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    input_data = request.json

    # Convert input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Scale the input data
    input_scaled = scaler.transform(input_df)

    # Make a prediction
    prediction = model.predict(input_scaled)

    # Return the prediction as JSON
    return jsonify({'predicted_house_price': prediction[0]})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)