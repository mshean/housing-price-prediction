import requests

# Define the input data
input_data = {
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
    "ocean_proximity_INLAND": 0,
    "ocean_proximity_ISLAND": 0,
    "ocean_proximity_NEAR BAY": 1,
    "ocean_proximity_NEAR OCEAN": 0
}

# Send a POST request to the API
response = requests.post('http://127.0.0.1:5000/predict', json=input_data, verify=False)

# Print the prediction
print(response.json())