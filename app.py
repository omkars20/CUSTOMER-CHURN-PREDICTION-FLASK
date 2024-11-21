from flask import Flask, request, jsonify
import joblib
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load models
logistic_model = joblib.load('models/logistic_model.pkl')
random_forest_model = joblib.load('models/random_forest_model.pkl')
xgboost_model = joblib.load('models/xgboost_model.pkl')
ann_model = tf.keras.models.load_model('models/ann_model.h5')

# Load the StandardScaler
scaler = joblib.load('models/scaler.pkl')

# Define the geography columns based on your `pd.get_dummies()` logic
geography_columns = ['Geography_Germany', 'Geography_Spain']

# Map of models
models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model,
    "XGBoost": xgboost_model,
    "ANN": ann_model
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the request data
        data = request.json
        selected_model = data["model"]  # Model selected from the dropdown
        raw_features = data["features"]  # Raw input features from the frontend

        # Encode Geography
        geography = raw_features['Geography']
        geography_encoded = [0, 0]  # Default [Geography_Germany, Geography_Spain]
        if geography == 'Germany':
            geography_encoded[0] = 1
        elif geography == 'Spain':
            geography_encoded[1] = 1
        # France is implicitly represented by [0, 0]

        # Extract and scale numerical features
        numerical_features = [
            raw_features['CreditScore'], raw_features['Age'], raw_features['Tenure'],
            raw_features['Balance'], raw_features['NumOfProducts'], raw_features['HasCrCard'],
            raw_features['IsActiveMember'], raw_features['EstimatedSalary']
        ]
        scaled_numerical_features = scaler.transform([numerical_features])

        # Combine scaled numerical features with encoded geography
        input_features = np.concatenate((scaled_numerical_features[0], geography_encoded))

        # Check if the selected model exists
        if selected_model not in models:
            return jsonify({"error": "Model not found"}), 400

        # Get the selected model
        model = models[selected_model]

        # Make predictions
        if selected_model == "ANN":
            prediction = model.predict([input_features]).tolist()
        else:
            prediction = model.predict([input_features]).tolist()

        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
