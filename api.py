from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = tf.keras.models.load_model('currency_exchange_model.h5')

# Load the scaler
scaler = MinMaxScaler(feature_range=(0, 1))
# Example: Fit the scaler on some data (replace with your actual data)
# scaler.fit(data['INDIA - INDIAN RUPEE/US$'].values.reshape(-1, 1))

# Initialize Flask app
app = Flask(__name__)

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    input_data = request.json['data']
    input_data = np.array(input_data).reshape(-1, 1)
    
    # Normalize the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Reshape for LSTM input (batch_size, seq_length, num_features)
    input_data_scaled = input_data_scaled.reshape((1, seq_length, 1))
    
    # Make predictions
    predictions_scaled = model.predict(input_data_scaled)
    
    # Inverse transform the predictions to original scale
    predictions_original = scaler.inverse_transform(predictions_scaled)
    
    # Return predictions as JSON
    return jsonify({'predictions': predictions_original.tolist()})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)