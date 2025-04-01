import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Configuration
SEQ_LENGTH = 5000  # Must match your training sequence length
MODEL_PATH = 'currency_exchange_model.h5'
SCALER_PATH = 'scaler.save'

@st.cache_data
def load_test_data():
    try:
        # Load model and scaler
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        # Generate test data - replace with your actual data loading
        test_days = SEQ_LENGTH + 100  # Generate extra days for sequences
        test_data = np.linspace(0.1, 0.9, test_days).reshape(-1, 1)
        
        # Scale the test data (2D input)
        test_data_scaled = scaler.transform(test_data)
        
        # Create sequences (results in 3D array)
        X_test = np.array([test_data_scaled[i:i+SEQ_LENGTH] 
                          for i in range(len(test_data_scaled) - SEQ_LENGTH)])
        
        # Get corresponding actual values (2D)
        y_test = test_data_scaled[SEQ_LENGTH:].reshape(-1, 1)

        # Make predictions
        predictions_scaled = model.predict(X_test)
        
        # Inverse transform requires 2D input
        predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
        y_test_original = scaler.inverse_transform(y_test)
        
        return pd.DataFrame({
            'Actual': y_test_original.flatten(),
            'Predicted': predictions.flatten()
        })

    except Exception as e:
        st.error(f"Error loading test data: {str(e)}")
        return pd.DataFrame()

# Streamlit App
st.title('Currency Exchange Rate Predictor')

if st.button('Generate Predictions'):
    predictions = load_test_data()
    
    if not predictions.empty:
        st.line_chart(predictions[['Actual', 'Predicted']])
        st.dataframe(predictions)
    else:
        st.warning("No predictions generated. Check error messages.")