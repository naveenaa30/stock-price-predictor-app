import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load pre-trained model
model = load_model('model/model.keras')  # Make sure your trained model is saved here

# Function to fetch stock data
def fetch_stock_data(ticker):
    data = yf.download(ticker, start='2020-01-01', end='2025-03-30', interval='1d')
    return data['Close'].values.reshape(-1, 1)

# Function to preprocess and predict
def predict_next_30_days(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    sequence_length = 60
    last_sequence = scaled_data[-sequence_length:]
    predictions = []

    for _ in range(30):
        pred = model.predict(last_sequence.reshape(1, sequence_length, 1))[0][0]
        predictions.append(pred)
        last_sequence = np.append(last_sequence[1:], pred)
    
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Streamlit app
st.title("Stock Price Predictor")
st.write("Predict the next 30 days of stock prices using an LSTM model.")

# User input
ticker = st.text_input("Enter stock ticker (e.g., AAPL for Apple):", "")

if ticker:
    st.write(f"Fetching data for {ticker}...")
    try:
        # Fetch data and display it
        data = fetch_stock_data(ticker)
        st.line_chart(data)

        # Predict and display results
        st.write("Predicting next 30 days of prices...")
        predictions = predict_next_30_days(data)
        st.write("Predicted Prices for Next 30 Days:")
        st.write(predictions)

        # Plot the predicted prices
        fig, ax = plt.subplots()
        ax.plot(range(1, 31), predictions, label="Predicted Prices")
        ax.set_xlabel("Day")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"An error occurred: {e}")
