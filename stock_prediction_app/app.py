import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
import numpy as np


# Load the trained model and data
def load_model_and_data():
    model = joblib.load("sp500_model.pkl")
    sp500 = yf.Ticker("^GSPC").history(period="max").loc["1990-01-01":]
    return model, sp500

# Calculate indicators (reuse from training)
def calculate_indicators(data):
    data["SMA_10"] = data["Close"].rolling(window=10).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["EMA_10"] = data["Close"].ewm(span=10, adjust=False).mean()
    data["RSI"] = calculate_rsi(data["Close"], window=14)
    data["MACD"] = data["Close"].ewm(span=12, adjust=False).mean() - data["Close"].ewm(span=26, adjust=False).mean()
    data["ATR"] = data["High"] - data["Low"]
    data["ROC"] = data["Close"].pct_change(periods=5) * 100
    data["OBV"] = calculate_obv(data)
    return data

def calculate_rsi(series, window):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_obv(data):
    obv = (np.sign(data["Close"].diff()) * data["Volume"]).fillna(0).cumsum()
    return obv

# Streamlit app
st.set_page_config(page_title="S&P 500 Prediction", layout="wide", page_icon="📈")
st.title("📊 S&P 500 Prediction App")

# Load model and data
model, sp500 = load_model_and_data()
sp500 = calculate_indicators(sp500)
latest_data = sp500.iloc[-1].copy()

# Sidebar
st.sidebar.header("📋 Model Information")
st.sidebar.markdown(
    """
    This application predicts **S&P 500**'s movement for the next day based on technical indicators:
    
    - **SMA (Simple Moving Average)**
    - **EMA (Exponential Moving Average)**
    - **RSI (Relative Strength Index)**
    - **MACD (Moving Average Convergence Divergence)**
    - **ATR (Average True Range)**
    - **ROC (Rate of Change)**
    - **OBV (On-Balance Volume)**
    
    💡 **Prediction**: The app predicts whether the S&P 500 will go **UP** or **DOWN** tomorrow.
    """
)

# Display latest data
st.subheader("📅 Latest Market Data")
latest_df = latest_data[["Open", "High", "Low", "Close", "Volume"]].to_frame().T
latest_df.index = ["Latest"]
st.table(latest_df)

# Predict
st.subheader("📈 Prediction for Tomorrow")
predictors = ["SMA_10", "SMA_50", "EMA_10", "RSI", "MACD", "ATR", "ROC", "OBV"]
prediction = model.predict(latest_data[predictors].values.reshape(1, -1))

if prediction[0] == 1:
    st.success("The S&P 500 is predicted to go **UP** tomorrow. 📈")
else:
    st.error("The S&P 500 is predicted to go **DOWN** tomorrow. 📉")

# Visualization section
st.subheader("📊 Market Trends and Technical Indicators")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Closing Price (Last 1 Year)")
    st.line_chart(sp500["Close"][-365:])

    st.markdown("### Volume (Last 1 Year)")
    st.bar_chart(sp500["Volume"][-365:])

with col2:
    st.markdown("### RSI (Relative Strength Index)")
    plt.figure(figsize=(10, 5))
    plt.plot(sp500["RSI"][-365:], label="RSI", color="orange")
    plt.axhline(70, color="red", linestyle="--", label="Overbought (70)")
    plt.axhline(30, color="green", linestyle="--", label="Oversold (30)")
    plt.legend(loc="upper left")
    st.pyplot(plt)

    st.markdown("### SMA & EMA (Last 1 Year)")
    plt.figure(figsize=(10, 5))
    plt.plot(sp500["Close"][-365:], label="Close", color="blue")
    plt.plot(sp500["SMA_10"][-365:], label="SMA_10", color="orange")
    plt.plot(sp500["EMA_10"][-365:], label="EMA_10", color="green")
    plt.legend(loc="upper left")
    st.pyplot(plt)

# Show historical data
st.subheader("🕒 Historical Market Data")
if st.checkbox("Show Raw Data"):
    st.write(sp500.tail(10))
