import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Fetch S&P 500 historical data
def fetch_data():
    sp500 = yf.Ticker("^GSPC").history(period="max")
    sp500 = sp500.loc["1990-01-01":].copy()  # Use data from 1990 onwards
    return sp500

# Calculate technical indicators
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

# Calculate RSI
def calculate_rsi(series, window):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Calculate OBV
def calculate_obv(data):
    obv = (np.sign(data["Close"].diff()) * data["Volume"]).fillna(0).cumsum()
    return obv

# Prepare data for training
def prepare_data(data):
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    data = data.dropna()
    predictors = ["SMA_10", "SMA_50", "EMA_10", "RSI", "MACD", "ATR", "ROC", "OBV"]
    return data, predictors

# Train the model and save it
def train_model(data, predictors):
    X = data[predictors]
    y = data["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    joblib.dump(model, "sp500_model.pkl")  # Save the model
    print("Model saved as sp500_model.pkl")

# Main script
if __name__ == "__main__":
    sp500 = fetch_data()
    sp500 = calculate_indicators(sp500)
    sp500, predictors = prepare_data(sp500)
    train_model(sp500, predictors)
