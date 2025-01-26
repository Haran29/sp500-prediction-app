# S&P 500 Prediction App

This repository contains a machine learning-based app that predicts whether the S&P 500 index will go up or down the next day using historical market data and technical indicators.

### Features:
- **Prediction Model**: Predicts the S&P 500's next-day movement (up or down) based on indicators.
- **Technical Indicators Used**:
  - Simple Moving Average (SMA)
  - Exponential Moving Average (EMA)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Average True Range (ATR)
  - Rate of Change (ROC)
  - On-Balance Volume (OBV)
- **Visualization**: Provides interactive charts to display historical market trends.

### Technologies Used:
- **Python**
- **Streamlit** (for the web app)
- **Scikit-learn** (for machine learning model)
- **yfinance** (for fetching historical market data)
- **Joblib** (for saving and loading the model)
- **Matplotlib** (for data visualization)
- **Pandas** and **NumPy** (for data processing)

### How It Works:
1. The app fetches historical data for the S&P 500 from Yahoo Finance.
2. It calculates several technical indicators based on this data.
3. A RandomForestClassifier model is trained on the indicators and used to predict the next-day movement.
4. The app displays the prediction and provides interactive charts for market trends.

### Running the App Locally:
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/sp500-prediction-app.git
    ```
2. Navigate to the project directory:
    ```bash
    cd sp500-prediction-app
    ```
3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the app:
    ```bash
    streamlit run app.py
    ```
