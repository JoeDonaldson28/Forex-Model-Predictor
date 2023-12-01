#%% Forex model predictor Updated 2.0
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Import necessary libraries
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf

#%% Define the currency pairs and API endpoint
currency_pairs = ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'USDCHF=X', 'AUDUSD=X', 'USDCAD=X']

# Define the start and end date for the time series data
start_date = '2000-11-7'
end_date = '2023-11-7'

# Function to fetch currency data from the API for a specific period
def fetch_currency_data(pair):
    currency_data = yf.download(pair, start=start_date, end=end_date)
    return currency_data

# Create a dictionary to store data
data = {}
future_predictions_dict_lstm = {}
future_predictions_dict_ann = {}

# Fetch data for each currency pair and store it in the 'data' dictionary
for pair in currency_pairs:
    data[pair] = fetch_currency_data(pair)

# Fetch interest rate data for a specific symbol (e.g., ^IRX for the 13-week Treasury bill rate)
interest_rate_data = fetch_currency_data('^IRX')

#%% Iterate through currency pairs
for pair, df in data.items():
    df.dropna(inplace=True)
    df['PercentChange'] = df['Close'].pct_change() * 100  # Calculate percent change in exchange rates
    df.dropna(inplace=True)

    X = df.drop(['Close', 'PercentChange'], axis=1)
    y = df['PercentChange']

    # Merge interest rate data with exchange rate data based on the 'Date' column
    df = pd.merge(df, interest_rate_data['Adj Close'], left_index=True, right_index=True, how='inner')

    X = df.drop(['Close', 'PercentChange'], axis=1)
    y = df['PercentChange']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

#%% ATR
# Calculate ATR for each currency pair
for pair, df in data.items():
    df.dropna(inplace=True)
    df['PercentChange'] = df['Close'].pct_change() * 100  # Calculate percent change in exchange rates
    df.dropna(inplace=True)

    # Calculate True Range
    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
    df['TrueRange'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)

    # Calculate ATR
    n = 14  # You can adjust the smoothing period as needed
    df['ATR'] = df['TrueRange'].rolling(window=n).mean()

    # Drop intermediate columns used for ATR calculation
    df.drop(['High-Low', 'High-PrevClose', 'Low-PrevClose', 'TrueRange'], axis=1, inplace=True)

    #%% Linear Regression
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_linear = model.predict(X_test)
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    rmse_linear = np.sqrt(mse_linear)
    mae_linear = mean_absolute_error(y_test, y_pred_linear)
    r2_linear = r2_score(y_test, y_pred_linear)

    #%% Visualize Linear Regression
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label="Actual Percent Change", color='blue')
    plt.plot(y_test.index, y_pred_linear, label="Predicted (Linear Regression)", linestyle='--', color='red')
    plt.xlabel('Date')
    plt.ylabel('Percent Change')
    plt.legend()
    plt.title(f'Percent Change Predictions for {pair} (Linear Regression)')
    plt.grid()
    plt.show()

    #%% LSTM Data Preparation
    prediction_days = 250
    X_lstm_future = []
    y_lstm_future = []
    for i in range(len(X_test) - prediction_days + 1):
        X_lstm_future.append(X_test.iloc[i:i + prediction_days].values)
        y_lstm_future.append(y_test.iloc[i + prediction_days - 1])

    X_lstm_future = np.array(X_lstm_future)
    y_lstm_future = np.array(y_lstm_future)

    # LSTM Model
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, input_shape=(X_lstm_future.shape[1], X_lstm_future.shape[2])))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_lstm_future, y_lstm_future, epochs=50, verbose=0)

    # LSTM Predictions
    y_lstm_future_pred = lstm_model.predict(X_lstm_future)
    mse_lstm = mean_squared_error(y_lstm_future, y_lstm_future_pred)
    rmse_lstm = np.sqrt(mse_lstm)
    mae_lstm = mean_absolute_error(y_lstm_future, y_lstm_future_pred)
    r2_lstm = r2_score(y_lstm_future, y_lstm_future_pred)

    # Store predictions in the dictionaries
    future_predictions_dict_lstm[pair] = {
        'actual': y_lstm_future,
        'predicted_lstm': y_lstm_future_pred.flatten()
    }

    #%% ANN Model
    ann_model = Sequential()
    ann_model.add(Dense(50, activation='relu', input_dim=X_train.shape[1]))
    ann_model.add(Dense(1))
    ann_model.compile(optimizer='adam', loss='mean_squared_error')
    ann_model.fit(X_train, y_train, epochs=50, verbose=0)
    y_ann_pred = ann_model.predict(X_test)
    mse_ann = mean_squared_error(y_test, y_ann_pred)
    rmse_ann = np.sqrt(mse_ann)
    mae_ann = mean_absolute_error(y_test, y_ann_pred)
    r2_ann = r2_score(y_test, y_ann_pred)

    # Store predictions in the dictionaries
    future_predictions_dict_ann[pair] = {
        'actual': y_test,
        'predicted_ann': y_ann_pred.flatten()
    }

    #%% ARIMA Forecasting
    result = adfuller(y)
    if result[1] > 0.05:
        y_diff = y.diff().dropna()
    else:
        y_diff = y.copy()

    # Fit an ARIMA model
    p = 1  # Order of AutoRegressive (AR) component
    d = 1  # Degree of differencing (I)
    q = 1  # Order of Moving Average (MA) component

    model = ARIMA(y_diff, order=(p, d, q))  # Create ARIMA model instance
    results = model.fit()  # Fit the ARIMA model

    # Forecast future values
    forecast_steps = 30  # Number of steps to forecast into the future
    forecast = results.forecast(steps=forecast_steps)

    #%% Visualize ARIMA Forecast
    plt.figure(figsize=(12, 6))
    plt.plot(y.index, y, label='Actual', color='blue')
    plt.plot(range(len(y), len(y) + forecast_steps), forecast, label='Forecast', color='red')
    plt.title(f'{pair} Exchange Rate Forecast (ARIMA)')
    plt.xlabel('Date')
    plt.ylabel('Percent Change')
    plt.legend()
    plt.grid(True)
    plt.show()

#%% Visualize predictions for each currency pair using LSTM model in separate graphs
# Visualize predictions for each currency pair using LSTM model in separate graphs
for pair, predictions in future_predictions_dict_lstm.items():
    actual = predictions['actual']
    predicted_lstm = predictions['predicted_lstm']

    # Convert actual to a pandas Series with the appropriate index
    actual = pd.Series(actual, index=y_test.index[-len(actual):])

    # Visualize predictions for the current currency pair in a separate graph
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual, label="Actual Percent Change", color='blue')
    plt.plot(actual.index, predicted_lstm, label="Predicted (LSTM)", linestyle='--', color='red')
    plt.xlabel('Date')
    plt.ylabel('Percent Change')
    plt.legend()
    plt.title(f'Percent Change Predictions for {pair} (LSTM)')
    plt.grid()
    plt.show()


#%% Visualize predictions for each currency pair using ANN model in separate graphs
for pair, predictions in future_predictions_dict_ann.items():
    actual = predictions['actual']
    predicted_ann = predictions['predicted_ann']

    #%% Visualize predictions for the current currency pair in a separate graph
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual, label="Actual Percent Change", color='blue')
    plt.plot(actual.index, predicted_ann, label="Predicted (ANN)", linestyle='--', color='green')
    plt.xlabel('Date')
    plt.ylabel('Percent Change')
    plt.legend()
    plt.title(f'Percent Change Predictions for {pair} (ANN)')
    plt.grid()
    plt.show()




#%% Evaluate Linear Regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Initialize dictionaries to store model evaluation metrics
model_metrics = {
    'Linear Regression': {},
    'LSTM': {},
    'ANN': {},
}

# Evaluate Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_linear = model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

model_metrics['Linear Regression']['MSE'] = mse_linear
model_metrics['Linear Regression']['RMSE'] = rmse_linear
model_metrics['Linear Regression']['MAE'] = mae_linear
model_metrics['Linear Regression']['R2'] = r2_linear

#%% Evaluate LSTM
y_lstm_future_pred = lstm_model.predict(X_lstm_future)
mse_lstm = mean_squared_error(y_lstm_future, y_lstm_future_pred)
rmse_lstm = np.sqrt(mse_lstm)
mae_lstm = mean_absolute_error(y_lstm_future, y_lstm_future_pred)
r2_lstm = r2_score(y_lstm_future, y_lstm_future_pred)

model_metrics['LSTM']['MSE'] = mse_lstm
model_metrics['LSTM']['RMSE'] = rmse_lstm
model_metrics['LSTM']['MAE'] = mae_lstm
model_metrics['LSTM']['R2'] = r2_lstm

#%% Evaluate ANN
y_ann_pred = ann_model.predict(X_test)
mse_ann = mean_squared_error(y_test, y_ann_pred)
rmse_ann = np.sqrt(mse_ann)
mae_ann = mean_absolute_error(y_test, y_ann_pred)
r2_ann = r2_score(y_test, y_ann_pred)

model_metrics['ANN']['MSE'] = mse_ann
model_metrics['ANN']['RMSE'] = rmse_ann
model_metrics['ANN']['MAE'] = mae_ann
model_metrics['ANN']['R2'] = r2_ann

#%% Print the evaluation metrics
for model_name, metrics in model_metrics.items():
    print(f'{model_name} Metrics:')
    print(f'MSE: {metrics.get("MSE", "N/A")}')
    print(f'RMSE: {metrics.get("RMSE", "N/A")}')
    print(f'MAE: {metrics.get("MAE", "N/A")}')
    print(f'R2: {metrics.get("R2", "N/A")}')
    print()

#%% Determine the best model based on a metric (e.g., lowest RMSE or highest R2)
best_model = min(model_metrics, key=lambda x: model_metrics[x]['RMSE'])
print(f'The best model for predicting percent change is: {best_model}')

# %%
