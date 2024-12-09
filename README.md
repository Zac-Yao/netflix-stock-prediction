# Netflix Stock Price Prediction

This project implements a stock price prediction model using LSTM neural networks, using both historical stock data and relative sentiment scores. The model predicts Netflix stock prices and evaluates its performance on training and testing datasets. Additionally, it forecasts future stock prices based on the most recent data.

## Features
- **Historical Prediction:** Predict stock prices based on past data.
- **Future Forecasting:** Predict stock prices for the next 60 days using a sliding window approach.
- **Performance Metrics:** Evaluate the model using MSE, RMSE, MAE, and R² score.
- **Visualizations:** Generate loss, RMSE, and predicted vs. actual stock prices graphs.

## Requirements
- Python 3.8 or higher
- Libraries: TensorFlow, NumPy, Pandas, Matplotlib, Scikit-learn

## Files
- `prediction_model.py`: The main script that trains, evaluates, and visualizes the LSTM model's performance.
- Input data: `Netflix_train.csv` and `Netflix_test.csv` (historical stock data and sentiment scores).
- Output files:
  - `train_predictions.csv` and `test_predictions.csv`: Actual vs. predicted stock prices for training and testing datasets.
  - Graphs:
    - `loss_rmse_training.png`: Training loss and RMSE during training.
    - `historical_predicted_future_prices.png`: Predicted stock prices for historical and future data.
    - `rmse_vs_date.png`: RMSE values over time.
