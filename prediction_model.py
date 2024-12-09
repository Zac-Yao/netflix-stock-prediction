import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data
train_data = pd.read_csv('C:/Users/clone/Desktop/Final Results/Input Datas/Netflix_train.csv')
test_data = pd.read_csv('C:/Users/clone/Desktop/Final Results/Input Datas/Netflix_test.csv')

# Convert date to datetime
train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])

# Sort data by date
train_data = train_data.sort_values('date')
test_data = test_data.sort_values('date')

def train_and_evaluate_model(output_dir):
    # Use the same feature set as the first code
    features = ['open', 'high', 'low', 'close', 'volume', 'rel_sent','rel_sent_score']
    target = 'close'

    # Prepare the data
    X_train = train_data[features].values
    y_train = train_data[target].values
    X_test = test_data[features].values
    y_test = test_data[target].values

    def create_dataset(X, y, time_step=60):
        X_out, y_out = [], []
        for i in range(len(X) - time_step):
            X_out.append(X[i:i+time_step])
            y_out.append(y[i+time_step])
        return np.array(X_out), np.array(y_out)

    # Create sequences
    X_train, y_train = create_dataset(X_train, y_train)
    X_test, y_test = create_dataset(X_test, y_test)

    # Print shapes to verify
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # Define the LSTM model
    model = Sequential([
        LSTM(units=400, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(units=400, return_sequences=False),
        Dropout(0.2),
        Dense(units=50, activation='relu'),
        Dense(units=1)
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    class RMSECallback(Callback):
        def __init__(self, X_test, y_test):
            self.X_test = X_test
            self.y_test = y_test
            self.rmse_scores = []

        def on_epoch_end(self, epoch, logs=None):
            y_pred = self.model.predict(self.X_test)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            self.rmse_scores.append(rmse)
            print(f'\nEpoch {epoch+1} - RMSE: {rmse:.4f}')

    # Create an instance of RMSECallback
    rmse_callback = RMSECallback(X_test, y_test)

    # Train the model (same epochs and batch_size as the first code)
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[rmse_callback],
        verbose=1
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)




# Save train predictions
    train_predictions = model.predict(X_train)
    train_actual = y_train
    train_dates = train_data['date'][60:60+len(train_actual)]
    train_results_df = pd.DataFrame({
        'Date': train_dates,
        'Actual': train_actual,
        'Predicted': train_predictions.flatten()
    })
    train_results_df.to_csv(os.path.join(output_dir, 'train_predictions.csv'), index=False)

    # Save test predictions
    test_predictions = model.predict(X_test)
    test_actual = y_test
    test_dates = test_data['date'][60:60+len(test_actual)]
    test_results_df = pd.DataFrame({
        'Date': test_dates,
        'Actual': test_actual,
        'Predicted': test_predictions.flatten()
    })
    test_results_df.to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)
    
    
    
    
    
    # Plot loss and RMSE during training
    plt.figure(figsize=(12, 10))

    # Loss subplot
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)

    # RMSE subplot
    plt.subplot(2, 1, 2)
    plt.plot(rmse_callback.rmse_scores, label='RMSE')
    plt.title('RMSE During Training')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_rmse_training.png'))
    plt.close()

    # Predict stock prices
    predicted_prices = model.predict(X_test)
    real_prices = y_test

    # Function to predict future prices (same as the first code)
    def predict_future_with_sliding_window(model, data, last_date, days=60):
        predictions = []
        prediction_dates = []
        current_batch = data[-1:].reshape((1, 60, 7))  # Use the last sequence from data

        for i in range(days):
            predicted_price = model.predict(current_batch)[0][0]
            predictions.append(predicted_price)
            next_date = last_date + pd.Timedelta(days=i+1)
            prediction_dates.append(next_date)
            
            # Prepare next batch (assume other features remain constant)
            next_batch = np.roll(current_batch, -1, axis=1)
            next_batch[0, -1, 3] = predicted_price  # Update the 'close' price (index 3)
            current_batch = next_batch

        return np.array(predictions), prediction_dates

    # Predict future stock prices
    last_date = test_data['date'].iloc[-1]
    future_predictions, future_dates = predict_future_with_sliding_window(model, X_test, last_date, days=60)

    # Plot historical, predicted, and future prices
    plt.figure(figsize=(15, 6))

    # Historical closing price (training data)
    plt.plot(train_data['date'], train_data['close'], label='Historical Closing Price (Train)', color='blue')

    # Historical closing price (test data)
    plt.plot(test_data['date'], test_data['close'], label='Historical Closing Price (Test)', color='blue')

    # Predicted closing price (for the test period)
    plt.plot(test_data['date'][60:60+len(predicted_prices)], predicted_prices, label='Predicted Closing Price', color='orange')

    # Future price predictions
    plt.plot(future_dates, future_predictions, label='Future Price Predictions', color='green')

    # Add a vertical line to show the train/test split
    plt.axvline(x=train_data['date'].max(), color='red', linestyle='--', label='Train/Test Split')

    plt.title('Netflix Stock Price Prediction - Historical, Predicted, and Future (60 days)')
    plt.xlabel('Date')
    plt.ylabel('Price (Normalized)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'historical_predicted_future_prices.png'))
    plt.close()

    # Predict the next day's price
    last_sequence = X_test[-1:].reshape((1, 60, 7))
    next_day_prediction = model.predict(last_sequence)[0][0]

    print(f"Predicted normalized price for the next day after {last_date.strftime('%Y-%m-%d')}: {next_day_prediction:.4f}")

    # Calculate performance metrics (only for the test period)
    test_mse = mean_squared_error(y_test, predicted_prices)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, predicted_prices)
    test_r2 = r2_score(y_test, predicted_prices)

    # Print and save final performance metrics
    performance_metrics = {
        "MSE": test_mse,
        "RMSE": test_rmse,
        "MAE": test_mae,
        "R2 Score": test_r2
    }

    metrics_text = "\n".join([f"{key}: {value:.4f}" for key, value in performance_metrics.items()])
    print("\nFinal Model Performance (Test Data):")
    print(metrics_text)

    # Save performance metrics to a text file
    with open(os.path.join(output_dir, 'performance_metrics.txt'), 'w') as f:
        f.write(metrics_text)

    # Calculate RMSE for each prediction
    rmse_values = []
    for i in range(len(real_prices)):
        rmse = np.sqrt(mean_squared_error([real_prices[i]], [predicted_prices[i]]))
        rmse_values.append(rmse)

    # Get the dates corresponding to the predictions
    prediction_dates = test_data['date'][60:60+len(real_prices)]

    # Create a DataFrame with dates and RMSE values
    rmse_df = pd.DataFrame({'Date': prediction_dates, 'RMSE': rmse_values})

    # Plot RMSE vs Date
    plt.figure(figsize=(12, 6))
    plt.plot(rmse_df['Date'], rmse_df['RMSE'])
    plt.title('RMSE vs Date for Netflix Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rmse_vs_date.png'))
    plt.close()

    # Calculate and print average RMSE
    average_rmse = np.mean(rmse_values)
    print(f"Average RMSE: {average_rmse:.4f}")

    # Save average RMSE to the metrics file
    with open(os.path.join(output_dir, 'performance_metrics.txt'), 'a') as f:
        f.write(f"\nAverage RMSE: {average_rmse:.4f}")

# Run the model
output_dir = 'outputs/netflix_stock_prediction'
train_and_evaluate_model(output_dir)