import tkinter as tk
from tkinter import Toplevel, Text, Scrollbar, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageSequence
import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
import wrds
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from tensorflow.keras.layers import Input # type: ignore
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter

def update_placeholder(event):
    if ticker_entry.get() == 'Enter ticker here':
        ticker_entry.delete(0, tk.END)

def restore_placeholder(event):
    if ticker_entry.get() == '':
        ticker_entry.insert(0, 'Enter ticker here')

def fetch_stock_data(ticker):
    db = wrds.Connection(wrds_username='username')
    sql_query = f"""
        SELECT 
            b.date,
            d.permno,
            d.htick AS ticker,
            b.ret AS return
        FROM 
            crsp.dsf AS b
        JOIN
            crsp_a_stock.dsfhdr AS d ON b.permno = d.permno 
        WHERE
            b.date BETWEEN '2018-01-01' AND '2023-12-01'
            AND d.htick = '{ticker}'
        """
    try:
            data = db.raw_sql(sql_query)
            dates = pd.to_datetime(data['date']).tolist()  # Or however you extract the dates
    finally:
            db.close()
    return data, dates

def calculate_historical_volatility(data):
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    # Scale returns before calculating volatility
    data['return_scaled'] = data['return'] * 100  # Scale by 100 as suggested
    data['volatility'] = data['return_scaled'].rolling(window=30).std() * np.sqrt(252)
    return data.dropna(subset=['volatility'])


def fit_garch_model(data, results_text):
    returns = data['return'].dropna()
    model = arch_model(returns, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp='off')

    # Update the ScrolledText widget with the GARCH model summary
    results_text.config(state=tk.NORMAL)  # Ensure the widget is editable
    results_text.delete('1.0', tk.END)  # Clear existing text
    results_text.insert(tk.END, str(model_fit.summary()))  # Insert model summary
    results_text.config(state=tk.DISABLED)  # Disable editing to prevent user changes

def prepare_and_train_lstm(data):
    # Split the data into training and testing datasets
    train_size = int(len(data) * 0.95)  # Using 95% of data for training
    train_data = data['volatility'].iloc[:train_size]
    test_data = data['volatility'].iloc[train_size:]

    # Fit the scaler on the training data only
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))  # Fit and transform training data

    sequence_length = 60
    x_train, y_train = [], []

    # Prepare training data using the scaled values
    for i in range(sequence_length, len(train_scaled)):
        x_train.append(train_scaled[i - sequence_length:i, 0])
        y_train.append(train_scaled[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build and train the LSTM model
    model = Sequential([
        Input(shape=(sequence_length, 1)),
        LSTM(100, return_sequences=True),
        Dropout(0.1),
        LSTM(100, return_sequences=False),
        Dropout(0.1),
        Dense(50),
        Dense(1, activation='relu')
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=1)  # Increased epochs

    # Prepare the full dataset for prediction, scaled with the same scaler
    full_scaled_data = scaler.transform(data['volatility'].values.reshape(-1, 1))

    return model, scaler, full_scaled_data, train_size

# Use this formatter to simplify the x-axis dates.
def format_date(index, pos, dates):
    if index < 0 or index >= len(dates):
        return ''
    return pd.to_datetime(str(dates[index])).strftime('%Y-%m-%d')

def make_predictions_and_plot(model, scaler, scaled_data, train_data_len, results_text, fig, ax, dates, number_of_future_points):
    sequence_length = 60
    last_input_sequence = scaled_data[train_data_len - sequence_length:train_data_len, :].reshape(1, sequence_length, 1)

    # Generate future predictions
    predictions = [model.predict(last_input_sequence)[0, 0]]  # Start with the first prediction
    for _ in range(1, number_of_future_points):
        last_prediction = np.array([[predictions[-1]]]).reshape(1, 1, 1)
        last_input_sequence = np.append(last_input_sequence[:, 1:, :], last_prediction, axis=1)
        predictions.append(model.predict(last_input_sequence)[0, 0])

    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)

    # Generate future dates for predictions; +1 because it includes the starting point
    future_dates = pd.date_range(start=dates[train_data_len - 1], periods=number_of_future_points + 1, freq='B')

    # Plot historical data
    historical_volatility = scaler.inverse_transform(scaled_data[:train_data_len]).flatten()
    ax.plot(dates[:train_data_len], historical_volatility, label='Historical Volatility', color='blue')

    # Plot future predicted volatility
    # Ensure the future_dates and the data array for plotting have the same length
    ax.plot(future_dates, np.concatenate([[historical_volatility[-1]], predictions.flatten()]), label='Future Predicted Volatility', color='red')

    # Setting y-axis format and x-axis formatting
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    ax.legend()
    canvas.draw()

def on_predict_button_clicked():
    ticker = ticker_entry.get().strip().upper()
    if not ticker or ticker == 'Enter ticker here':
        messagebox.showinfo("Input needed", "Please enter a stock ticker.")
        return
    
    # Fetch and prepare data
    data, dates = fetch_stock_data(ticker)  # Get stock data
    volatility_data = calculate_historical_volatility(data)  # Calculate volatility
    fit_garch_model(volatility_data, results_text)  # Fit GARCH model and display results

    # Prepare and train LSTM model
    model, scaler, scaled_volatility, train_data_len = prepare_and_train_lstm(volatility_data)
    
    # Set the number of future points you want to predict
    number_of_future_points = 30

    # Convert date strings to datetime if necessary
    dates = pd.to_datetime(dates)

    # Plot predictions
    make_predictions_and_plot(model, scaler, scaled_volatility, train_data_len, results_text, fig, ax, dates, number_of_future_points)


# The initialization and widget setup appears correctly configured.
root = tk.Tk()
root.title("Stock Volatility Predictor")
root.geometry('1024x768')  # Set a larger initial size for the window

ticker_entry = tk.Entry(root, width=50)
ticker_entry.insert(0, 'Enter ticker here')
ticker_entry.bind("<FocusIn>", update_placeholder)
ticker_entry.bind("<FocusOut>", restore_placeholder)
ticker_entry.pack(pady=20)

predict_button = tk.Button(root, text="Predict", command=on_predict_button_clicked)
predict_button.pack(pady=10)

results_text = scrolledtext.ScrolledText(root, height=10)
results_text.pack(pady=20)

fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

root.mainloop()
