import datetime
import pandas as pd
import matplotlib.pyplot as plt

from utils import schwab_utils, tradeui_utils
from analysis import indicators

from sklearn.linear_model import LinearRegression
import numpy as np

import torch
import csv


ticker = 'MSFT'

def extract_graph_components(symbol):
    stock_data = schwab_utils.get_daily_price_history(symbol)

    datetimes, close_prices, volumes = zip(
        *[(
            datetime.datetime.fromtimestamp(data_point['datetime'] / 1000),
            data_point['close'],
            data_point['volume']
        ) for data_point in stock_data])

    # Calculate moving averages
    close_prices_series = pd.Series(close_prices, index=datetimes)
    ma20 = close_prices_series.rolling(window=20).mean()
    ma50 = close_prices_series.rolling(window=50).mean()
    ma200 = close_prices_series.rolling(window=200).mean()

    # Calculate MACD
    weekly_close_prices_series = close_prices_series.resample('W').last()
    ema12 = weekly_close_prices_series.ewm(span=12, adjust=False).mean()
    ema26 = weekly_close_prices_series.ewm(span=26, adjust=False).mean()
    weekly_datetimes = weekly_close_prices_series.index

    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    return datetimes, close_prices, volumes, ma20, ma50, ma200, weekly_datetimes, macd_line, signal_line, histogram

def plot(datetimes, close_prices, volumes, ma20, ma50, ma200, weekly_datetimes, macd_line, signal_line, histogram, indicators=None):
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3,1]})

    # Plot moving averages
    color = '#34A853'
    ax1.plot(datetimes, close_prices, color=color, linewidth=2)
    ax1.fill_between(datetimes, close_prices, color=color, alpha=0.2)
    ax1.set_xlabel('Date', fontsize=12, color='black')
    ax1.set_ylabel('Close Price', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.plot(datetimes, ma20, color='magenta', linewidth=1, label='MA20')
    ax1.plot(datetimes, ma50, color='blue', linewidth=1, label='MA50')
    ax1.plot(datetimes, ma200, color='red', linewidth=1, label='MA200')
    ax1.legend(loc='upper left')
    
    # Plot overlaid trading volumes
    ax2 = ax1.twinx()
    color = 'tab:gray'
    ax2.set_ylabel('Volume', fontsize=12, color='black')
    ax2.bar(datetimes, volumes, color=color, alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='black')

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Plot MACD
    ax3.plot(weekly_datetimes, macd_line, color='black', linewidth=1, label='MACD')
    ax3.plot(weekly_datetimes, signal_line, color='red', linewidth=1, label='Signal')
    ax3.bar(weekly_datetimes, histogram, color='green', alpha=0.5, width=1.0)
    ax3.axhline(0, color='grey', linewidth=0.8)
    ax3.legend(loc='upper left')
    ax3.set_ylabel('MACD', fontsize=12)

    # Annotate signals
    if indicators is not None:
        for indicator in indicators:
        	color = 'gold' if indicator.trend == 'BULLISH' else 'brown'
        	for date in indicator.dates:
        		closest_date_index = min(range(len(datetimes)), key=lambda i: abs(datetimes[i] - date))

        		price = close_prices[closest_date_index]

        		print(date, price)
        		ax1.plot(date, price, 'o', markersize=10, markerfacecolor='none', markeredgecolor=color, markeredgewidth=2, label=indicator.description)


    fig.tight_layout()
    plt.savefig('msft.png')
    #plt.show()

def bullish_crossover(weekly_datetimes, histogram):
	indicator = indicators.Indicator('BULLISH', 'Bullish Crossover')
	prev_h = histogram.iloc[0]
	for d, h in zip(weekly_datetimes, histogram):
		if prev_h < 0 and h > 0:
			indicator.add_date(d)
		prev_h = h
	return indicator


def bearish_crossover(weekly_datetimes, histogram):
	indicator = indicators.Indicator('BEARISH', 'Bearish Crossover')
	prev_h = histogram.iloc[0]
	for d, h in zip(weekly_datetimes, histogram):
		if prev_h > 0 and h < 0:
			indicator.add_date(d)
		prev_h = h
	return indicator

def zero_cross():
	pass

# Signal combining crossover + MACD raw value

def plot_log(datetimes, close_prices):

    X = np.arange(len(datetimes)).reshape(-1, 1)
    y_log = np.log(close_prices)
    model = LinearRegression()
    model.fit(X, y_log)

    y_pred = model.predict(X)
    print(y_log)
    print(y_pred)


    color = '#34A853'
    
    plt.plot(datetimes, y_log, color=color, linewidth=2)
    #plt.fill_between(datetimes, close_prices, color=color, alpha=0.2)
    plt.xlabel('Date', fontsize=12, color='black')
    plt.ylabel('Close Price', fontsize=12, color='black')

    plt.yscale('log')


    plt.plot(datetimes, y_pred, color='red')

    plt.show()

def build_dataset(symbol):
    stock_data = schwab_utils.get_daily_price_history(symbol)

    datetimes, close_prices, volumes = zip(
        *[(
            datetime.datetime.fromtimestamp(data_point['datetime'] / 1000),
            data_point['close'],
            data_point['volume']
        ) for data_point in stock_data])

    close_prices_series = pd.Series(close_prices, index=datetimes)
    ma20 = close_prices_series.rolling(window=20).mean()
    ma50 = close_prices_series.rolling(window=50).mean()
    ma200 = close_prices_series.rolling(window=200).mean()

    weekly_close_prices_series = close_prices_series.resample('W').last()
    ema12 = weekly_close_prices_series.ewm(span=12, adjust=False).mean()
    ema26 = weekly_close_prices_series.ewm(span=26, adjust=False).mean()
    weekly_datetimes = weekly_close_prices_series.index


    ma20 = tuple(ma20)
    ma50 = tuple(ma50)
    ma200 = tuple(ma200)
    print(type(ma20))

    tensor = torch.tensor([close_prices, volumes, ma20, ma50, ma200], dtype=torch.float32).T
    
    print(tensor[0])
    return tensor

    # with open("msft.csv", "w", newline="") as file:
    #     writer = csv.writer(file)
    #     tensor_list = tensor.tolist()
    #     print(tensor_list)
    #     writer.writerows(tensor_list)
def create_test_train_datasets(data, lookback, train_ratio=0.8):
    print("create dataset")
    sequence_length, num_features = data.shape
    print(data.shape)
    x_sequences = []
    y_sequences = []
    start_index = 230
    for t in range(start_index, sequence_length):
        x = data[t-lookback:t, :].float()
        y = data[t, 0].reshape(1).float()
        x_sequences.append(x)
        y_sequences.append(y)
    x_all = torch.stack(x_sequences)
    y_all = torch.stack(y_sequences)
    print("dims")
    print(x_all.shape)
    print(y_all.shape)

    num_samples = x_all.shape[0]
    split_index = int(num_samples * train_ratio)

    train_x = x_all[:split_index]
    train_y = y_all[:split_index]
    val_x = x_all[split_index:]
    val_y = y_all[split_index:]

    return train_x, train_y, val_x, val_y



def run():
    print("Running technical analysis ...")
    components = extract_graph_components(ticker)
    #plot_log(components[0], components[1])

    #bull_cross = bullish_crossover(components[6], components[9])
    #bear_cross = bearish_crossover(components[6], components[9])
    #plot(*components, [bull_cross, bear_cross]
    #    )

    # datetimes, close_prices, volumes, ma20, ma50, ma200, weekly_datetimes, macd_line, signal_line, histogram
    
    #print(components)
    data = build_dataset(ticker)
    train_x, train_y, val_x, val_y = create_test_train_datasets(data, 30)

    return train_x, train_y, val_x, val_y

    

