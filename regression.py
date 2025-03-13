import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import shap
import sys
import torch

from utils import schwab_utils

class TransformerRegression(torch.nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerRegression, self).__init__()
        self.embedding = torch.nn.Linear(input_dim, model_dim)
        self.pos_encoding = PositionalEncoding(model_dim)
        self.transformer_encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers
        )
        self.fc_out = torch.nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        x = self.fc_out(x)
        return x

class PositionalEncoding(torch.nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.model_dim = model_dim

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        pe = torch.zeros(seq_len, self.model_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.model_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        if self.model_dim % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        x = x + pe
        return x

def sequential_train_test_split(X, y, train_ratio):
    num_samples = X.shape[0]
    split_index = int(num_samples * train_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test

def linear_regression(symbol):
    # Daily Price History: open, high, low, close, volume, datetime
    data = schwab_utils.get_daily_price_history(symbol)
    df = pd.DataFrame.from_dict(data)

    # Moving Average: 20 day, 50 day, 200 day
    df['MA 20'] = df['close'].rolling(window=20).mean()
    df['MA 50'] = df['close'].rolling(window=50).mean()
    df['MA 200'] = df['close'].rolling(window=200).mean()

    # Moving Average Slope: 20 day, 50 day, 200 day
    df['MAS 20'] = df['MA 20'].diff(1)
    df['MAS 50'] = df['MA 50'].diff(1)
    df['MAS 200'] = df['MA 200'].diff(1)

    # Volume
    df.rename(columns={'volume': 'Volume'}, inplace=True)

    # Relative Strength Index
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Ichimoku Cloud
    df['Tenkan Sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
    df['Kijun Sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
    df['Senkou A'] = ((df['Tenkan Sen'] + df['Kijun Sen']) / 2).shift(26)
    df['Senkou B'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
    df['Chikou Span'] = df['close'].shift(-26)

    # Average True Range
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = pd.DataFrame({
        'tr_1': df['high'] - df['low'],
        'tr_2': abs(df['high'] - df['prev_close']),
        'tr_3': abs(df['low'] - df['prev_close'])
    }).max(axis=1)
    df['ATR'] = df['tr'].rolling(window=14).mean()
    df.drop(columns=['prev_close', 'tr'], inplace=True)

    # Baseline: Assume the previous day's change holds for the next day.
    df['prev_close'] = df['close'].shift(1)
    df['prev_close_diff'] = df['close'].shift(1).diff()
    df['baseline'] = df['prev_close'] + df['prev_close_diff']
    df.drop(columns=['prev_close', 'prev_close_diff'], inplace=True)

    # Generate X and y from the data.
    df.dropna(inplace=True)
    y = df[['datetime', 'close', 'baseline']]
    X = df.drop(['open', 'high', 'low', 'close', 'baseline'], axis=1)

    # Train Test Split: 80/20
    X_train, X_test, y_train, y_test = sequential_train_test_split(X, y, 0.8)
    X_datetime = pd.to_datetime(X['datetime'], unit='ms')
    X_test_datetime = pd.to_datetime(X_test['datetime'], unit='ms')
    y_baseline = y_test[['baseline']]
    X_train = X_train.drop(columns=['datetime'])
    X_test = X_test.drop(columns=['datetime'])
    y_train = y_train.drop(columns=['datetime', 'baseline'])
    y_test = y_test.drop(columns=['datetime', 'baseline'])

    # Linear Regression
    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    # Mean Squared Error
    mse = sklearn.metrics.mean_squared_error(y_test, y_predict)
    mse_baseline = sklearn.metrics.mean_squared_error(y_test, y_baseline)
    print(f'Linear MSE: {mse}')
    print(f'Baseline MSE: {mse_baseline}')

    # Actual vs Prediction vs Baseline Plot
    plt.plot(X_datetime, y['close'], color='gray', alpha=0.5, linewidth=1)
    plt.scatter(X_test_datetime, y_test['close'], color='blue', s=5, label='Actual')
    plt.scatter(X_test_datetime, y_predict, color='red', s=5, label='Prediction')
    plt.scatter(X_test_datetime, y_baseline, color='yellow', s=5, label='Baseline')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend(loc='upper left')
    plt.title('Linear Regression: Actual vs Prediction vs Baseline')
    plt.show()

    # SHAP Decision Plot
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer.shap_values(X_test)
    expected_value = explainer.expected_value
    features_display = X_train.columns
    shap.decision_plot(expected_value, shap_values, features_display)

def transformer_regression(symbol):
    # Daily Price History: open, high, low, close, volume, datetime
    data = schwab_utils.get_daily_price_history(symbol)
    df = pd.DataFrame.from_dict(data)

    # Moving Average: 20 day, 50 day, 200 day
    df['MA 20'] = df['close'].rolling(window=20).mean()
    df['MA 50'] = df['close'].rolling(window=50).mean()
    df['MA 200'] = df['close'].rolling(window=200).mean()

    # Moving Average Slope: 20 day, 50 day, 200 day
    df['MAS 20'] = df['MA 20'].diff(1)
    df['MAS 50'] = df['MA 50'].diff(1)
    df['MAS 200'] = df['MA 200'].diff(1)

    # Volume
    df.rename(columns={'volume': 'Volume'}, inplace=True)

    # Relative Strength Index
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    epsilon = 1e-8
    rs = gain / (loss + epsilon)
    df['RSI'] = 100 - (100 / (1 + rs))

    # Ichimoku Cloud
    df['Tenkan Sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
    df['Kijun Sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
    df['Senkou A'] = ((df['Tenkan Sen'] + df['Kijun Sen']) / 2).shift(26)
    df['Senkou B'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
    df['Chikou Span'] = df['close'].shift(-26)

    # Average True Range
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = pd.DataFrame({
        'tr_1': df['high'] - df['low'],
        'tr_2': abs(df['high'] - df['prev_close']),
        'tr_3': abs(df['low'] - df['prev_close'])
    }).max(axis=1)
    df['ATR'] = df['tr'].rolling(window=14).mean()
    df.drop(columns=['prev_close', 'tr'], inplace=True)

    # Baseline: Assume the previous day's change holds for the next day.
    df['prev_close'] = df['close'].shift(1)
    df['prev_close_diff'] = df['close'].shift(1).diff()
    df['baseline'] = df['prev_close'] + df['prev_close_diff']
    df.drop(columns=['prev_close', 'prev_close_diff'], inplace=True)

    # Generate X and y from the data.
    df.dropna(inplace=True)
    y = df[['datetime', 'close', 'baseline']]
    X = df.drop(['open', 'high', 'low', 'baseline'], axis=1)

    # Random Seed
    torch.manual_seed(42)

    # Sequence Length: 7 day lookback
    seq_len = 8
    X.set_index('datetime', inplace=True)
    y.set_index('datetime', inplace=True)
    seq = [X]
    for i in range(1, seq_len):
        for c in X.columns:
            seq.append(X[c].shift(i).rename(f'{c}_{i}'))
    X_seq = pd.concat(seq, axis=1)
    X_seq.dropna(inplace=True)
    y_seq = y.loc[X_seq.index].reset_index()
    X_seq = X_seq.reset_index()
    #X_seq['close'] = 0
    X_seq['close'] = X_seq['close_1']

    # Train Test Split: 80/20
    X_train, X_test, y_train, y_test = sequential_train_test_split(X_seq, y_seq, 0.8)
    X_datetime = pd.to_datetime(X_seq['datetime'], unit='ms')
    X_test_datetime = pd.to_datetime(X_test['datetime'], unit='ms')
    y_baseline = y_test[['baseline']]
    X_train = X_train.drop(columns=['datetime'])
    X_test = X_test.drop(columns=['datetime'])
    y_train = y_train.drop(columns=['datetime', 'baseline'])
    y_test = y_test.drop(columns=['datetime', 'baseline'])

    # Normalization
    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    y_train_mean = y_train.mean(axis=0)
    y_train_std = y_train.std(axis=0)
    X_train_scaled = (X_train - X_train_mean) / X_train_std
    X_test_scaled = (X_test - X_train_mean) / X_train_std
    y_train_scaled = (y_train - y_train_mean) / y_train_std
    y_test_scaled = (y_test - y_train_mean) / y_train_std

    # DataLoader
    batch_size = 32
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train_scaled.values, dtype=torch.float32), torch.tensor(y_train_scaled.values, dtype=torch.float32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Transformer Regression
    input_dim = 15
    model_dim = 64
    num_heads = 4
    num_layers = 2
    output_dim = 1
    model = TransformerRegression(input_dim, model_dim, num_heads, num_layers, output_dim)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    # Train
    model.train()
    num_epochs = 1000
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch, (X_batch, y_batch) in enumerate(dataloader):
            X_batch_seq = X_batch.reshape(batch_size, seq_len, input_dim).flip(dims=[1])
            y_predict = model(X_batch_seq)
            loss = criterion(y_predict, y_batch)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch: {epoch} Average Loss: {epoch_loss / len(dataloader)}')
        scheduler.step(epoch_loss / len(dataloader))

    # Evaluate
    model.eval()
    with torch.no_grad():
        X_test_seq = torch.tensor(X_test_scaled.values, dtype=torch.float32).reshape(X_test_scaled.shape[0], seq_len, input_dim).flip(dims=[1])
        y_predict_scaled = model(X_test_seq)
        y_predict = (y_predict_scaled * y_train_std.values) + y_train_mean.values
        loss = criterion(y_predict, torch.tensor(y_test.to_numpy(), dtype=torch.float32))
        print(f'Final Loss: {loss}')

    # Actual vs Prediction vs Baseline Plot
    plt.plot(X_datetime, y_seq['close'], color='gray', alpha=0.5, linewidth=1)
    plt.scatter(X_test_datetime, y_test['close'], color='blue', s=5, label='Actual')
    plt.scatter(X_test_datetime, y_predict, color='red', s=5, label='Prediction')
    plt.scatter(X_test_datetime, y_baseline, color='yellow', s=5, label='Baseline')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend(loc='upper left')
    plt.title('Transformer Regression: Actual vs Prediction vs Baseline')
    plt.show()

    # SHAP Decision Plot
    explainer = shap.DeepExplainer(model, torch.tensor(X_train_scaled.head(100).values, dtype=torch.float32).reshape(100, seq_len, input_dim).flip(dims=[1]))
    shap_values = explainer.shap_values(torch.tensor(X_test_scaled.head(10).values, dtype=torch.float32).reshape(10, seq_len, input_dim).flip(dims=[1]), check_additivity=False)
    shap_values_reshaped = shap_values[0][:, :, 0]
    shap.decision_plot(
        base_value=np.mean(shap_values_reshaped),
        shap_values=shap_values_reshaped,
        feature_names=X_train.columns[:input_dim].tolist(),
        feature_order=list(range(shap_values_reshaped.shape[1]))
    )

def main():
    if len(sys.argv) != 3:
        print("Usage: python regression.py <model_type> <stock_symbol>")
        sys.exit(1)

    model_type = sys.argv[1].strip().lower()
    stock_symbol = sys.argv[2].strip().upper()

    if model_type not in ['linear', 'transformer']:
        print("Invalid model type. Please enter either 'linear' or 'transformer'.")
        sys.exit(1)

    if model_type == 'linear':
        linear_regression(stock_symbol)
    elif model_type == 'transformer':
        transformer_regression(stock_symbol)

if __name__ == '__main__':
    main()