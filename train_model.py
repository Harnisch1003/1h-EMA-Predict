import ccxt, time, joblib, os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Config
COINS = ["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT"]
EMA_FAST = 12
EMA_SLOW = 26
CANDLES_PER_SYMBOL = 2000  # adjust as needed
exchange = ccxt.binance()

def fetch_ohlcv(symbol, timeframe="1h", limit=2000):
    all_data = []
    since = None
    while len(all_data) < limit:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break
        all_data += batch
        since = batch[-1][0] + 1
        time.sleep(0.15)
        if len(all_data) >= limit:
            break
    df = pd.DataFrame(all_data[:limit], columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

for coin in COINS:
    print("Training:", coin)
    df = fetch_ohlcv(coin, timeframe="1h", limit=CANDLES_PER_SYMBOL)
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna().reset_index(drop=True)
    X = df[["ema_fast","ema_slow"]].values
    y = df["target"].values
    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LogisticRegression(solver="liblinear", max_iter=2000)
    model.fit(X_train_s, y_train)
    acc = model.score(X_test_s, y_test)
    name = coin.split("/")[0].lower()
    artifact = {"model": model, "scaler": scaler, "features": ["ema_fast","ema_slow"], "trained_at": pd.Timestamp.utcnow()}
    joblib.dump(artifact, f"{name}_model.joblib")
    print(f"Saved {name}_model.joblib (test acc: {acc:.3f})")
print("All models trained.")