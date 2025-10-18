import os, time, joblib, requests, pandas as pd
from datetime import datetime, timezone
import ccxt

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
TIMEFRAME = os.getenv("TIMEFRAME", "1h")
THRESHOLD = 0.6

ASSETS = {"BTC": "BTC/USDT", "ETH": "ETH/USDT", "SOL": "SOL/USDT", "XRP": "XRP/USDT"}
MODEL_FILES = {
    "BTC": "btc_model.joblib",
    "ETH": "eth_model.joblib",
    "SOL": "sol_model.joblib",
    "XRP": "xrp_model.joblib"
}
LOG_FILE = "predictions_log.csv"
exchange = ccxt.binance()

def send_telegram(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"})
    except Exception as e:
        print("Telegram Error:", e)

def fetch_latest(symbol, limit=300):
    data = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["close"] = df["close"].astype(float)
    return df

def prepare_features(df):
    df["ema_fast"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=26, adjust=False).mean()
    return df.dropna()

def predict(artifact, df_recent):
    df = prepare_features(df_recent)
    row = df.iloc[-1:][["ema_fast","ema_slow"]]
    X = artifact["scaler"].transform(row.values)
    prob = float(artifact["model"].predict_proba(X)[:,1][0])
    direction = "UP" if prob >= 0.5 else "DOWN"
    return direction, prob

def load_models():
    models, mtimes = {}, {}
    for coin, path in MODEL_FILES.items():
        if os.path.exists(path):
            models[coin] = joblib.load(path)
            mtimes[coin] = os.path.getmtime(path)
            print(f"✅ Modell für {coin} geladen.")
        else:
            models[coin] = None
            mtimes[coin] = 0
            print(f"⚠️ Kein Modell gefunden: {path}")
    return models, mtimes

def check_reload(models, mtimes):
    updated = []
    for coin, path in MODEL_FILES.items():
        if os.path.exists(path):
            new_time = os.path.getmtime(path)
            if new_time > mtimes.get(coin, 0):
                models[coin] = joblib.load(path)
                mtimes[coin] = new_time
                updated.append(coin)
    return updated

if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["timestamp","coin","signal","prob","prev_close","actual_close","correct"]).to_csv(LOG_FILE, index=False)

models, mtimes = load_models()
predicted_hours = set()
print(f"🚀 Bot gestartet – Timeframe: {TIMEFRAME}, Schwelle: {THRESHOLD*100:.0f}% (UTC)")

while True:
    now = datetime.now(timezone.utc)
    updated = check_reload(models, mtimes)
    if updated:
        send_telegram("🔄 Modelle aktualisiert: " + ", ".join(updated))

    if now.minute == 59 and now.hour not in predicted_hours:
        strong_signals = []
        skipped = []
        for coin, symbol in ASSETS.items():
            model = models.get(coin)
            if model is None:
                skipped.append(f"{coin}: kein Modell")
                continue
            try:
                df = fetch_latest(symbol)
                direction, prob = predict(model, df)
                if prob < THRESHOLD:
                    skipped.append(f"{coin}: no strong signal ({prob*100:.1f}%)")
                    continue

                prev_close = float(df["close"].iloc[-1])
                strong_signals.append((coin, direction, prob, prev_close))
                row = {
                    "timestamp": now.isoformat(),
                    "coin": coin,
                    "signal": direction,
                    "prob": prob,
                    "prev_close": prev_close,
                    "actual_close": "",
                    "correct": ""
                }
                pd.DataFrame([row]).to_csv(LOG_FILE, mode="a", header=False, index=False)
            except Exception as e:
                skipped.append(f"{coin}: Fehler {e}")

        if strong_signals:
            msg_lines = [
                f"🕐 <b>Next {TIMEFRAME} Prediction</b> ({now.strftime('%Y-%m-%d %H:%M UTC')})",
                ""
            ]
            for coin, direction, prob, _ in strong_signals:
                msg_lines.append(f"{coin}: {direction} ({prob*100:.1f}%)")
            try:
                df = pd.read_csv(LOG_FILE)
                df = df[df["prob"] >= THRESHOLD]
                stats = []
                for c in ASSETS.keys():
                    d = df[df["coin"] == c]
                    total = len(d[d["correct"].notna()])
                    correct = int(d["correct"].sum()) if total > 0 else 0
                    acc = (correct/total*100) if total > 0 else 0
                    stats.append(f"{c}: {correct}/{total} ({acc:.1f}%)")
                msg_lines.append("")
                msg_lines.append("📊 Trefferquote (≥60%)")
                msg_lines.extend(stats)
            except Exception:
                pass
            send_telegram("\n".join(msg_lines))

        predicted_hours.add(now.hour)
    if now.hour == 0 and now.minute < 2:
        predicted_hours.clear()
    time.sleep(20)
