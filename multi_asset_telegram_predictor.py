import os, time, joblib, requests, pandas as pd
from datetime import datetime, timezone, timedelta
import ccxt

# Config via env
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
TIMEFRAME = os.getenv("TIMEFRAME", "1h")  # default 1h, set to "4h" or "1d" optionally

if not BOT_TOKEN or not CHAT_ID:
    raise SystemExit("Set BOT_TOKEN and CHAT_ID environment variables.")

ASSETS = {"BTC":"BTC/USDT","ETH":"ETH/USDT","SOL":"SOL/USDT","XRP":"XRP/USDT"}
MODEL_FILES = {"BTC":"btc_model.joblib","ETH":"eth_model.joblib","SOL":"sol_model.joblib","XRP":"xrp_model.joblib"}
LOG_FILE = "predictions_log.csv"
exchange = ccxt.binance()

def send_telegram(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id":CHAT_ID, "text":text})
    except Exception as e:
        print("Telegram error:", e)

def fetch_latest(symbol, limit=300):
    data = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
    df["open_time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["close"] = df["close"].astype(float)
    return df

def prepare_features(df):
    df["ema_fast"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=26, adjust=False).mean()
    return df.dropna()

def predict(artifact, df_recent):
    df = prepare_features(df_recent)
    row = df.iloc[-1:][["ema_fast","ema_slow"]]
    X = row.values
    Xs = artifact["scaler"].transform(X)
    prob = float(artifact["model"].predict_proba(Xs)[:,1][0])
    sig = "UP" if prob >= 0.5 else "DOWN"
    return sig, prob

# Logging init
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["pred_time_utc","coin","target_open_utc","pred_signal","pred_prob","prev_close","actual_close","correct"]).to_csv(LOG_FILE, index=False)

# Model management with auto-reload
def load_models():
    models = {}
    mtimes = {}
    for k, path in MODEL_FILES.items():
        if os.path.exists(path):
            try:
                models[k] = joblib.load(path)
                mtimes[k] = os.path.getmtime(path)
                print(f"Loaded model for {k}")
            except Exception as e:
                models[k] = None
                mtimes[k] = 0
                print("Error loading", path, e)
        else:
            models[k] = None
            mtimes[k] = 0
            print("Model missing:", path)
    return models, mtimes

def check_reload(models, mtimes):
    updated = []
    for k, path in MODEL_FILES.items():
        if os.path.exists(path):
            newt = os.path.getmtime(path)
            if newt > mtimes.get(k, 0):
                try:
                    models[k] = joblib.load(path)
                    mtimes[k] = newt
                    updated.append(k)
                except Exception as e:
                    print("Reload error", k, e)
    return updated

models, mtimes = load_models()

predicted_targets = set()

print("Bot running. TIMEFRAME =", TIMEFRAME, "UTC timebase.")

while True:
    now = datetime.now(timezone.utc)
    # auto reload models
    updated = check_reload(models, mtimes)
    if updated:
        send_telegram("🔄 Models reloaded: " + ", ".join(updated))
        print("Reloaded models:", updated)

    # Determine the "minute before timeframe candle starts"
    if TIMEFRAME == "1h":
        trigger_minute = 59
        next_open = (now + timedelta(minutes=1)).replace(minute=0, second=0, microsecond=0)
    elif TIMEFRAME == "4h":
        # trigger 1 minute before nearest 4h hour (0,4,8,...)
        # find next hour aligned to 4h
        hour = ((now.hour // 4) + 1) * 4 % 24
        next_open = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        # if next_open <= now, add 4 hours
        if next_open <= now:
            next_open = next_open + timedelta(hours=4)
        trigger_minute = (next_open - now).seconds // 60 - 1  # approximate, we use minute==59 for simplicity
        # fallback: use minute==59 as safe trigger
        trigger_minute = 59
    elif TIMEFRAME == "1d":
        # 1 minute before UTC midnight
        trigger_minute = 23*60 + 59  # not used directly; we'll check hour/minute below
    else:
        trigger_minute = 59

    # simple trigger logic: for 1h use minute 59; for 4h/day user can set TIMEFRAME accordingly but we'll still use minute==59
    if now.minute == 59:
        # target open is next timeframe start
        if TIMEFRAME == "1h":
            target_open = (now + timedelta(minutes=1)).replace(minute=0, second=0, microsecond=0)
        elif TIMEFRAME == "4h":
            # round up to next 4h
            next_hour = ((now.hour // 4) + 1) * 4 % 24
            target_open = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
            if target_open <= now:
                target_open = target_open + timedelta(hours=4)
        elif TIMEFRAME == "1d":
            target_open = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            target_open = (now + timedelta(minutes=1)).replace(minute=0, second=0, microsecond=0)

        if target_open in predicted_targets:
            time.sleep(20)
            continue

        preds = []
        for coin, symbol in ASSETS.items():
            artifact = models.get(coin)
            if artifact is None:
                preds.append((coin, None, None, None))
                continue
            try:
                df = fetch_latest(symbol, limit=300)
                sig, prob = predict(artifact, df)
                prev_close = float(df["close"].iloc[-1])
                preds.append((coin, sig, prob, prev_close))
                # log
                row = {
                    "pred_time_utc": now.isoformat(),
                    "coin": coin,
                    "target_open_utc": target_open.isoformat(),
                    "pred_signal": sig,
                    "pred_prob": prob,
                    "prev_close": prev_close,
                    "actual_close": None,
                    "correct": None
                }
                pd.DataFrame([row]).to_csv(LOG_FILE, mode="a", header=False, index=False)
            except Exception as e:
                preds.append((coin, None, None, None))
                print("Prediction error", coin, e)

        # build message
        header = f"🔔 Next prediction ({TIMEFRAME}) — {now.strftime('%Y-%m-%d %H:%M UTC')}"
        body = []
        for coin, sig, prob, prev in preds:
            if sig is None:
                body.append(f"{coin}: NO_MODEL/ERROR")
            else:
                body.append(f"{coin}: {sig} ({prob*100:.1f}%)")
        # stats summary
        # compute stats from log
        try:
            df_log = pd.read_csv(LOG_FILE)
            stats = []
            for coin in ASSETS.keys():
                d = df_log[df_log["coin"] == coin]
                done = d["correct"].notna().sum()
                correct = int(d["correct"].sum()) if done>0 else 0
                pct = (correct/done*100) if done>0 else 0.0
                stats.append(f"{coin}: {correct}/{done} ({pct:.1f}%)")
            stats_text = "\\n".join(stats)
        except Exception:
            stats_text = ""

        msg = header + "\\n" + "\\n".join(body) + "\\n\\nStats:\\n" + stats_text
        send_telegram(msg)
        print("Sent:", msg)
        predicted_targets.add(target_open)

    # cleanup old predicted targets (older than 6 hours)
    predicted_targets = {t for t in predicted_targets if (datetime.now(timezone.utc) - pd.to_datetime(t)) < timedelta(hours=6)}
    time.sleep(20)