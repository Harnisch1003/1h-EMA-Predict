# Crypto Multi-Timeframe Telegram Predictor (Starter Package)

Dieses Paket enthält einen Telegram-Bot für 1h-Vorhersagen (standard) mit optionaler Unterstützung für 4h und 1d.
Das System unterstützt BTC, ETH, SOL, XRP und lädt **automatisch** neue Modelle, wenn du sie hochlädst (Hot-reload).

## Inhalt
- `train_model.py` : Trainingsskript (erstellt Modelle per Coin)
- `multi_asset_telegram_predictor.py` : Bot-Skript mit Auto-Reload & Multi-Timeframe Support
- `requirements.txt` : benötigte Python-Pakete
- `predictions_log.csv` : leere Log-Datei (wird vom Bot gefüllt)
- Dummy-Modelle: `*_model.joblib` & `*_scaler.joblib` (btc/eth/sol/xrp)
- `.gitignore`

## Schnellstart (GitHub → Replit)
1. Neues GitHub-Repo erstellen und Dateien hochladen (oder direkt in Replit importieren).
2. In Replit: Secrets (Tools → Secrets) hinzufügen:
   - `BOT_TOKEN` : Telegram Bot Token (von @BotFather)
   - `CHAT_ID`  : Deine Chat-ID (Empfänger der Nachrichten)
   - Optional: `TIMEFRAME` : Standard ist `1h`. Setze auf `4h` oder `1d` wenn gewünscht.
3. (Optional) Führe `python train_model.py` einmal aus, um echte Modelle zu trainieren. Alternativ: lade deine eigenen `*_model.joblib` & `*_scaler.joblib` hoch.
4. Starte `multi_asset_telegram_predictor.py`. Der Bot prüft jede Minute und sendet 1 Minute vor jeder neuen Timeframe-Candle (z. B. für `1h` bei `xx:59` UTC) Meldungen.

## Modelle erzeugen
- Starte `train_model.py`. Es lädt historische Candles von Binance, berechnet EMA-Features und trainiert für jeden Coin ein Logistic Regression Modell.
- Ergebnis-Dateien (Beispiel):
  - `btc_model.joblib`, `btc_scaler.joblib`
  - `eth_model.joblib`, `eth_scaler.joblib`
  - `sol_model.joblib`, `sol_scaler.joblib`
  - `xrp_model.joblib`, `xrp_scaler.joblib`

## Hinweise
- Dummy-Modelle sind enthalten; ersetze sie mit deinen echten Modellen.
- Zeitbasis ist UTC.
- Replit kann schlafen; nutze UptimeRobot oder Replit "Always On" für konstante Verfügbarkeit.