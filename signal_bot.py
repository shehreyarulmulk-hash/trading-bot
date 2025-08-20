"""
AI Data Center + Rebound Signal Bot (NYSE)
=========================================

What this does
--------------
- Watches a curated list of NYSE/NASDAQ tickers tied to AI data centers (builders, lessors, power/cooling) + rebound candidates.
- Computes technical triggers on multiple timeframes (Daily + 1h):
  * RSI(14) oversold cross-up ( <30 → >30 )
  * 20/50 SMA Golden Cross
  * MACD line crossing above Signal line
  * Bollinger Band squeeze breakout
  * 52-week bounce (near prior low with bullish candle)
- (Optional) Insider activity boost: If you set `SEC_API_KEY`, we’ll look up recent insider **buys** and boost signal confidence.
- Sends concise Telegram alerts with the reason(s) to **BUY** or **SELL**.
- Designed to run on PythonAnywhere (or any cron) every N minutes.

Quick start
-----------
1) Install deps:
   pip install --upgrade yfinance pandas numpy requests python-dateutil

2) Set environment variables (never hardcode secrets!):
   export TELEGRAM_BOT_TOKEN="<your_bot_token>"
   export TELEGRAM_CHAT_ID="<your_chat_id>"
   # Optional (insider data via sec-api.com)
   export SEC_API_KEY="<your_sec_api_key>"
   # Optional (for polite SEC headers if you later query EDGAR directly)
   export SEC_USER_AGENT="yourname youremail@example.com"

3) Edit TICKERS below if you want. Default list includes:
   DLR, J, FLR, KBR, PWR, CARR, AMAT, MRVL, LITE, VVX, ATNF

4) Run locally to test:
   python signal_bot.py

5) Schedule on PythonAnywhere:
   - Files → upload this file.
   - Consoles → Bash: set env vars with `echo 'export KEY=VALUE' >> ~/.bashrc && source ~/.bashrc`
   - Tasks → Add a scheduled task: `python /home/youruser/signal_bot.py`
     (every 30 minutes is a good start)

Notes
-----
- This script **alerts**; it does not auto-trade. Hook your broker/exchange as desired.
- For equities price data: uses Yahoo Finance via `yfinance` (sufficient for signal timing; not for execution).
- Timeframes used: Daily (primary), 1h (confirmation). You can adjust.
- Treat outputs as **signals**, not guarantees. Use risk controls.
"""

import os
import time
import math
import json
import sys
import traceback
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional

import requests
import pandas as pd
import numpy as np
import yfinance as yf

# =====================
# Configuration
# =====================
TICKERS: List[str] = [
    # NYSE data-center & infra plays
    "DLR",  # Digital Realty Trust – data center REIT
    "J",    # Jacobs Solutions – engineering/design
    "FLR",  # Fluor – EPC for hyperscale infra
    "KBR",  # KBR – secure/government data centers
    "PWR",  # Quanta Services – power infrastructure
    "CARR", # Carrier Global – cooling/thermal mgmt
    # Rebound candidates with AI/infrastructure linkage or recent volatility
    "AMAT", # Applied Materials – semicap; dip/rebound setups
    "MRVL", # Marvell – AI/data center chips
    "LITE", # Lumentum – optics/lasers for AI infra
    "VVX",  # V2X – defense/logistics (contract catalysts)
    "ATNF", # 180 Life / ETHZilla – high risk/volatility
]

# Technical thresholds
RSI_PERIOD = 14
RSI_OVERSOLD = 30
SMA_FAST = 20
SMA_SLOW = 50
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
BB_SQUEEZE_PCTL = 0.20  # BB width < 20th percentile considered "squeeze"

# Score weights (tune as you like)
WEIGHTS = {
    "rsi_cross": 2.0,
    "golden_cross": 2.5,
    "macd_cross": 1.5,
    "bb_breakout": 2.0,
    "wk52_bounce": 1.5,
    "insider_buy": 1.0,  # bonus if recent insider buys
}

# Min score to alert BUY
BUY_SCORE_THRESHOLD = 3.0
# Simple SELL signal: RSI >= 70 or price hits TP levels; you can extend this.
RSI_OVERBOUGHT = 70

# Timeframes
DAILY_PERIOD = "6mo"
DAILY_INTERVAL = "1d"
INTRADAY_PERIOD = "60d"
INTRADAY_INTERVAL = "60m"  # use 60m for reliability; 15m may be rate-limited

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Optional insider data (sec-api.com)
SEC_API_KEY = os.getenv("SEC_API_KEY", "")
SEC_API_ENDPOINT = "https://api.sec-api.io/insider-transactions"

# Misc
APP_NAME = "AI-DC-Signal-Bot"
TZ = timezone.utc  # keep UTC in logs; your client can convert


# =====================
# Indicator utilities
# =====================

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=series.index).rolling(period).mean()
    roll_down = pd.Series(loss, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(method="bfill").fillna(50)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(series: pd.Series, period: int = 20, num_std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    mid = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid
    return upper, mid, lower, width


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


# =====================
# Data fetch
# =====================

def fetch_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna().copy()


# =====================
# Insider activity (optional)
# =====================

def _insider_buy_score_sec_api(ticker: str, lookback_days: int = 21) -> Tuple[float, List[str]]:
    """Use sec-api.io to count recent insider BUY transactions (code: P). Returns (bonus_score, notes)."""
    if not SEC_API_KEY:
        return 0.0, ["insider: skipped (no SEC_API_KEY)"]
    try:
        since = (datetime.now(tz=TZ) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        payload = {
            "query": {
                "query_string": {
                    "query": f"ticker:{ticker} AND transactionCoding.transactionAcquiredDisposedCode:P AND filedAt:[{since} TO now]"
                }
            },
            "from": 0,
            "size": 50,
            "sort": [{"filedAt": {"order": "desc"}}]
        }
        headers = {"Authorization": SEC_API_KEY, "Content-Type": "application/json"}
        resp = requests.post(SEC_API_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=20)
        resp.raise_for_status()
        data = resp.json()
        total = data.get("total", 0) or 0
        # Bonus score capped to avoid overpowering TA
        bonus = min(WEIGHTS["insider_buy"], 0.25 * total)
        notes = [f"insider buys: {total} in last {lookback_days}d (+{bonus:.2f} score)"]
        return bonus, notes
    except Exception as e:
        return 0.0, [f"insider: error {e}"]


# =====================
# Signal engine
# =====================

def compute_signals(df: pd.DataFrame) -> Dict[str, any]:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    out = {"score": 0.0, "reasons": [], "levels": {}}

    # Indicators
    r = rsi(close, RSI_PERIOD)
    sma_fast = sma(close, SMA_FAST)
    sma_slow = sma(close, SMA_SLOW)
    macd_line, macd_sig, macd_hist = macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    bb_up, bb_mid, bb_lo, bb_w = bollinger_bands(close, BB_PERIOD, BB_STD)

    # 1) RSI cross-up from oversold
    if len(r) >= 2 and (r.iloc[-2] < RSI_OVERSOLD) and (r.iloc[-1] >= RSI_OVERSOLD):
        out["score"] += WEIGHTS["rsi_cross"]
        out["reasons"].append(f"RSI cross-up: {r.iloc[-2]:.1f}→{r.iloc[-1]:.1f}")

    # 2) 20/50 Golden Cross
    if len(sma_fast) >= 2 and len(sma_slow) >= 2:
        prev_cross = sma_fast.iloc[-2] - sma_slow.iloc[-2]
        curr_cross = sma_fast.iloc[-1] - sma_slow.iloc[-1]
        if prev_cross <= 0 and curr_cross > 0:
            out["score"] += WEIGHTS["golden_cross"]
            out["reasons"].append("20/50 SMA Golden Cross")

    # 3) MACD bull cross
    if len(macd_line) >= 2 and len(macd_sig) >= 2:
        if (macd_line.iloc[-2] <= macd_sig.iloc[-2]) and (macd_line.iloc[-1] > macd_sig.iloc[-1]):
            out["score"] += WEIGHTS["macd_cross"]
            out["reasons"].append("MACD cross-up")

    # 4) BB squeeze breakout: width in lowest 20% AND close > upper band today
    width_pctl = (bb_w.dropna().rank(pct=True)).iloc[-1] if not bb_w.dropna().empty else 1.0
    if width_pctl <= BB_SQUEEZE_PCTL and close.iloc[-1] > (bb_up.iloc[-1] if not math.isnan(bb_up.iloc[-1]) else float("inf")):
        out["score"] += WEIGHTS["bb_breakout"]
        out["reasons"].append("Bollinger squeeze breakout")

    # 5) 52-week bounce: within 3% of 52w low recently and bullish candle today
    rolling_52w_low = close.rolling(252, min_periods=100).min()
    if not rolling_52w_low.dropna().empty:
        near_low = (close.iloc[-2] - rolling_52w_low.iloc[-2]) / rolling_52w_low.iloc[-2] if rolling_52w_low.iloc[-2] > 0 else 1.0
        bullish_candle = close.iloc[-1] > close.iloc[-2] and close.iloc[-1] > (high.iloc[-2] + low.iloc[-2]) / 2
        if near_low <= 0.03 and bullish_candle:
            out["score"] += WEIGHTS["wk52_bounce"]
            out["reasons"].append("Near 52w low + bullish follow-through")

    # Risk levels (ATR-based)
    a = atr(df)
    if not np.isnan(a.iloc[-1]):
        stop = close.iloc[-1] - 1.2 * a.iloc[-1]
        tp1 = close.iloc[-1] + 1.2 * a.iloc[-1]
        tp2 = close.iloc[-1] + 2.0 * a.iloc[-1]
        out["levels"].update({"stop": round(float(stop), 2), "tp1": round(float(tp1), 2), "tp2": round(float(tp2), 2)})

    # Simple SELL condition: RSI overbought now (optional)
    sell = r.iloc[-1] >= RSI_OVERBOUGHT
    out["sell"] = bool(sell)
    out["rsi_now"] = float(r.iloc[-1]) if not np.isnan(r.iloc[-1]) else None

    return out


def combine_daily_intraday(daily_sig: Dict, intraday_sig: Dict) -> Dict:
    """Combine daily + intraday signals to emphasize alignment."""
    score = daily_sig["score"]
    reasons = list(daily_sig["reasons"])  # copy

    # If intraday MACD or RSI confirm, add a small bonus
    if any("MACD" in r for r in intraday_sig["reasons"]):
        score += 0.5
        reasons.append("Intraday MACD confirm")
    if any("RSI" in r for r in intraday_sig["reasons"]):
        score += 0.5
        reasons.append("Intraday RSI confirm")

    # Carry levels / sell flag from daily by default
    combined = {
        "score": score,
        "reasons": reasons,
        "levels": daily_sig.get("levels", {}),
        "sell": daily_sig.get("sell", False),
        "rsi_now": daily_sig.get("rsi_now")
    }
    return combined


# =====================
# Alerts
# =====================

def send_telegram(msg: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] Telegram not configured; printing message instead:\n" + msg)
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            print(f"[ERROR] Telegram send failed: {r.status_code} {r.text}")
    except Exception:
        print("[ERROR] Telegram send exception:\n" + traceback.format_exc())


# =====================
# Main loop
# =====================

def build_message(ticker: str, price: float, combined: Dict, insider_notes: List[str]) -> str:
    header = f"*{ticker}* signal @ {price:.2f} ({datetime.now(tz=TZ).strftime('%Y-%m-%d %H:%M UTC')})"
    lines = []
    lines.append(f"Score: *{combined['score']:.2f}*  | RSI: {combined.get('rsi_now', float('nan')):.1f}")
    if combined.get("sell"):
        lines.append("⚠️ RSI overbought — consider *trim/sell* into strength")
    if combined["reasons"]:
        reasons_txt = "\n".join([f"• {r}" for r in combined["reasons"]])
        lines.append("Reasons:\n" + reasons_txt)
    if insider_notes:
        lines.append("Insider:\n" + "\n".join([f"• {n}" for n in insider_notes]))
    if combined.get("levels"):
        L = combined["levels"]
        lines.append(f"Risk: stop ~{L.get('stop','?')}, TP1 ~{L.get('tp1','?')}, TP2 ~{L.get('tp2','?')}")
    lines.append("—")
    lines.append("Not financial advice. Manage risk.")
    return "\n".join([header] + lines)


def latest_price(df: pd.DataFrame) -> float:
    return float(df["Close"].iloc[-1])


def run_once():
    alerts: List[str] = []

    for ticker in TICKERS:
        try:
            # Daily frame
            ddf = fetch_ohlcv(ticker, DAILY_PERIOD, DAILY_INTERVAL)
            if ddf.empty or len(ddf) < 60:
                print(f"[INFO] insufficient daily data for {ticker}")
                continue
            daily_sig = compute_signals(ddf)

            # Intraday frame (confirmation)
            idf = fetch_ohlcv(ticker, INTRADAY_PERIOD, INTRADAY_INTERVAL)
            intraday_sig = compute_signals(idf) if not idf.empty and len(idf) > 50 else {"score": 0.0, "reasons": [], "sell": False}

            combined = combine_daily_intraday(daily_sig, intraday_sig)

            # Optional insider buy bonus
            insider_bonus, insider_notes = _insider_buy_score_sec_api(ticker)
            combined["score"] += insider_bonus

            price = latest_price(ddf)

            # BUY alert threshold
            if combined["score"] >= BUY_SCORE_THRESHOLD:
                msg = build_message(ticker, price, combined, insider_notes)
                alerts.append(msg)
            # SELL heads-up (simple — refine as needed)
            elif combined.get("sell"):
                msg = build_message(ticker, price, combined, insider_notes)
                alerts.append(msg)

        except Exception:
            print(f"[ERROR] processing {ticker}:\n" + traceback.format_exc())
            continue

    # Dispatch
    if alerts:
        for m in alerts:
            send_telegram(m)
    else:
        print("[INFO] No alerts this run.")


if __name__ == "__main__":
    try:
        run_once()
    except Exception:
        print("[FATAL] Uncaught exception:\n" + traceback.format_exc())
        sys.exit(1)

