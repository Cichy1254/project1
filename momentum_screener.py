import numpy as np
setattr(np, "NaN", np.nan)      # ensure pandas_ta import compatibility

import re, csv, os, datetime
import pandas as pd
import requests, feedparser
from bs4 import BeautifulSoup
import yfinance as yf
import pandas_ta as ta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from rich import print

nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# ──────────────────────────────────────────────────────────────────────────────
# Helper Modules: New Data Points
# ──────────────────────────────────────────────────────────────────────────────

def get_put_call_ratio(ticker):
    """Fetch put and call volumes for the next option expiry and return ratio."""
    try:
        tkr = yf.Ticker(ticker)
        exp = tkr.options[0]
        chain = tkr.option_chain(exp)
        call_vol = chain.calls['volume'].sum()
        put_vol  = chain.puts['volume'].sum()
        ratio = round(put_vol / call_vol, 2) if call_vol else None
        return ratio, int(call_vol), int(put_vol)
    except Exception:
        return None, None, None

def attach_tech_indicators(df):
    """Attach MACD, Bollinger Bands, BB percent, and BB position to df."""
    # MACD
    macd = ta.macd(df['Close'])
    df['MACD']        = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']

    # Bollinger Bands + percent band
    bb = ta.bbands(df['Close'], length=20, std=2)
    df = df.join(bb)

    # Rename for clarity
    df.rename(columns={
        'BBL_20_2.0':  'BB_LOWER',
        'BBM_20_2.0':  'BB_MID',
        'BBU_20_2.0':  'BB_UPPER',
        'BBP_20_2.0':  'BB_PERC'
    }, inplace=True)

    # BB Position: 0 = at lower band, 1 = at upper band
    df['BB_POS'] = (
        (df['Close'] - df['BB_LOWER'])
        / (df['BB_UPPER'] - df['BB_LOWER'])
    )

    return df

def get_fundamentals_yf(ticker):
    """Pull trailing P/E and EBITDA from yfinance info."""
    info = yf.Ticker(ticker).info
    return {
        'pe': info.get('trailingPE'),
        'ebitda': info.get('ebitda')
    }


def get_calendar_events(ticker):
    """
    Return a dict of upcoming calendar events for a ticker,
    e.g. Earnings Date, Ex-Dividend Date, Dividend Date, Report Date.
    Works around yfinance returning either a dict or a DataFrame.
    """
    tkr = yf.Ticker(ticker)
    cal = tkr.calendar
    events = {}

    # yfinance may return dict
    if isinstance(cal, dict):
        for name, ts in cal.items():
            if ts is None or (isinstance(ts, float) and np.isnan(ts)):
                continue
            try:
                dt = pd.to_datetime(ts)
                events[name] = dt.strftime("%Y-%m-%d")
            except Exception:
                events[name] = str(ts)

    # or return DataFrame
    elif isinstance(cal, pd.DataFrame) and not cal.empty:
        for name, ts in cal.iloc[:, 0].items():
            if pd.isna(ts):
                continue
            if isinstance(ts, pd.Timestamp):
                events[name] = ts.strftime("%Y-%m-%d")
            else:
                events[name] = str(ts)

    return events

# ──────────────────────────────────────────────────────────────────────────────
# Original Helpers
# ──────────────────────────────────────────────────────────────────────────────

def tag_headline(title):
    t = title.lower()
    if "earnings" in t or "results" in t or re.search(r"\bq\d\b", t):
        return "[EARNINGS]"
    if any(k in t for k in ["acquisition", "merger", "deal"]):
        return "[ACQUISITION]"
    if any(k in t for k in ["analyst", "upgrade", "downgrade"]):
        return "[ANALYST COVERAGE]"
    if any(k in t for k in ["jump", "surge", "soar", "rally"]):
        return "[PRICE ACTION]"
    if any(k in t for k in ["ev", "solar", "green energy"]):
        return "[SECTOR TAILWIND]"
    return "[OTHER]"

def sentiment_label(title):
    score = sia.polarity_scores(title)["compound"]
    if score > 0.25: return "[BULLISH]"
    if score < -0.25: return "[BEARISH]"
    return "[NEUTRAL]"

def get_price_trend(ticker):
    try:
        df = yf.Ticker(ticker).history(period="7d", interval="1d")
        closes = df["Close"].tail(5).tolist()
        return "".join(
            "📈" if closes[i] > closes[i-1] else "📉"
            for i in range(1, len(closes))
        )
    except:
        return ""

def get_derivative_signals(ticker):
    info = yf.Ticker(ticker).info
    sr = info.get("shortRatio")
    sf = info.get("shortPercentOfFloat")
    mc = info.get("marketCap")
    avg_equity_vol = info.get("averageVolume")

    # Options note
    option_note = None
    call_vol = None
    iv = None
    try:
        exp = yf.Ticker(ticker).options[0]
        chain = yf.Ticker(ticker).option_chain(exp)
        call_vol = chain.calls["volume"].sum()
        iv = chain.calls["impliedVolatility"].mean() * 100
        option_note = f"📊 Options: Calls {call_vol:,} | Avg IV {iv:.2f}%"
        if call_vol > 10000:
            option_note += " 🚀 Unusual"
    except:
        option_note = "📊 Options: Unavailable"

    # Short-squeeze note
    short_note = None
    if sr and sf:
        short_note = f"🧨 Float {sf*100:.2f}% | Ratio {sr:.2f}d"
    elif sf:
        short_note = f"🧨 Float {sf*100:.2f}%"
    elif sr:
        short_note = f"🧨 Ratio {sr:.2f}d"

    # Market cap classification
    cap_note = None
    if mc:
        if mc < 3e8:
            cap_tag = "[MICROCAP]"
        elif mc < 1e10:
            cap_tag = "[MIDCAP]"
        elif mc > 2e11:
            cap_tag = "[MEGACAP]"
        else:
            cap_tag = "[LARGECAP]"
        vol_tag = "[LOW VOL]" if avg_equity_vol and avg_equity_vol < 1e6 else "[HIGH VOL]"
        cap_note = f"{cap_tag} | {vol_tag} | Cap ${mc/1e9:.2f}B"

    return option_note, short_note, cap_note, sr, sf, call_vol, iv

def score_combined_signals(metrics, headlines, sr=None, sf=None, cv=None, iv=None):
    score = min(metrics["pct"]/2, 5)
    if metrics["rsi"] and 55 < metrics["rsi"] < 70:
        score += 1
    if metrics["vol"] > 1.5 * metrics["avg_vol"]:
        score += 1

    tags = {t for t, _, _ in headlines}
    score += len(tags)
    score += sum(1 if s=="[BULLISH]" else -1 if s=="[BEARISH]" else 0
                 for _, s, _ in headlines)

    if len(tags) >= 2 and any(s=="[BULLISH]" for _, s, _ in headlines):
        score += 2
    if len(tags) >= 3:
        score += 3
    if sr and sr > 5:
        score += 1
    if sf and sf > 0.2:
        score += 1
    if cv and cv > 10000:
        score += 1
    if iv and iv > 200:
        score += 0.5

    return round(score, 2)

def scrape_news_for_ticker(ticker, max_items=10):
    feed = feedparser.parse(f"https://news.google.com/rss/search?q={ticker}+stock")
    results = []
    for e in feed.entries[:max_items]:
        title = e.title
        results.append((
            tag_headline(title),
            sentiment_label(title),
            title
        ))
    return results

def scrape_tickers(url):
    headers = {"User-Agent":"Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.select("table tbody tr")
        return [row.find("td").text.strip() for row in rows if row.find("td")]
    except:
        return []

def screen_tickers(tickers):
    perfect = []
    for tk in tickers:
        try:
            df = yf.Ticker(tk).history(period="60d", interval="1d", auto_adjust=True)
            if df.shape[0] < 40:
                continue
            df["RSI"]   = ta.rsi(df["Close"], length=14)
            df["SMA20"] = df["Close"].rolling(20).mean()
            df = attach_tech_indicators(df)
        except:
            continue

        latest    = df.iloc[-1]
        prev      = df.iloc[-2]
        price     = latest["Close"]
        vol       = latest["Volume"]
        rsi       = latest["RSI"]
        pct       = (price - prev["Close"]) / prev["Close"] * 100
        sma20     = latest["SMA20"]
        avg_vol_30d = df["Volume"].rolling(30).mean().iloc[-1]
        bb_per    = latest.get("BB_PERC", np.nan)
        bb_pos    = latest.get("BB_POS", np.nan)

        if (
            price < 30 and
            vol > avg_vol_30d and
            pct > 1 and
            (rsi is None or 45 < rsi < 75) and
            (sma20 is None or price > sma20)
        ):
            metrics = dict(
                pct=pct,
                rsi=rsi,
                vol=vol,
                avg_vol=avg_vol_30d,
                bbperc=bb_per,
                bbpos=bb_pos
            )
            perfect.append((tk, price, rsi, vol, pct, metrics))
    return perfect

def log_signals_to_csv(entries):
    fn = "signal_history.csv"
    fieldnames = ["Timestamp","Ticker","Price","SignalScore"]
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [{"Timestamp": now,
             "Ticker":    e[1],
             "Price":     e[2],
             "SignalScore": e[0]} for e in entries]
    write_header = not os.path.isfile(fn)
    with open(fn, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(rows)

def display_results(perfect):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[bold cyan]🚀 Signals @ {now}[/bold cyan]\n")
    scored = []

    for tk, price, rsi, vol, pct, metrics in perfect:
        heads           = scrape_news_for_ticker(tk)
        opt, short, cap, sr, sf, cv, iv = get_derivative_signals(tk)
        pcr, call_vol, put_vol             = get_put_call_ratio(tk)
        funds                               = get_fundamentals_yf(tk)
        calendar_events                     = get_calendar_events(tk)

        score = score_combined_signals(metrics, heads, sr, sf, cv, iv)
        scored.append((
            score, tk, price, rsi, vol, pct, metrics,
            heads, opt, short, cap, sr, sf, cv, iv,
            pcr, call_vol, put_vol, funds, calendar_events
        ))

    log_signals_to_csv(scored)

    for idx, (
        score, tk, price, rsi, vol, pct, metrics,
        heads, opt, short, cap, sr, sf, cv, iv,
        pcr, call_vol, put_vol, funds, calendar_events
    ) in enumerate(sorted(scored, key=lambda x: x[0], reverse=True), 1):

        hist = yf.Ticker(tk).history(period="30d", interval="1d", auto_adjust=True)
        avg_vol_30d = hist["Volume"].mean() if not hist.empty else None
        vol_pct = (vol / avg_vol_30d * 100) if avg_vol_30d else None

        print(f"[bold yellow]#{idx} {tk}[/bold yellow]  "
              f"${price:.2f}  RSI {rsi:.1f}  Vol +{vol_pct:.1f}%  Δ{pct:.2f}%  "
              f"P/C {pcr or 'N/A'}  PE {funds['pe'] or 'N/A'}  "
              f"Score {score}  BB% {metrics['bbperc']:.1f}%  BBpos {metrics['bbpos']:.2f}")

        if calendar_events:
            print("Calendar Events:")
            for name, date in calendar_events.items():
                print(f" • {name}: {date}")

        if call_vol and iv is not None:
            call_pct = None
            try:
                exps = yf.Ticker(tk).options[-5:]
                vols = [yf.Ticker(tk).option_chain(exp).calls["volume"].sum() for exp in exps]
                avg_call = sum(vols) / len(vols)
                call_pct = (call_vol / avg_call * 100) if avg_call else None
            except:
                pass
            if call_pct:
                opt_text = f"📊 Options: +{call_pct:.1f}%  | Avg IV {iv:.2f}%"
                if call_vol > 10000:
                    opt_text += " 🚀 Unusual"
                print(opt_text)
            else:
                print(opt)
        else:
            print(opt)

        if short or cap:
            print(f"{short or ''} {cap or ''}".strip())

        trend = get_price_trend(tk)
        if trend:
            print(f"Trend {trend}")

        print("News:")
        for tag, senti, title in heads:
            print(f" • {tag} {senti} — {title}")


        print()

def run_screener(return_data=False):
    urls = {
        "Gainers":   "https://finance.yahoo.com/gainers",
        "Active":    "https://finance.yahoo.com/most-active",
        "PreMarket": "https://finance.yahoo.com/premarket",
        "AfterHours":"https://finance.yahoo.com/after-hours",
        "Trending":  "https://finance.yahoo.com/markets/stocks/trending/"
    }

    raw = set()
    for url in urls.values():
        raw.update(scrape_tickers(url))

    if os.path.exists("watchlist.csv"):
        try:
            watch = pd.read_csv("watchlist.csv")["Ticker"].dropna().tolist()
            raw.update(watch)
        except:
            pass

    tickers = [t for t in raw if re.match(r"^[A-Za-z\.]{1,6}$", t)]
    perfect = screen_tickers(tickers)

    if return_data:
        dashboard_data = []
        for tk, price, rsi, vol, pct, metrics in perfect:
            hist = yf.Ticker(tk).history(period="30d", interval="1d", auto_adjust=True)
            avg_vol_30d = hist["Volume"].mean() if not hist.empty else None

            call_vols = []
            tkr = yf.Ticker(tk)
            for exp in tkr.options[-5:]:
                try:
                    call_vols.append(tkr.option_chain(exp).calls["volume"].sum())
                except:
                    continue
            avg_call_vol_30d = sum(call_vols)/len(call_vols) if call_vols else None

            opt, short, cap, sr, sf, cv, iv = get_derivative_signals(tk)
            pcr, call_vol, put_vol = get_put_call_ratio(tk)
            funds = get_fundamentals_yf(tk)
            heads = scrape_news_for_ticker(tk)
            calendar_events = get_calendar_events(tk)

            vol_pct      = vol / avg_vol_30d if avg_vol_30d else None
            call_vol_pct = call_vol / avg_call_vol_30d if avg_call_vol_30d and call_vol else None

            score = score_combined_signals(metrics, heads, sr, sf, call_vol, iv)

            dashboard_data.append({
                "Ticker":         tk,
                "Price":          round(price, 2),
                "RSI":            round(rsi or 0, 2),
                "Volume":         int(vol),
                "AvgVol30d":      int(avg_vol_30d) if avg_vol_30d else None,
                "VolPct":         round(vol_pct, 2) if vol_pct else None,
                "Change (%)":     round(pct, 2),
                "PutCallRatio":   pcr,
                "CallVol":        call_vol,
                "PutVol":         put_vol,
                "SignalScore":    score,
                "ShortFloat":     round(sf * 100, 2) if sf else None,
                "ShortRatio":     round(sr, 2) if sr else None,
                "AvgCallVol30d":  int(avg_call_vol_30d) if avg_call_vol_30d else None,
                "CallVolPct":     round(call_vol_pct, 2) if call_vol_pct else None,
                "IV":             round(iv, 2) if iv else None,
                "PE":             funds['pe'],
                "EBITDA":         funds['ebitda'],
                "OptionNote":     opt,
                "ShortNote":      short,
                "CapNote":        cap,
                "Trend":          get_price_trend(tk),
                "CalendarEvents": calendar_events,
                "BB_Perc":        metrics['bbperc'],
                "BB_Pos":         metrics['bbpos']
            })
        return dashboard_data
    else:
        display_results(perfect)

if __name__ == "__main__":
    run_screener()