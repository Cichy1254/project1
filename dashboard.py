import numpy as np
setattr(np, "NaN", np.nan)      # ensure pandas_ta import compatibility

import pandas_ta as ta
import streamlit as st
import pandas as pd
import time
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from momentum_screener import (
    run_screener,
    scrape_news_for_ticker,
    get_put_call_ratio,
    get_fundamentals_yf,
    attach_tech_indicators,
    get_calendar_events
)

def style_df(df):
    """Apply common text-color formatting and number formats."""
    # 1) number formatting map
    fmt_map = {
        "Price":       "{:.2f}",
        "SignalScore": "{:.2f}",
        "Change (%)":  "{:.0f}%",
        "RSI":         "{:.2f}",
        "VolPct":      "{:.1%}",
        "CallVolPct":  "{:.1%}",
        "PutCallRatio":"{:.2f}",
        "IV":          "{:.0f}%",
        "ShortFloat":  "{:.2f}%",
        "ShortRatio":  "{:.2f}",
        "Volume":      "{:,}",
        "AvgVol30d":   "{:,}",
        "CallVol":     "{:,}",
        "AvgCallVol30d":"{:,}",
        "PutVol":      "{:,}",
        "MACD":       "{:.2f}",
        "MACD_signal":       "{:.2f}",
        "BB_UPPER":       "{:.2f}",
        "BB_LOWER":       "{:.2f}",
    }

    # 2) start the Styler
    styler = df.style.format(fmt_map)
    # 3) text-color mappers (reuse your existing functions)
    styler = (
        styler
          .map(sig_text_color,    subset=["SignalScore"])
          .map(change_text_color, subset=["Change (%)"])
          .map(rsi_text_color,    subset=["RSI"])
          .map(volpct_text_color, subset=["VolPct"])
          .map(pcr_text_color,    subset=["PutCallRatio"])
          .map(callvolpct_text_color, subset=["CallVolPct"])
          .map(iv_text_color,         subset=["IV"])
          .map(shortfloat_text_color, subset=["ShortFloat"])
          .map(shortratio_text_color, subset=["ShortRatio"])
          .apply(price_bb_text_color,      axis=1)
    )
    return styler


# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cichy's Quick Picks", layout="wide")
import streamlit as st

# 1) Inject a tiny CSS rule for centering
st.markdown(
    """
    <style>
      .centered {
        text-align: center;
        margin: 0 auto;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# 2) Use HTML tags + the .centered class instead of plain st.title/st.markdown
st.markdown("<h1 class='centered'>🚀 Momentum Stock Screener</h1>", unsafe_allow_html=True)

st.markdown(
    "<p class='centered'>Identify sub-$30 stocks with strong volume, sentiment, and derivative signals.</p>",
    unsafe_allow_html=True
)

st.markdown("<h3 class='centered'>📘 Indicator Cheat Sheet</h3>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

c1.markdown("""
**Signal Score (0–10)**
- Composite of price, volume, sentiment, & derivatives  
- 0–3 = weak / reversal risk  
- 4–6 = neutral / consolidation  
- 7–10 = strong momentum

**Price Change (%)**
- % move vs prior close  
- Positive = bullish momentum  
- Negative = bearish momentum
""")

c2.markdown("""
**RSI (14)**
- Measures overbought/oversold  
- 0–30 = oversold / reversal zone  
- 30–55 = neutral  
- 55–70 = bullish momentum  
- 70–100 = overbought / pullback risk

**MACD (12,26,9)**
- Trend-strength oscillator  
- MACD > Signal = bullish cross  
- MACD < Signal = bearish cross
""")

c3.markdown("""
**VolPct (30d)**
- Today’s volume ÷ 30-day avg  
- >150% = heavy participation  
- <50% = low conviction

**CallVolPct (30d)**
- Options call volume spike often leads price moves

**Put/Call Ratio**
- Bullish skew <1  
- Bearish skew >1
""")

c4.markdown("""
**Bollinger Bands (20,2σ)**
- Price vs bands  
  - Near upper = overbought  
  - Near lower = oversold

**Fundamentals (P/E, EBITDA)**
- P/E: <15 = value | >30 = growth  
- EBITDA: higher = stronger earnings

**Upcoming Catalysts**
- Earnings dates  
- Stock splits  
- Dividend announcements
""")


# ─── Cached Helpers ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_history(ticker: str) -> pd.DataFrame:
    return yf.Ticker(ticker).history(period="14d", interval="1h", auto_adjust=True)

@st.cache_data(ttl=300)
def get_event_flags(ticker: str):
    try:
        cal = yf.Ticker(ticker).calendar
        earn = cal.loc["Earnings Date"][0].date() if "Earnings Date" in cal.index else None
        divi = cal.loc["Ex-Dividend Date"][0].date() if "Ex-Dividend Date" in cal.index else None
        return earn, divi
    except:
        return None, None

# ─── Tech Chart Helper ─────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def plot_tech_chart(ticker: str):
    df = yf.Ticker(ticker).history(period="60d", interval="1d", auto_adjust=True)
    if df.empty:
        return None
    df = attach_tech_indicators(df)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.5],
        vertical_spacing=0.06,
        subplot_titles=[f"{ticker} — Price + BBands", "MACD"]
    )
    # Price + Bollinger
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_UPPER"],
        line_color="lightblue", name="BB Upper"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_LOWER"], fill="tonexty",
        line_color="lightblue", name="BB Lower"
    ), row=1, col=1)
    # MACD Panel
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"],
        line_color="green", name="MACD"
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD_signal"],
        line_color="red", name="MACD Signal"
    ), row=2, col=1)
    fig.update_layout(
        height=600,
        margin=dict(t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=.08, xanchor="right", x=1)
    )
    return fig

# ─── Run Screener ───────────────────────────────────────────────────────────────
with st.spinner("Running screener…"):
    results = run_screener(return_data=True)

# ─── Main Table & Visuals ───────────────────────────────────────────────────────
if not results:
    st.warning("⚠️ No momentum tickers found.")
else:
    df = pd.DataFrame(results)

    # fetch static info
    @st.cache_data(ttl=600)
    def fetch_ticker_info(tickers: list) -> pd.DataFrame:
        infos = []
        for t in tickers:
            try:
                info = yf.Ticker(t).info
                infos.append({
                    'Ticker': t,
                    'Sector': info.get('sector') or 'Unknown',
                    'Industry': info.get('industry') or 'Unknown',
                    'MarketCap': info.get('marketCap') or np.nan
                })
            except:
                infos.append({'Ticker': t, 'Sector':'Unknown','Industry':'Unknown','MarketCap':np.nan})
        return pd.DataFrame(infos)

    infos    = fetch_ticker_info(df['Ticker'].tolist())

    # fetch tech signals
    @st.cache_data(ttl=600)
    def fetch_tech_signals(tickers: list) -> pd.DataFrame:
        rows = []
        for t in tickers:
            try:
                hist = yf.Ticker(t).history(period="60d", interval="1d", auto_adjust=True)
                hist = attach_tech_indicators(hist)
                last = hist.iloc[-1]
                rows.append({
                    'Ticker': t,
                    'MACD': last['MACD'],
                    'MACD_signal': last['MACD_signal'],
                    'BB_MID': last['BB_MID'],
                    'BB_UPPER': last['BB_UPPER'],
                    'BB_LOWER': last['BB_LOWER']
                })
            except:
                continue
        return pd.DataFrame(rows)

    tech_df = fetch_tech_signals(df['Ticker'].tolist())

    # merge & process
    df = df.merge(infos, on='Ticker', how='left').merge(tech_df, on='Ticker', how='left')
    bins   = [0, 300e6, 2e9, 10e9, 200e9, float('inf')]
    labels = ['Micro','Small','Mid','Large','Mega']
    df['MarketCapCategory'] = pd.cut(df['MarketCap'], bins=bins, labels=labels)

    # round everything to 2 decimals
    df = df.round(2)

    # select & reorder cols (remove EBITDA)
    order = [
        "SignalScore","Ticker","Price","Change (%)",
        "VolPct","Volume","AvgVol30d","CallVolPct","CallVol",
        "AvgCallVol30d","PutCallRatio","PutVol",
        "IV","RSI","ShortFloat","ShortRatio",
        "MACD","MACD_signal","BB_UPPER","BB_LOWER","Calendar_Events",
        "Trend","Sector","Industry","MarketCapCategory"
    ]
    df = df[[c for c in order if c in df.columns]].copy()

    # formatting maps
    percent_fmt = {
        "Change (%)": "{:.2f}%",
        "IV":         "{:.0f}%",
        "VolPct":     "{:.0%}",
        "CallVolPct": "{:.0f}%",
        "CallVol":    "{:,.0f}",
        "Volume":     "{:,.0f}",
        "PutVol":     "{:,.0f}",
        "AvgVol30d":    "{:,.0f}",
        "AvgCallVol30d":"{:,.0f}",
        
    }
    default_fmt = {
        col: "{:.2f}" for col in df.select_dtypes(include="number").columns
        if col not in percent_fmt
    }
    fmt_map = {**default_fmt, **percent_fmt}

    # summary metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Signal Score", f"{df.SignalScore.mean():.2f}")
    c2.metric("Total Tickers",     len(df))
    c3.metric("Abn Vol >200%",     int((df.VolPct > 2).sum()))

    st.caption(f"Last updated: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}")
import pandas as pd
import streamlit as st


# ─── 1) Text-only color functions ───────────────────────────────
# ─── 1) Text‐color functions for each new column ────────────────
def callvolpct_text_color(v):
    if pd.isna(v): return ""
    # >200% call‐vol spike = bullish
    return "color: #2A9D8F" if v >= 2 else ""

def iv_text_color(v):
    if pd.isna(v): return ""
    # Extremely high IV (>80%) = red (costly)
    # Extremely low IV (<20%)  = green (cheap)
    if v > 0.8: return "color: #E63946"
    if v < 0.2: return "color: #2A9D8F"
    return ""

def shortfloat_text_color(v):
    if pd.isna(v): return ""
    # High short float (>10%) = red
    # Low short float (<2%)  = green
    if v > 0.10: return "color: #E63946"
    if v < 0.02: return "color: #2A9D8F"
    return ""

def shortratio_text_color(v):
    if pd.isna(v): return ""
    # Days to cover > 5 = red, < 2 = green
    if v > 5:   return "color: #E63946"
    if v < 2:   return "color: #2A9D8F"
    return ""


# ─── 2) Price‐by‐BB_Pos text color via a row‐wise apply ────────
def price_bb_text_color(row):
    bb_pos = row.get("BB_Pos", None)
    if bb_pos is None or pd.isna(bb_pos):
        color = ""
    elif bb_pos > 0.8:
        color = "#E63946"  # near upper band → red
    elif bb_pos < 0.2:
        color = "#2A9D8F"  # near lower band → green
    else:
        color = "#A8A8A8"  # middle → gray
    # return a list of styles, only Price gets colored
    return [
        f"color: {color}" if col == "Price" else ""
        for col in row.index
    ]
def sig_text_color(v):
    if pd.isna(v): return ""
    if v <= 3:     return "color: #E63946"
    if v <= 6:     return "color: #F4A261"
    return "color: #2A9D8F"

def change_text_color(v):
    if pd.isna(v): return ""
    return "color: #2A9D8F" if v > 0 else "color: #E63946"

def rsi_text_color(v):
    if pd.isna(v): return ""
    if v < 30:     return "color: #2A9D8F"
    if v < 55:     return "color: #A8A8A8"
    if v < 70:     return "color: #2A9D8F"
    return "color: #E63946"

def volpct_text_color(v):
    if pd.isna(v): return ""
    if v >= 1.5:   return "color: #2A9D8F"
    if v < 0.5:    return "color: #E63946"
    return "color: #A8A8A8"

def pcr_text_color(v):
    if pd.isna(v): return ""
    return "color: #2A9D8F" if v < 1 else "color: #E63946"

# ─── 2) Build styled DataFrame ─────────────────────────────────
styled = (
    df.style
      .format("{:.2f}", subset=[
          "Price","SignalScore","Change (%)","RSI","VolPct",
          "CallVolPct","PutCallRatio","IV","ShortFloat",
          "ShortRatio"
      ])
      # your existing text mappers
      .map(sig_text_color,    subset=["SignalScore"])
      .map(change_text_color, subset=["Change (%)"])
      .map(rsi_text_color,    subset=["RSI"])
      .map(volpct_text_color, subset=["VolPct"])
      .map(pcr_text_color,    subset=["PutCallRatio"])
      # new text mappers
      .map(callvolpct_text_color, subset=["CallVolPct"])
      .map(iv_text_color,         subset=["IV"])
      .map(shortfloat_text_color, subset=["ShortFloat"])
      .map(shortratio_text_color, subset=["ShortRatio"])
      # row‐wise Price coloring
      .apply(price_bb_text_color, axis=1)
)

# ─── 4) Render it ─────────────────────────────────────────────
st.dataframe(style_df(df), use_container_width=True)

abn_vol = df[df.VolPct > 2].sort_values("VolPct", ascending=False).head(10)
st.subheader("🔎 Abnormal Volume Leaders >200%")
st.dataframe(style_df(abn_vol), use_container_width=True)

abn_call = df[df.CallVolPct > 2].sort_values("CallVolPct", ascending=False).head(10)
st.subheader("🔎 Abnormal Call Volume Leaders >200%")
st.dataframe(style_df(abn_call), use_container_width=True)

# ─── Top Momentum Picks ────────────────────────────────────────────────────
st.subheader("🏅 Top Momentum Picks")

def _normalize(dt):
    if dt is None:
        return None
    if isinstance(dt, (pd.Series, pd.Index, list, tuple)):
        for v in dt:
            if pd.notna(v):
                dt = v
                break
        else:
            return None
    if isinstance(dt, pd.Timestamp):
        return dt.strftime("%Y-%m-%d")
    if hasattr(dt, "date"):
        return str(dt)
    return str(dt)

top = df.sort_values("SignalScore", ascending=False).head(10)

for _, r in top.iterrows():
    with st.container():
        st.markdown(f"#### {r['Ticker']}")

        # ─── Metrics ─────────────────────────────────────────────────────
        cols = st.columns(12)
        m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12 = cols

        m1.metric("Price",    f"${r['Price']:.2f}")
        m2.metric("Change %", f"{r['Change (%)']:.2f}%")
        m3.metric("Vol×",     f"{r['VolPct']*100:.0f}%")
        m4.metric("RSI",      f"{r['RSI']:.2f}")

        m5.metric("MACD",     f"{r['MACD']:.2f}")
        m6.metric("Signal",   f"{r['MACD_signal']:.2f}")

        mid, hi, lo = r.get("BB_MID"), r.get("BB_UPPER"), r.get("BB_LOWER")
        if mid is not None and hi is not None and lo is not None:
            pct = (r["Price"] - mid) / (hi - lo) * 100
            pos = f"{pct:.2f}% above mid" if pct > 0 else f"{abs(pct):.2f}% below mid"
        else:
            pos = "N/A"
        m7.metric("BB Pos", pos)

        m8.metric("IV %",      f"{r.get('IV', 0):.0f}%")
        m9.metric("CallVol %", f"{r.get('CallVolPct', 0)*100:.0f}%")
        m10.metric("Short %",   f"{r.get('ShortFloat', 0):.2f}%")

        cal  = get_calendar_events(r["Ticker"])
        earn = _normalize(cal.get("Earnings Date"))
        exd  = _normalize(cal.get("Ex-Dividend Date"))
        if earn:
            m11.write(f"🗓 Earnings: {earn}")
        if exd:
            m12.write(f"💰 Ex-Dividend: {exd}")

        # ─── Inline Tech Chart ──────────────────────────────────────────
        fig = plot_tech_chart(r["Ticker"])
        if fig:
            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        # ─── Headlines ───────────────────────────────────────────────────
        heads = scrape_news_for_ticker(r["Ticker"], max_items=15)
        if heads:
            st.markdown("🗞️ News:")
            col1, col2 = st.columns(2)
            half = (len(heads) + 1) // 2

            for tag, senti, title in heads[:half]:
                emoji = {"[BULLISH]": "🟢", "[BEARISH]": "🔴"}.get(senti, "⚪️")
                col1.markdown(f"- {emoji} {title}")

            for tag, senti, title in heads[half:]:
                emoji = {"[BULLISH]": "🟢", "[BEARISH]": "🔴"}.get(senti, "⚪️")
                col2.markdown(f"- {emoji} {title}")
        else:
            st.info(f"No headlines found for {r['Ticker']}")


