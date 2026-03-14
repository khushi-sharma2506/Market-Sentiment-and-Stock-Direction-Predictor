import streamlit as st

st.set_page_config(
    page_title="StockSense — Market Sentiment & Stock Direction Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── NSE Stock Universe ──────────────────────────────────────────────────────────
NSE_STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Wipro": "WIPRO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "HUL": "HINDUNILVR.NS",
    "Tata Motors": "TATAMOTORS.NS",
}

analyzer = SentimentIntensityAnalyzer()

# ── Helper Functions ────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, period="6mo"):
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


def compute_indicators(df):
    df = df.copy()
    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal"]
    # Bollinger Bands
    ma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_Upper"] = ma20 + 2 * std20
    df["BB_Lower"] = ma20 - 2 * std20
    df["BB_Mid"] = ma20
    # MA Cross
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA_Cross"] = (df["MA10"] > df["MA50"]).astype(int)
    # Volatility
    df["Volatility"] = df["Close"].pct_change().rolling(14).std()
    return df.dropna()


@st.cache_data(ttl=1800)
def fetch_news_sentiment(query="Indian stock market NSE"):
    try:
        url = (
            f"https://newsapi.org/v2/everything?q={query}"
            f"&language=en&sortBy=publishedAt&pageSize=20"
            f"&apiKey=demo"
        )
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            articles = r.json().get("articles", [])
            results = []
            for a in articles[:10]:
                title = a.get("title", "")
                score = analyzer.polarity_scores(title)["compound"]
                results.append({"headline": title, "score": score,
                                 "source": a.get("source", {}).get("name", ""),
                                 "url": a.get("url", "")})
            return results
    except Exception:
        pass
    # Fallback mock headlines
    headlines = [
        ("Nifty50 surges 1.2% on strong FII inflows", 0.72),
        ("RBI holds repo rate, market reacts positively", 0.55),
        ("IT sector faces headwinds amid global slowdown", -0.45),
        ("Reliance Q3 results beat analyst estimates", 0.68),
        ("HDFC Bank reports record profits in FY25", 0.76),
        ("Sensex drops 400 points on weak global cues", -0.62),
        ("Bajaj Finance raises growth guidance for FY26", 0.58),
        ("TCS wins $1.5B deal, stock jumps 3%", 0.80),
        ("Inflation data raises rate hike fears", -0.40),
        ("Auto sector sales hit 3-year high in February", 0.65),
    ]
    return [{"headline": h, "score": s, "source": "Mock Feed", "url": "#"}
            for h, s in headlines]


def get_sentiment_score(ticker_name):
    news = fetch_news_sentiment(ticker_name)
    if not news:
        return 0.0
    return round(np.mean([n["score"] for n in news]), 4)


def build_signal(df, sentiment=0.0):
    if df.empty:
        return "NO DATA", ["Stock data could not be fetched"]
    latest = df.iloc[-1]
    score = 0
    reasons = []
    if latest["RSI"] < 40:
        score += 1; reasons.append("RSI oversold")
    elif latest["RSI"] > 65:
        score -= 1; reasons.append("RSI overbought")
    if latest["MACD"] > latest["Signal"]:
        score += 1; reasons.append("MACD bullish crossover")
    else:
        score -= 1; reasons.append("MACD bearish")
    if latest["MA_Cross"] == 1:
        score += 1; reasons.append("MA10 > MA50 (uptrend)")
    else:
        score -= 1; reasons.append("MA10 < MA50 (downtrend)")
    if sentiment > 0.1:
        score += 1; reasons.append(f"Positive sentiment ({sentiment:.2f})")
    elif sentiment < -0.1:
        score -= 1; reasons.append(f"Negative sentiment ({sentiment:.2f})")
    if score >= 2:
        signal = "🟢 BUY"
    elif score <= -2:
        signal = "🔴 SELL"
    else:
        signal = "🟡 HOLD"
    return signal, reasons


def train_models(df, sentiment=0.0):
    df = df.copy()
    df["Sentiment"] = sentiment
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    features = ["RSI", "MACD", "Signal", "MA_Cross", "Volatility", "Sentiment"]
    df = df.dropna(subset=features + ["Target"])
    X = df[features]
    y = df["Target"]
    if len(X) < 30:
        return None, None, 0, 0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_s, y_train)
    lr_acc = round(accuracy_score(y_test, lr.predict(X_test_s)) * 100, 1)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = round(accuracy_score(y_test, rf.predict(X_test)) * 100, 1)
    return lr, rf, lr_acc, rf_acc


# ── Sidebar ─────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/000000/stock-share.png", width=64)
st.sidebar.title("StockSense")
st.sidebar.caption("NSE Market Sentiment & Direction Predictor")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Market Dashboard", "📰 Sentiment Feed",
     "📈 Stock Analyzer", "🔄 Stock Comparison", "🤖 Model Comparison"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Built by Khushi Sharma | B.Tech AIML")
st.sidebar.caption("Data: Yahoo Finance · Sentiment: VADER")
st.sidebar.caption("ML: Scikit-learn · Charts: Plotly")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Market Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Market Dashboard":
    st.title("🏠 Market Dashboard")
    st.caption(f"NSE Overview — {datetime.now().strftime('%d %b %Y, %H:%M IST')}")

    cols = st.columns(5)
    rows = []
    with st.spinner("Fetching live data for 10 NSE stocks..."):
        for i, (name, ticker) in enumerate(NSE_STOCKS.items()):
            df = fetch_stock_data(ticker, period="3mo")
            if df.empty:
                continue
            df = compute_indicators(df)
            sentiment = get_sentiment_score(name)
            signal, _ = build_signal(df, sentiment)
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            chg = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100
            rows.append({
                "Stock": name,
                "Ticker": ticker,
                "Price (₹)": round(float(latest["Close"]), 2),
                "Change %": round(float(chg), 2),
                "RSI": round(float(latest["RSI"]), 1),
                "Signal": signal,
                "Sentiment": round(float(sentiment), 2),
            })

    if rows:
        df_dash = pd.DataFrame(rows)

        def color_signal(val):
            if "BUY" in val: return "color: #00c853; font-weight:bold"
            if "SELL" in val: return "color: #ff1744; font-weight:bold"
            return "color: #ffd600; font-weight:bold"

        def color_change(val):
            return "color: #00c853" if val >= 0 else "color: #ff1744"

        st.dataframe(
            df_dash.style
            .map(color_signal, subset=["Signal"])
            .map(color_change, subset=["Change %"]),
            use_container_width=True, height=420
        )

        # Mini sparklines for top 5
        st.subheader("Price Trend — Top 5 Stocks")
        spark_cols = st.columns(5)
        for i, row in df_dash.head(5).iterrows():
            df_sp = fetch_stock_data(row["Ticker"], "1mo")
            if not df_sp.empty:
                fig = go.Figure(go.Scatter(
                    x=df_sp.index, y=df_sp["Close"],
                    mode="lines", line=dict(width=2, color="#00b4d8")))
                fig.update_layout(margin=dict(l=0, r=0, t=24, b=0),
                                  height=120, title=row["Stock"][:12],
                                  xaxis=dict(visible=False),
                                  yaxis=dict(visible=False),
                                  plot_bgcolor="#0e1117",
                                  paper_bgcolor="#0e1117",
                                  font=dict(color="white", size=10))
                spark_cols[i].plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Sentiment Feed
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📰 Sentiment Feed":
    st.title("📰 Live Sentiment Feed")
    query = st.text_input("Search news topic:", value="NSE Indian stocks")
    news = fetch_news_sentiment(query)

    if news:
        scores = [n["score"] for n in news]
        avg = np.mean(scores)
        pos = sum(1 for s in scores if s > 0.05)
        neg = sum(1 for s in scores if s < -0.05)
        neu = len(scores) - pos - neg

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Sentiment", f"{avg:.3f}", delta=f"{'Bullish' if avg>0 else 'Bearish'}")
        m2.metric("🟢 Positive", pos)
        m3.metric("🔴 Negative", neg)
        m4.metric("🟡 Neutral", neu)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Market Sentiment Score"},
            gauge={
                "axis": {"range": [-1, 1]},
                "bar": {"color": "#00b4d8"},
                "steps": [
                    {"range": [-1, -0.05], "color": "#ff4444"},
                    {"range": [-0.05, 0.05], "color": "#ffd700"},
                    {"range": [0.05, 1], "color": "#00c853"},
                ],
                "threshold": {"line": {"color": "white", "width": 4},
                              "thickness": 0.75, "value": avg}
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="#0e1117", font=dict(color="white"), height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.subheader("Headlines")
        for n in news:
            color = "#00c853" if n["score"] > 0.05 else ("#ff4444" if n["score"] < -0.05 else "#ffd700")
            emoji = "🟢" if n["score"] > 0.05 else ("🔴" if n["score"] < -0.05 else "🟡")
            st.markdown(
                f"""<div style="background:#1e1e2e;padding:12px;border-radius:8px;
                border-left:4px solid {color};margin-bottom:8px;">
                {emoji} <b>{n['headline']}</b><br>
                <small style="color:grey;">Source: {n['source']} &nbsp;|&nbsp; 
                Sentiment: <span style="color:{color}">{n['score']:.3f}</span></small>
                </div>""", unsafe_allow_html=True
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Stock Analyzer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Stock Analyzer":
    st.title("📈 Stock Analyzer")

    col1, col2 = st.columns([2, 1])
    stock_name = col1.selectbox("Select Stock", list(NSE_STOCKS.keys()))
    period = col2.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=2)
    ticker = NSE_STOCKS[stock_name]

    with st.spinner(f"Loading {stock_name}..."):
        df = fetch_stock_data(ticker, period)
        if df.empty:
            st.error("Could not fetch data. Try again later.")
            st.stop()
        df = compute_indicators(df)
        sentiment = get_sentiment_score(stock_name)
        signal, reasons = build_signal(df, sentiment)

    # Signal banner
    color = "#00c853" if "BUY" in signal else ("#ff1744" if "SELL" in signal else "#ffd600")
    st.markdown(
        f"""<div style="background:{color}22;border:2px solid {color};padding:16px;
        border-radius:12px;text-align:center;font-size:24px;font-weight:bold;color:{color}">
        {signal} &nbsp;|&nbsp; Sentiment Score: {sentiment:.3f}
        </div>""", unsafe_allow_html=True
    )
    st.markdown("")
    with st.expander("Signal Reasoning"):
        for r in reasons:
            st.write(f"• {r}")

    # Candlestick + Bollinger
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"],
        line=dict(color="rgba(100,200,255,0.4)", width=1), name="BB Upper"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"],
        line=dict(color="rgba(100,200,255,0.4)", width=1),
        fill="tonexty", fillcolor="rgba(100,200,255,0.05)", name="BB Lower"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA10"],
        line=dict(color="#ff9800", width=1.5, dash="dot"), name="MA10"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"],
        line=dict(color="#e040fb", width=1.5, dash="dot"), name="MA50"))
    fig.update_layout(
        title=f"{stock_name} — Candlestick + Bollinger Bands",
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"), height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    # RSI + MACD
    c1, c2 = st.columns(2)
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"],
        line=dict(color="#00b4d8", width=2), name="RSI"))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ff4444", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="#00c853", annotation_text="Oversold")
    fig_rsi.update_layout(title="RSI (14)", plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117", font=dict(color="white"), height=280, showlegend=False)
    c1.plotly_chart(fig_rsi, use_container_width=True)

    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"],
        line=dict(color="#00c853", width=2), name="MACD"))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df["Signal"],
        line=dict(color="#ff9800", width=1.5, dash="dot"), name="Signal"))
    fig_macd.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"],
        marker_color=["#00c853" if v >= 0 else "#ff4444" for v in df["MACD_Hist"]],
        name="Histogram"))
    fig_macd.update_layout(title="MACD", plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117", font=dict(color="white"), height=280,
        legend=dict(orientation="h", yanchor="bottom", y=1.02))
    c2.plotly_chart(fig_macd, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Stock Comparison
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔄 Stock Comparison":
    st.title("🔄 Stock Comparison")

    c1, c2, c3 = st.columns([2, 2, 1])
    stock1 = c1.selectbox("Stock A", list(NSE_STOCKS.keys()), index=0)
    stock2 = c2.selectbox("Stock B", list(NSE_STOCKS.keys()), index=2)
    period = c3.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=2)

    with st.spinner("Fetching comparison data..."):
        df1 = fetch_stock_data(NSE_STOCKS[stock1], period)
        df2 = fetch_stock_data(NSE_STOCKS[stock2], period)

    if df1.empty or df2.empty:
        st.error("Could not fetch data for one or both stocks.")
        st.stop()

    # Normalised price chart
    norm1 = (df1["Close"] / df1["Close"].iloc[0]) * 100
    norm2 = (df2["Close"] / df2["Close"].iloc[0]) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df1.index, y=norm1,
        line=dict(color="#00b4d8", width=2), name=stock1))
    fig.add_trace(go.Scatter(x=df2.index, y=norm2,
        line=dict(color="#ff9800", width=2), name=stock2))
    fig.update_layout(
        title="Normalised Price Comparison (Base = 100)",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"), height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis_title="Indexed Price"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Volume comparison
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=df1.index, y=df1["Volume"],
        marker_color="#00b4d8", name=stock1, opacity=0.7))
    fig_vol.add_trace(go.Bar(x=df2.index, y=df2["Volume"],
        marker_color="#ff9800", name=stock2, opacity=0.7))
    fig_vol.update_layout(
        title="Volume Comparison", barmode="overlay",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"), height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # Stats table
    df1c = compute_indicators(df1)
    df2c = compute_indicators(df2)
    l1, l2 = df1c.iloc[-1], df2c.iloc[-1]
    ret1 = round(float(((df1["Close"].iloc[-1] - df1["Close"].iloc[0]) / df1["Close"].iloc[0]) * 100), 2)
    ret2 = round(float(((df2["Close"].iloc[-1] - df2["Close"].iloc[0]) / df2["Close"].iloc[0]) * 100), 2)

    stats = pd.DataFrame({
        "Metric": ["Current Price (₹)", f"Return ({period})", "RSI", "MACD", "Volatility"],
        stock1: [
            round(float(l1["Close"]), 2), f"{ret1}%",
            round(float(l1["RSI"]), 1), round(float(l1["MACD"]), 3),
            round(float(l1["Volatility"]), 4)
        ],
        stock2: [
            round(float(l2["Close"]), 2), f"{ret2}%",
            round(float(l2["RSI"]), 1), round(float(l2["MACD"]), 3),
            round(float(l2["Volatility"]), 4)
        ]
    })
    st.dataframe(stats.set_index("Metric"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.title("🤖 ML Model Comparison")
    st.caption("Logistic Regression vs Random Forest — predicting next-day price direction")

    stock_name = st.selectbox("Select Stock to Train On", list(NSE_STOCKS.keys()))
    ticker = NSE_STOCKS[stock_name]

    with st.spinner("Training models..."):
        df = fetch_stock_data(ticker, "1y")
        if df.empty:
            st.error("Could not fetch data.")
            st.stop()
        df = compute_indicators(df)
        sentiment = get_sentiment_score(stock_name)
        lr_model, rf_model, lr_acc, rf_acc = train_models(df, sentiment)

    if lr_model is None:
        st.warning("Not enough data to train. Try a longer period.")
        st.stop()

    # Accuracy bar chart
    fig_acc = go.Figure(go.Bar(
        x=["Logistic Regression", "Random Forest"],
        y=[lr_acc, rf_acc],
        marker_color=["#00b4d8", "#ff9800"],
        text=[f"{lr_acc}%", f"{rf_acc}%"],
        textposition="outside"
    ))
    fig_acc.update_layout(
        title=f"Model Accuracy on {stock_name}",
        yaxis=dict(range=[0, 100], title="Accuracy %"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"), height=380
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    # Metrics cards
    m1, m2 = st.columns(2)
    winner = "Logistic Regression" if lr_acc >= rf_acc else "Random Forest"
    m1.metric("Logistic Regression", f"{lr_acc}%", delta=f"{'Winner 🏆' if winner == 'Logistic Regression' else ''}")
    m2.metric("Random Forest", f"{rf_acc}%", delta=f"{'Winner 🏆' if winner == 'Random Forest' else ''}")

    st.info(f"🏆 **{winner}** performs better on {stock_name} with **{max(lr_acc, rf_acc)}% accuracy**")

    # Feature importance (RF)
    features = ["RSI", "MACD", "Signal", "MA_Cross", "Volatility", "Sentiment"]
    importances = rf_model.feature_importances_
    fig_fi = go.Figure(go.Bar(
        x=importances, y=features, orientation="h",
        marker_color="#00c853",
        text=[f"{v:.3f}" for v in importances], textposition="outside"
    ))
    fig_fi.update_layout(
        title="Random Forest — Feature Importance",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"), height=320,
        xaxis_title="Importance Score"
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # Model info
    with st.expander("ℹ️ About the Models"):
        st.markdown("""
        **Features used for prediction:**
        - `RSI` — Relative Strength Index (14-day)
        - `MACD` — Moving Average Convergence Divergence
        - `Signal` — MACD Signal Line (9-day EMA)
        - `MA_Cross` — MA10 > MA50 crossover flag
        - `Volatility` — 14-day rolling std of returns
        - `Sentiment` — VADER compound score from news

        **Target variable:**  
        `1` = Next day Close > Today's Close (Price UP)  
        `0` = Price DOWN

        **Train/Test split:** 80/20 (time-ordered, no shuffle)
        """)
