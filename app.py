"""
app.py — StockSense | Market Sentiment & Stock Direction Predictor
------------------------------------------------------------------
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from stock_utils import (
    INDIAN_STOCKS, generate_stock_data, compute_rsi,
    compute_macd, compute_bollinger, generate_news,
    get_sentiment_score, get_buy_sell_signal, train_models
)

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    section[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #21262d; }
    div[data-testid="metric-container"] {
        background: #161b22; border: 1px solid #21262d;
        border-radius: 12px; padding: 16px;
    }
    div[data-testid="metric-container"] label {
        color: #8b949e !important; font-size: 0.72rem !important;
        text-transform: uppercase; letter-spacing: 1px;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important; font-weight: 800 !important;
    }
    .stButton > button {
        background: #1f6feb; color: #fff; font-weight: 700;
        border: none; border-radius: 8px; width: 100%;
    }
    .stButton > button:hover { background: #388bfd; }
    .buy-box  { background: rgba(63,185,80,0.1);  border: 1px solid rgba(63,185,80,0.4);  border-radius: 12px; padding: 20px; text-align: center; }
    .sell-box { background: rgba(248,81,73,0.1);  border: 1px solid rgba(248,81,73,0.4);  border-radius: 12px; padding: 20px; text-align: center; }
    .hold-box { background: rgba(255,193,7,0.1);  border: 1px solid rgba(255,193,7,0.4);  border-radius: 12px; padding: 20px; text-align: center; }
    .news-pos  { border-left: 3px solid #3fb950; padding: 10px 14px; margin: 6px 0; background: rgba(63,185,80,0.05); border-radius: 0 8px 8px 0; }
    .news-neg  { border-left: 3px solid #f85149; padding: 10px 14px; margin: 6px 0; background: rgba(248,81,73,0.05); border-radius: 0 8px 8px 0; }
    .news-neu  { border-left: 3px solid #8b949e; padding: 10px 14px; margin: 6px 0; background: rgba(139,148,158,0.05); border-radius: 0 8px 8px 0; }
</style>
""", unsafe_allow_html=True)

SENT_COLOR = {"Positive": "#3fb950", "Negative": "#f85149", "Neutral": "#8b949e"}
SENT_EMOJI = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 StockSense")
    st.markdown("*Market Sentiment & Stock Direction Predictor*")
    st.divider()

    page = st.radio("Navigate", [
        "🏠 Market Dashboard",
        "📰 Sentiment Feed",
        "📈 Stock Analyzer",
        "🔄 Stock Comparison",
        "🤖 Model Comparison"
    ])

    st.divider()
    st.markdown("**Quick Stats**")
    st.success("🟢 NSE Market Open")
    st.caption(f"Updated: {datetime.now().strftime('%d %b %Y, %H:%M')}")
    st.divider()
    st.caption("Stack: Python · Scikit-learn · Streamlit · Plotly")
    st.caption("Built by Khushi Sharma | B.Tech AIML")


# ══════════════════════════════════════════════════════════════
# PAGE 1 — MARKET DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "🏠 Market Dashboard":
    st.title("🏠 Indian Stock Market Dashboard")
    st.caption("Real-time overview of top NSE stocks with sentiment signals")
    st.divider()

    # Generate market overview
    market_data = []
    for sym, info in INDIAN_STOCKS.items():
        df   = generate_stock_data(sym, 5)
        sig  = get_buy_sell_signal(sym)
        news = generate_news(sym, 3)
        avg_sent = np.mean([n['compound'] for n in news])

        price   = df['Close'].iloc[-1]
        prev    = df['Close'].iloc[-2]
        change  = ((price - prev) / prev * 100)

        market_data.append({
            "Symbol"    : sym,
            "Company"   : info['name'],
            "Sector"    : info['sector'],
            "Price (₹)" : round(price, 2),
            "Change %"  : round(change, 2),
            "Signal"    : sig['action'],
            "Sentiment" : "Positive" if avg_sent > 0.05 else "Negative" if avg_sent < -0.05 else "Neutral",
            "RSI"       : sig['rsi'],
        })

    df_market = pd.DataFrame(market_data)

    # KPIs
    bullish = len(df_market[df_market['Signal'] == 'BUY'])
    bearish = len(df_market[df_market['Signal'] == 'SELL'])
    neutral = len(df_market[df_market['Signal'] == 'HOLD'])
    pos_sent= len(df_market[df_market['Sentiment'] == 'Positive'])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks Tracked", str(len(df_market)))
    c2.metric("BUY Signals",    str(bullish),  delta=f"{bullish} stocks")
    c3.metric("SELL Signals",   str(bearish),  delta=f"{bearish} stocks", delta_color="inverse")
    c4.metric("Positive Sentiment", str(pos_sent), delta=f"of {len(df_market)}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Signals Overview")
        fig = go.Figure(go.Bar(
            x=df_market['Symbol'],
            y=df_market['Change %'],
            marker_color=['#3fb950' if c > 0 else '#f85149' for c in df_market['Change %']],
            text=[f"{c:+.2f}%" for c in df_market['Change %']],
            textposition='outside'
        ))
        fig.update_layout(
            plot_bgcolor='#161b22', paper_bgcolor='#161b22',
            font_color='#e6edf3', height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(showgrid=True, gridcolor='#21262d')
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Signal Distribution")
        sig_counts = df_market['Signal'].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=sig_counts.index, values=sig_counts.values,
            hole=0.55,
            marker_colors=['#3fb950', '#f85149', '#ffc107']
        ))
        fig2.update_layout(
            plot_bgcolor='#161b22', paper_bgcolor='#161b22',
            font_color='#e6edf3', height=300,
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("📋 All Stocks Overview")

    for _, row in df_market.iterrows():
        c1, c2, c3, c4, c5, c6 = st.columns([2, 3, 1, 1, 1, 1])
        c1.markdown(f"**{row['Symbol']}**")
        c2.markdown(f"<span style='color:#8b949e;font-size:0.85rem'>{row['Company']}</span>", unsafe_allow_html=True)
        c3.markdown(f"₹{row['Price (₹)']:,.2f}")
        color = "#3fb950" if row['Change %'] > 0 else "#f85149"
        c4.markdown(f"<span style='color:{color}'>{row['Change %']:+.2f}%</span>", unsafe_allow_html=True)
        sig_color = "#3fb950" if row['Signal']=="BUY" else "#f85149" if row['Signal']=="SELL" else "#ffc107"
        c5.markdown(f"<span style='color:{sig_color};font-weight:700'>{row['Signal']}</span>", unsafe_allow_html=True)
        sent_color = SENT_COLOR[row['Sentiment']]
        c6.markdown(f"<span style='color:{sent_color}'>{SENT_EMOJI[row['Sentiment']]} {row['Sentiment']}</span>", unsafe_allow_html=True)
        st.divider()


# ══════════════════════════════════════════════════════════════
# PAGE 2 — SENTIMENT FEED
# ══════════════════════════════════════════════════════════════
elif page == "📰 Sentiment Feed":
    st.title("📰 Live News Sentiment Feed")
    st.markdown("AI-powered sentiment analysis on latest Indian market news")
    st.divider()

    col1, col2 = st.columns([1, 3])
    with col1:
        selected_stocks = st.multiselect("Select Stocks", list(INDIAN_STOCKS.keys()),
                                          default=["TCS", "INFY", "RELIANCE"])
    with col2:
        filter_sent = st.multiselect("Filter Sentiment", ["Positive", "Negative", "Neutral"],
                                      default=["Positive", "Negative", "Neutral"])

    st.divider()

    all_news = []
    for sym in selected_stocks:
        all_news.extend(generate_news(sym, 6))

    filtered_news = [n for n in all_news if n['sentiment'] in filter_sent]

    # Sentiment summary
    if filtered_news:
        pos = sum(1 for n in filtered_news if n['sentiment'] == 'Positive')
        neg = sum(1 for n in filtered_news if n['sentiment'] == 'Negative')
        neu = sum(1 for n in filtered_news if n['sentiment'] == 'Neutral')

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Headlines", str(len(filtered_news)))
        c2.metric("🟢 Positive", str(pos))
        c3.metric("🔴 Negative", str(neg))
        c4.metric("⚪ Neutral",  str(neu))

        st.divider()

        # Sentiment chart
        sent_scores = [n['compound'] for n in filtered_news]
        symbols     = [n['symbol'] for n in filtered_news]
        headlines_short = [n['headline'][:40] + "..." for n in filtered_news]

        fig = go.Figure(go.Bar(
            x=sent_scores,
            y=headlines_short,
            orientation='h',
            marker_color=['#3fb950' if s > 0.05 else '#f85149' if s < -0.05 else '#8b949e' for s in sent_scores],
            text=[f"{s:+.3f}" for s in sent_scores],
            textposition='outside'
        ))
        fig.update_layout(
            title="Sentiment Scores (VADER)",
            plot_bgcolor='#161b22', paper_bgcolor='#161b22',
            font_color='#e6edf3', height=max(300, len(filtered_news) * 35),
            margin=dict(l=0, r=60, t=40, b=0),
            xaxis=dict(range=[-1.2, 1.2], showgrid=True, gridcolor='#21262d')
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("📰 Headlines")

        for news in filtered_news:
            css_class = f"news-{'pos' if news['sentiment']=='Positive' else 'neg' if news['sentiment']=='Negative' else 'neu'}"
            st.markdown(f"""
            <div class="{css_class}">
                <strong style="color:#e6edf3">{news['headline']}</strong><br>
                <span style="color:#8b949e;font-size:0.78rem">
                    {SENT_EMOJI[news['sentiment']]} {news['sentiment']} ({news['compound']:+.3f}) &nbsp;|&nbsp;
                    📰 {news['source']} &nbsp;|&nbsp; 🕐 {news['time']} &nbsp;|&nbsp;
                    📊 {news['symbol']}
                </span>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — STOCK ANALYZER
# ══════════════════════════════════════════════════════════════
elif page == "📈 Stock Analyzer":
    st.title("📈 Stock Analyzer")
    st.markdown("Deep analysis with price charts, technical indicators and AI signal")
    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.selectbox("Select Stock", list(INDIAN_STOCKS.keys()),
                               format_func=lambda x: f"{x} — {INDIAN_STOCKS[x]['name']}")
    with col2:
        period = st.selectbox("Period", ["1 Month", "3 Months", "6 Months"], index=1)

    days_map = {"1 Month": 22, "3 Months": 66, "6 Months": 132}
    days     = days_map[period]
    df       = generate_stock_data(symbol, days + 50).tail(days)

    # Technical indicators
    df['RSI']                        = compute_rsi(df['Close'])
    df['MACD'], df['Signal_L'], df['Hist'] = compute_macd(df['Close'])
    df['BB_Upper'], df['BB_Mid'], df['BB_Lower'] = compute_bollinger(df['Close'])
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()

    sig  = get_buy_sell_signal(symbol)
    news = generate_news(symbol, 4)
    avg_sent = np.mean([n['compound'] for n in news])

    # Signal box
    box_class = "buy-box" if sig['action']=="BUY" else "sell-box" if sig['action']=="SELL" else "hold-box"
    sig_color = sig['color']
    st.markdown(f"""
    <div class="{box_class}">
        <h1 style="color:{sig_color};margin:0;font-size:2.5rem">
            {"📈" if sig['action']=="BUY" else "📉" if sig['action']=="SELL" else "⏸️"} {sig['action']}
        </h1>
        <p style="color:#e6edf3;margin:6px 0">
            <strong>{INDIAN_STOCKS[symbol]['name']}</strong> &nbsp;|&nbsp;
            Confidence: <strong>{sig['confidence']}%</strong> &nbsp;|&nbsp;
            RSI: <strong>{sig['rsi']}</strong> &nbsp;|&nbsp;
            Sentiment: <strong>{avg_sent:+.3f}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Price + Bollinger chart
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=["Price + Bollinger Bands", "RSI", "MACD"])

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        increasing_line_color='#3fb950', decreasing_line_color='#f85149',
        name="Price"
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='#388bfd', width=1, dash='dot'), name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='#388bfd', width=1, dash='dot'), name='BB Lower', fill='tonexty', fillcolor='rgba(56,139,253,0.05)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='#ffc107', width=1.5), name='MA20'), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#a371f7', width=1.5), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#f85149", line_width=1, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#3fb950", line_width=1, row=2, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'],     line=dict(color='#1f6feb', width=1.5), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal_L'], line=dict(color='#f85149', width=1.5), name='Signal'), row=3, col=1)
    fig.add_bar(x=df.index, y=df['Hist'],
                marker_color=['#3fb950' if v >= 0 else '#f85149' for v in df['Hist']],
                name='Histogram', row=3, col=1)

    fig.update_layout(
        plot_bgcolor='#161b22', paper_bgcolor='#0d1117',
        font_color='#e6edf3', height=620,
        xaxis_rangeslider_visible=False,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    for i in range(1, 4):
        fig.update_xaxes(showgrid=True, gridcolor='#21262d', row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor='#21262d', row=i, col=1)

    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader(f"📰 Latest News — {symbol}")
    for n in news:
        css = f"news-{'pos' if n['sentiment']=='Positive' else 'neg' if n['sentiment']=='Negative' else 'neu'}"
        st.markdown(f"""
        <div class="{css}">
            <strong>{n['headline']}</strong><br>
            <span style="color:#8b949e;font-size:0.78rem">{SENT_EMOJI[n['sentiment']]} {n['sentiment']} ({n['compound']:+.3f}) | {n['source']} | {n['time']}</span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 4 — STOCK COMPARISON
# ══════════════════════════════════════════════════════════════
elif page == "🔄 Stock Comparison":
    st.title("🔄 Stock Comparison")
    st.markdown("Compare two Indian stocks side by side")
    st.divider()

    col1, col2 = st.columns(2)
    sym1 = col1.selectbox("Stock 1", list(INDIAN_STOCKS.keys()), index=0,
                           format_func=lambda x: f"{x} — {INDIAN_STOCKS[x]['name']}")
    sym2 = col2.selectbox("Stock 2", list(INDIAN_STOCKS.keys()), index=1,
                           format_func=lambda x: f"{x} — {INDIAN_STOCKS[x]['name']}")

    if sym1 == sym2:
        st.warning("Please select two different stocks.")
    else:
        df1  = generate_stock_data(sym1, 90)
        df2  = generate_stock_data(sym2, 90)
        sig1 = get_buy_sell_signal(sym1)
        sig2 = get_buy_sell_signal(sym2)
        news1 = generate_news(sym1, 3)
        news2 = generate_news(sym2, 3)
        sent1 = np.mean([n['compound'] for n in news1])
        sent2 = np.mean([n['compound'] for n in news2])

        # Metric comparison
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{sym1} Price", f"₹{df1['Close'].iloc[-1]:,.2f}")
        c2.metric(f"{sym2} Price", f"₹{df2['Close'].iloc[-1]:,.2f}")
        c3.metric(f"{sym1} Signal", sig1['action'])
        c4.metric(f"{sym2} Signal", sig2['action'])

        st.divider()

        # Normalized price comparison
        norm1 = df1['Close'] / df1['Close'].iloc[0] * 100
        norm2 = df2['Close'] / df2['Close'].iloc[0] * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df1.index, y=norm1, name=sym1,
                                  line=dict(color='#1f6feb', width=2)))
        fig.add_trace(go.Scatter(x=df2.index, y=norm2, name=sym2,
                                  line=dict(color='#f78166', width=2)))
        fig.update_layout(
            title="Normalized Price Comparison (Base = 100)",
            plot_bgcolor='#161b22', paper_bgcolor='#161b22',
            font_color='#e6edf3', height=360,
            yaxis_title="Normalized Price",
            xaxis=dict(showgrid=True, gridcolor='#21262d'),
            yaxis=dict(showgrid=True, gridcolor='#21262d'),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        # RSI comparison
        rsi1 = compute_rsi(df1['Close'])
        rsi2 = compute_rsi(df2['Close'])

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df1.index, y=rsi1, name=f"{sym1} RSI", line=dict(color='#1f6feb', width=2)))
        fig_rsi.add_trace(go.Scatter(x=df2.index, y=rsi2, name=f"{sym2} RSI", line=dict(color='#f78166', width=2)))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#f85149", line_width=1)
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#3fb950", line_width=1)
        fig_rsi.update_layout(
            title="RSI Comparison",
            plot_bgcolor='#161b22', paper_bgcolor='#161b22',
            font_color='#e6edf3', height=280,
            yaxis=dict(range=[0, 100], showgrid=True, gridcolor='#21262d'),
            xaxis=dict(showgrid=True, gridcolor='#21262d'),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

        # Side by side stats
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"📊 {sym1} — {INDIAN_STOCKS[sym1]['name']}")
            st.markdown(f"**Signal:** <span style='color:{sig1['color']};font-weight:700'>{sig1['action']}</span>", unsafe_allow_html=True)
            st.markdown(f"**RSI:** {sig1['rsi']}")
            st.markdown(f"**Sentiment:** {sent1:+.3f}")
            st.markdown(f"**Confidence:** {sig1['confidence']}%")
            for n in news1:
                st.caption(f"{SENT_EMOJI[n['sentiment']]} {n['headline'][:60]}...")

        with col2:
            st.subheader(f"📊 {sym2} — {INDIAN_STOCKS[sym2]['name']}")
            st.markdown(f"**Signal:** <span style='color:{sig2['color']};font-weight:700'>{sig2['action']}</span>", unsafe_allow_html=True)
            st.markdown(f"**RSI:** {sig2['rsi']}")
            st.markdown(f"**Sentiment:** {sent2:+.3f}")
            st.markdown(f"**Confidence:** {sig2['confidence']}%")
            for n in news2:
                st.caption(f"{SENT_EMOJI[n['sentiment']]} {n['headline'][:60]}...")


# ══════════════════════════════════════════════════════════════
# PAGE 5 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.title("🤖 Model Comparison")
    st.markdown("Logistic Regression vs Random Forest — which predicts better?")
    st.divider()

    symbol = st.selectbox("Select Stock to Analyze",
                           list(INDIAN_STOCKS.keys()),
                           format_func=lambda x: f"{x} — {INDIAN_STOCKS[x]['name']}")

    if st.button("⚡ Run Model Comparison"):
        with st.spinner(f"Training LR & RF models on {symbol} data..."):
            results = train_models(symbol)

        st.divider()

        # Accuracy comparison
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LR Accuracy",    f"{results['lr_accuracy']}%")
        c2.metric("RF Accuracy",    f"{results['rf_accuracy']}%")
        c3.metric("LR Signal",      results['lr_signal'])
        c4.metric("RF Signal",      results['rf_signal'])

        winner = "Random Forest" if results['rf_accuracy'] > results['lr_accuracy'] else "Logistic Regression"
        st.success(f"🏆 **{winner}** performs better on {INDIAN_STOCKS[symbol]['name']} data!")

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            # Accuracy bar chart
            fig = go.Figure(go.Bar(
                x=["Logistic Regression", "Random Forest"],
                y=[results['lr_accuracy'], results['rf_accuracy']],
                marker_color=['#1f6feb', '#3fb950'],
                text=[f"{results['lr_accuracy']}%", f"{results['rf_accuracy']}%"],
                textposition='outside', textfont_size=16
            ))
            fig.update_layout(
                title="Model Accuracy Comparison",
                plot_bgcolor='#161b22', paper_bgcolor='#161b22',
                font_color='#e6edf3', height=320,
                yaxis=dict(range=[0, 110], showgrid=True, gridcolor='#21262d'),
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Feature importance
            fi = results['feature_importance']
            fig_fi = go.Figure(go.Bar(
                x=list(fi.values())[::-1],
                y=list(fi.keys())[::-1],
                orientation='h', marker_color='#3fb950'
            ))
            fig_fi.update_layout(
                title="RF Feature Importance",
                plot_bgcolor='#161b22', paper_bgcolor='#161b22',
                font_color='#e6edf3', height=320,
                xaxis=dict(showgrid=True, gridcolor='#21262d'),
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig_fi, use_container_width=True)

        st.divider()
        st.subheader("📊 Prediction History (Test Set)")

        pred_df = pd.DataFrame({
            "Actual"             : results['y_test'],
            "LR Prediction"      : results['lr_pred_history'],
            "RF Prediction"      : results['rf_pred_history'],
        })
        pred_df['Actual']        = pred_df['Actual'].map({1: "UP ↑", 0: "DOWN ↓"})
        pred_df['LR Prediction'] = pred_df['LR Prediction'].map({1: "UP ↑", 0: "DOWN ↓"})
        pred_df['RF Prediction'] = pred_df['RF Prediction'].map({1: "UP ↑", 0: "DOWN ↓"})
        pred_df['LR Correct']    = pred_df['Actual'] == pred_df['LR Prediction']
        pred_df['RF Correct']    = pred_df['Actual'] == pred_df['RF Prediction']

        st.dataframe(pred_df.head(20), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("🧠 How Each Model Works")
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("📘 Logistic Regression"):
                st.markdown("""
                - Learns a **linear boundary** between UP and DOWN
                - Fast to train, easy to interpret
                - Uses sigmoid function for probability
                - Works well when relationship is linear
                - **Best for:** Simple, interpretable predictions
                """)
        with col2:
            with st.expander("🌲 Random Forest"):
                st.markdown("""
                - Builds **100 decision trees** and averages them
                - Captures **non-linear patterns** in data
                - Less prone to overfitting
                - Shows feature importance
                - **Best for:** Complex market patterns
                """)
