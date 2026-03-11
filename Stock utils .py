"""
stock_utils.py
--------------
Handles stock data fetching, technical indicators,
sentiment analysis and ML model training.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── Indian Stocks ─────────────────────────────────────────────
INDIAN_STOCKS = {
    "TCS":        {"name": "Tata Consultancy Services", "symbol": "TCS.NS",    "sector": "IT"},
    "INFY":       {"name": "Infosys",                   "symbol": "INFY.NS",   "sector": "IT"},
    "RELIANCE":   {"name": "Reliance Industries",       "symbol": "RELIANCE.NS","sector": "Energy"},
    "HDFCBANK":   {"name": "HDFC Bank",                 "symbol": "HDFCBANK.NS","sector": "Banking"},
    "WIPRO":      {"name": "Wipro",                     "symbol": "WIPRO.NS",  "sector": "IT"},
    "TATAMOTORS": {"name": "Tata Motors",               "symbol": "TATAMOTORS.NS","sector": "Auto"},
    "ICICIBANK":  {"name": "ICICI Bank",                "symbol": "ICICIBANK.NS","sector": "Banking"},
    "BAJFINANCE": {"name": "Bajaj Finance",             "symbol": "BAJFINANCE.NS","sector": "Finance"},
    "SUNPHARMA":  {"name": "Sun Pharma",                "symbol": "SUNPHARMA.NS","sector": "Pharma"},
    "ONGC":       {"name": "ONGC",                      "symbol": "ONGC.NS",   "sector": "Energy"},
}

# ── Synthetic News Headlines ──────────────────────────────────
NEWS_TEMPLATES = {
    "positive": [
        "{company} reports strong Q3 results, profit up 18% YoY",
        "{company} wins major contract worth ₹2,500 crore",
        "{company} expands into new markets, stock hits 52-week high",
        "Analysts upgrade {company} to BUY with target price hike",
        "{company} announces special dividend, investors cheer",
        "{company} Q2 revenue beats estimates by 12%",
        "{company} signs strategic partnership with global tech giant",
        "Foreign investors increase stake in {company} significantly",
    ],
    "negative": [
        "{company} misses Q3 earnings, revenue down 8% QoQ",
        "{company} faces regulatory scrutiny, stock under pressure",
        "Analysts downgrade {company} amid margin concerns",
        "{company} CEO resignation triggers sell-off",
        "{company} warns of slowdown in key markets",
        "{company} quarterly profit falls short of estimates",
        "Global headwinds impact {company} export revenue",
        "{company} stock hits 52-week low on weak guidance",
    ],
    "neutral": [
        "{company} to announce Q4 results next week",
        "{company} board meeting scheduled for Friday",
        "{company} completes share buyback program",
        "NSE includes {company} in new index category",
        "{company} management meets institutional investors",
        "{company} files annual report with SEBI",
        "Market watchers eye {company} for next quarter guidance",
        "{company} maintains stable dividend payout ratio",
    ]
}


def generate_stock_data(symbol: str, days: int = 180) -> pd.DataFrame:
    """Generate realistic synthetic stock price data."""
    np.random.seed(hash(symbol) % 1000)

    base_prices = {
        "TCS": 3800, "INFY": 1500, "RELIANCE": 2400,
        "HDFCBANK": 1650, "WIPRO": 480, "TATAMOTORS": 780,
        "ICICIBANK": 1100, "BAJFINANCE": 7200,
        "SUNPHARMA": 1600, "ONGC": 280
    }

    base = base_prices.get(symbol, 1000)
    dates = pd.date_range(end=datetime.today(), periods=days, freq='B')

    # Simulate realistic price movement
    returns  = np.random.normal(0.0004, 0.015, days)
    prices   = base * np.exp(np.cumsum(returns))
    highs    = prices * (1 + np.abs(np.random.normal(0, 0.008, days)))
    lows     = prices * (1 - np.abs(np.random.normal(0, 0.008, days)))
    opens    = prices * (1 + np.random.normal(0, 0.005, days))
    volumes  = np.random.randint(1_000_000, 10_000_000, days)

    df = pd.DataFrame({
        'Date': dates, 'Open': opens, 'High': highs,
        'Low': lows, 'Close': prices, 'Volume': volumes
    }).set_index('Date')

    return df.round(2)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-10)
    return (100 - 100 / (1 + rs)).round(2)


def compute_macd(series: pd.Series):
    """Compute MACD line, signal line and histogram."""
    ema12   = series.ewm(span=12, adjust=False).mean()
    ema26   = series.ewm(span=26, adjust=False).mean()
    macd    = ema12 - ema26
    signal  = macd.ewm(span=9, adjust=False).mean()
    hist    = macd - signal
    return macd.round(4), signal.round(4), hist.round(4)


def compute_bollinger(series: pd.Series, period: int = 20):
    """Compute Bollinger Bands."""
    sma    = series.rolling(period).mean()
    std    = series.rolling(period).std()
    upper  = (sma + 2 * std).round(2)
    lower  = (sma - 2 * std).round(2)
    return upper, sma.round(2), lower


def get_sentiment_score(text: str) -> dict:
    """Simple rule-based sentiment scorer (simulates VADER)."""
    positive_words = ['strong', 'profit', 'wins', 'high', 'upgrade', 'buy',
                      'dividend', 'beats', 'partnership', 'increase', 'growth',
                      'record', 'expand', 'strategic', 'surge', 'rally']
    negative_words = ['misses', 'down', 'scrutiny', 'downgrade', 'resignation',
                      'slowdown', 'falls', 'weak', 'low', 'sell', 'concern',
                      'pressure', 'drop', 'loss', 'decline', 'warn']

    text_lower = text.lower()
    pos = sum(1 for w in positive_words if w in text_lower)
    neg = sum(1 for w in negative_words if w in text_lower)

    total = pos + neg + 1
    compound = (pos - neg) / total

    if compound > 0.1:   sentiment = "Positive"
    elif compound < -0.1: sentiment = "Negative"
    else:                 sentiment = "Neutral"

    return {
        "compound": round(compound, 3),
        "positive": round(pos / total, 3),
        "negative": round(neg / total, 3),
        "sentiment": sentiment
    }


def generate_news(symbol: str, n: int = 8) -> list:
    """Generate synthetic news headlines for a stock."""
    np.random.seed(hash(symbol + str(datetime.today().date())) % 9999)
    company = INDIAN_STOCKS.get(symbol, {}).get("name", symbol)
    news    = []

    categories = np.random.choice(
        ["positive", "negative", "neutral"],
        size=n, p=[0.4, 0.3, 0.3]
    )

    for i, cat in enumerate(categories):
        template = np.random.choice(NEWS_TEMPLATES[cat])
        headline = template.format(company=company)
        score    = get_sentiment_score(headline)
        hours_ago= np.random.randint(1, 48)

        news.append({
            "headline"  : headline,
            "sentiment" : score["sentiment"],
            "compound"  : score["compound"],
            "source"    : np.random.choice(["Economic Times", "Moneycontrol", "LiveMint", "Business Standard", "NDTV Profit"]),
            "time"      : f"{hours_ago}h ago",
            "symbol"    : symbol
        })

    return sorted(news, key=lambda x: abs(x['compound']), reverse=True)


def build_features(symbol: str) -> pd.DataFrame:
    """Build feature matrix for ML model training."""
    df = generate_stock_data(symbol, days=300)

    df['RSI']       = compute_rsi(df['Close'])
    df['MACD'], df['Signal'], df['MACD_Hist'] = compute_macd(df['Close'])
    df['MA5']       = df['Close'].rolling(5).mean()
    df['MA20']      = df['Close'].rolling(20).mean()
    df['MA50']      = df['Close'].rolling(50).mean()
    df['Return']    = df['Close'].pct_change()
    df['Volatility']= df['Return'].rolling(10).std()
    df['MA_Cross']  = (df['MA5'] > df['MA20']).astype(int)
    df['Above_MA50']= (df['Close'] > df['MA50']).astype(int)

    # Simulate sentiment feature
    np.random.seed(hash(symbol) % 777)
    df['Sentiment'] = np.random.uniform(-1, 1, len(df))

    # Target: 1 if next day price goes up
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    return df.dropna()


def train_models(symbol: str):
    """Train Logistic Regression and Random Forest, return both."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report

    df = build_features(symbol)
    feature_cols = ['RSI', 'MACD', 'Signal', 'MACD_Hist', 'MA_Cross',
                    'Above_MA50', 'Return', 'Volatility', 'Sentiment']

    X = df[feature_cols].values
    y = df['Target'].values

    scaler  = StandardScaler()
    X_scaled= scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=False
    )

    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_acc  = accuracy_score(y_test, lr_pred)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc  = accuracy_score(y_test, rf_pred)

    # Latest prediction
    latest      = scaler.transform(df[feature_cols].iloc[-1:].values)
    lr_signal   = lr.predict(latest)[0]
    rf_signal   = rf.predict(latest)[0]
    lr_proba    = lr.predict_proba(latest)[0]
    rf_proba    = rf.predict_proba(latest)[0]

    return {
        "lr_accuracy" : round(lr_acc * 100, 2),
        "rf_accuracy" : round(rf_acc * 100, 2),
        "lr_signal"   : "UP ↑" if lr_signal == 1 else "DOWN ↓",
        "rf_signal"   : "UP ↑" if rf_signal == 1 else "DOWN ↓",
        "lr_confidence": round(max(lr_proba) * 100, 1),
        "rf_confidence": round(max(rf_proba) * 100, 1),
        "lr_pred_history": lr_pred.tolist(),
        "rf_pred_history": rf_pred.tolist(),
        "y_test"         : y_test.tolist(),
        "feature_importance": dict(zip(feature_cols, rf.feature_importances_.round(4)))
    }


def get_buy_sell_signal(symbol: str) -> dict:
    """Generate combined Buy/Sell/Hold signal."""
    df      = generate_stock_data(symbol, 60)
    rsi     = compute_rsi(df['Close']).iloc[-1]
    macd, signal, _ = compute_macd(df['Close'])
    news    = generate_news(symbol, 5)

    avg_sentiment = np.mean([n['compound'] for n in news])
    macd_cross    = macd.iloc[-1] > signal.iloc[-1]
    rsi_ok        = 30 < rsi < 70

    score = 0
    if avg_sentiment > 0.1 : score += 2
    if avg_sentiment < -0.1: score -= 2
    if macd_cross           : score += 1
    if rsi < 40             : score += 1  # oversold = buy opportunity
    if rsi > 65             : score -= 1  # overbought = sell signal

    if score >= 2  : action = "BUY";  color = "#3fb950"
    elif score <= -1: action = "SELL"; color = "#f85149"
    else            : action = "HOLD"; color = "#ffc107"

    return {
        "action"       : action,
        "color"        : color,
        "score"        : score,
        "rsi"          : round(rsi, 2),
        "macd_cross"   : macd_cross,
        "avg_sentiment": round(avg_sentiment, 3),
        "confidence"   : min(abs(score) * 20 + 50, 95)
    }
