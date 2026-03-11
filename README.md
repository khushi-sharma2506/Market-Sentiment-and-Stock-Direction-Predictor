# 📈 StockSense — Market Sentiment & Stock Direction Predictor

An AI-powered stock market analysis platform for Indian NSE stocks. Combines NLP sentiment analysis on financial news with technical indicators (RSI, MACD, Bollinger Bands) and ML models to predict stock price direction.

---

## 📸 Features

| Page | Description |
|---|---|
| 🏠 Market Dashboard | Overview of 10 NSE stocks with signals & sentiment |
| 📰 Sentiment Feed | Live news headlines with VADER sentiment scores |
| 📈 Stock Analyzer | Candlestick chart + RSI + MACD + Buy/Sell signal |
| 🔄 Stock Comparison | Compare 2 stocks with normalised price charts |
| 🤖 Model Comparison | Logistic Regression vs Random Forest accuracy |

---

## 🧠 ML Models

- **Logistic Regression** — linear classifier for direction prediction
- **Random Forest** — ensemble of 100 trees for complex pattern detection
- **Features:** RSI, MACD, Signal Line, MA Cross, Volatility, Sentiment Score
- **Target:** 1 = Price goes UP tomorrow, 0 = Price goes DOWN

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Scikit-learn | LR + RF ML models |
| Streamlit | Web dashboard |
| Plotly | Candlestick + interactive charts |
| VADER | Sentiment scoring |
| yfinance | Live NSE stock data |
| Pandas / NumPy | Data processing |

---

## 🚀 Run Locally

```bash
git clone https://github.com/khushi-sharma2506/Market-Sentiment-and-Stock-Direction-Predictor.git
cd Market-Sentiment-and-Stock-Direction-Predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select repo → `app.py` → Deploy ✅

---

## 👤 Author

**Khushi Sharma** — B.Tech CSE (AI & ML), Graphic Era University
