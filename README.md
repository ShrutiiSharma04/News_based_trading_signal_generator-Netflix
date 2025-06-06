﻿# News_based_trading_signal_generator-Netflix
🎬 Netflix Sentiment-Based Trading Signal Dashboard
This project is a simple yet powerful Streamlit dashboard that combines real-time news sentiment and stock market data to generate basic trading signals (BUY / SELL / HOLD) for Netflix (NFLX).

💡 Idea: Stock prices aren't just driven by numbers — they’re influenced by news, public mood, and social sentiment.
This dashboard shows how we can track that sentiment using AI and turn it into insights for stock trading.

📊 What It Does
✅ Pulls Netflix stock price data from Yahoo Finance
✅ Fetches live news headlines using NewsAPI
✅ Analyzes headline sentiment using VADER (NLP tool)
✅ Generates daily sentiment scores
✅ Combines sentiment & price to generate:

✅ BUY when sentiment is strongly positive

❌ SELL when sentiment is negative

⚖️ HOLD when sentiment is neutral
✅ Visualizes:

1. Netflix price with moving averages
2. Sentiment bar chart
3. Signal markers on the price chart
4. Latest sentiment gauge
5. Word cloud from news headlines

🛠️ Technologies Used: 
Python
Streamlit — interactive dashboard
NewsAPI — fetches real-time headlines
yfinance — pulls Netflix stock data
VADER Sentiment — NLP-based sentiment analyzer
NLTK — for text preprocessing
Matplotlib & Plotly — for visualizations
WordCloud — to show most common headline words

