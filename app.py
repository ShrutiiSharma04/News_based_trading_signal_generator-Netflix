import streamlit as st
import yfinance as yf
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import requests
from datetime import date
import plotly.graph_objects as go
from wordcloud import WordCloud

# We start with the NLTK Setup
nltk.download('punkt')
nltk.download('stopwords')

# Next step is streamlit Page Configuration
st.set_page_config(page_title="Netflix Sentiment Signals", layout="wide")
st.markdown("<h1 style='text-align: center;'>🎬 Netflix Sentiment-Based Trading Signal Dashboard</h1>", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=150)
    st.markdown("### ⚙️ Settings")
    start_date = st.date_input("Start Date", date(2025, 5, 1))
    end_date = st.date_input("End Date", date(2025, 6, 1))
    api_key = st.text_input("🔑 NewsAPI Key", type="password")

# Loading the Stock Data for your choice: mine was Netflix
st.header("1. 📈 Netflix Stock Price & Moving Averages")
netflix = yf.Ticker("NFLX")
stock_data = netflix.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
stock_data.reset_index(inplace=True)
stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
stock_data['MA_7'] = stock_data['Close'].rolling(window=7).mean()
stock_data['MA_14'] = stock_data['Close'].rolling(window=14).mean()
st.line_chart(stock_data.set_index('Date')[['Close', 'MA_7', 'MA_14']])

# News Section
st.header("2. 📰 News Sentiment Analysis")
if api_key:
    with st.spinner("Fetching News..."):
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q': 'Netflix',
            'from': start_date.strftime("%Y-%m-%d"),
            'to': end_date.strftime("%Y-%m-%d"),
            'sortBy': 'relevancy',
            'apiKey': api_key,
            'pageSize': 100,
            'language': 'en'
        }
        response = requests.get(url, params=params)
        data = response.json()

    if data['status'] != 'ok':
        st.error(f"❌ NewsAPI error: {data['message']}")
    else:
        articles = data['articles']
        news_data = pd.DataFrame(articles)[['publishedAt', 'title']]
        news_data.columns = ['date', 'headline']
        news_data['date'] = pd.to_datetime(news_data['date']).dt.date

        # Preprocess Headlines 
        import nltk
        nltk.download('punkt_tab')
        stop_words = set(stopwords.words('english'))
        def preprocess_text(text):
            words = word_tokenize(text)
            words = [word for word in words if word.isalpha()]
            words = [word.lower() for word in words if word.lower() not in stop_words]
            return ' '.join(words)
        news_data['cleaned_headline'] = news_data['headline'].apply(preprocess_text)

        # Sentiment Scoring
        analyzer = SentimentIntensityAnalyzer()
        news_data['sentiment_score'] = news_data['cleaned_headline'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

        daily_sentiment = news_data.groupby('date')['sentiment_score'].mean().reset_index()
        combined = pd.merge(stock_data, daily_sentiment, left_on='Date', right_on='date', how='inner')

        def generate_signal(score):
            if score > 0.2: return "BUY"
            elif score < -0.2: return "SELL"
            else: return "HOLD"

        combined['Signal'] = combined['sentiment_score'].apply(generate_signal)

        st.subheader("📅 Signal Table")
        st.dataframe(combined[['Date', 'Close', 'sentiment_score', 'Signal']])

        # Plot: Price vs Sentiment
        st.subheader("📉 Netflix Price & Sentiment")
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(combined['Date'], combined['Close'], label='Close Price', color='blue')
        ax1.set_ylabel('Price', color='blue')
        ax2 = ax1.twinx()
        ax2.bar(combined['Date'], combined['sentiment_score'], alpha=0.4,
                color=['green' if s >= 0 else 'red' for s in combined['sentiment_score']])
        ax2.set_ylabel('Sentiment Score', color='gray')
        st.pyplot(fig)

        # Plot: Trading Signal Markers
        st.subheader("📍 Trading Signals on Price Chart")
        fig2, ax = plt.subplots(figsize=(10, 5))
        ax.plot(combined['Date'], combined['Close'], color='black', label='Close Price')
        buy = combined[combined['Signal'] == 'BUY']
        sell = combined[combined['Signal'] == 'SELL']
        hold = combined[combined['Signal'] == 'HOLD']
        ax.scatter(buy['Date'], buy['Close'], color='green', label='BUY', marker='^', s=100)
        ax.scatter(sell['Date'], sell['Close'], color='red', label='SELL', marker='v', s=100)
        ax.scatter(hold['Date'], hold['Close'], color='blue', label='HOLD', marker='o', s=50)
        ax.legend()
        st.pyplot(fig2)

        # Gauge Sentiment
        st.subheader("🌡️ Latest Sentiment Gauge")
        latest_sentiment = daily_sentiment.iloc[-1]['sentiment_score'] if not daily_sentiment.empty else 0
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest_sentiment,
            title={'text': "Latest Sentiment"},
            gauge={'axis': {'range': [-1, 1]},
                   'steps': [
                       {'range': [-1, -0.2], 'color': "red"},
                       {'range': [-0.2, 0.2], 'color': "lightgray"},
                       {'range': [0.2, 1], 'color': "green"}]}
        ))
        st.plotly_chart(fig_gauge)

        # WordCloud
        st.subheader("☁️ Word Cloud from Headlines")
        text = " ".join(news_data['cleaned_headline'].values)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)

        # Expander & CSV
        with st.expander("📄 Raw Headlines & Scores"):
            st.dataframe(news_data[['date', 'headline', 'sentiment_score']])
        st.download_button("📥 Download Signals as CSV", combined.to_csv(index=False), "netflix_signals.csv")
else:
    st.warning("🔐 Please enter your NewsAPI key in the sidebar to begin.")
