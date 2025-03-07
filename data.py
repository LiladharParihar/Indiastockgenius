import yfinance as yf
import pandas as pd
from datetime import datetime
from sentiment import analyze_sentiment  # Import the sentiment function
import requests
from bs4 import BeautifulSoup

def fetch_stock_data(ticker, period="5y"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        return data[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()

def fetch_realtime_data(ticker):
    historical = fetch_stock_data(ticker, period="1d")
    if not historical.empty:
        last_price = historical['Close'].iloc[-1]
        volume = historical['Volume'].iloc[-1]
        return {"last_price": last_price, "volume": volume}
    return {"last_price": "N/A", "volume": "N/A"}

def fetch_news_sentiment(ticker):
    try:
        # Your NewsAPI key is already added here
        NEWSAPI_KEY = "YOUR_API_KEY"  # Replace with your actual key
        url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&domains=economictimes.indiatimes.com,moneycontrol.com&apiKey={NEWSAPI_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json()['articles']
        titles = [article['title'] for article in articles[:5]]  # Top 5 headlines
        return analyze_sentiment(titles)
    except Exception as e:
        print(f"Error fetching news: {e}")
        return "Error fetching sentiment"