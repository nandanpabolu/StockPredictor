import streamlit as st
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import openai
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define important stock tickers
important_tickers = [
    "AAPL - Apple Inc.",
    "TSLA - Tesla Inc.",
    "GOOGL - Alphabet Inc. (Google)",
    "MSFT - Microsoft Corp.",
    "AMZN - Amazon.com Inc.",
    "META - Meta Platforms Inc. (Facebook)",
    "NFLX - Netflix Inc.",
    "NVIDIA - NVIDIA Corp.",
    "INTC - Intel Corp.",
    "AMD - Advanced Micro Devices Inc.",
    "BABA - Alibaba Group",
    "DIS - Walt Disney Co.",
    "NVDA - NVIDIA Corporation",
    "JPM - JPMorgan Chase & Co.",
    "BRK.B - Berkshire Hathaway Inc.",
    "V - Visa Inc."
]

# Define functions
def get_historical_prices(ticker, total_days=30, used_days=10):
    stock = yf.Ticker(ticker)
    stock_info = stock.history(period="1mo")
    closing_prices = stock_info['Close'].values[-used_days:]
    full_data = stock_info['Close'].values[-total_days:]
    return closing_prices, full_data, stock_info

def moving_average(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def predict_next_day_price(ticker):
    past_10_days, full_month_data, stock_info = get_historical_prices(ticker)
    
    X = np.arange(1, len(past_10_days) + 1).reshape(-1, 1)
    linear_model = LinearRegression().fit(X, past_10_days)
    
    next_day = np.array([[len(past_10_days) + 1]])
    predicted_price_lr = linear_model.predict(next_day)[0]
    
    # Hybrid Model (Linear Regression + Moving Average)
    ma_prediction = moving_average(full_month_data, window=5)[-1]  # 5-day moving average
    predicted_price = (predicted_price_lr + ma_prediction) / 2
    
    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(1, len(full_month_data) + 1), y=full_month_data, mode='lines+markers', name='Full Month Prices', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=np.arange(len(full_month_data) - len(past_10_days) + 1, len(full_month_data) + 1), y=past_10_days, mode='lines+markers', name='Past 10 Days Prices', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=[len(full_month_data) + 1], y=[predicted_price], mode='markers', name='Predicted Price', marker=dict(color='red', size=10)))
    
    # Add indicators
    ma_values = moving_average(full_month_data, window=5)
    fig.add_trace(go.Scatter(x=np.arange(5, len(ma_values) + 5), y=ma_values, mode='lines', name='5-day Moving Average', line=dict(color='orange')))
    
    fig.update_layout(
        title=f"Stock Price Prediction for {ticker}",
        xaxis_title="Day",
        yaxis_title="Price (USD)",
        hovermode="x",
        legend=dict(x=0, y=1),
    )
    
    st.plotly_chart(fig)
    return predicted_price

def fetch_articles(ticker):
    search_url = f"https://www.google.com/search?q={ticker} stock news&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = soup.find_all('div', class_='BNeawe vvjwJb AP7Wnd')
    return [article.get_text() for article in articles]

def analyze_sentiment(articles):
    filtered_articles = [article for article in articles if len(article) > 20 and "stock" in article.lower()]
    sentiment_scores = [TextBlob(article).sentiment.polarity for article in filtered_articles]
    positive = sum(score > 0 for score in sentiment_scores)
    negative = sum(score < 0 for score in sentiment_scores)
    total = positive + negative
    return (positive - negative) / total if total > 0 else 0

def adjust_prediction(predicted_price, sentiment_score, alpha=0.05):
    adjusted_price = predicted_price * (1 + alpha * sentiment_score)
    confidence_score = 50 + (sentiment_score * 50)
    return adjusted_price, confidence_score

def generate_response(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return str(e)

# Streamlit UI
st.sidebar.title("Stock Chatbot")
selected_ticker = st.sidebar.selectbox("Select a Stock", important_tickers)
ticker = selected_ticker.split(" - ")[0]

option = st.sidebar.selectbox("Choose a function", ["Home", "Stock Price Prediction", "Sentiment Analysis", "Sentiment-Adjusted Prediction", "Chat"])

if option == "Home":
    st.title("Welcome to the Stock Chatbot!")
    st.write("Select an option from the sidebar to get started.")

elif option == "Stock Price Prediction":
    st.title(f"Stock Price Prediction for {ticker}")
    predicted_price = predict_next_day_price(ticker)
    st.write(f"The predicted price for {ticker} on the next trading day is ${predicted_price:.2f}.")

elif option == "Sentiment Analysis":
    st.title(f"Sentiment Analysis for {ticker}")
    articles = fetch_articles(ticker)
    sentiment_score = analyze_sentiment(articles)
    st.write(f"The sentiment score for {ticker} is {sentiment_score:.2f}.")
    st.write("Sample Articles Analyzed:")
    for i, article in enumerate(articles[:5], start=1):
        st.write(f"{i}. {article}")

elif option == "Sentiment-Adjusted Prediction":
    st.title(f"Sentiment-Adjusted Prediction for {ticker}")
    predicted_price = predict_next_day_price(ticker)
    articles = fetch_articles(ticker)
    sentiment_score = analyze_sentiment(articles)
    
    # Dynamic sentiment impact adjustment
    alpha = st.sidebar.slider("Adjust Sentiment Impact (Alpha)", 0.01, 0.20, 0.05)
    
    adjusted_price, confidence_score = adjust_prediction(predicted_price, sentiment_score, alpha)
    
    st.write(f"### Original Predicted Price: ${predicted_price:.2f}")
    st.write(f"### Sentiment Score: {sentiment_score:.2f}")
    st.write(f"### Confidence Level: {confidence_score:.2f}%")
    st.write(f"### Sentiment-Adjusted Predicted Price: ${adjusted_price:.2f}")
    st.write("### Calculations Overview:")
    st.write(f"- **Positive Sentiment Influence:** {sentiment_score * 100:.2f}%")
    st.markdown("- **Adjusted Price Formula:** `Original Price * (1 + α * Sentiment Score)`")
    st.markdown("- **Confidence Level:** `50 + (Sentiment Score * 50)`")

elif option == "Chat":
    st.title("Chat with the Stock Bot")
    user_input = st.text_input("You:", "")
    if user_input:
        response = generate_response(user_input)
        st.write(f"**You:** {user_input}")
        st.write(f"**Bot:** {response}")