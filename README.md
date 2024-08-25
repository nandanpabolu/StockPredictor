
# Stock Prediction and Sentiment Analysis Chatbot

This project is a Streamlit-based web application that predicts stock prices using historical data and sentiment analysis from news articles. It also features a chatbot powered by OpenAI's GPT-3 that can answer questions about stocks.

## Features

- **Stock Price Prediction:** Predicts the next day's stock price using a hybrid model combining Linear Regression and Moving Averages.
- **Sentiment Analysis:** Analyzes news articles related to a selected stock to determine market sentiment.
- **Sentiment-Adjusted Prediction:** Adjusts the predicted stock price based on sentiment analysis.
- **Chatbot:** An interactive chatbot that can answer questions related to stocks and market trends.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/nandanpabolu/your-repo-name.git
    cd your-repo-name
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**

    Create a `.env` file in the root directory and add your OpenAI API key:

    ```bash
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

1. **Run the application:**

    ```bash
    streamlit run app.py
    ```

2. **Navigate through the sidebar:**
   - **Home:** Introduction to the app.
   - **Stock Price Prediction:** Predict the stock price for the next trading day.
   - **Sentiment Analysis:** Analyze the sentiment of news articles for the selected stock.
   - **Sentiment-Adjusted Prediction:** Get the stock price prediction adjusted by sentiment analysis.
   - **Chat:** Interact with the chatbot for stock-related queries.

## Project Structure

```
├── app.py               # Main application file
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables
├── README.md            # Project documentation
└── ...
```

## Contributing

Feel free to open issues or submit pull requests for any improvements or bug fixes.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- [OpenAI](https://openai.com/)


testing changes 

