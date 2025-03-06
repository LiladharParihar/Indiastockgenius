import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox, QTextEdit, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtWebEngineWidgets import QWebEngineView
import yfinance as yf
import pandas as pd
import numpy as np
from model import predict_lstm
from data import fetch_stock_data, fetch_realtime_data, fetch_news_sentiment
from indicators import calculate_indicators
from portfolio import manage_portfolio
from risk import assess_risk
from backtest import backtest_strategy
import pyttsx3
from sentiment import analyze_sentiment
import os
import json
import plotly.graph_objects as go
from plotly.io import to_html
import qdarkstyle
from concurrent.futures import ThreadPoolExecutor

class IndiaStockGenius(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IndiaStockGenius - Stock Analyzer by Liladhar Parihar")
        self.setGeometry(100, 100, 900, 700)
        self.engine = pyttsx3.init()
        self.lstm_model = None  # Cache LSTM model
        self.lstm_scaler = None  # Cache scaler
        self.setup_gui()
        self.portfolio = {}
        self.ensure_config()

    def ensure_config(self):
        default_config = {"forecast_days": 30, "sequence_length": 60, "chart_style": "dark", "volume_overlay": True, "prediction_range_percent": 20}
        if not os.path.exists("config.json"):
            with open("config.json", "w") as f:
                json.dump(default_config, f, indent=2)

    def setup_gui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignTop)

        header = QLabel("IndiaStockGenius")
        header.setStyleSheet("font: bold 18pt 'Segoe UI'; color: #4CAF50;")
        left_layout.addWidget(header)

        input_layout = QHBoxLayout()
        ticker_label = QLabel("Stock Ticker (e.g., RELIANCE.NS):")
        ticker_label.setStyleSheet("font: 12pt 'Segoe UI'; color: white;")
        self.ticker_entry = QLineEdit()
        self.ticker_entry.setStyleSheet("font: 11pt 'Segoe UI'; padding: 5px;")
        self.ticker_entry.setFixedWidth(200)
        analyze_button = QPushButton("Analyze")
        analyze_button.setStyleSheet("font: bold 11pt 'Segoe UI'; padding: 5px; background-color: #4CAF50; color: white;")
        analyze_button.clicked.connect(self.analyze)
        input_layout.addWidget(ticker_label)
        input_layout.addWidget(self.ticker_entry)
        input_layout.addWidget(analyze_button)
        left_layout.addLayout(input_layout)

        self.volume_check = QCheckBox("Volume Overlay")
        self.volume_check.setChecked(True)
        self.volume_check.setStyleSheet("font: 11pt 'Segoe UI'; color: white;")
        self.volume_check.stateChanged.connect(self.update_config)
        left_layout.addWidget(self.volume_check)

        button_layout = QHBoxLayout()
        portfolio_button = QPushButton("Portfolio")
        portfolio_button.setStyleSheet("font: bold 11pt 'Segoe UI'; padding: 5px; background-color: #4CAF50; color: white;")
        portfolio_button.clicked.connect(self.show_portfolio)
        backtest_button = QPushButton("Backtest")
        backtest_button.setStyleSheet("font: bold 11pt 'Segoe UI'; padding: 5px; background-color: #4CAF50; color: white;")
        backtest_button.clicked.connect(self.backtest)
        voice_button = QPushButton("Voice")
        voice_button.setStyleSheet("font: bold 11pt 'Segoe UI'; padding: 5px; background-color: #4CAF50; color: white;")
        voice_button.clicked.connect(self.voice_command)
        button_layout.addWidget(portfolio_button)
        button_layout.addWidget(backtest_button)
        button_layout.addWidget(voice_button)
        left_layout.addLayout(button_layout)

        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setStyleSheet("font: 12pt 'Segoe UI'; background-color: #333333; color: white; border: 1px solid #555555;")
        left_layout.addWidget(self.result_display)

        self.chart_view = QWebEngineView()
        self.chart_view.setMinimumSize(400, 400)
        self.chart_view.loadFinished.connect(self.on_chart_load)  # Debug loading
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(self.chart_view, 2)

        self.setStyleSheet("background-color: #2E2E2E;")

    def on_chart_load(self, ok):
        print(f"Chart load finished: {'Success' if ok else 'Failed'}")

    def update_config(self):
        config = {
            "forecast_days": 30,
            "sequence_length": 60,
            "chart_style": "dark",
            "volume_overlay": self.volume_check.isChecked(),
            "prediction_range_percent": 20
        }
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)

    def analyze(self):
        ticker = self.ticker_entry.text().upper()
        if not ticker:
            QMessageBox.critical(self, "Error", "Enter a ticker!")
            return

        with ThreadPoolExecutor() as executor:
            future_stock = executor.submit(fetch_stock_data, ticker, period="1y")
            historical = future_stock.result()

        if historical.empty:
            self.result_display.setText(f"No historical data for {ticker}")
            return

        try:
            with open("tooltips.json", "r") as f:
                tooltips = json.load(f)
        except FileNotFoundError:
            tooltips = {
                "close_price": "The closing price of the stock for the day.",
                "prediction": "Predicted price in 30 days based on LSTM model.",
                "ma50": "50-day Simple Moving Average.",
                "ma100": "100-day Simple Moving Average.",
                "ma200": "200-day Simple Moving Average.",
                "macd": "Moving Average Convergence Divergence, trend indicator.",
                "bollinger": "Middle Bollinger Band, volatility indicator.",
                "volatility": "Price range (±1 SD) over 30 days.",
                "rsi": "Relative Strength Index (0-100), measures momentum."
            }

        realtime = fetch_realtime_data(ticker)
        news_sentiment = fetch_news_sentiment(ticker)
        indicators = calculate_indicators(historical)
        result = predict_lstm(historical['Close'].values, model=self.lstm_model, scaler=self.lstm_scaler)
        if isinstance(result, dict):
            predictions = result['predictions']
            self.lstm_model = result['model']
            self.lstm_scaler = result['scaler']
        else:
            predictions = result
        risk = assess_risk(historical, predictions)

        last_price = historical['Close'].iloc[-1]
        realtime_price = realtime.get('last_price', 'N/A')
        realtime_display = f"₹{realtime_price:.2f}" if realtime_price != "N/A" else "N/A"

        pred_direction = "Go Long" if predictions[-1] > last_price else "Go Short"
        rsi_status = "Neutral (30-70)" if 30 <= indicators['RSI'] <= 70 else "Overbought (>70)" if indicators['RSI'] > 70 else "Oversold (<30)"
        volatility_range = f"{last_price - risk['volatility']:.2f} - {last_price + risk['volatility']:.2f}"

        result_text = f"Last Price: ₹{last_price:.2f}\n"
        result_text += f"Real-Time (Mock): {realtime_display}\n"
        result_text += f"Future Prediction: ₹{predictions[-1]:.2f} ({pred_direction}, Confidence: {risk['confidence']:.0%}, Sentiment: {news_sentiment})\n"
        result_text += f"Indicators: RSI={indicators['RSI']:.2f} ({rsi_status}), MA50={indicators['MA50']:.2f}, MACD={indicators['MACD']:.2f}, Bollinger={indicators['Bollinger']:.2f}\n"
        result_text += f"Risk: Volatility=₹{volatility_range}, Stop-Loss: ₹{risk['stop_loss']:.2f}"
        self.result_display.setText(result_text)

        self.plot_chart(historical, predictions, indicators, ticker)

    def plot_chart(self, historical, predictions, indicators, ticker):
        config = json.load(open("config.json", "r"))
        close_prices = historical['Close']
        pred_dates = pd.date_range(start=historical.index[-1], periods=31, freq="B")[1:]
        pred_prices = pd.Series(predictions, index=pred_dates)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=historical.index, y=close_prices, mode='lines', name='Historical Price', line=dict(color='blue'), hovertemplate="₹%{y:.2f}"))
        fig.add_trace(go.Scatter(x=pred_dates, y=pred_prices, mode='lines', name='Predicted Price (30 Days)', line=dict(color='orange', dash='dash'), hovertemplate="₹%{y:.2f}"))
        fig.add_trace(go.Scatter(x=historical.index, y=[indicators['MA50']] * len(historical), mode='lines', name='MA50', line=dict(color='green', dash='dot'), hovertemplate="₹%{y:.2f}"))

        if config["volume_overlay"]:
            fig.add_trace(go.Bar(x=historical.index, y=historical['Volume'], name='Volume', yaxis='y2', opacity=0.3, marker_color='gray'))

        layout = dict(
            title=f"{ticker} Analysis",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            template="plotly_dark" if config["chart_style"] == "dark" else "plotly",
            yaxis2=dict(title="Volume", overlaying="y", side="right") if config["volume_overlay"] else None,
            legend=dict(x=0.01, y=0.99),
            height=500,
            width=600
        )
        fig.update_layout(**layout)
        html = to_html(fig, full_html=False)
        print("Plotly HTML:", html[:100])
        self.chart_view.setHtml(html)
        fig.write_html("test_chart.html")
        print("Chart saved to test_chart.html")

    def show_portfolio(self):
        if not self.portfolio:
            QMessageBox.information(self, "Portfolio", "No stocks in portfolio. Add via code or GUI later.")
            return
        result = manage_portfolio(self.portfolio)
        self.result_display.setText(f"Portfolio Value: ₹{result['total_value']:.2f}\nPredicted Gain: {result['predicted_gain']:.2f}%")

    def backtest(self):
        ticker = self.ticker_entry.text().upper()
        if not ticker:
            QMessageBox.critical(self, "Error", "Enter a ticker!")
            return
        historical = fetch_stock_data(ticker, period="2y")
        if historical.empty:
            self.result_display.setText(f"No data for backtest on {ticker}")
            return
        result = backtest_strategy(historical)
        self.result_display.setText(f"Backtest for {ticker}: Profit={result['profit']:.2f}%, Accuracy={result['accuracy']:.2f}%")

    def voice_command(self):
        text = input("Enter voice command: ").lower()
        if "analyze" in text and "stock" in text:
            ticker = text.replace("analyze", "").replace("stock", "").strip()
            self.ticker_entry.setText(ticker.upper() + ".NS")
            self.analyze()
        elif "portfolio" in text:
            self.show_portfolio()
        elif "backtest" in text:
            self.backtest()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = IndiaStockGenius()
    window.show()
    sys.exit(app.exec_())