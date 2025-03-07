import sys
from check_dependencies import check_dependencies

# Check dependencies before importing other modules
if not check_dependencies(auto_install="--install" in sys.argv):
    print("Missing required dependencies. Please run: python check_dependencies.py --install")
    sys.exit(1)

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QLineEdit, QPushButton, QCheckBox, QTextEdit, QMessageBox,
                           QTabWidget, QComboBox, QCompleter, QSizePolicy)
from PyQt5.QtCore import Qt, QUrl, QStringListModel, QTimer
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
        self.setGeometry(100, 100, 1200, 800)  # Made window larger
        self.engine = pyttsx3.init()
        self.lstm_model = None
        self.lstm_scaler = None
        
        # Setup timer for processing events
        self.process_timer = QTimer()
        self.process_timer.timeout.connect(lambda: QApplication.processEvents())
        self.process_timer.start(100)  # Process events every 100ms
        
        self.setup_gui()
        self.portfolio = {}
        self.ensure_config()
        self.console_log = []  # Store console logs
        self.setup_stock_autocomplete()
        self.update_market_data()  # Initial market data update

    def ensure_config(self):
        default_config = {"forecast_days": 30, "sequence_length": 60, "chart_style": "dark", "volume_overlay": True, "prediction_range_percent": 20}
        if not os.path.exists("config.json"):
            with open("config.json", "w") as f:
                json.dump(default_config, f, indent=2)

    def setup_stock_autocomplete(self):
        """Setup autocomplete for stock ticker input"""
        # Common Indian stocks with NS suffix
        common_stocks = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "HINDUNILVR.NS", "HDFC.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS",
            "KOTAKBANK.NS", "LT.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS",
            "WIPRO.NS", "HCLTECH.NS", "ULTRACEMCO.NS", "TITAN.NS", "BAJFINANCE.NS",
            "SUNPHARMA.NS", "TATAMOTORS.NS", "ADANIENT.NS", "TATASTEEL.NS", "NTPC.NS",
            "POWERGRID.NS", "TECHM.NS", "BAJAJFINSV.NS", "HINDALCO.NS", "ONGC.NS"
        ]
        
        # Create completer
        completer = QCompleter(common_stocks)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        
        # Set completer for ticker entry
        self.ticker_entry.setCompleter(completer)
        self.ticker_entry.setPlaceholderText("Enter stock ticker (e.g., RELIANCE.NS)")

    def setup_gui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel setup
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignTop)

        # Header
        header = QLabel("IndiaStockGenius")
        header.setStyleSheet("font: bold 18pt 'Segoe UI'; color: #4CAF50;")
        left_layout.addWidget(header)

        # Input section
        input_layout = QHBoxLayout()
        ticker_label = QLabel("Stock Ticker:")
        ticker_label.setStyleSheet("font: 12pt 'Segoe UI'; color: white;")
        self.ticker_entry = QLineEdit()
        self.ticker_entry.setStyleSheet("""
            QLineEdit {
                font: 11pt 'Segoe UI';
                padding: 5px;
                border: 1px solid #555;
                border-radius: 3px;
                background: #333;
                color: white;
            }
            QLineEdit:focus {
                border: 1px solid #4CAF50;
                background: #3a3a3a;
            }
        """)
        self.ticker_entry.setFixedWidth(200)
        
        # Add prediction term selector
        self.prediction_term = QComboBox()
        self.prediction_term.addItems(["Short Term (7 days)", "Medium Term (30 days)", "Long Term (90 days)"])
        self.prediction_term.setStyleSheet("""
            QComboBox {
                font: 11pt 'Segoe UI';
                padding: 5px;
                border: 1px solid #555;
                border-radius: 3px;
                background: #333;
                color: white;
                min-width: 150px;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 10px;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
            }
            QComboBox:on {
                border: 1px solid #4CAF50;
            }
            QComboBox QAbstractItemView {
                background-color: #333;
                color: white;
                selection-background-color: #4CAF50;
            }
        """)
        
        analyze_button = QPushButton("Analyze")
        analyze_button.setStyleSheet("""
            QPushButton {
                font: bold 11pt 'Segoe UI';
                padding: 8px 16px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        analyze_button.clicked.connect(self.analyze)
        
        input_layout.addWidget(ticker_label)
        input_layout.addWidget(self.ticker_entry)
        input_layout.addWidget(self.prediction_term)
        input_layout.addWidget(analyze_button)
        left_layout.addLayout(input_layout)

        # Analysis Summary Box
        self.summary_frame = QWidget()
        self.summary_frame.setStyleSheet("""
            QWidget {
                background-color: #333333;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 10px;
            }
            QLabel {
                font: 11pt 'Segoe UI';
                color: white;
                padding: 2px;
            }
        """)
        self.summary_layout = QVBoxLayout(self.summary_frame)
        
        # Market Status
        self.market_status = QLabel("Market Status: Open")
        self.market_status.setStyleSheet("font: 11pt 'Segoe UI'; color: #4CAF50;")
        self.summary_layout.addWidget(self.market_status)
        
        # Sensex/Nifty Status with rates
        self.sensex_status = QLabel("SENSEX: 72,500 ▲ 0.5%")
        self.nifty_status = QLabel("NIFTY: 22,100 ▲ 0.3%")
        self.sensex_status.setStyleSheet("color: #4CAF50;")  # Green for positive
        self.nifty_status.setStyleSheet("color: #4CAF50;")   # Green for positive
        self.summary_layout.addWidget(self.sensex_status)
        self.summary_layout.addWidget(self.nifty_status)
        
        # Analysis Results
        self.analysis_results = QTextEdit()
        self.analysis_results.setReadOnly(True)
        self.analysis_results.setStyleSheet("""
            QTextEdit {
                font: 11pt 'Segoe UI';
                background-color: #2A2A2A;
                color: white;
                border: none;
                padding: 5px;
            }
        """)
        self.analysis_results.setMinimumHeight(200)
        self.summary_layout.addWidget(self.analysis_results)
        
        # Add the summary frame to main layout
        left_layout.addWidget(self.summary_frame)

        # Action buttons
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

        # Console Log
        console_container = QWidget()
        console_layout = QVBoxLayout(console_container)
        self.console_display = QTextEdit()
        self.console_display.setReadOnly(True)
        self.console_display.setStyleSheet("font: 11pt 'Consolas'; background-color: #1E1E1E; color: #D4D4D4; border: 1px solid #555555;")
        console_layout.addWidget(self.console_display)
        left_layout.addWidget(console_container)

        # Chart view with responsive size
        self.chart_view = QWebEngineView()
        self.chart_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.chart_view.loadFinished.connect(self.on_chart_load)

        # Add widgets to main layout with proper sizing
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(self.chart_view, 2)

        self.setStyleSheet("background-color: #2E2E2E;")

    def log_to_console(self, message):
        """Log message to console with immediate display"""
        self.console_log.append(message)
        self.console_display.append(message)
        self.console_display.verticalScrollBar().setValue(
            self.console_display.verticalScrollBar().maximum()
        )  # Auto-scroll to bottom
        print(message)  # Still print to actual console for debugging
        QApplication.processEvents()  # Process events immediately

    def on_chart_load(self, ok):
        self.log_to_console(f"Chart load finished: {'Success' if ok else 'Failed'}")

    def update_config(self):
        config = {
            "forecast_days": 30,
            "sequence_length": 60,
            "chart_style": "dark",
            "volume_overlay": True,  # Always enabled
            "prediction_range_percent": 20
        }
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)

    def analyze(self):
        ticker = self.ticker_entry.text().upper()
        if not ticker:
            QMessageBox.critical(self, "Error", "Enter a ticker!")
            return

        self.log_to_console(f"\nAnalyzing {ticker}...")
        
        # Get prediction term
        pred_term = self.prediction_term.currentText()
        if "Short Term" in pred_term:
            forecast_days = 7
        elif "Medium Term" in pred_term:
            forecast_days = 30
        else:  # Long Term
            forecast_days = 90
            
        self.log_to_console(f"Selected {pred_term} analysis ({forecast_days} days forecast)")

        try:
            with ThreadPoolExecutor() as executor:
                self.log_to_console("Fetching historical data...")
                future_stock = executor.submit(fetch_stock_data, ticker, period="2y")  # Extended period for better training
                historical = future_stock.result()

            if historical.empty:
                self.analysis_results.setText(f"No historical data for {ticker}")
                return

            self.log_to_console("Calculating technical indicators...")
            indicators = calculate_indicators(historical)
            
            self.log_to_console("Fetching real-time data...")
            realtime = fetch_realtime_data(ticker)
            
            self.log_to_console("Analyzing news sentiment...")
            news_sentiment = fetch_news_sentiment(ticker)
            
            self.log_to_console("Running LSTM prediction model...")
            result = predict_lstm(historical['Close'].values, forecast_days=forecast_days, model=self.lstm_model, scaler=self.lstm_scaler)
            if isinstance(result, dict):
                predictions = result['predictions']
                confidence = result['confidence']
                self.lstm_model = result['model']
                self.lstm_scaler = result['scaler']
            else:
                predictions = result
                confidence = 0.5  # Default if not provided
            
            risk = assess_risk(historical, predictions)

            last_price = historical['Close'].iloc[-1]
            realtime_price = realtime.get('last_price', 'N/A')
            realtime_display = f"₹{realtime_price:.2f}" if realtime_price != "N/A" else "N/A"

            pred_direction = "Go Long" if predictions[-1] > last_price else "Go Short"
            pred_change = ((predictions[-1] - last_price) / last_price) * 100
            
            rsi_status = "Neutral (30-70)" if 30 <= indicators['RSI'] <= 70 else "Overbought (>70)" if indicators['RSI'] > 70 else "Oversold (<30)"
            volatility_range = f"{last_price - risk['volatility']:.2f} - {last_price + risk['volatility']:.2f}"

            # Detailed analysis output
            result_text = f"Analysis Results for {ticker} ({pred_term})\n"
            result_text += f"{'='*50}\n\n"
            result_text += f"Current Price: ₹{last_price:.2f}\n"
            result_text += f"Real-Time Price: {realtime_display}\n\n"
            
            result_text += f"Prediction ({forecast_days} days):\n"
            result_text += f"Direction: {pred_direction} ({pred_change:+.2f}%)\n"
            result_text += f"Target Price: ₹{predictions[-1]:.2f}\n"
            result_text += f"Model Confidence: {confidence:.1%}\n"
            result_text += f"Market Sentiment: {news_sentiment}\n\n"
            
            result_text += f"Technical Indicators:\n"
            result_text += f"RSI = {indicators['RSI']:.2f} ({rsi_status})\n"
            result_text += f"MA50 = ₹{indicators['MA50']:.2f}\n"
            result_text += f"MACD = {indicators['MACD']:.2f}\n"
            result_text += f"Bollinger = ₹{indicators['Bollinger']:.2f}\n\n"
            
            result_text += f"Risk Analysis:\n"
            result_text += f"Volatility Range: ₹{volatility_range}\n"
            result_text += f"Suggested Stop-Loss: ₹{risk['stop_loss']:.2f}"
            
            self.analysis_results.setText(result_text)
            self.log_to_console("Analysis complete. Generating chart...")
            
            self.plot_chart(historical, predictions, indicators, ticker)
            
        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            self.log_to_console(error_msg)
            self.analysis_results.setText(error_msg)

    def plot_chart(self, historical, predictions, indicators, ticker):
        config = json.load(open("config.json", "r"))
        close_prices = historical['Close']
        
        # Use the actual length of predictions instead of hardcoding
        pred_dates = pd.date_range(start=historical.index[-1], periods=len(predictions) + 1, freq="B")[1:]
        pred_prices = pd.Series(predictions, index=pred_dates)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=historical.index, y=close_prices, mode='lines', name='Historical Price', line=dict(color='blue'), hovertemplate="₹%{y:.2f}"))
        fig.add_trace(go.Scatter(x=pred_dates, y=pred_prices, mode='lines', name=f'Predicted Price ({len(predictions)} Days)', line=dict(color='orange', dash='dash'), hovertemplate="₹%{y:.2f}"))
        fig.add_trace(go.Scatter(x=historical.index, y=[indicators['MA50']] * len(historical), mode='lines', name='MA50', line=dict(color='green', dash='dot'), hovertemplate="₹%{y:.2f}"))

        if config["volume_overlay"]:
            fig.add_trace(go.Bar(x=historical.index, y=historical['Volume'], name='Volume', yaxis='y2', opacity=0.3, marker_color='gray'))

        # Add copyright text
        fig.add_annotation(
            text="© Liladhar Parihar",
            xref="paper",
            yref="paper",
            x=1,
            y=-0.15,  # Move it lower to avoid overlap
            showarrow=False,
            font=dict(size=10, color="gray"),
            xanchor="right"
        )

        layout = dict(
            title=f"{ticker} Analysis",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            template="plotly_dark",
            yaxis2=dict(title="Volume", overlaying="y", side="right") if config["volume_overlay"] else None,
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=50, r=50, t=30, b=50),  # Increased bottom margin for copyright
            autosize=True,
            xaxis=dict(
                rangeslider=dict(visible=False),
                showgrid=True,
            ),
            yaxis=dict(
                showgrid=True,
            ),
        )
        fig.update_layout(**layout)
        
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    html, body {{
                        margin: 0;
                        padding: 0;
                        width: 100%;
                        height: 100vh;
                        overflow: hidden;
                        background-color: #2E2E2E;
                    }}
                    #chart-container {{
                        position: absolute;
                        top: 0;
                        left: 0;
                        right: 0;
                        bottom: 0;
                        width: 100%;
                        height: 100%;
                    }}
                    .js-plotly-plot, .plot-container, .plotly {{
                        width: 100% !important;
                        height: 100% !important;
                    }}
                </style>
            </head>
            <body>
                <div id="chart-container">
                    {to_html(fig, full_html=False, include_plotlyjs=False, config={
                        'responsive': True,
                        'displayModeBar': True,
                        'scrollZoom': True,
                        'displaylogo': False,
                    })}
                </div>
                <script>
                    function resizeChart() {{
                        var d3 = Plotly.d3;
                        var gd3 = d3.select('.plotly-graph-div');
                        var gd = gd3.node();
                        if (gd) {{
                            var rect = gd.getBoundingClientRect();
                            Plotly.relayout(gd, {{
                                width: rect.width,
                                height: rect.height
                            }});
                        }}
                    }}
                    window.addEventListener('resize', resizeChart);
                    document.addEventListener('DOMContentLoaded', resizeChart);
                    resizeChart();
                </script>
            </body>
            </html>
            """
            
            self.chart_view.setHtml(html_content, baseUrl=QUrl("https://cdn.plot.ly/"))
            
        except Exception as e:
            print(f"Error generating chart: {e}")
            self.analysis_results.append("\nError: Failed to generate chart. Check console for details.")

    def show_portfolio(self):
        if not self.portfolio:
            QMessageBox.information(self, "Portfolio", "No stocks in portfolio. Add via code or GUI later.")
            return
        result = manage_portfolio(self.portfolio)
        self.analysis_results.setText(f"Portfolio Value: ₹{result['total_value']:.2f}\nPredicted Gain: {result['predicted_gain']:.2f}%")

    def backtest(self):
        ticker = self.ticker_entry.text().upper()
        if not ticker:
            QMessageBox.critical(self, "Error", "Enter a ticker!")
            return
        historical = fetch_stock_data(ticker, period="2y")
        if historical.empty:
            self.analysis_results.setText(f"No data for backtest on {ticker}")
            return
        result = backtest_strategy(historical)
        self.analysis_results.setText(f"Backtest for {ticker}: Profit={result['profit']:.2f}%, Accuracy={result['accuracy']:.2f}%")

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

    def update_market_data(self):
        """Update market data with real-time values"""
        try:
            # Fetch Sensex data
            sensex = yf.Ticker("^BSESN")
            sensex_data = sensex.history(period="1d")
            sensex_current = sensex_data['Close'].iloc[-1]
            sensex_prev = sensex_data['Open'].iloc[-1]
            sensex_change = ((sensex_current - sensex_prev) / sensex_prev) * 100
            
            # Fetch Nifty data
            nifty = yf.Ticker("^NSEI")
            nifty_data = nifty.history(period="1d")
            nifty_current = nifty_data['Close'].iloc[-1]
            nifty_prev = nifty_data['Open'].iloc[-1]
            nifty_change = ((nifty_current - nifty_prev) / nifty_prev) * 100
            
            # Update labels with colors only for arrows
            sensex_color = "#4CAF50" if sensex_change >= 0 else "#FF5252"
            nifty_color = "#4CAF50" if nifty_change >= 0 else "#FF5252"
            
            # Create colored arrow spans
            sensex_arrow = f'<span style="color: {sensex_color}">{"▲" if sensex_change >= 0 else "▼"}</span>'
            nifty_arrow = f'<span style="color: {nifty_color}">{"▲" if nifty_change >= 0 else "▼"}</span>'
            
            # Set text with HTML formatting
            self.sensex_status.setText(f'SENSEX: {sensex_current:,.0f} {sensex_arrow} {abs(sensex_change):.1f}%')
            self.nifty_status.setText(f'NIFTY: {nifty_current:,.0f} {nifty_arrow} {abs(nifty_change):.1f}%')
            
            # Enable rich text interpretation
            self.sensex_status.setTextFormat(Qt.RichText)
            self.nifty_status.setTextFormat(Qt.RichText)
            
            # Set base style (white text)
            self.sensex_status.setStyleSheet("color: white;")
            self.nifty_status.setStyleSheet("color: white;")
            
            # Update market status
            current_time = pd.Timestamp.now('Asia/Kolkata').time()
            market_open = pd.Timestamp('09:15:00').time()
            market_close = pd.Timestamp('15:30:00').time()
            
            if market_open <= current_time <= market_close:
                self.market_status.setText("Market Status: Open")
                self.market_status.setStyleSheet("color: #4CAF50;")
            else:
                self.market_status.setText("Market Status: Closed")
                self.market_status.setStyleSheet("color: #FF5252;")
                
        except Exception as e:
            self.log_to_console(f"Error updating market data: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = IndiaStockGenius()
    window.show()
    sys.exit(app.exec_())