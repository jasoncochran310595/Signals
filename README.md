# 📈 SmartBuy Signals App

**Real-time buy signals for XAUUSD (Gold), Top Cryptocurrencies, and Memecoins.**

This Streamlit dashboard provides technical analysis-based trading signals to help you identify potential buying opportunities. It uses:

- **Moving Averages** (SMA 50/200) for trend detection
- **RSI (14)** for overbought/oversold conditions
- **Volume analysis** and recent momentum
- **Composite Buy Score** (0-100) with clear recommendations

⚠️ **IMPORTANT DISCLAIMER**: This is **NOT financial advice**. Cryptocurrencies, especially memecoins, are extremely volatile and risky. You can lose all your capital. Always do your own research (DYOR), consider your risk tolerance, and consult licensed financial advisors. Past performance ≠ future results. The signals are for educational and informational purposes only.

## Features

- **XAUUSD (Gold)**: Professional forex/commodity analysis with candlestick charts
- **Top Cryptos**: BTC, ETH, SOL, XRP, ADA, AVAX with detailed signals and charts
- **Memecoins**: Popular high-risk/high-reward tokens like DOGE, SHIB, PEPE, WIF, etc.
- Interactive charts with price + SMAs + RSI
- Buy/Sell signals with explanations
- Auto-cached data (refreshes every 10 mins)

## Installation & Running

1. **Clone or download** this folder.

2. **Install dependencies** (Python 3.10+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   streamlit run app.py
   ```

4. Open your browser at the local URL shown (usually http://localhost:8501)

## How to Use

1. Navigate between the 3 tabs.
2. For detailed view, select a specific asset.
3. Click **Refresh** buttons to fetch latest data.
4. Hover on charts for exact values.
5. Read the "Why this signal?" for insights.
6. Use signals as **one data point** among many in your trading strategy.

## Customization

You can easily add more coins by editing the lists in `app.py`:
- `TOP_COINS`
- `MEME_COINS`

For custom coin, in Memecoins tab you can analyze by entering CoinGecko ID (e.g. `myro`).

## Data Sources

- Gold (XAUUSD): Polygon.io API
- Cryptocurrencies & Memecoins: CoinGecko API
- Technical indicators calculated locally with pandas

## License

MIT - Free to use and modify for personal use.

Built with ❤️ by Grok for smart traders.
