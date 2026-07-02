import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from polygon import RESTClient
from coingecko_sdk import Coingecko
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="SmartBuy Signals | XAUUSD + Crypto + Memecoins",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better look
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1f77b4;}
    .signal-strong {background-color: #d4edda; padding: 10px; border-radius: 8px; border-left: 6px solid #28a745;}
    .signal-buy {background-color: #d1ecf1; padding: 10px; border-radius: 8px; border-left: 6px solid #17a2b8;}
    .signal-hold {background-color: #fff3cd; padding: 10px; border-radius: 8px; border-left: 6px solid #ffc107;}
    .signal-sell {background-color: #f8d7da; padding: 10px; border-radius: 8px; border-left: 6px solid #dc3545;}
    .metric-card {background-color: #f8f9fa; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .disclaimer {font-size: 0.85rem; color: #6c757d; border-top: 1px solid #dee2e6; padding-top: 10px; margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

@st.cache_data(ttl=600, show_spinner=False)
def get_gold_data(days: int = 365) -> pd.DataFrame:
    """Fetch daily OHLCV for XAUUSD (Gold spot) from Polygon"""
    try:
        client = RESTClient()
        to_date = datetime.now().date()
        from_date = to_date - timedelta(days=days)
        aggs = list(client.list_aggs(
            "C:XAUUSD", 1, "day",
            from_=from_date.isoformat(),
            to=to_date.isoformat(),
            limit=50000
        ))
        if not aggs:
            return pd.DataFrame()
        df = pd.DataFrame([{
            "timestamp": a.timestamp,
            "open": a.open,
            "high": a.high,
            "low": a.low,
            "close": a.close,
            "volume": getattr(a, 'volume', 0) or 0
        } for a in aggs])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error fetching Gold data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def get_crypto_ohlcv(coin_id: str, days: int = 200) -> pd.DataFrame:
    """Fetch OHLCV from CoinGecko (daily or aggregated)"""
    try:
        client = Coingecko()
        ohlc_data = client.coins.ohlc.get(id=coin_id, vs_currency="usd", days=days)
        if not ohlc_data:
            return pd.DataFrame()
        df = pd.DataFrame(ohlc_data, columns=["timestamp", "open", "high", "low", "close"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        
        # Add volume from market_chart
        try:
            chart = client.coins.market_chart.get(id=coin_id, vs_currency="usd", days=days)
            if hasattr(chart, "total_volumes") and chart.total_volumes:
                vol_df = pd.DataFrame(chart.total_volumes, columns=["timestamp", "volume"])
                df = df.merge(vol_df, on="timestamp", how="left")
            else:
                df["volume"] = np.nan
        except:
            df["volume"] = np.nan
            
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def get_coin_market_data(coin_id: str) -> dict:
    """Get current market stats from CoinGecko"""
    try:
        client = Coingecko()
        markets = client.coins.markets.get(
            vs_currency="usd",
            ids=coin_id,
            per_page=1,
            page=1
        )
        if not markets:
            return {}
        m = markets[0]
        return {
            "current_price": m.current_price or 0,
            "market_cap": m.market_cap or 0,
            "total_volume": m.total_volume or 0,
            "price_change_24h": m.price_change_percentage_24h or 0,
            "market_cap_rank": getattr(m, "market_cap_rank", None),
            "name": m.name,
            "symbol": m.symbol.upper() if m.symbol else coin_id.upper()
        }
    except Exception:
        return {}

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SMA50, SMA200, RSI(14)"""
    if df.empty or len(df) < 10:
        return df
    df = df.copy()
    n = len(df)
    
    # SMAs - adaptive periods
    if n >= 50:
        df["sma_50"] = df["close"].rolling(window=50, min_periods=20).mean()
    else:
        df["sma_50"] = df["close"].rolling(window=max(10, n//3), min_periods=5).mean()
    
    if n >= 200:
        df["sma_200"] = df["close"].rolling(window=200, min_periods=50).mean()
    else:
        df["sma_200"] = np.nan
    
    # RSI(14)
    if n > 14:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=14, min_periods=7).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=7).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
    else:
        df["rsi"] = np.nan
    
    return df

def get_signal(df: pd.DataFrame) -> tuple:
    """Generate buy signal, score (0-100), and list of reasons"""
    if df.empty or len(df) < 5:
        return "⚪ INSUFFICIENT DATA", 0, ["Not enough price history for reliable signals."]
    
    latest = df.iloc[-1]
    score = 0
    reasons = []
    
    price = latest["close"]
    rsi = latest.get("rsi", np.nan)
    sma50 = latest.get("sma_50", np.nan)
    sma200 = latest.get("sma_200", np.nan)
    volume = latest.get("volume", np.nan)
    
    # 1. TREND (up to 45 points)
    has_long_trend = pd.notna(sma200) and sma200 > 0
    if has_long_trend:
        if pd.notna(sma50) and sma50 > sma200:
            score += 25
            reasons.append("✅ Bullish trend: SMA50 > SMA200 (Golden Cross territory)")
        else:
            reasons.append("⚠️ Bearish or flat long-term trend (SMA50 < SMA200)")
        
        if pd.notna(sma50) and price > sma50:
            score += 20
            reasons.append("✅ Short-term bullish: Price trading above SMA50")
        elif pd.notna(sma50):
            reasons.append("⚠️ Price below SMA50 (short-term weakness)")
    else:
        # Short history fallback
        if pd.notna(sma50) and price > sma50:
            score += 30
            reasons.append("✅ Price above short-term moving average (bullish momentum)")
        else:
            reasons.append("⚠️ Price below short-term MA")
    
    # 2. RSI (up to 30 points)
    if pd.notna(rsi):
        if rsi < 30:
            score += 30
            reasons.append(f"🟢 RSI = {rsi:.1f} → OVERSOLD (strong potential reversal/buy)")
        elif rsi < 45:
            score += 18
            reasons.append(f"🟢 RSI = {rsi:.1f} → Neutral-bullish zone")
        elif rsi > 70:
            score -= 15
            reasons.append(f"🔴 RSI = {rsi:.1f} → OVERBOUGHT (risk of pullback)")
        else:
            score += 8
            reasons.append(f"➖ RSI = {rsi:.1f} → Neutral")
    
    # 3. MOMENTUM (up to 15 points)
    if len(df) >= 8:
        price_7d_ago = df["close"].iloc[-8]
        if price_7d_ago > 0:
            change_7d = ((price - price_7d_ago) / price_7d_ago) * 100
            if change_7d > 8:
                score += 15
                reasons.append(f"🚀 Strong momentum: +{change_7d:.1f}% over last week")
            elif change_7d > 3:
                score += 8
                reasons.append(f"📈 Positive momentum: +{change_7d:.1f}% last week")
            elif change_7d < -12:
                score -= 8
                reasons.append(f"📉 Weak momentum: {change_7d:.1f}% last week")
    
    # 4. VOLUME (up to 10 points)
    if pd.notna(volume) and len(df) > 20:
        avg_vol = df["volume"].rolling(20, min_periods=5).mean().iloc[-1]
        if avg_vol > 0 and volume > (avg_vol * 1.8):
            score += 10
            reasons.append("📊 High volume spike (confirmation of interest)")
    
    # Clamp score
    score = max(0, min(100, int(score)))
    
    # Final signal
    if score >= 75:
        signal = "🟢 STRONG BUY"
    elif score >= 55:
        signal = "🟢 BUY"
    elif score >= 40:
        signal = "🟡 HOLD / WATCH"
    else:
        signal = "🔴 SELL / AVOID"
    
    return signal, score, reasons

def make_price_rsi_chart(df: pd.DataFrame, title: str, symbol: str = "") -> go.Figure:
    """Create professional candlestick + MA + RSI chart"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.68, 0.32],
        subplot_titles=(f"{title} ({symbol})", "RSI (14-period)")
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ),
        row=1, col=1
    )
    
    # Moving Averages
    if "sma_50" in df.columns and df["sma_50"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df["sma_50"],
                name="SMA 50", line=dict(color="#ff9800", width=2)
            ),
            row=1, col=1
        )
    if "sma_200" in df.columns and df["sma_200"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df["sma_200"],
                name="SMA 200", line=dict(color="#2196f3", width=2)
            ),
            row=1, col=1
        )
    
    # RSI
    if "rsi" in df.columns and df["rsi"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df["rsi"],
                name="RSI", line=dict(color="#9c27b0", width=2)
            ),
            row=2, col=1
        )
        # Overbought/Oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="#f44336", line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#4caf50", line_width=1, row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1, title="RSI")
    
    fig.update_layout(
        height=620,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified"
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    
    return fig

# ==================== DATA LISTS ====================

TOP_COINS = [
    {"name": "Bitcoin", "id": "bitcoin", "symbol": "BTC"},
    {"name": "Ethereum", "id": "ethereum", "symbol": "ETH"},
    {"name": "Solana", "id": "solana", "symbol": "SOL"},
    {"name": "XRP", "id": "ripple", "symbol": "XRP"},
    {"name": "Cardano", "id": "cardano", "symbol": "ADA"},
    {"name": "Avalanche", "id": "avalanche-2", "symbol": "AVAX"},
]

MEME_COINS = [
    {"name": "Dogecoin", "id": "dogecoin", "symbol": "DOGE"},
    {"name": "Shiba Inu", "id": "shiba-inu", "symbol": "SHIB"},
    {"name": "Pepe", "id": "pepe", "symbol": "PEPE"},
    {"name": "dogwifhat", "id": "dogwifhat", "symbol": "WIF"},
    {"name": "Bonk", "id": "bonk", "symbol": "BONK"},
    {"name": "FLOKI", "id": "floki", "symbol": "FLOKI"},
    {"name": "Brett (Based)", "id": "based-brett", "symbol": "BRETT"},
    {"name": "Popcat", "id": "popcat", "symbol": "POPCAT"},
]

# ==================== MAIN APP ====================

st.sidebar.title("📈 SmartBuy Signals")
st.sidebar.markdown("**Gold • Top Coins • Memecoins**")
st.sidebar.divider()

st.sidebar.markdown("### Quick Tips")
st.sidebar.info("""
- **Strong Buy (75+)**: High conviction entry
- **Buy (55-74)**: Good risk/reward
- **Hold**: Wait for better setup
- **Sell/Avoid**: High risk or overextended
""")

st.sidebar.markdown("### Risk Warning")
st.sidebar.error("""
Memecoins are **extremely speculative**. 
Most go to zero. Only invest what you can afford to lose.
""")

st.sidebar.divider()
st.sidebar.caption("Data refreshes automatically every ~10 minutes. Built with Polygon + CoinGecko APIs.")

# Header
st.markdown('<p class="main-header">📈 SmartBuy Signals</p>', unsafe_allow_html=True)
st.markdown("**Technical Analysis Buy Signals** for XAUUSD, Major Cryptos & Popular Memecoins • Updated live")

# Disclaimer
with st.expander("⚠️ Important Disclaimer - Read Before Using"):
    st.markdown("""
    This application provides **informational signals only** based on historical price action and technical indicators (RSI, Moving Averages, Volume). 
    
    **It is NOT financial, investment, or trading advice.** 
    
    - Cryptocurrency markets are highly volatile. You can lose 100% of your investment.
    - Memecoins are particularly risky and often driven by hype/social media rather than fundamentals.
    - Past performance does not guarantee future results.
    - Always conduct your own research, consider multiple timeframes, on-chain data, news, and your personal financial situation.
    - The developers are not responsible for any financial losses incurred based on these signals.
    """)

# ==================== TABS ====================
tab_gold, tab_top, tab_meme = st.tabs(["🥇 XAUUSD (Gold)", "🪙 Top Cryptocurrencies", "🐶 Memecoins"])

# ---------- GOLD TAB ----------
with tab_gold:
    st.header("🥇 Gold Spot (XAUUSD)")
    st.caption("Professional analysis for the safe-haven asset")
    
    col_left, col_right = st.columns([0.38, 0.62])
    
    with col_left:
        if st.button("🔄 Refresh Gold Data", key="refresh_gold"):
            st.cache_data.clear()
            st.rerun()
        
        gold_df = get_gold_data(days=400)
        
        if gold_df.empty:
            st.error("Could not load Gold data. Please try again later.")
        else:
            gold_df = calculate_indicators(gold_df)
            latest = gold_df.iloc[-1]
            
            # Metrics
            prev_close = gold_df.iloc[-2]["close"] if len(gold_df) > 1 else latest["close"]
            chg_24h = ((latest["close"] - prev_close) / prev_close * 100) if prev_close > 0 else 0
            
            st.metric(
                label="Current Price (USD/oz)",
                value=f"${latest['close']:,.2f}",
                delta=f"{chg_24h:+.2f}% (24h)"
            )
            
            # Signal
            signal, score, reasons = get_signal(gold_df)
            
            if "STRONG BUY" in signal:
                st.markdown(f'<div class="signal-strong"><h3>{signal}</h3><p>Buy Score: <b>{score}/100</b></p></div>', unsafe_allow_html=True)
            elif "BUY" in signal:
                st.markdown(f'<div class="signal-buy"><h3>{signal}</h3><p>Buy Score: <b>{score}/100</b></p></div>', unsafe_allow_html=True)
            elif "HOLD" in signal:
                st.markdown(f'<div class="signal-hold"><h3>{signal}</h3><p>Buy Score: <b>{score}/100</b></p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="signal-sell"><h3>{signal}</h3><p>Buy Score: <b>{score}/100</b></p></div>', unsafe_allow_html=True)
            
            # Key stats
            st.markdown("**Key Indicators**")
            rsi_val = latest.get("rsi", np.nan)
            sma50_val = latest.get("sma_50", np.nan)
            sma200_val = latest.get("sma_200", np.nan)
            
            ind_cols = st.columns(3)
            ind_cols[0].metric("RSI (14)", f"{rsi_val:.1f}" if pd.notna(rsi_val) else "N/A")
            ind_cols[1].metric("SMA 50", f"${sma50_val:,.0f}" if pd.notna(sma50_val) else "N/A")
            ind_cols[2].metric("SMA 200", f"${sma200_val:,.0f}" if pd.notna(sma200_val) else "N/A (short history)")
            
            with st.expander("📋 Why this signal? (Click to expand)", expanded=True):
                for reason in reasons:
                    st.write(f"• {reason}")
                st.caption("Signals combine trend, momentum, RSI and volume. Higher score = stronger buy opportunity.")
    
    with col_right:
        if not gold_df.empty:
            fig = make_price_rsi_chart(gold_df, "XAUUSD Gold Spot", "XAUUSD")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Chart will appear once data loads.")

# ---------- TOP CRYPTOS TAB ----------
with tab_top:
    st.header("🪙 Top Cryptocurrencies")
    st.caption("Major coins by market cap with technical buy signals")
    
    # Selector
    coin_names = [c["name"] for c in TOP_COINS]
    selected_name = st.selectbox("Select coin for detailed analysis:", coin_names, index=0)
    selected_coin = next((c for c in TOP_COINS if c["name"] == selected_name), TOP_COINS[0])
    
    if st.button("🔄 Refresh Selected Coin Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Detailed view for selected
    st.subheader(f"Analysis: {selected_coin['name']} ({selected_coin['symbol']})")
    
    ohlcv = get_crypto_ohlcv(selected_coin["id"], days=200)
    market = get_coin_market_data(selected_coin["id"])
    
    if ohlcv.empty:
        st.warning("Could not fetch data for this coin. It may be a temporary API issue.")
    else:
        ohlcv = calculate_indicators(ohlcv)
        signal, score, reasons = get_signal(ohlcv)
        latest = ohlcv.iloc[-1]
        
        # Top metrics row
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        
        price = market.get("current_price", latest["close"])
        chg = market.get("price_change_24h", 0)
        
        mcol1.metric("Price", f"${price:,.4f}" if price < 10 else f"${price:,.2f}", f"{chg:+.2f}%")
        mcol2.metric("Market Cap Rank", f"#{market.get('market_cap_rank', '?')}")
        mcol3.metric("24h Volume", f"${market.get('total_volume', 0):,.0f}")
        mcol4.metric("Buy Score", f"{score}/100")
        
        # Signal box
        if "STRONG BUY" in signal:
            st.markdown(f'<div class="signal-strong"><h2 style="margin:0">{signal}</h2></div>', unsafe_allow_html=True)
        elif "BUY" in signal:
            st.markdown(f'<div class="signal-buy"><h2 style="margin:0">{signal}</h2></div>', unsafe_allow_html=True)
        elif "HOLD" in signal:
            st.markdown(f'<div class="signal-hold"><h2 style="margin:0">{signal}</h2></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="signal-sell"><h2 style="margin:0">{signal}</h2></div>', unsafe_allow_html=True)
        
        # Chart
        fig = make_price_rsi_chart(ohlcv, selected_coin["name"], selected_coin["symbol"])
        st.plotly_chart(fig, use_container_width=True)
        
        # Reasons
        with st.expander("📋 Detailed Signal Breakdown"):
            for r in reasons:
                st.markdown(f"- {r}")
    
    # Quick comparison table for all top coins
    st.divider()
    st.subheader("📊 Quick Comparison - All Top Coins")
    
    overview_data = []
    for coin in TOP_COINS:
        try:
            mkt = get_coin_market_data(coin["id"])
            ohl = get_crypto_ohlcv(coin["id"], days=90)
            if not ohl.empty:
                ohl = calculate_indicators(ohl)
                sig, sc, _ = get_signal(ohl)
            else:
                sig, sc = "⚪ N/A", 0
            overview_data.append({
                "Coin": f"{coin['name']} ({coin['symbol']})",
                "Price": mkt.get("current_price", 0),
                "24h Change %": round(mkt.get("price_change_24h", 0), 2),
                "Signal": sig,
                "Score": sc
            })
        except:
            overview_data.append({
                "Coin": f"{coin['name']} ({coin['symbol']})",
                "Price": 0, "24h Change %": 0, "Signal": "⚪ Error", "Score": 0
            })
    
    overview_df = pd.DataFrame(overview_data)
    
    def highlight_signal(val):
        if "STRONG BUY" in str(val) or ("BUY" in str(val) and "SELL" not in str(val)):
            return "background-color: #d4edda; color: black; font-weight: bold"
        elif "HOLD" in str(val):
            return "background-color: #fff3cd; color: black"
        elif "SELL" in str(val):
            return "background-color: #f8d7da; color: black"
        return ""
    
    styled_df = overview_df.style.format({
        "Price": "${:,.4f}" if overview_df["Price"].max() < 10 else "${:,.2f}",
        "24h Change %": "{:+.2f}%"
    }).map(highlight_signal, subset=["Signal"])
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )

# ---------- MEMECOINS TAB ----------
with tab_meme:
    st.header("🐶 Memecoins")
    st.caption("High-risk, high-reward tokens. Signals are **less reliable** due to low liquidity & hype-driven moves. Trade with extreme caution!")
    
    st.warning("⚠️ Memecoins can drop 80-100% in hours. These signals are for entertainment/educational purposes primarily.")
    
    if st.button("🔄 Refresh Memecoin Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Select meme for detail
    meme_names = [m["name"] for m in MEME_COINS]
    selected_meme_name = st.selectbox("Select memecoin for full analysis + chart:", meme_names, index=0)
    selected_meme = next((m for m in MEME_COINS if m["name"] == selected_meme_name), MEME_COINS[0])
    
    # Detailed analysis
    st.subheader(f"🔍 {selected_meme['name']} ({selected_meme['symbol']})")
    
    meme_ohlcv = get_crypto_ohlcv(selected_meme["id"], days=120)
    meme_market = get_coin_market_data(selected_meme["id"])
    
    if meme_ohlcv.empty:
        st.error("Data temporarily unavailable for this memecoin.")
    else:
        meme_ohlcv = calculate_indicators(meme_ohlcv)
        meme_signal, meme_score, meme_reasons = get_signal(meme_ohlcv)
        latest_m = meme_ohlcv.iloc[-1]
        
        # Metrics
        mm1, mm2, mm3 = st.columns(3)
        price_m = meme_market.get("current_price", latest_m["close"])
        chg_m = meme_market.get("price_change_24h", 0)
        
        mm1.metric("Current Price", f"${price_m:,.8f}" if price_m < 0.01 else f"${price_m:,.4f}", f"{chg_m:+.1f}%")
        mm2.metric("Buy Score", f"{meme_score}/100")
        mm3.metric("24h Volume", f"${meme_market.get('total_volume', 0):,.0f}")
        
        # Big signal
        if "STRONG BUY" in meme_signal:
            st.markdown(f'<div class="signal-strong"><h2>{meme_signal} — Score {meme_score}</h2></div>', unsafe_allow_html=True)
        elif "BUY" in meme_signal:
            st.markdown(f'<div class="signal-buy"><h2>{meme_signal} — Score {meme_score}</h2></div>', unsafe_allow_html=True)
        elif "HOLD" in meme_signal:
            st.markdown(f'<div class="signal-hold"><h2>{meme_signal} — Score {meme_score}</h2></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="signal-sell"><h2>{meme_signal} — Score {meme_score}</h2></div>', unsafe_allow_html=True)
        
        # Chart
        fig_meme = make_price_rsi_chart(meme_ohlcv, selected_meme["name"], selected_meme["symbol"])
        st.plotly_chart(fig_meme, use_container_width=True)
        
        with st.expander("📋 Why this signal for this memecoin?"):
            for r in meme_reasons:
                st.write(f"• {r}")
            st.markdown("""
            **Memecoin Note**: These assets often move on social sentiment, celebrity tweets, and hype cycles rather than traditional TA. 
            Use signals together with X/Twitter sentiment, community activity, and liquidity checks.
            """)
    
    # Top memecoins overview table
    st.divider()
    st.subheader("🔥 Top Memecoins Overview (from CoinGecko Meme category)")
    
    try:
        client = Coingecko()
        top_meme_markets = client.coins.markets.get(
            vs_currency="usd", category="meme-token",
            order="market_cap_desc", per_page=10, page=1
        )
        meme_overview = []
        for m in top_meme_markets:
            if m.current_price and m.current_price > 0:
                meme_overview.append({
                    "Name": f"{m.name} ({m.symbol.upper()})",
                    "Price": m.current_price,
                    "24h %": round(m.price_change_percentage_24h or 0, 1),
                    "Market Cap": m.market_cap or 0
                })
        if meme_overview:
            meme_df = pd.DataFrame(meme_overview)
            st.dataframe(
                meme_df.style.format({
                    "Price": lambda x: f"${x:,.8f}" if x < 0.01 else (f"${x:,.4f}" if x < 1 else f"${x:,.2f}"),
                    "24h %": "{:+.1f}%",
                    "Market Cap": "${:,.0f}"
                }),
                use_container_width=True, hide_index=True
            )
    except Exception as e:
        st.info("Could not load live top memecoins list. The detailed selector above still works for popular ones.")
    
    st.caption("You can analyze any other memecoin by knowing its CoinGecko ID (e.g. 'myro', 'gigachad', 'mog-coin'). Add to MEME_COINS list in code if needed.")

# Footer
st.divider()
st.markdown("""
<div class="disclaimer">
<b>SmartBuy Signals</b> v1.0 • Data via Polygon.io & CoinGecko • Indicators calculated in real-time • 
This tool does not execute trades. Use responsibly. Not financial advice.
</div>
""", unsafe_allow_html=True)
