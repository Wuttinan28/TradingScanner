import streamlit as st
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go

# ==========================================
# 1. ตั้งค่าหน้าเพจ Streamlit
# ==========================================
st.set_page_config(page_title="Pro Trading Dashboard", layout="wide")
st.title("📈 Pro Multi-Asset Scanner (Fibo & Momentum)")
st.markdown("ระบบคัดกรองสัญญาณ Real-time ด้วย Twelve Data (รองรับ Forex, คริปโต, หุ้นสหรัฐฯ)")

# ==========================================
# 2. ฟังก์ชันคำนวณ Indicators & Fibonacci
# ==========================================
def calculate_indicators(df: pd.DataFrame, slope_lookback: int = 5, fib_lookback: int = 20) -> pd.DataFrame:
    data = df.copy()
    close, high, low = data['Close'], data['High'], data['Low']

    # EMAs
    data['EMA20'] = close.ewm(span=20, adjust=False).mean()
    data['EMA50'] = close.ewm(span=50, adjust=False).mean()
    data['EMA200'] = close.ewm(span=200, adjust=False).mean()

    # ATR (14)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data['ATR14'] = tr.ewm(alpha=1/14, adjust=False).mean()

    # EMA Angle
    ema_diff = data['EMA20'] - data['EMA20'].shift(slope_lookback)
    norm_slope = ema_diff / (slope_lookback * data['ATR14'])
    data['EMA20_Angle'] = np.degrees(np.arctan(norm_slope))

    # RSI & MACD
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    data['RSI14'] = 100 - (100 / (1 + rs))

    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    macd, macd_sig = ema12 - ema26, (ema12 - ema26).ewm(span=9, adjust=False).mean()
    data['MACD_Hist_Rising'] = (macd - macd_sig) > (macd - macd_sig).shift(1)

    # จุด Swing สำหรับ Fibo
    data['Rolling_High'] = high.rolling(window=fib_lookback).max()
    data['Rolling_Low'] = low.rolling(window=fib_lookback).min()

    return data

def evaluate_signal(row: pd.Series) -> dict:
    close = float(row['Close'])
    ema20 = float(row['EMA20'])
    ema50 = float(row['EMA50'])
    ema200 = float(row['EMA200'])
    ema_angle = float(row['EMA20_Angle'])
    macd_rising = bool(row['MACD_Hist_Rising'])
    
    recent_high = float(row['Rolling_High'])
    recent_low = float(row['Rolling_Low'])
    atr = float(row['ATR14'])

    # เช็ค Trend ระยะสั้น (Momentum)
    strong_bull = (close > ema20) and (ema20 > ema50) and (ema50 > ema200)
    strong_bear = (close < ema20) and (ema20 < ema50) and (ema50 < ema200)
    
    # เช็ค Trend ภาพใหญ่ (สำหรับรอรับ Fibo)
    macro_bull = (close > ema200) and (ema50 > ema200)
    macro_bear = (close < ema200) and (ema50 < ema200)

    # กาง Fibonacci
    diff = recent_high - recent_low
    fibo_up_500 = recent_high - (diff * 0.500)
    fibo_up_618 = recent_high - (diff * 0.618)
    fibo_dn_500 = recent_low + (diff * 0.500)
    fibo_dn_618 = recent_low + (diff * 0.618)

    # ระยะคลาดเคลื่อน (1 ATR)
    tolerance = atr * 1.0 
    is_near_fibo_up = (abs(close - fibo_up_500) < tolerance) or (abs(close - fibo_up_618) < tolerance)
    is_near_fibo_dn = (abs(close - fibo_dn_500) < tolerance) or (abs(close - fibo_dn_618) < tolerance)

    trend_status = "Strong Uptrend" if strong_bull else "Strong Downtrend" if strong_bear else "Sideway/Pullback"
    
    # 🎯 เงื่อนไขเข้าเทรด
    if macro_bull and is_near_fibo_up and macd_rising and (row['RSI14'] < 55):
        action = "BUY (Fibo 🚀)"
    elif strong_bull and ema_angle > 10 and macd_rising:
        action = "BUY (Momentum)"
    elif macro_bear and is_near_fibo_dn and not macd_rising and (row['RSI14'] > 45):
        action = "SELL (Fibo 🩸)"
    elif strong_bear and ema_angle < -10 and not macd_rising:
        action = "SELL (Momentum)"
    else:
        action = "WAIT"

    return {
        "Price": round(close, 4),
        "Trend": trend_status,
        "EMA_Angle(°)": round(ema_angle, 1),
        "RSI": round(float(row['RSI14']), 1),
        "ATR": round(atr, 4),
        "Signal": action
    }

# ==========================================
# 3. ฟังก์ชันดึงข้อมูล (Twelve Data) & กราฟ
# ==========================================
def fetch_twelvedata(symbol: str, interval: str, api_key: str):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=250&apikey={api_key}"
    response = requests.get(url).json()
    
    if 'values' in response:
        df = pd.DataFrame(response['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.sort_index(ascending=True)
        df = df.astype(float)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        return df
    else:
        raise Exception(response.get('message', 'ไม่สามารถดึงข้อมูลได้ โปรดเช็คชื่อย่อหรือ API Key'))

def plot_chart(df: pd.DataFrame, ticker: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='blue', width=1.5), name='EMA20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='orange', width=1.5), name='EMA50'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='red', width=1.5), name='EMA200'))
    fig.update_layout(title=f"Chart: {ticker}", xaxis_rangeslider_visible=False, height=450, margin=dict(l=0, r=0, t=40, b=0))
    return fig

# ==========================================
# 4. Sidebar: ตั้งค่าระบบ & จัดการ Watchlist
# ==========================================
st.sidebar.header("🔑 API Setup")
api_key = st.sidebar.text_input("Twelve Data API Key", type="password", help="สมัครฟรีที่ twelvedata.com")

st.sidebar.header("⚙️ Settings")
tf_options = {"15 นาที": "15min", "1 ชั่วโมง": "1h", "4 ชั่วโมง": "4h", "1 วัน": "1day", "1 สัปดาห์": "1week"}
selected_tf_label = st.sidebar.selectbox("เลือก Timeframe", list(tf_options.keys()), index=3)
selected_tf = tf_options[selected_tf_label]

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Risk Management")
