import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tvDatafeed import TvDatafeed, Interval  # ตัว D ต้องเป็นตัวพิมพ์ใหญ่

# ==========================================
# 1. ตั้งค่าหน้าเพจ & เชื่อมต่อ TradingView
# ==========================================
st.set_page_config(page_title="Pro Trading Dashboard", layout="wide")
st.title("📈 Pro Multi-Asset Scanner (Powered by TradingView)")
st.markdown("ระบบสแกนหุ้นไทย, คริปโต, Forex แบบ Real-time")

@st.cache_resource
def get_tv_connection():
    return TvDatafeed()

tv = get_tv_connection()

# ==========================================
# 2. ฟังก์ชันคำนวณ Indicators & Fibonacci
# ==========================================
def calculate_indicators(df: pd.DataFrame, slope_lookback: int = 5, fib_lookback: int = 20) -> pd.DataFrame:
    data = df.copy()
    close, high, low = data['Close'], data['High'], data['Low']

    data['EMA20'] = close.ewm(span=20, adjust=False).mean()
    data['EMA50'] = close.ewm(span=50, adjust=False).mean()
    data['EMA200'] = close.ewm(span=200, adjust=False).mean()

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data['ATR14'] = tr.ewm(alpha=1/14, adjust=False).mean()

    ema_diff = data['EMA20'] - data['EMA20'].shift(slope_lookback)
    norm_slope = ema_diff / (slope_lookback * data['ATR14'])
    data['EMA20_Angle'] = np.degrees(np.arctan(norm_slope))

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    data['RSI14'] = 100 - (100 / (1 + rs))

    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    macd, macd_sig = ema12 - ema26, (ema12 - ema26).ewm(span=9, adjust=False).mean()
    data['MACD_Hist_Rising'] = (macd - macd_sig) > (macd - macd_sig).shift(1)

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

    strong_bull = (close > ema20) and (ema20 > ema50) and (ema50 > ema200)
    strong_bear = (close < ema20) and (ema20 < ema50) and (ema50 < ema200)
    
    macro_bull = (close > ema200) and (ema50 > ema200)
    macro_bear = (close < ema200) and (ema50 < ema200)

    diff = recent_high - recent_low
    fibo_up_500 = recent_high - (diff * 0.500)
    fibo_up_618 = recent_high - (diff * 0.618)
    fibo_dn_500 = recent_low + (diff * 0.500)
    fibo_dn_618 = recent_low + (diff * 0.618)

    tolerance = atr * 1.0 
    is_near_fibo_up = (abs(close - fibo_up_500) < tolerance) or (abs(close - fibo_up_618) < tolerance)
    is_near_fibo_dn = (abs(close - fibo_dn_500) < tolerance) or (abs(close - fibo_dn_618) < tolerance)

    trend_status = "Strong Uptrend" if strong_bull else "Strong Downtrend" if strong_bear else "Sideway/Pullback"
    
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
# 3. ฟังก์ชันดึงข้อมูล (TradingView) & กราฟ
# ==========================================
def fetch_tradingview(ticker: str, interval: Interval):
    # TradingView ต้องการข้อมูลรูปแบบ EXCHANGE:SYMBOL เช่น SET:PTT
    if ":" in ticker:
        exchange, symbol = ticker.split(":")
    else:
        # ระบบคาดเดา Exchange อัตโนมัติหากผู้ใช้ไม่ได้พิมพ์มา
        exchange = "BINANCE" if "USD" in ticker else "NASDAQ"
        symbol = ticker

    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=250)
    
    if df is None or df.empty:
        raise Exception("ไม่พบข้อมูลบน TradingView ตรวจสอบชื่อย่อหรือตลาดอีกครั้ง")
    
    # เปลี่ยนชื่อคอลัมน์ให้ตรงกับระบบคำนวณ
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    return df

def plot_chart(df: pd.DataFrame, ticker: str):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        subplot_titles=(f"{ticker}", "RSI (14)"),
                        row_heights=[0.75, 0.25])

    # กราฟแท่งเทียน & EMA
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='dodgerblue', width=1.5), name='EMA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='orange', width=1.5), name='EMA50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='red', width=1.5), name='EMA200'), row=1, col=1)

    # Fibonacci Lines
    latest_high = df['Rolling_High'].iloc[-1]
    latest_low = df['Rolling_Low'].iloc[-1]
    diff = latest_high - latest_low
    fibo_500 = latest_high - (diff * 0.500)
    fibo_618 = latest_high - (diff * 0.618)

    fig.add_hline(y=latest_high, line_dash="dot", line_color="gray", annotation_text="Swing High", row=1, col=1)
    fig.add_hline(y=latest_low, line_dash="dot", line_color="gray", annotation_text="Swing Low", row=1, col=1)
    fig.add_hline(y=fibo_500, line_dash="dash", line_color="cyan", annotation_text="Fibo 50.0%", row=1, col=1)
    fig.add_hline(y=fibo_618, line_dash="dash", line_color="magenta", annotation_text="Fibo 61.8% (Golden Zone)", row=1, col=1)

    # กราฟ RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI14'], line=dict(color='purple', width=1.5), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False, height=550, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
    return fig

# ==========================================
# 4. Sidebar: ตั้งค่าระบบ & จัดการ Watchlist
# ==========================================
st.sidebar.header("⚙️ Settings")

# อัปเดต Timeframe ให้ตรงกับไลบรารี TradingView
tf_options = {
    "15 นาที": Interval.in_15_minute, 
    "1 ชั่วโมง": Interval.in_1_hour, 
    "4 ชั่วโมง": Interval.in_4_hour, 
    "1 วัน": Interval.in_daily, 
    "1 สัปดาห์": Interval.in_weekly
}
selected_tf_label = st.sidebar.selectbox("เลือก Timeframe", list(tf_options.keys()), index=3)
selected_tf = tf_options[selected_tf_label]

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Risk Management")
account_size = st.sidebar.number_input("เงินทุน / Account Size ($)", min_value=100.0, value=1000.0, step=100.0)
risk_pct = st.sidebar.slider("ความเสี่ยงต่อไม้ (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
risk_amount = account_size * (risk_pct / 100)

st.sidebar.markdown("---")
st.sidebar.subheader("📝 My Watchlist (TradingView Format)")
st.sidebar.caption("รูปแบบการพิมพ์: EXCHANGE:SYMBOL")

# อัปเดต Watchlist เบื้องต้นให้เป็นรูปแบบ TradingView
if 'my_watchlist' not in st.session_state:
    st.session_state['my_watchlist'] = [
        "SET:PTT",       # หุ้นไทย ปตท.
        "SET:AOT",       # หุ้นไทย ท่าอากาศยาน
        "OANDA:EURUSD",  # คู่เงิน Forex
        "OANDA:XAUUSD",  # ทองคำ
        "BINANCE:BTCUSD",# คริปโต
        "NASDAQ:AAPL"    # หุ้นอเมริกา
    ]

new_ticker = st.sidebar.text_input("เพิ่มชื่อ (เช่น SET:CPALL)")
if st.sidebar.button("➕ Add Ticker"):
    ticker_upper = new_ticker.upper().strip()
    if ticker_upper and ticker_upper not in st.session_state['my_watchlist']:
        st.session_state['my_watchlist'].append(ticker_upper)
        st.sidebar.success(f"เพิ่ม {ticker_upper} แล้ว")

ticker_to_remove = st.sidebar.selectbox("เลือก Ticker ที่ลบ", [""] + st.session_state['my_watchlist'])
if st.sidebar.button("🗑️ Remove Ticker") and ticker_to_remove:
    st.session_state['my_watchlist'].remove(ticker_to_remove)
    st.rerun()

# ==========================================
# 5. ส่วนแสดงผลหลัก: สแกนเนอร์ & กราฟ (Auto-Scan)
# ==========================================
st.subheader(f"📊 สแกนเนอร์ตลาด (Timeframe: {selected_tf_label})")

# ไม่มี st.button แล้ว ระบบจะทำงานทันทีที่โหลดมาถึงบรรทัดนี้
with st.spinner(f'กำลังดึงข้อมูล Real-time และวิเคราะห์สัญญาณ...'):
    results = []
    charts = {}
    
    for ticker in st.session_state['my_watchlist']:
        try:
            df = fetch_tradingview(ticker, selected_tf)
            
            if not df.empty and len(df) > 200:
                df_ind = calculate_indicators(df)
                latest = df_ind.iloc[-1]
                res = evaluate_signal(latest)
                price = res['Price']
                atr = res['ATR']
                
                sl_dist = 1.5 * atr
                tp_dist = sl_dist * 1.5
                
                if "BUY" in res['Signal']:
                    trade_setup = f"SL: {price - sl_dist:.4f} | TP: {price + tp_dist:.4f}"
                elif "SELL" in res['Signal']:
                    trade_setup = f"SL: {price + sl_dist:.4f} | TP: {price - tp_dist:.4f}"
                else:
                    trade_setup = "-"
                
                if "SET:" in ticker or "NASDAQ:" in ticker:
                    position = risk_amount / sl_dist
                    pos_text = f"{int(position)} หุ้น"
                else:
                    position = risk_amount / (sl_dist * 100000)
                    pos_text = f"{max(0.01, round(position, 2))} Lot"
                    
                res["Ticker"] = ticker
                res["Action_Plan (SL/TP)"] = trade_setup
                res["Position_Size"] = pos_text if res['Signal'] != "WAIT" else "-"
                results.append(res)
                
                charts[ticker] = plot_chart(df_ind, ticker)
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดกับ {ticker}: {e}")

    if results:
        summary_df = pd.DataFrame(results)
        cols = ["Ticker", "Price", "Trend", "EMA_Angle(°)", "RSI", "Signal", "Action_Plan (SL/TP)", "Position_Size"]
        summary_df = summary_df[cols]

        def highlight_signal(val):
            if pd.isna(val): return ''
            val_str = str(val)
            if 'BUY' in val_str: return 'background-color: #d4edda; color: green; font-weight: bold'
            elif 'SELL' in val_str: return 'background-color: #f8d7da; color: red; font-weight: bold'
            return 'color: gray'

        # --- แยกตัวที่ต้อง Action ออกมาโชว์เด่นๆ ---
        actionable_df = summary_df[summary_df['Signal'] != "WAIT"]
        
        if not actionable_df.empty:
            st.success(f"🔥 พบสัญญาณเข้าเทรด (Action Required): {len(actionable_df)} รายการ")
            st.dataframe(actionable_df.style.map(highlight_signal, subset=['Signal']), use_container_width=True)
        else:
            st.info("⏳ ยังไม่มีสัญญาณเข้าเทรดในขณะนี้ (ตลาดยังพักตัว หรือกำลังสร้างเทรนด์ใหม่)")
            
        # --- ซ่อนตัวที่ให้รอก่อน (WAIT) ไว้ในเมนูกดขยาย ---
        with st.expander("📋 ดูข้อมูลสินทรัพย์ที่กำลังเฝ้าระวังทั้งหมด (All Watchlist)"):
            st.dataframe(summary_df.style.map(highlight_signal, subset=['Signal']), use_container_width=True)
        
        st.markdown("---")
        st.subheader("📈 กราฟราคา Fibo & RSI (Real-time)")
        cols = st.columns(2)
        col_idx = 0
        for tkr, fig in charts.items():
            with cols[col_idx % 2]:
                st.plotly_chart(fig, use_container_width=True)
            col_idx += 1

# --- ระบบ Auto-Refresh ทุกๆ 60 วินาที ---
import time
st.markdown("---")
st.caption("🔄 Dashboard จะอัปเดตข้อมูลอัตโนมัติทุกๆ 60 วินาที...")
time.sleep(60)
st.rerun()
