import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tvDatafeed import TvDatafeed, Interval
from supabase import create_client, Client
import time
import os

# ==========================================
# 1. ตั้งค่าหน้าเพจ & เชื่อมต่อ Supabase / TradingView
# ==========================================
st.set_page_config(page_title="Pro Trading Dashboard", layout="wide")

SUPABASE_URL = os.environ.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY", "")

@st.cache_resource
def init_supabase() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("⚠️ ยังไม่ได้ตั้งค่า SUPABASE_URL และ SUPABASE_KEY")
        st.stop()
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

@st.cache_resource
def get_tv_connection():
    return TvDatafeed()

tv = get_tv_connection()

# ==========================================
# 2. ฟังก์ชันจัดการ Database (Watchlist & Order)
# ==========================================
def fetch_user_watchlist(email: str):
    res = supabase.table("user_watchlists").select("ticker").eq("user_email", email).execute()
    # ดึงข้อมูลออกมาตามลำดับที่บันทึกไว้ (เนื่องจากเราใช้วิธีลบและเขียนใหม่เวลาสลับตำแหน่ง)
    return [row["ticker"] for row in res.data]

def add_ticker_to_db(email: str, ticker: str):
    existing = fetch_user_watchlist(email)
    if ticker not in existing:
        supabase.table("user_watchlists").insert({"user_email": email, "ticker": ticker}).execute()

def remove_ticker_from_db(email: str, ticker: str):
    supabase.table("user_watchlists").delete().eq("user_email", email).eq("ticker", ticker).execute()

def rebuild_watchlist_db(email: str, new_list: list):
    # ฟังก์ชันสำหรับจัดเรียงตำแหน่ง: ลบของเดิมทั้งหมด แล้วเขียนใหม่ตามลำดับ
    supabase.table("user_watchlists").delete().eq("user_email", email).execute()
    for t in new_list:
        supabase.table("user_watchlists").insert({"user_email": email, "ticker": t}).execute()

# ==========================================
# 3. ระบบ Authentication (แก้ปัญหา F5 หลุดล็อกอิน)
# ==========================================
if 'user' not in st.session_state:
    st.session_state['user'] = None

# ตรวจสอบ URL Query Parameter ว่ามี Session ค้างอยู่ไหม (ช่วยจำตอนกด F5)
if st.session_state['user'] is None and "session_email" in st.query_params:
    class SessionUser:
        def __init__(self, email):
            self.email = email
    st.session_state['user'] = SessionUser(st.query_params["session_email"])

if st.session_state['user'] is None:
    st.title("🔐 เข้าสู่ระบบ (Pro Multi-Asset Scanner)")
    st.markdown("กรุณาเข้าสู่ระบบเพื่อเข้าถึง Dashboard และจัดการ Watchlist ส่วนตัวของคุณ")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Login / Sign Up")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("เข้าสู่ระบบ (Login)", use_container_width=True):
                try:
                    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state['user'] = res.user
                    # ฝังอีเมลไว้ใน URL เพื่อให้ระบบจำได้ตอนรีเฟรชหน้าเว็บ
                    st.query_params["session_email"] = res.user.email
                    st.rerun()
                except Exception as e:
                    st.error("อีเมลหรือรหัสผ่านไม่ถูกต้อง (หรือยังไม่ได้สมัครสมาชิก)")
        with c2:
            if st.button("สมัครสมาชิก (Sign Up)", use_container_width=True):
                try:
                    res = supabase.auth.sign_up({"email": email, "password": password})
                    st.success("สมัครสมาชิกสำเร็จ! กรุณากดปุ่ม 'เข้าสู่ระบบ' อีกครั้ง")
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาด: {e}")
    st.stop()

# ==========================================
# 4. ฟังก์ชันคำนวณ Indicators & Fibonacci
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
    close, ema20, ema50, ema200, ema_angle, atr = float(row['Close']), float(row['EMA20']), float(row['EMA50']), float(row['EMA200']), float(row['EMA20_Angle']), float(row['ATR14'])
    macd_rising = bool(row['MACD_Hist_Rising'])
    recent_high, recent_low = float(row['Rolling_High']), float(row['Rolling_Low'])
    strong_bull = (close > ema20) and (ema20 > ema50) and (ema50 > ema200)
    strong_bear = (close < ema20) and (ema20 < ema50) and (ema50 < ema200)
    macro_bull = (close > ema200) and (ema50 > ema200)
    macro_bear = (close < ema200) and (ema50 < ema200)
    diff = recent_high - recent_low
    fibo_up_500, fibo_up_618 = recent_high - (diff * 0.500), recent_high - (diff * 0.618)
    fibo_dn_500, fibo_dn_618 = recent_low + (diff * 0.500), recent_low + (diff * 0.618)
    tolerance = atr * 1.0 
    is_near_fibo_up = (abs(close - fibo_up_500) < tolerance) or (abs(close - fibo_up_618) < tolerance)
    is_near_fibo_dn = (abs(close - fibo_dn_500) < tolerance) or (abs(close - fibo_dn_618) < tolerance)
    trend_status = "Strong Uptrend" if strong_bull else "Strong Downtrend" if strong_bear else "Sideway/Pullback"
    
    if macro_bull and is_near_fibo_up and macd_rising and (row['RSI14'] < 55): action = "BUY (Fibo 🚀)"
    elif strong_bull and ema_angle > 10 and macd_rising: action = "BUY (Momentum)"
    elif macro_bear and is_near_fibo_dn and not macd_rising and (row['RSI14'] > 45): action = "SELL (Fibo 🩸)"
    elif strong_bear and ema_angle < -10 and not macd_rising: action = "SELL (Momentum)"
    else: action = "WAIT"

    return {"Price": round(close, 4), "Trend": trend_status, "EMA_Angle(°)": round(ema_angle, 1), "RSI": round(float(row['RSI14']), 1), "ATR": round(atr, 4), "Signal": action}

# ==========================================
# 5. ฟังก์ชันดึงข้อมูล & กราฟ
# ==========================================
def fetch_tradingview(ticker: str, interval: Interval):
    if ":" in ticker:
        exchange, symbol = ticker.split(":")
    else:
        exchange = "BINANCE" if "USD" in ticker else "NASDAQ"
        symbol = ticker
    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=250)
    if df is None or df.empty: raise Exception("ไม่พบข้อมูลบน TradingView ตรวจสอบชื่อย่อหรือตลาดอีกครั้ง")
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    return df

def plot_chart(df: pd.DataFrame, ticker: str):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=(f"{ticker}", "RSI (14)"), row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='dodgerblue', width=1.5), name='EMA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='orange', width=1.5), name='EMA50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='red', width=1.5), name='EMA200'), row=1, col=1)
    
    latest_high, latest_low = df['Rolling_High'].iloc[-1], df['Rolling_Low'].iloc[-1]
    diff = latest_high - latest_low
    fig.add_hline(y=latest_high, line_dash="dot", line_color="gray", annotation_text="Swing High", row=1, col=1)
    fig.add_hline(y=latest_low, line_dash="dot", line_color="gray", annotation_text="Swing Low", row=1, col=1)
    fig.add_hline(y=latest_high - (diff * 0.500), line_dash="dash", line_color="cyan", annotation_text="Fibo 50.0%", row=1, col=1)
    fig.add_hline(y=latest_high - (diff * 0.618), line_dash="dash", line_color="magenta", annotation_text="Fibo 61.8%", row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI14'], line=dict(color='purple', width=1.5), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, height=550, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
    return fig

# ==========================================
# 6. Sidebar (ตั้งค่า, เพิ่มลบ และ จัดเรียงตำแหน่ง)
# ==========================================
st.sidebar.markdown(f"👤 **ผู้ใช้งาน:** `{st.session_state['user'].email}`")
if st.sidebar.button("🚪 ออกจากระบบ (Logout)"):
    supabase.auth.sign_out()
    st.session_state['user'] = None
    if "session_email" in st.query_params:
        del st.query_params["session_email"]
    st.rerun()

st.sidebar.header("⚙️ Settings")
tf_options = {"15 นาที": Interval.in_15_minute, "1 ชั่วโมง": Interval.in_1_hour, "4 ชั่วโมง": Interval.in_4_hour, "1 วัน": Interval.in_daily, "1 สัปดาห์": Interval.in_weekly}
selected_tf_label = st.sidebar.selectbox("เลือก Timeframe", ["15 นาที", "1 ชั่วโมง", "4 ชั่วโมง", "1 วัน", "1 สัปดาห์"], index=3)
selected_tf = tf_options[selected_tf_label]

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Risk Management")
account_size = st.sidebar.number_input("เงินทุน / Account Size ($)", min_value=100.0, value=1000.0, step=100.0)
risk_pct = st.sidebar.slider("ความเสี่ยงต่อไม้ (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
risk_amount = account_size * (risk_pct / 100)

st.sidebar.markdown("---")
st.sidebar.subheader("📝 จัดการ Watchlist")

user_email = st.session_state['user'].email
current_watchlist = fetch_user_watchlist(user_email)

if not current_watchlist:
    default_tickers = ["NASDAQ:NVDA", "NASDAQ:TSLA", "BINANCE:BTCUSDT", "BINANCE:SOLUSDT", "OANDA:XAUUSD", "TVC:USOIL"]
    for t in default_tickers:
        add_ticker_to_db(user_email, t)
    current_watchlist = default_tickers

# 6.1 เพิ่ม / ลบ สินทรัพย์
new_ticker = st.sidebar.text_input("เพิ่มชื่อ (เช่น NASDAQ:AAPL)")
if st.sidebar.button("➕ Add Ticker"):
    ticker_upper = new_ticker.upper().strip()
    if ticker_upper and ticker_upper not in current_watchlist:
        add_ticker_to_db(user_email, ticker_upper)
        st.sidebar.success(f"เพิ่ม {ticker_upper} แล้ว")
        time.sleep(1)
        st.rerun()

ticker_to_remove = st.sidebar.selectbox("เลือก Ticker ที่ลบ", [""] + current_watchlist)
if st.sidebar.button("🗑️ Remove Ticker") and ticker_to_remove:
    remove_ticker_from_db(user_email, ticker_to_remove)
    st.rerun()

# 6.2 ฟังก์ชันใหม่: จัดเรียงตำแหน่ง
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 สลับตำแหน่งสินทรัพย์")
ticker_to_move = st.sidebar.selectbox("เลือก Ticker ที่ต้องการย้าย", current_watchlist, key="move_ticker")
col_up, col_dn = st.sidebar.columns(2)

if col_up.button("⬆️ เลื่อนขึ้น") and ticker_to_move:
    idx = current_watchlist.index(ticker_to_move)
    if idx > 0:
        # สลับตำแหน่งในลิสต์
        current_watchlist[idx], current_watchlist[idx-1] = current_watchlist[idx-1], current_watchlist[idx]
        rebuild_watchlist_db(user_email, current_watchlist)
        st.rerun()

if col_dn.button("⬇️ เลื่อนลง") and ticker_to_move:
    idx = current_watchlist.index(ticker_to_move)
    if idx < len(current_watchlist) - 1:
        # สลับตำแหน่งในลิสต์
        current_watchlist[idx], current_watchlist[idx+1] = current_watchlist[idx+1], current_watchlist[idx]
        rebuild_watchlist_db(user_email, current_watchlist)
        st.rerun()

# ==========================================
# 7. ส่วนแสดงผลหลัก (สแกนแบบ Background)
# ==========================================
st.title("📈 Pro Multi-Asset Scanner")

# ใช้ @st.fragment เพื่อจำกัดให้ทำงานรีเฟรชเฉพาะส่วนกราฟ/ตารางด้านล่างนี้ โดยหน้าเว็บหลักไม่กะพริบ
@st.fragment(run_every="60s")
def run_background_scanner():
    st.subheader(f"📊 สแกนเนอร์ตลาด (Timeframe: {selected_tf_label})")
    st.caption(f"⏱️ ข้อมูลอัปเดตล่าสุด: {time.strftime('%H:%M:%S')} (ระบบกำลังแอบตรวจจับสัญญาณเบื้องหลัง...)")
    
    with st.spinner(f'กำลังประมวลผลสัญญาณ...'):
        results = []
        charts = {}
        
        for ticker in current_watchlist:
            try:
                df = fetch_tradingview(ticker, selected_tf)
                if not df.empty and len(df) > 200:
                    df_ind = calculate_indicators(df)
                    latest = df_ind.iloc[-1]
                    res = evaluate_signal(latest)
                    price, atr = res['Price'], res['ATR']
                    sl_dist, tp_dist = 1.5 * atr, (1.5 * atr) * 1.5
                    
                    if "BUY" in res['Signal']: trade_setup = f"SL: {price - sl_dist:.4f} | TP: {price + tp_dist:.4f}"
                    elif "SELL" in res['Signal']: trade_setup = f"SL: {price + sl_dist:.4f} | TP: {price - tp_dist:.4f}"
                    else: trade_setup = "-"
                    
                    pos_text = f"{int(risk_amount / sl_dist)} หุ้น" if "SET:" in ticker or "NASDAQ:" in ticker or "AMEX:" in ticker else f"{max(0.01, round(risk_amount / (sl_dist * 100000), 2))} Lot"
                        
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
                val_str = str(val)
                if 'BUY' in val_str: return 'background-color: #d4edda; color: green; font-weight: bold'
                elif 'SELL' in val_str: return 'background-color: #f8d7da; color: red; font-weight: bold'
                return 'color: gray'

            actionable_df = summary_df[summary_df['Signal'] != "WAIT"]
            if not actionable_df.empty:
                st.success(f"🔥 พบสัญญาณเข้าเทรด (Action Required): {len(actionable_df)} รายการ")
                st.dataframe(actionable_df.style.map(highlight_signal, subset=['Signal']), use_container_width=True)
            else:
                st.info("⏳ ยังไม่มีสัญญาณเข้าเทรดในขณะนี้")
                
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

# เรียกใช้ฟังก์ชันแอบสแกน
run_background_scanner()
