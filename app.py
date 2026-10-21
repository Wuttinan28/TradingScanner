import streamlit as st
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. ตั้งค่าหน้าเพจ Streamlit
# ==========================================
st.set_page_config(page_title="Pro Trading Dashboard", layout="wide")
st.title("📈 Pro Multi-Asset Scanner (Powered by Twelve Data)")
st.markdown("ระบบวิเคราะห์และคัดกรองสัญญาณ Real-time สำหรับ Forex, คริปโต, หุ้นสหรัฐฯ")

# ==========================================
# 2. ฟังก์ชันคำนวณ Indicators (อัปเกรด Fibonacci)
# ==========================================
def calculate_indicators(df: pd.DataFrame, slope_lookback: int = 5, fib_lookback: int = 20) -> pd.DataFrame:
    data = df.copy()
    close, high, low = data['Close'], data['High'], data['Low']

    # 1. EMAs เดิม
    data['EMA20'] = close.ewm(span=20, adjust=False).mean()
    data['EMA50'] = close.ewm(span=50, adjust=False).mean()
    data['EMA200'] = close.ewm(span=200, adjust=False).mean()

    # 2. คำนวณ ATR และ EMA Angle
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data['ATR14'] = tr.ewm(alpha=1/14, adjust=False).mean()

    ema_diff = data['EMA20'] - data['EMA20'].shift(slope_lookback)
    norm_slope = ema_diff / (slope_lookback * data['ATR14'])
    data['EMA20_Angle'] = np.degrees(np.arctan(norm_slope))

    # 3. RSI & MACD
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    data['RSI14'] = 100 - (100 / (1 + rs))

    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    macd, macd_sig = ema12 - ema26, (ema12 - ema26).ewm(span=9, adjust=False).mean()
    data['MACD_Hist_Rising'] = (macd - macd_sig) > (macd - macd_sig).shift(1)

    # 4. 🌟 เพิ่มการคำนวณจุด Swing สำหรับ Fibo 🌟
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
    
    # ข้อมูลสำหรับ Fibo
    recent_high = float(row['Rolling_High'])
    recent_low = float(row['Rolling_Low'])
    atr = float(row['ATR14'])

    # โครงสร้าง Trend
    bull_stack = (close > ema200) and (ema50 > ema200)
    bear_stack = (close < ema200) and (ema50 < ema200)

    # 🌟 สมการกาง Fibo (ขาขึ้นวัดจากบนลงล่าง, ขาลงวัดจากล่างขึ้นบน)
    diff = recent_high - recent_low
    
    # ระดับการย่อตัว (Retracement Levels) สำหรับขาขึ้น
    fibo_up_382 = recent_high - (diff * 0.382)
    fibo_up_500 = recent_high - (diff * 0.500)
    fibo_up_618 = recent_high - (diff * 0.618) # Golden Zone

    # ระดับการเด้ง (Retracement Levels) สำหรับขาลง
    fibo_dn_382 = recent_low + (diff * 0.382)
    fibo_dn_500 = recent_low + (diff * 0.500)
    fibo_dn_618 = recent_low + (diff * 0.618)

    # ตั้งระยะความคลาดเคลื่อน (Tolerance) อนุญาตให้ราคาแกว่งใกล้เส้น Fibo ได้ในระยะ 1 ATR
    tolerance = atr * 1.0 
    
    is_near_fibo_up = (abs(close - fibo_up_500) < tolerance) or (abs(close - fibo_up_618) < tolerance)
    is_near_fibo_dn = (abs(close - fibo_dn_500) < tolerance) or (abs(close - fibo_dn_618) < tolerance)

    # ==========================
    # 🎯 ตัดสินใจเข้าเทรด (Decision Logic)
    # ==========================
    trend_status = "Strong Uptrend" if bull_stack else "Strong Downtrend" if bear_stack else "Sideway/Neutral"
    
    # 1. สัญญาณเข้าซื้อแบบย่อตัว (Fibo Pullback) - คุ้มค่าที่สุด
    if bull_stack and is_near_fibo_up and macd_rising and (row['RSI14'] < 55):
        action = "BUY (Fibo Pullback 🚀)"
        
    # 2. สัญญาณเข้าซื้อแบบตามน้ำ/เบรคเอาท์ (Momentum) - รถกำลังซิ่ง
    elif bull_stack and ema_angle > 10 and macd_rising:
        action = "BUY (Momentum)"
        
    # 3. สัญญาณขายชอร์ตแบบเด้งเพื่อลงต่อ (Fibo Pullback)
    elif bear_stack and is_near_fibo_dn and not macd_rising and (row['RSI14'] > 45):
        action = "SELL (Fibo Pullback 🩸)"
        
    # 4. สัญญาณขายแบบตามน้ำ (Momentum)
    elif bear_stack and ema_angle < -10 and not macd_rising:
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
# 3. ฟังก์ชันดึงข้อมูลจาก Twelve Data
# ==========================================
def fetch_twelvedata(symbol: str, interval: str, api_key: str):
    # ดึงข้อมูลย้อนหลัง 250 แท่ง (เพียงพอสำหรับ EMA200)
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=250&apikey={api_key}"
    response = requests.get(url).json()
    
    if 'values' in response:
        df = pd.DataFrame(response['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        # Twelve Data ส่งข้อมูลล่าสุดขึ้นก่อน ต้องกลับด้านให้ข้อมูลเก่าอยู่บน (เพื่อคำนวณ EMA)
        df = df.sort_index(ascending=True)
        df = df.astype(float)
        # เปลี่ยนชื่อคอลัมน์ให้เข้ากับฟังก์ชันคำนวณเดิม
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        return df
    else:
        raise Exception(response.get('message', 'ไม่สามารถดึงข้อมูลได้ โปรดเช็คชื่อย่อหรือ API Key'))

def plot_chart(df: pd.DataFrame, ticker: str):
    # สร้างกราฟแบบแยกส่วน (2 แถว: ราคาด้านบน, RSI ด้านล่าง)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        subplot_titles=(f"{ticker}", "RSI (14)"),
                        row_heights=[0.75, 0.25]) # แบ่งสัดส่วน กราฟราคา 75%, RSI 25%

    # 1. กราฟราคา (Candlestick) และเส้น EMA
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='dodgerblue', width=1.5), name='EMA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='orange', width=1.5), name='EMA50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='red', width=1.5), name='EMA200'), row=1, col=1)

    # 2. คำนวณและวาดเส้น Fibonacci (จากรอบการสวิงล่าสุด)
    latest_high = df['Rolling_High'].iloc[-1]
    latest_low = df['Rolling_Low'].iloc[-1]
    diff = latest_high - latest_low
    fibo_500 = latest_high - (diff * 0.500)
    fibo_618 = latest_high - (diff * 0.618)

    # วาดเส้น Fibo ลงบนกราฟ
    fig.add_hline(y=latest_high, line_dash="dot", line_color="gray", annotation_text="Swing High", row=1, col=1)
    fig.add_hline(y=latest_low, line_dash="dot", line_color="gray", annotation_text="Swing Low", row=1, col=1)
    fig.add_hline(y=fibo_500, line_dash="dash", line_color="cyan", annotation_text="Fibo 50.0%", row=1, col=1)
    fig.add_hline(y=fibo_618, line_dash="dash", line_color="magenta", annotation_text="Fibo 61.8% (Golden Zone)", row=1, col=1)

    # 3. กราฟ RSI ด้านล่าง (Row 2)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI14'], line=dict(color='purple', width=1.5), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1) # เส้น Overbought
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1) # เส้น Oversold

    # ตั้งค่าเค้าโครงกราฟ
    fig.update_layout(xaxis_rangeslider_visible=False, height=550, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    
    # ปิด slider ในหน้าต่าง RSI ด้วย
    fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
    
    return fig

# ==========================================
# 4. Sidebar: ตั้งค่าระบบ & จัดการ Watchlist
# ==========================================
st.sidebar.header("🔑 API Setup")
api_key = st.sidebar.text_input("Twelve Data API Key", type="password", help="สมัครฟรีที่ twelvedata.com")

st.sidebar.header("⚙️ Settings")
# ปรับ Timeframe ให้ตรงกับรูปแบบของ Twelve Data
tf_options = {"15 นาที": "15min", "1 ชั่วโมง": "1h", "4 ชั่วโมง": "4h", "1 วัน": "1day", "1 สัปดาห์": "1week"}
selected_tf_label = st.sidebar.selectbox("เลือก Timeframe", list(tf_options.keys()), index=3)
selected_tf = tf_options[selected_tf_label]

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Risk Management")
account_size = st.sidebar.number_input("เงินทุน / Account Size ($)", min_value=100.0, value=1000.0, step=100.0)
risk_pct = st.sidebar.slider("ความเสี่ยงต่อไม้ (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
risk_amount = account_size * (risk_pct / 100)

st.sidebar.markdown("---")
st.sidebar.subheader("📝 My Watchlist (Twelve Data Format)")

# นำหุ้นไทยออก และใช้รูปแบบ Ticker ของ Twelve Data (ใช้ / คั่น)
if 'my_watchlist' not in st.session_state:
    st.session_state['my_watchlist'] = ["EUR/USD", "XAU/USD", "BTC/USD", "AAPL", "NVDA", "SPY"]

new_ticker = st.sidebar.text_input("เพิ่มชื่อย่อ (เช่น JPY/USD, MSFT)")
if st.sidebar.button("➕ Add Ticker"):
    ticker_upper = new_ticker.upper().strip()
    if ticker_upper and ticker_upper not in st.session_state['my_watchlist']:
        st.session_state['my_watchlist'].append(ticker_upper)
        st.sidebar.success(f"เพิ่ม {ticker_upper} แล้ว")

ticker_to_remove = st.sidebar.selectbox("เลือก Ticker ที่ต้องการลบ", [""] + st.session_state['my_watchlist'])
if st.sidebar.button("🗑️ Remove Ticker") and ticker_to_remove:
    st.session_state['my_watchlist'].remove(ticker_to_remove)
    st.rerun()

# ==========================================
# 5. ส่วนแสดงผลหลัก: สแกนเนอร์ & กราฟ
# ==========================================
st.subheader(f"📊 สแกนเนอร์ตลาด (Timeframe: {selected_tf_label})")

if st.button("🚀 เริ่มสแกน (Scan Now)", use_container_width=True):
    if not api_key:
        st.error("⚠️ โปรดใส่ Twelve Data API Key ที่แถบด้านซ้ายมือขวาก่อนสแกน")
    else:
        with st.spinner(f'กำลังดึงข้อมูล Real-time ผ่าน API...'):
            results = []
            charts = {}
            
            for ticker in st.session_state['my_watchlist']:
                try:
                    # ดึงข้อมูลจากฟังก์ชัน Twelve Data
                    df = fetch_twelvedata(ticker, selected_tf, api_key)
                    
                    if not df.empty and len(df) > 200:
                        df_ind = calculate_indicators(df)
                        latest = df_ind.iloc[-1]
                        res = evaluate_signal(latest)
                        price = res['Price']
                        atr = res['ATR']
                        
                        sl_dist = 1.5 * atr
                        tp_dist = sl_dist * 1.5
                        
                        if res['Signal'] == "BUY":
                            sl_price = price - sl_dist
                            tp_price = price + tp_dist
                            trade_setup = f"SL: {sl_price:.4f} | TP: {tp_price:.4f}"
                        elif res['Signal'] == "SELL":
                            sl_price = price + sl_dist
                            tp_price = price - tp_dist
                            trade_setup = f"SL: {sl_price:.4f} | TP: {tp_price:.4f}"
                        else:
                            trade_setup = "-"
                        
                        # คำนวณ Position Sizing
                        if "/" in ticker and ("USD" in ticker or "EUR" in ticker or "JPY" in ticker):
                            # สำหรับ Forex และ คริปโต
                            position = risk_amount / (sl_dist * 100000)
                            pos_text = f"{max(0.01, round(position, 2))} Lot"
                        else:
                            # สำหรับ หุ้น (Shares)
                            position = risk_amount / sl_dist
                            pos_text = f"{int(position)} หุ้น"
                            
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

                st.dataframe(summary_df.style.map(highlight_signal, subset=['Signal']), use_container_width=True)
                
                st.markdown("---")
                st.subheader("📈 กราฟราคา Real-time")
                cols = st.columns(2)
                col_idx = 0
                for tkr, fig in charts.items():
                    with cols[col_idx % 2]:
                        st.plotly_chart(fig, use_container_width=True)
                    col_idx += 1
