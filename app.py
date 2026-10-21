import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# ==========================================
# 1. ตั้งค่าหน้าเพจ Streamlit
# ==========================================
st.set_page_config(page_title="Pro Trading Dashboard", layout="wide")
st.title("📈 Pro Multi-Asset Scanner & Charting")
st.markdown("ระบบวิเคราะห์และคัดกรองสัญญาณพร้อมกราฟ (รองรับ Forex, คริปโต, หุ้น, กองทุน)")

# ==========================================
# 2. ฟังก์ชันคำนวณ Indicators
# ==========================================
def calculate_indicators(df: pd.DataFrame, slope_lookback: int = 5) -> pd.DataFrame:
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

    return data

def evaluate_signal(row: pd.Series) -> dict:
    close = float(row['Close'])
    ema20 = float(row['EMA20'])
    ema50 = float(row['EMA50'])
    ema200 = float(row['EMA200'])
    ema_angle = float(row['EMA20_Angle'])
    macd_rising = bool(row['MACD_Hist_Rising'])

    bull_stack = (close > ema20) and (ema20 > ema50) and (ema50 > ema200)
    bear_stack = (close < ema20) and (ema20 < ema50) and (ema50 < ema200)

    trend = "Strong Uptrend" if bull_stack else "Strong Downtrend" if bear_stack else "Sideway/Neutral"
    
    if bull_stack and ema_angle > 20 and macd_rising:
        action = "BUY"
    elif bear_stack and ema_angle < -20 and not macd_rising:
        action = "SELL"
    else:
        action = "WAIT"

    return {
        "Price": round(close, 4),
        "Trend": trend,
        "EMA_Angle(°)": round(ema_angle, 1),
        "RSI": round(float(row['RSI14']), 1),
        "ATR": round(float(row['ATR14']), 4),
        "Signal": action
    }

# ==========================================
# 3. สร้างกราฟแท่งเทียนด้วย Plotly
# ==========================================
def plot_chart(df: pd.DataFrame, ticker: str):
    fig = go.Figure()
    
    # แท่งเทียน
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    
    # เส้น EMA
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='blue', width=1.5), name='EMA20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='orange', width=1.5), name='EMA50'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='red', width=1.5), name='EMA200'))
    
    fig.update_layout(title=f"Chart: {ticker}", xaxis_rangeslider_visible=False, height=450, margin=dict(l=0, r=0, t=40, b=0))
    return fig

# ==========================================
# 4. Sidebar: ตั้งค่าระบบ & จัดการ Watchlist
# ==========================================
st.sidebar.header("⚙️ Settings & Watchlist")

# เลือก Timeframe
tf_options = {"15 นาที": "15m", "1 ชั่วโมง": "1h", "4 ชั่วโมง": "4h", "1 วัน": "1d", "1 สัปดาห์": "1wk"}
selected_tf_label = st.sidebar.selectbox("เลือก Timeframe", list(tf_options.keys()), index=3)
selected_tf = tf_options[selected_tf_label]

# คำนวณความเสี่ยง
st.sidebar.markdown("---")
st.sidebar.subheader("💰 Risk Management")
account_size = st.sidebar.number_input("เงินทุน / Account Size ($)", min_value=100.0, value=1000.0, step=100.0)
risk_pct = st.sidebar.slider("ความเสี่ยงต่อไม้ (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
risk_amount = account_size * (risk_pct / 100)
st.sidebar.info(f"เสียได้สูงสุดต่อไม้: **${risk_amount:,.2f}**")

# จัดการ Watchlist ด้วย Session State
st.sidebar.markdown("---")
st.sidebar.subheader("📝 My Watchlist")

if 'my_watchlist' not in st.session_state:
    st.session_state['my_watchlist'] = ["EURUSD=X", "GC=F", "BTC-USD", "NVDA", "AAPL", "PTT.BK"]

new_ticker = st.sidebar.text_input("เพิ่มชื่อย่อ (เช่น TSLA, AOT.BK)")
if st.sidebar.button("➕ Add Ticker"):
    ticker_upper = new_ticker.upper().strip()
    if ticker_upper and ticker_upper not in st.session_state['my_watchlist']:
        st.session_state['my_watchlist'].append(ticker_upper)
        st.sidebar.success(f"เพิ่ม {ticker_upper} แล้ว")

# ปุ่มลบ Ticker
ticker_to_remove = st.sidebar.selectbox("เลือก Ticker ที่ต้องการลบ", [""] + st.session_state['my_watchlist'])
if st.sidebar.button("🗑️ Remove Ticker") and ticker_to_remove:
    st.session_state['my_watchlist'].remove(ticker_to_remove)
    st.rerun()

st.sidebar.caption("ปัจจุบันมีสินทรัพย์ในลิสต์: " + ", ".join(st.session_state['my_watchlist']))

# ==========================================
# 5. ส่วนแสดงผลหลัก: สแกนเนอร์ & กราฟ
# ==========================================
# กำหนดระยะดูข้อมูลย้อนหลัง (Period) อิงตาม Timeframe เพื่อป้องกัน Error ขอข้อมูลเยอะเกิน
period_map = {"15m": "5d", "1h": "1mo", "4h": "1mo", "1d": "1y", "1wk": "2y"}
selected_period = period_map[selected_tf]

st.subheader(f"📊 สแกนเนอร์ตลาด (Timeframe: {selected_tf_label})")

if st.button("🚀 เริ่มสแกน (Scan Now)", use_container_width=True):
    with st.spinner(f'กำลังดึงข้อมูล Timeframe {selected_tf}...'):
        results = []
        charts = {}
        
        for ticker in st.session_state['my_watchlist']:
            try:
                df = yf.download(ticker, period=selected_period, interval=selected_tf, progress=False)
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # เช็คว่ามีข้อมูลพอที่จะคำนวณ EMA200 ไหม
                if not df.empty and len(df) > 200:
                    df_ind = calculate_indicators(df)
                    latest = df_ind.iloc[-1]
                    res = evaluate_signal(latest)
                    price = res['Price']
                    atr = res['ATR']
                    
                    # --- คำนวณ SL / TP อิงตาม ATR (สมมติ R:R = 1:1.5) ---
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
                    
                    # --- คำนวณ Position Sizing ---
                    if "X" in ticker or "USD" in ticker: # คร่าวๆ สำหรับ Forex/Crypto
                        position = risk_amount / (sl_dist * 100000)
                        pos_text = f"{max(0.01, round(position, 2))} Lot"
                    else:
                        position = risk_amount / sl_dist
                        pos_text = f"{int(position)} หุ้น"
                        
                    res["Ticker"] = ticker
                    res["Action_Plan (SL/TP)"] = trade_setup
                    res["Position_Size"] = pos_text if res['Signal'] != "WAIT" else "-"
                    results.append(res)
                    
                    # เก็บกราฟไว้แสดงผล
                    charts[ticker] = plot_chart(df_ind, ticker)

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดกับ {ticker}: ไม่พบข้อมูลหรือมีปัญหากับ Timeframe นี้")

        # 5.1 แสดงตาราง
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
            
            # 5.2 แสดงกราฟของตัวที่มีสัญญาณ (หรือแสดงทั้งหมด)
            st.markdown("---")
            st.subheader("📈 กราฟราคา (แยกตามสินทรัพย์)")
            
            # เรียงกราฟเป็นคอลัมน์ (2 กราฟต่อ 1 แถว)
            cols = st.columns(2)
            col_idx = 0
            for tkr, fig in charts.items():
                with cols[col_idx % 2]:
                    st.plotly_chart(fig, use_container_width=True)
                col_idx += 1
