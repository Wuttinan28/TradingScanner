import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# ==========================================
# 1. ตั้งค่าหน้าเพจ Streamlit
# ==========================================
st.set_page_config(page_title="Multi-Asset Trading Dashboard", layout="wide")
st.title("📈 Multi-Asset Scanner & Position Calculator")
st.markdown("ระบบวิเคราะห์และคัดกรองสัญญาณเทรดสำหรับ **Forex, หุ้น, และ ETF กองทุน**")

# ==========================================
# 2. ฟังก์ชันคำนวณ Indicators และ Signal
# ==========================================
def calculate_indicators(df: pd.DataFrame, slope_lookback: int = 5) -> pd.DataFrame:
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

    # EMA20 Angle (ATR-Normalized)
    ema_diff = data['EMA20'] - data['EMA20'].shift(slope_lookback)
    norm_slope = ema_diff / (slope_lookback * data['ATR14'])
    data['EMA20_Angle'] = np.degrees(np.arctan(norm_slope))

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    data['RSI14'] = 100 - (100 / (1 + rs))

    # MACD & ADX (แบบรวบรัดเพื่อประหยัดบรรทัด)
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    macd, macd_sig = ema12 - ema26, (ema12 - ema26).ewm(span=9, adjust=False).mean()
    data['MACD_Hist_Rising'] = (macd - macd_sig) > (macd - macd_sig).shift(1)
    data['ADX14'] = np.random.uniform(20, 40, size=len(data)) # *Mock ADX for simplicity in demo*

    return data

def evaluate_signal(row: pd.Series) -> dict:
    # เพิ่มการแปลงค่าให้เป็น float เพื่อป้องกันปัญหาประเภทข้อมูลจาก pandas
    close = float(row['Close'])
    ema20 = float(row['EMA20'])
    ema50 = float(row['EMA50'])
    ema200 = float(row['EMA200'])
    ema_angle = float(row['EMA20_Angle'])
    macd_rising = bool(row['MACD_Hist_Rising'])

    bull_stack = (close > ema20) and (ema20 > ema50) and (ema50 > ema200)
    bear_stack = (close < ema20) and (ema20 < ema50) and (ema50 < ema200)

    trend_status = "Strong Uptrend" if bull_stack else "Strong Downtrend" if bear_stack else "Sideway/Neutral"
    
    if bull_stack and ema_angle > 20 and macd_rising:
        action = "BUY (Trend Follow)"
    elif bear_stack and ema_angle < -20 and not macd_rising:
        action = "SELL (Trend Follow)"
    else:
        action = "WAIT"

    return {
        "Price": round(close, 4),
        "Trend": trend_status,
        "EMA_Angle(°)": round(ema_angle, 1),
        "RSI": round(float(row['RSI14']), 1),
        "ATR": round(float(row['ATR14']), 4),
        "Signal": action
    }

# ==========================================
# 3. Sidebar: ตั้งค่า Position Sizing
# ==========================================
st.sidebar.header("⚙️ Risk Management")
account_size = st.sidebar.number_input("Account Balance ($)", min_value=100.0, value=1000.0, step=100.0)
risk_pct = st.sidebar.slider("Risk per Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
risk_amount = account_size * (risk_pct / 100)
st.sidebar.info(f"💰 ความเสี่ยงที่รับได้ต่อ 1 ไม้: **${risk_amount:,.2f}**")

# ==========================================
# 4. ส่วนแสดงผลตารางสแกนเนอร์
# ==========================================
watchlist = {
    "EURUSD=X": "Forex (EUR/USD)",
    "GC=F": "Gold (XAU/USD)",
    "NVDA": "US Stock (Nvidia)",
    "AAPL": "US Stock (Apple)",
    "SPY": "ETF (S&P 500)"
}

st.subheader("📊 Live Market Scanner (Daily Timeframe)")

if st.button("🚀 สแกนตลาดเดี๋ยวนี้ (Scan Now)"):
    with st.spinner('กำลังดึงข้อมูลและคำนวณสัญญาณ...'):
        results = []
        for ticker, name in watchlist.items():
            try:
                df = yf.download(ticker, period="1y", interval="1d", progress=False)
                
                # --- ส่วนที่แก้ไข: จัดการปัญหา MultiIndex ของ yfinance ---
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                # ----------------------------------------------------
                
                if not df.empty and len(df) > 200:
                    df_ind = calculate_indicators(df)
                    latest = df_ind.iloc[-1]
                    res = evaluate_signal(latest)
                    
                    # คำนวณ Position Sizing
                    sl_dist = 1.5 * res['ATR']
                    if "Forex" in name or "Gold" in name:
                        position = risk_amount / (sl_dist * 100000)
                        pos_text = f"{max(0.01, round(position, 2))} Lot"
                    else:
                        position = risk_amount / sl_dist
                        pos_text = f"{int(position)} Shares"
                        
                    res["Asset"] = name
                    res["Ticker"] = ticker
                    res["StopLoss_Dist"] = round(sl_dist, 4)
                    res["Suggested_Position"] = pos_text
                    results.append(res)
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดกับ {ticker}: {e}")

        # สร้าง DataFrame และไฮไลท์สี
        if results:
            summary_df = pd.DataFrame(results)
            cols = ["Asset", "Ticker", "Price", "Trend", "EMA_Angle(°)", "RSI", "Signal", "StopLoss_Dist", "Suggested_Position"]
            summary_df = summary_df[cols]

            def highlight_signal(val):
                if pd.isna(val): return ''
                val_str = str(val)
                if 'BUY' in val_str: return 'background-color: #d4edda; color: green; font-weight: bold'
                elif 'SELL' in val_str: return 'background-color: #f8d7da; color: red; font-weight: bold'
                return 'color: gray'

            st.dataframe(summary_df.style.map(highlight_signal, subset=['Signal']), use_container_width=True)
            
            st.success("✅ สแกนข้อมูลสำเร็จ! (อิงจากราคาปิดล่าสุด)")
