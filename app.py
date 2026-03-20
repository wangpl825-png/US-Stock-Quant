import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 美股代號與全名對照表 (VOO 作為大盤基準)
# ==========================================
TICKER_NAME_MAP = {
    "VOO": "Vanguard S&P 500 ETF (大盤基準)",
    "AAPL": "Apple Inc. (蘋果)", 
    "MSFT": "Microsoft Corp. (微軟)", 
    "NVDA": "NVIDIA Corp. (輝達)", 
    "TSLA": "Tesla Inc. (特斯拉)", 
    "AMZN": "Amazon.com Inc. (亞馬遜)", 
    "GOOGL": "Alphabet Inc. (Google)", 
    "META": "Meta Platforms Inc.", 
    "AMD": "Advanced Micro Devices", 
    "SMCI": "Super Micro Computer",
    "PLTR": "Palantir Technologies",
    "COIN": "Coinbase Global"
}

def get_stock_name(ticker):
    name = TICKER_NAME_MAP.get(ticker, "")
    return f"{ticker} {name}".strip()

# ==========================================
# 網頁基本設定 & 初始化 Session State
# ==========================================
st.set_page_config(page_title="美股量化預測與回測系統", page_icon="📈", layout="wide")
st.title("📈 美股量化預測與回測系統")
st.markdown("結合蒙地卡羅模擬、技術指標圖表與歷史回測，精準掌握投資勝率。")

# 初始化觀察名單 (Watchlist)
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["AAPL", "NVDA", "TSLA"]

# ==========================================
# 獲取即時美金轉台幣匯率
# ==========================================
@st.cache_data(ttl=3600)
def get_exchange_rate():
    try:
        twd_data = yf.Ticker("USDTWD=X").history(period="1d")
        return float(twd_data['Close'].iloc[-1])
    except:
        return 32.0 # 備用預設匯率

USD_TO_TWD = get_exchange_rate()

# ==========================================
# 側邊欄：功能選單、動態參數與 Watchlist
# ==========================================
st.sidebar.header("🎯 策略與功能選擇")
strategy = st.sidebar.radio(
    "請選擇分析模式：",
    ("歷史回溯投資試算 💰", "自選股蒙地卡羅 (含圖表) 🎲", "大盤觀測: 成交量 Top 10", "大盤觀測: 強勢噴出 (單日>15%)")
)

st.sidebar.markdown("---")
st.sidebar.subheader("📋 我的觀察名單 (Watchlist)")

# 新增股票到觀察名單
col1, col2 = st.sidebar.columns([2, 1])
new_ticker = col1.text_input("新增代號", placeholder="例如 MSFT").upper()
if col2.button("加入") and new_ticker:
    if new_ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(new_ticker)
        st.sidebar.success(f"已加入 {new_ticker}")
    else:
        st.sidebar.warning("已在名單中")

# 顯示並允許刪減當前觀察名單
current_watchlist = st.sidebar.multiselect(
    "目前名單 (點擊 'x' 移除)",
    options=st.session_state.watchlist,
    default=st.session_state.watchlist
)
st.session_state.watchlist = current_watchlist

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 蒙地卡羅參數設定")
SIMULATION_RUNS = 10000
DAYS_AHEAD = st.sidebar.slider("預測未來天數 (天)", min_value=7, max_value=365, value=30, step=1)

if strategy == "歷史回溯投資試算 💰":
    st.sidebar.markdown("---")
    st.sidebar.subheader("回測設定")
    backtest_ticker = st.sidebar.text_input("輸入要回測的美股代號", "NVDA").upper()
    backtest_days = st.sidebar.number_input("回溯天數 (天)", min_value=1, max_value=1000, value=30)
    backtest_amount_usd = st.sidebar.number_input("投資金額 (USD)", min_value=100, value=10000, step=1000)

WATCHLIST_TICKERS = list(TICKER_NAME_MAP.keys())[1:] # 排除VOO (預設掃描池)
st.sidebar.info(f"💱 目前匯率參考: 1 USD ≈ {USD_TO_TWD:.2f} TWD")

# ==========================================
# 資料獲取與運算函式
# ==========================================
@st.cache_data(ttl=1800)
def get_stock_data(tickers):
    data_dict = {}
    for t in tickers:
        df = yf.download(t.strip(), period="2y", progress=False)
        if not df.empty and len(df) > 100:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            data_dict[t.strip()] = df
    return data_dict

@st.cache_data(ttl=86400)
def get_analyst_ratings(ticker):
    try:
        info = yf.Ticker(ticker).info
        recommendation = info.get('recommendationKey', 'N/A').capitalize()
        target_price = info.get('targetMeanPrice', 'N/A')
        return recommendation, target_price
    except:
        return "N/A", "N/A"

def calculate_monte_carlo(df, runs, days):
    daily_returns = df['Close'].pct_change().dropna()
    hist_returns = daily_returns.tail(252)
    mu = hist_returns.mean()
    sigma = hist_returns.std()
    last_price = float(df['Close'].iloc[-1])
    
    simulated_returns = np.random.normal(mu, sigma, (days, runs))
    price_paths = last_price * np.exp(np.cumsum(simulated_returns, axis=0))
    final_prices = price_paths[-1, :]
    returns_dist = (final_prices - last_price) / last_price
    
    p90_return = np.percentile(returns_dist, 90)
    p50_return = np.percentile(returns_dist, 50)
    p10_return = np.percentile(returns_dist, 10)
    p1_return = np.percentile(returns_dist, 1)
    win_rate = np.sum(returns_dist > 0.005) / runs
    
    sampled_paths = price_paths[:, np.random.choice(runs, 100, replace=False)]
    return p90_return, p50_return, p10_return, p1_return, win_rate, last_price, final_prices, sampled_paths

def calculate_indicators(df):
    try:
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        sma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        df['BBU_20'] = sma20 + (std20 * 2)
        df['BBL_20'] = sma20 - (std20 * 2)
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
        df['K'] = rsv.ewm(com=2, adjust=False).mean()
        df['D'] = df['K'].ewm(com=2, adjust=False).mean()
        return df.dropna()
    except:
        return df

# ==========================================
# ★ 新增：動態化技術指標白話文解讀函數
# ==========================================
def generate_technical_summary(df, ticker):
    """讀取最新一天的數據，產生易懂的白話文分析"""
    latest = df.iloc[-1]
    
    close = float(latest['Close'])
    sma5 = float(latest['SMA_5'])
    macd = float(latest['MACD'])
    signal = float(latest['MACD_Signal'])
    hist = float(latest['MACD_Hist'])
    rsi = float(latest['RSI_14'])
    k = float(latest['K'])
    d = float(latest['D'])
    bbu = float(latest['BBU_20'])
    bbl = float(latest['BBL_20'])

    summary = f"#### 🤖 {get_stock_name(ticker)} 最新技術指標 AI 診斷 (收盤價: ${close:.2f})\n\n"
    
    # 1. 趨勢與布林通道判斷
    trend_status = "**短期強勢** (站上5日均線)" if close > sma5 else "**短期弱勢** (跌破5日均線)"
    bb_status = "，目前股價在布林通道**正常區間內**震盪。"
    if close > bbu:
        bb_status = "，且突破布林通道上軌，**短期可能過熱，留意拉回風險**。"
    elif close < bbl:
        bb_status = "，且跌破布林通道下軌，**可能出現超跌反彈的契機**。"
        
    summary += f"* **📈 趨勢與型態**：目前股價屬於{trend_status}{bb_status}\n"
    
    # 2. MACD 判斷
    macd_trend = "多頭排列 (MACD線 > 訊號線)" if macd > signal else "空頭排列 (MACD線 < 訊號線)"
    macd_mom = "正向，買盤動能較強" if hist > 0 else "負向，賣壓相對較重"
    summary += f"* **🌊 MACD 動能**：呈現**{macd_trend}**，柱狀圖顯示目前動能**{macd_mom}**。\n"
    
    # 3. RSI 判斷
    if rsi >= 70:
        rsi_status = f"**超買區 ({rsi:.1f} > 70)**，市場情緒極度樂觀，需留意短線獲利了結賣壓。"
    elif rsi <= 30:
        rsi_status = f"**超賣區 ({rsi:.1f} < 30)**，市場情緒悲觀，隨時有機會醞釀底部反彈。"
    else:
        rsi_status = f"**中立區間 ({rsi:.1f})**，多空雙方目前力道相對均衡，未見極端情緒。"
    summary += f"* **⚖️ RSI 買賣力道**：目前處於{rsi_status}\n"
    
    # 4. KD 判斷
    kd_cross = "黃金交叉 (K值大於D值)，有利多方表態" if k > d else "死亡交叉 (K值小於D值)，走勢偏向弱勢整理"
    summary += f"* **🎯 KD 隨機指標**：目前呈現**{kd_cross}** (K: {k:.1f}, D: {d:.1f})。\n"
    
    return summary

# ==========================================
# 圖表繪製函式
# ==========================================
def plot_comparison_chart(stock_data, tickers):
    fig = go.Figure()
    for t in tickers:
        if t in stock_data and not stock_data[t].empty:
            df = stock_data[t].tail(252)
            if len(df) > 0:
                perf = (df['Close'] / df['Close'].iloc[0] - 1) * 100
                if t == "VOO":
                    fig.add_trace(go.Scatter(x=df.index, y=perf, mode='lines', line=dict(color='black', width=3, dash='dash'), name=f'{get_stock_name("VOO")}'))
                else:
                    fig.add_trace(go.Scatter(x=df.index, y=perf, mode='lines', name=get_stock_name(t)))
    fig.update_layout(title="📈 多檔股票與大盤(VOO) 過去一年同期績效比較 (%)", xaxis_title="日期", yaxis_title="累積報酬率 (%)", hovermode="x unified", height=400)
    return fig

def plot_mc_distribution(last_price, sampled_paths, final_prices, ticker, days_ahead):
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], 
                        subplot_titles=(f"{get_stock_name(ticker)} 模擬未來 {days_ahead} 日走勢 (抽樣100條)", "最終預測價格分佈"))
    
    for i in range(sampled_paths.shape[1]):
        fig.add_trace(go.Scatter(y=sampled_paths[:, i], mode='lines', line=dict(color='rgba(100, 150, 250, 0.1)')), row=1, col=1)
    fig.add_hline(y=last_price, line_dash="dash", line_color="red", row=1, col=1, annotation_text="現價起點")
    
    fig.add_trace(go.Histogram(y=final_prices, orientation='h', marker_color='royalblue', opacity=0.7), row=1, col=2)
    fig.add_hline(y=last_price, line_dash="dash", line_color="red", row=1, col=2)
    
    fig.update_layout(showlegend=False, height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_technical_chart(df, ticker):
    plot_df = df.tail(100)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                        row_heights=[0.5, 0.15, 0.15, 0.2],
                        subplot_titles=(f"{get_stock_name(ticker)} K線與布林通道", "MACD", "RSI", "KD 指標"))

    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_5'], line=dict(color='orange', width=1.5), name='5MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BBU_20'], line=dict(color='rgba(150,150,150,0.5)', dash='dash'), name='BB Up'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BBL_20'], line=dict(color='rgba(150,150,150,0.5)', dash='dash'), fill='tonexty', fillcolor='rgba(150,150,150,0.1)', name='BB Low'), row=1, col=1)

    colors = ['crimson' if val >= 0 else 'forestgreen' for val in plot_df['MACD_Hist']]
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['MACD_Hist'], marker_color=colors, name='Histogram'), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MACD'], line=dict(color='blue', width=1), name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MACD_Signal'], line=dict(color='orange', width=1), name='Signal'), row=2, col=1)

    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI_14'], line=dict(color='purple', width=1.5), name='RSI'), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", row=3, col=1, line_color="red")
    fig.add_hline(y=30, line_dash="dot", row=3, col=1, line_color="green")

    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['K'], line=dict(color='blue', width=1.5), name='K'), row=4, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['D'], line=dict(color='orange', width=1.5), name='D'), row=4, col=1)

    fig.update_layout(height=800, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    return fig

# ==========================================
# 核心邏輯路由
# ==========================================
if strategy == "歷史回溯投資試算 💰":
    st.subheader(f"🕰️ 歷史投資時光機：{get_stock_name(backtest_ticker)}")
    with st.spinner("正在穿越時空調取資料..."):
        data = get_stock_data([backtest_ticker])
        df = data.get(backtest_ticker)
        
        if df is None or df.empty:
            st.error(f"找不到代號 {backtest_ticker} 的歷史資料。")
        else:
            target_date = datetime.date.today() - datetime.timedelta(days=backtest_days)
            target_date_pd = pd.to_datetime(target_date)
            past_df = df[df.index <= target_date_pd]
            
            if past_df.empty:
                st.error("回溯天數過長，超過歷史資料範圍。")
            else:
                past_price = float(past_df['Close'].iloc[-1])
                past_actual_date = past_df.index[-1].strftime('%Y-%m-%d')
                current_price = float(df['Close'].iloc[-1])
                
                shares_bought = backtest_amount_usd / past_price
                current_value_usd = shares_bought * current_price
                profit_usd = current_value_usd - backtest_amount_usd
                roi = (profit_usd / backtest_amount_usd) * 100
                
                current_value_twd = current_value_usd * USD_TO_TWD
                profit_twd = profit_usd * USD_TO_TWD
                
                st.info(f"如果你在 **{backtest_days} 天前** ({past_actual_date}) 以收盤價 **${past_price:.2f} USD** 買入 **${backtest_amount_usd:,.0f} USD** 的 **{get_stock_name(backtest_ticker)}**...")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("今日現值", f"${current_value_usd:,.0f} USD", f"約 NT$ {current_value_twd:,.0f}")
                col2.metric("投資報酬率 (ROI)", f"{roi:.2f}%", f"+ ${profit_usd:,.0f} USD (約 NT$ {profit_twd:,.0f})" if roi > 0 else f"- ${abs(profit_usd):,.0f} USD")
                col3.metric("今日現價", f"${current_price:.2f} USD", f"價差: ${current_price - past_price:.2f}")

                df_ta = calculate_indicators(df.copy())
                st.plotly_chart(plot_technical_chart(df_ta, backtest_ticker), use_container_width=True)
                
                # ★ 在圖表下方顯示動態文字解讀
                st.info(generate_technical_summary(df_ta, backtest_ticker))

else:
    with st.spinner('正在載入市場資料與執行分析...'):
        if strategy == "自選股蒙地卡羅 (含圖表) 🎲":
            target_tickers = st.session_state.watchlist.copy()
            if not target_tickers:
                st.warning("您的觀察名單目前是空的，請從左側邊欄新增！")
                st.stop()
            if "VOO" not in target_tickers:
                target_tickers.append("VOO")
        else:
            target_tickers = WATCHLIST_TICKERS
            
        stock_data = get_stock_data(target_tickers)
        filtered_tickers = []
        
        if strategy == "大盤觀測: 成交量 Top 10":
            vol_dict = {t: df['Volume'].iloc[-1] for t, df in stock_data.items() if not df.empty}
            sorted_tickers = sorted(vol_dict, key=vol_dict.get, reverse=True)
            filtered_tickers = sorted_tickers[:10]
        elif strategy == "大盤觀測: 強勢噴出 (單日>15%)":
            for t, df in stock_data.items():
                if len(df) >= 2:
                    daily_change = (df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1
                    if daily_change > 0.15:
                        filtered_tickers.append(t)
        else:
            filtered_tickers = [t for t in target_tickers if t in stock_data and (t != "VOO" or len(target_tickers) == 1)]

        if strategy == "自選股蒙地卡羅 (含圖表) 🎲" and len(target_tickers) > 1:
            st.plotly_chart(plot_comparison_chart(stock_data, target_tickers), use_container_width=True)
            st.divider()

        results = []
        if not filtered_tickers:
             st.warning(f"目前沒有符合條件的股票。")
        else:
            for t in filtered_tickers:
                df = stock_data[t]
                df_ta = calculate_indicators(df.copy())
                p90, p50, p10, p1, win_rate, price, final_prices, sampled_paths = calculate_monte_carlo(df_ta, SIMULATION_RUNS, DAYS_AHEAD)
                rec, target = get_analyst_ratings(t)
                
                results.append({
                    "Ticker": t, "Price": round(price, 2), "Rec": rec, "Target": target,
                    "P90_Return": p90, "P50_Return": p50, "P10_Return": p10, 
                    "P1_Return": p1, "Win_Rate": win_rate,
                    "DF_TA": df_ta, "final_prices": final_prices, "sampled_paths": sampled_paths
                })

    if results:
        res_df = pd.DataFrame(results).sort_values(by="P50_Return", ascending=False).reset_index(drop=True)
        st.subheader(f"📌 分析結果清單 (依預測中位數排序)")
        
        for idx, row in res_df.iterrows():
            price_twd = row['Price'] * USD_TO_TWD
            header_text = f"**{get_stock_name(row['Ticker'])}** | 現價: ${row['Price']} USD (約 NT${price_twd:,.0f}) | 🏆 勝率: {row['Win_Rate']*100:.1f}%"
            
            with st.expander(header_text):
                st.markdown(f"**機構共識評級:** `{row['Rec']}` | **平均目標價:** `${row['Target']}`")
                
                st.markdown(f"##### 🎲 蒙地卡羅未來 {DAYS_AHEAD} 天預測")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("樂觀預期 (P90)", f"{row['P90_Return']*100:.2f}%")
                c2.metric("中性預期 (P50)", f"{row['P50_Return']*100:.2f}%")
                c3.metric("保守預期 (P10)", f"{row['P10_Return']*100:.2f}%")
                c4.metric("極端風險 (P1)", f"{row['P1_Return']*100:.2f}%", delta_color="inverse")
                
                st.markdown("---")
                
                st.plotly_chart(plot_mc_distribution(row['Price'], row['sampled_paths'], row['final_prices'], row['Ticker'], DAYS_AHEAD), use_container_width=True)
                st.plotly_chart(plot_technical_chart(row['DF_TA'], row['Ticker']), use_container_width=True)
                
                # ★ 在圖表下方顯示動態文字解讀 (使用 st.info 產生淺藍色提示框)
                st.info(generate_technical_summary(row['DF_TA'], row['Ticker']))
