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
st.markdown("結合蒙地卡羅模擬、技術指標圖表、歷史回測與投資組合分析，精準掌握投資勝率。")

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
    ("歷史回溯投資試算 💰", "自選股蒙地卡羅 (含圖表) 🎲", "投資組合蒙地卡羅 (一年期) 💼", "大盤觀測: 成交量 Top 10", "大盤觀測: 強勢噴出 (單日>15%)")
)

st.sidebar.markdown("---")
st.sidebar.subheader("📋 我的觀察名單 (Watchlist)")

col1, col2 = st.sidebar.columns([2, 1])
new_ticker = col1.text_input("新增代號", placeholder="例如 MSFT").upper()
if col2.button("加入") and new_ticker:
    if new_ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(new_ticker)
        st.sidebar.success(f"已加入 {new_ticker}")
    else:
        st.sidebar.warning("已在名單中")

current_watchlist = st.sidebar.multiselect(
    "目前名單 (點擊 'x' 移除)",
    options=st.session_state.watchlist,
    default=st.session_state.watchlist
)
st.session_state.watchlist = current_watchlist

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 蒙地卡羅參數設定")
SIMULATION_RUNS = 10000

# 只有在非投資組合模式下，才讓使用者調天數 (投資組合固定為一年 252 天)
if strategy != "投資組合蒙地卡羅 (一年期) 💼":
    DAYS_AHEAD = st.sidebar.slider("預測未來天數 (天)", min_value=7, max_value=365, value=30, step=1)
else:
    DAYS_AHEAD = 252

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

def calculate_portfolio_mc(stock_data, tickers, weights, amount, runs=10000, days=252):
    """計算投資組合的蒙地卡羅模擬 (預設一年 252 天)"""
    df_list = []
    for t in tickers:
        df = stock_data[t]['Close'].rename(t)
        df_list.append(df)
    
    prices_df = pd.concat(df_list, axis=1).dropna()
    returns_df = prices_df.pct_change().dropna().tail(252) # 取過去 252 天做基準
    
    weight_arr = np.array(weights) / 100.0
    port_daily_returns = returns_df.dot(weight_arr)
    
    mu = port_daily_returns.mean()
    sigma = port_daily_returns.std()
    
    simulated_returns = np.random.normal(mu, sigma, (days, runs))
    port_paths = amount * np.exp(np.cumsum(simulated_returns, axis=0))
    final_values = port_paths[-1, :]
    returns_dist = (final_values - amount) / amount
    
    p90 = np.percentile(returns_dist, 90)
    p50 = np.percentile(returns_dist, 50)
    p10 = np.percentile(returns_dist, 10)
    
    # 計算虧損百分位數 (低於 0% 報酬的路徑比例)
    loss_prob = np.sum(returns_dist < 0) / runs
    loss_percentile = loss_prob * 100
    win_rate = 1 - loss_prob
    
    sampled_paths = port_paths[:, np.random.choice(runs, 100, replace=False)]
    
    return p90, p50, p10, loss_percentile, win_rate, final_values, sampled_paths

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
# ★ 更新：納入蒙地卡羅與綜合建議的技術指標解讀
# ==========================================
def generate_technical_summary(df, ticker, p90, p50, p10, win_rate, days):
    latest = df.iloc[-1]
    close = float(latest['Close'])
    sma5 = float(latest['SMA_5'])
    macd = float(latest['MACD'])
    signal = float(latest['MACD_Signal'])
    rsi = float(latest['RSI_14'])
    k = float(latest['K'])
    d = float(latest['D'])
    bbu = float(latest['BBU_20'])

    summary = f"#### 🤖 {get_stock_name(ticker)} 綜合 AI 診斷報告 (收盤價: ${close:.2f})\n\n"
    
    # 1. 技術面解析
    trend_status = "短期站上5日均線" if close > sma5 else "短期跌破5日均線"
    bb_status = "，並突破布林上軌(需防拉回)。" if close > bbu else "。"
    macd_trend = "MACD 多頭排列" if macd > signal else "MACD 空頭排列"
    kd_cross = "KD 黃金交叉" if k > d else "KD 死亡交叉"
    
    summary += f"**📊 技術指標解析**\n"
    summary += f"* **趨勢動能**：目前股價{trend_status}{bb_status}{macd_trend}，且呈現 {kd_cross}。\n"
    if rsi >= 70:
        summary += f"* **買賣力道**：RSI ({rsi:.1f}) 進入超買區，短線市場情緒過熱。\n"
    elif rsi <= 30:
        summary += f"* **買賣力道**：RSI ({rsi:.1f}) 進入超賣區，醞釀跌深反彈契機。\n"
    else:
        summary += f"* **買賣力道**：RSI ({rsi:.1f}) 處於中立區間，多空力道均衡。\n\n"

    # 2. 蒙地卡羅解析
    summary += f"**🎲 蒙地卡羅統計模型 (未來 {days} 天預測)**\n"
    summary += f"* 基於過去一年歷史波動，未來上漲勝率為 **{win_rate*100:.1f}%**。\n"
    summary += f"* **預期報酬分佈**：中位數(P50)預期報酬為 **{p50*100:.2f}%**；樂觀情境(P90)上看 **{p90*100:.2f}%**；悲觀情境(P10)則下探 **{p10*100:.2f}%**。\n\n"

    # 3. 綜合評估與建議邏輯
    summary += f"**💡 綜合評估與操作建議**\n"
    
    tech_score = 0
    if close > sma5: tech_score += 1
    if macd > signal: tech_score += 1
    if k > d: tech_score += 1
    if rsi < 70 and rsi > 30: tech_score += 1 # 穩健不極端
    
    if tech_score >= 3 and win_rate > 0.55 and p50 > 0:
        conclusion = "🟢 **積極偏多**：技術面呈強勢多頭格局，且統計機率支持正向期望值，建議可逢均線支撐偏多操作或續抱。"
    elif tech_score >= 2 and win_rate >= 0.5:
        conclusion = "🟡 **中性偏多**：技術面或統計勝率其中一項略顯掙扎，但整體仍具備上漲潛力，建議控制部位，採分批佈局策略。"
    elif tech_score < 2 and win_rate < 0.45:
        conclusion = "🔴 **保守觀望**：技術面呈現弱勢或修正格局，且蒙地卡羅顯示下行風險較高 (勝率偏低)，建議暫停加碼，耐心等待底部型態確立。"
    else:
        if rsi >= 70:
            conclusion = "🟠 **居高思危**：雖然期望值或趨勢仍向上，但技術指標嚴重超買，短線追高風險大，建議等待拉回量縮後再行評估。"
        else:
            conclusion = "⚪ **震盪整理**：多空訊號分歧，短線可能陷入箱型震盪，適合以區間高拋低吸策略應對。"

    summary += f"> {conclusion}"
    return summary

# ==========================================
# 圖表繪製函式
# ==========================================
def plot_portfolio_mc(amount, sampled_paths, final_values):
    """繪製投資組合專屬的蒙地卡羅圖"""
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], 
                        subplot_titles=(f"投資組合模擬未來 1 年走勢 (抽樣100條)", "最終預測總價值分佈"))
    
    for i in range(sampled_paths.shape[1]):
        fig.add_trace(go.Scatter(y=sampled_paths[:, i], mode='lines', line=dict(color='rgba(46, 204, 113, 0.1)')), row=1, col=1)
    fig.add_hline(y=amount, line_dash="dash", line_color="red", row=1, col=1, annotation_text="投入本金")
    
    fig.add_trace(go.Histogram(y=final_values, orientation='h', marker_color='seagreen', opacity=0.7), row=1, col=2)
    fig.add_hline(y=amount, line_dash="dash", line_color="red", row=1, col=2)
    
    fig.update_layout(showlegend=False, height=450, margin=dict(l=20, r=20, t=40, b=20), yaxis_title="投資組合總價值 (USD)")
    return fig

# (其餘原有的 plot_comparison_chart, plot_mc_distribution, plot_technical_chart 沿用)
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
                        subplot_titles=(f"{get_stock_name(ticker)} 模擬未來 {days_ahead} 日走勢", "最終預測價格分佈"))
    for i in range(sampled_paths.shape[1]):
        fig.add_trace(go.Scatter(y=sampled_paths[:, i], mode='lines', line=dict(color='rgba(100, 150, 250, 0.1)')), row=1, col=1)
    fig.add_hline(y=last_price, line_dash="dash", line_color="red", row=1, col=1, annotation_text="現價起點")
    fig.add_trace(go.Histogram(y=final_prices, orientation='h', marker_color='royalblue', opacity=0.7), row=1, col=2)
    fig.add_hline(y=last_price, line_dash="dash", line_color="red", row=1, col=2)
    fig.update_layout(showlegend=False, height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_technical_chart(df, ticker):
    plot_df = df.tail(100)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.15, 0.15, 0.2], subplot_titles=(f"{get_stock_name(ticker)} K線與布林通道", "MACD", "RSI", "KD 指標"))
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
    with st.spinner("正在調取資料..."):
        data = get_stock_data([backtest_ticker])
        df = data.get(backtest_ticker)
        if df is None or df.empty:
            st.error("找不到歷史資料。")
        else:
            target_date = datetime.date.today() - datetime.timedelta(days=backtest_days)
            past_df = df[df.index <= pd.to_datetime(target_date)]
            if past_df.empty: st.error("回溯天數過長。")
            else:
                past_price = float(past_df['Close'].iloc[-1])
                current_price = float(df['Close'].iloc[-1])
                profit_usd = (backtest_amount_usd / past_price) * current_price - backtest_amount_usd
                st.info(f"如果你在 {backtest_days} 天前以 ${past_price:.2f} 買入 ${backtest_amount_usd:,.0f} 的 {backtest_ticker}...")
                col1, col2 = st.columns(2)
                col1.metric("今日現值", f"${backtest_amount_usd + profit_usd:,.0f}", f"{profit_usd / backtest_amount_usd * 100:.2f}%")
                
                df_ta = calculate_indicators(df.copy())
                p90, p50, p10, p1, wr, _, _, _ = calculate_monte_carlo(df_ta, SIMULATION_RUNS, DAYS_AHEAD)
                st.plotly_chart(plot_technical_chart(df_ta, backtest_ticker), use_container_width=True)
                # 呼叫更新版的 AI 診斷
                st.info(generate_technical_summary(df_ta, backtest_ticker, p90, p50, p10, wr, DAYS_AHEAD))

elif strategy == "投資組合蒙地卡羅 (一年期) 💼":
    st.subheader("💼 自訂投資組合蒙地卡羅模擬 (預估未來 1 年 / 252個交易日)")
    st.markdown("設定你的專屬投資組合，系統將根據各成份股過去一年的相關性與波動度，運算 10,000 次可能路徑。")
    
    port_amount = st.number_input("投資總金額 (USD)", min_value=1000, value=100000, step=10000)
    selected_port_tickers = st.multiselect("選擇投資組合標的 (從您的 Watchlist 中挑選)", 
                                           options=st.session_state.watchlist, 
                                           default=st.session_state.watchlist)
    
    if len(selected_port_tickers) == 0:
        st.warning("請至少選擇一檔股票來建立投資組合。")
    else:
        st.markdown("##### ⚖️ 設定配置權重 (總和需為 100%)")
        cols = st.columns(len(selected_port_tickers))
        weights = []
        for i, t in enumerate(selected_port_tickers):
            default_w = 100 // len(selected_port_tickers)
            w = cols[i].number_input(f"{t} 權重 (%)", min_value=0, max_value=100, value=default_w)
            weights.append(w)
            
        total_weight = sum(weights)
        if total_weight != 100:
            st.error(f"⚠️ 目前權重總和為 {total_weight}%，請調整至剛好 100%。")
        else:
            if st.button("🚀 執行投資組合量化運算", type="primary"):
                with st.spinner("正在抓取資料並執行 10,000 次蒙地卡羅矩陣運算..."):
                    stock_data = get_stock_data(selected_port_tickers)
                    
                    # 確保所有選擇的股票都有抓到資料
                    valid_data = True
                    for t in selected_port_tickers:
                        if t not in stock_data or stock_data[t].empty:
                            st.error(f"無法獲取 {t} 的歷史資料，請重新選擇。")
                            valid_data = False
                            break
                            
                    if valid_data:
                        # 執行投組運算
                        p90, p50, p10, loss_percentile, win_rate, final_vals, sampled_paths = calculate_portfolio_mc(
                            stock_data, selected_port_tickers, weights, port_amount, SIMULATION_RUNS, 252)
                        
                        st.divider()
                        st.subheader("📊 投資組合分析報告")
                        
                        # 四個關鍵指標卡片
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("中位數預期報酬 (P50)", f"{p50*100:.2f}%", f"預期終值: ${port_amount*(1+p50):,.0f}")
                        c2.metric("樂觀情境報酬 (P90)", f"{p90*100:.2f}%", f"預期終值: ${port_amount*(1+p90):,.0f}")
                        c3.metric("悲觀情境報酬 (P10)", f"{p10*100:.2f}%", f"預期終值: ${port_amount*(1+p10):,.0f}", delta_color="inverse")
                        
                        # 虧損百分位數特別設計
                        c4.metric("📉 開始出現虧損的百分位", f"第 {loss_percentile:.1f} 百分位", "代表下行風險機率", delta_color="off")
                        
                        st.plotly_chart(plot_portfolio_mc(port_amount, sampled_paths, final_vals), use_container_width=True)
                        
                        st.info(f"""
                        **💡 投資組合風險解讀：** 根據歷史共變異數矩陣推算，您的這組資產配置在未來一年有 **{win_rate*100:.1f}%** 的機率獲得正報酬。
                        當市場發生最差的情境時，從第 **{loss_percentile:.1f} 百分位**（也就是最慘的 {loss_percentile:.1f}% 機率）開始，您的總資產將會面臨虧損。
                        如果這個虧損機率讓您感到不安，建議您調降高波動股票的權重，或加入如 VOO 等大盤 ETF 以分散風險。
                        """)

else:
    with st.spinner('正在載入市場資料與執行分析...'):
        if strategy == "自選股蒙地卡羅 (含圖表) 🎲":
            target_tickers = st.session_state.watchlist.copy()
            if not target_tickers: st.stop()
        else:
            target_tickers = WATCHLIST_TICKERS
            
        stock_data = get_stock_data(target_tickers)
        filtered_tickers = []
        
        if strategy == "大盤觀測: 成交量 Top 10":
            vol_dict = {t: df['Volume'].iloc[-1] for t, df in stock_data.items() if not df.empty}
            filtered_tickers = sorted(vol_dict, key=vol_dict.get, reverse=True)[:10]
        elif strategy == "大盤觀測: 強勢噴出 (單日>15%)":
            for t, df in stock_data.items():
                if len(df) >= 2 and (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) > 0.15:
                    filtered_tickers.append(t)
        else:
            filtered_tickers = [t for t in target_tickers if t in stock_data]

        if strategy == "自選股蒙地卡羅 (含圖表) 🎲" and len(target_tickers) > 1:
            st.plotly_chart(plot_comparison_chart(stock_data, target_tickers), use_container_width=True)

        results = []
        for t in filtered_tickers:
            df = stock_data[t]
            df_ta = calculate_indicators(df.copy())
            p90, p50, p10, p1, win_rate, price, final_prices, sampled_paths = calculate_monte_carlo(df_ta, SIMULATION_RUNS, DAYS_AHEAD)
            rec, target = get_analyst_ratings(t)
            results.append({
                "Ticker": t, "Price": price, "Rec": rec, "Target": target, "Win_Rate": win_rate,
                "P90": p90, "P50": p50, "P10": p10, "P1": p1, "DF_TA": df_ta, 
                "final_prices": final_prices, "sampled_paths": sampled_paths
            })

    if results:
        res_df = pd.DataFrame(results).sort_values(by="P50", ascending=False).reset_index(drop=True)
        st.subheader(f"📌 分析結果清單 (依預期中位數排序)")
        for idx, row in res_df.iterrows():
            with st.expander(f"**{get_stock_name(row['Ticker'])}** | 現價: ${row['Price']:.2f} | 🏆 勝率: {row['Win_Rate']*100:.1f}%"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("樂觀預期 (P90)", f"{row['P90']*100:.2f}%")
                c2.metric("中位預期 (P50)", f"{row['P50']*100:.2f}%")
                c3.metric("悲觀預期 (P10)", f"{row['P10']*100:.2f}%")
                c4.metric("極端風險 (P1)", f"{row['P1']*100:.2f}%", delta_color="inverse")
                
                st.plotly_chart(plot_mc_distribution(row['Price'], row['sampled_paths'], row['final_prices'], row['Ticker'], DAYS_AHEAD), use_container_width=True)
                st.plotly_chart(plot_technical_chart(row['DF_TA'], row['Ticker']), use_container_width=True)
                
                # 呼叫更新版的 AI 診斷
                st.info(generate_technical_summary(row['DF_TA'], row['Ticker'], row['P90'], row['P50'], row['P10'], row['Win_Rate'], DAYS_AHEAD))
