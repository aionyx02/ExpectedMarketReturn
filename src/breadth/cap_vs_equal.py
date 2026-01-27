import yfinance as yf
import pandas as pd
import os

def breadth_signal_logic(cap_ret, equal_ret):
    """
    邏輯核心：
    1. Healthy: 兩者都漲 (健康的牛市)
    2. Fragile: 權值股漲 (虛胖)，但等權重跌 (背離) -> 這是最危險的信號
    3. Weak: 兩者都跌 (空頭)
    """
    threshold = -0.01 # 容忍度，跌超過 1% 才算跌

    if cap_ret > 0 and equal_ret > threshold:
        return "HEALTHY" # 健康
    elif cap_ret > 0 and equal_ret <= threshold:
        return "FRAGILE" # 脆弱 (背離警示!)
    else:
        return "WEAK"    # 疲弱

def calc_breadth_pipeline():
    print("   [Breadth] Fetching Cap-Weighted vs Equal-Weighted data...")

    # 1. 下載數據
    # ^GSPC = S&P 500 (受 NVDA, MSFT 影響大)
    # RSP = 等權重 ETF (代表市場真實平均體溫)
    try:
        df_cap = yf.download("^GSPC", period="5y", interval="1d", progress=False)['Close']
        df_equal = yf.download("RSP", period="5y", interval="1d", progress=False)['Close']
    except Exception as e:
        print(f"❌ Breadth download failed: {e}")
        return

    # 整理數據
    df = pd.DataFrame()
    df["cap_price"] = df_cap
    df["equal_price"] = df_equal
    df = df.dropna()

    # 2. 計算月度動能 (20日報酬率)
    # 我們看的是「趨勢」，所以用月線級別的漲跌幅來比較
    df["cap_ret_1m"] = df["cap_price"].pct_change(20)
    df["equal_ret_1m"] = df["equal_price"].pct_change(20)

    # 3. 產生信號
    df["breadth_signal"] = df.apply(
        lambda row: breadth_signal_logic(row["cap_ret_1m"], row["equal_ret_1m"]),
        axis=1
    )

    # 重置索引以便存檔
    df.index.name = "date"
    df = df.reset_index()

    # 存檔
    output_path = "data/processed/breadth.csv"
    # 確保目錄存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"   ✅ [Breadth] Signal generated. Saved to {output_path}")

    # 顯示最新的狀態
    latest = df.iloc[-1]
    print(f"      Running Status ({latest['date'].strftime('%Y-%m-%d')}):")
    print(f"      Cap Return: {latest['cap_ret_1m']:.2%} | Equal Return: {latest['equal_ret_1m']:.2%}")
    print(f"      Market Breadth: {latest['breadth_signal']}")

if __name__ == "__main__":
    calc_breadth_pipeline()