import yfinance as yf
import pandas as pd
import os
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
def breadth_signal_logic(cap_ret, equal_ret):

    threshold = -0.01 # 容忍度，跌超過 1% 才算跌

    if cap_ret > 0 and equal_ret > threshold:
        return "HEALTHY" # 健康
    elif cap_ret > 0 and equal_ret <= threshold:
        return "FRAGILE" # 脆弱 (背離警示!)
    else:
        return "WEAK"    # 疲弱

def calc_breadth_pipeline():
    logging.info("   [Breadth] Fetching Cap-Weighted vs Equal-Weighted data...")

    try:
        df_cap = yf.download("^GSPC", period="5y", interval="1d", progress=False)['Close']
        df_equal = yf.download("RSP", period="5y", interval="1d", progress=False)['Close']
    except Exception as e:
        logging.error(f" Breadth download failed: {e}")
        return

    # 整理數據
    df = pd.DataFrame()
    df["cap_price"] = df_cap
    df["equal_price"] = df_equal
    df = df.dropna()

    df["cap_ret_1m"] = df["cap_price"].pct_change(20)
    df["equal_ret_1m"] = df["equal_price"].pct_change(20)

    #  產生信號
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
    logging.info(f"   [Breadth] Signal generated. Saved to {output_path}")

    # 顯示最新的狀態
    latest = df.iloc[-1]
    logging.info(f"      Running Status ({latest['date'].strftime('%Y-%m-%d')}):")
    logging.info(f"      Cap Return: {latest['cap_ret_1m']:.2%} | Equal Return: {latest['equal_ret_1m']:.2%}")
    logging.info(f"      Market Breadth: {latest['breadth_signal']}")

if __name__ == "__main__":
    calc_breadth_pipeline()