import pandas as pd
import os
import logging

from config.path import PathConfig

PathConfig.ensure_dir()
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
def calc_final_signal_pipeline(macro_path : str = None ,
                               market_path : str = None ,
                               breadth_path : str = None ,
                               output_path : str = None):
    global breadth
    logging.info("   [Decision] Merging Macro, Market, and Breadth data...")

    try:
        macro = pd.read_csv(macro_path)
        market = pd.read_csv(market_path)

        if os.path.exists(breadth_path):
            breadth = pd.read_csv(breadth_path)
            breadth["date"] = pd.to_datetime(breadth["date"])
            has_breadth = True
        else:
            logging.warning(" Warning: Breadth data not found. Skipping breadth check.")
            has_breadth = False

    except FileNotFoundError:
        logging.error("Error: Missing files. Run Macro and Market steps first.")
        return

    # 確保日期格式一致
    macro["date"] = pd.to_datetime(macro["date"])
    market["date"] = pd.to_datetime(market["date"])

    # 合併 Macro 和 Market
    macro = macro.sort_values("date")
    market = market.sort_values("date")
    df = pd.merge_asof(macro, market, on="date", direction="backward")

    # 合併 Breadth
    if has_breadth:
        breadth = breadth.sort_values("date")
        df = pd.merge_asof(df, breadth[["date", "breadth_signal"]], on="date", direction="backward")
        df["breadth_signal"] = df["breadth_signal"].fillna("HEALTHY") # 預設健康
    else:
        df["breadth_signal"] = "HEALTHY"

    # 填補與計算
    df["expected_return"] = df["expected_return"].fillna(0.07)
    if "trend_signal" not in df.columns:
        df["trend_signal"] = True

    df["final_return"] = df["expected_return"] * df["macro_factor"]

    def get_signal(row):
        #  宏觀風控
        if row["macro_factor"] < 0.8: # 嚴格一點
            return "BEAR"

        #  技術面趨勢風控
        if not row["trend_signal"]:
            return "BEAR"

        # 市場廣度風控
        # 如果大盤漲但廣度差，禁止積極做多，強制轉為 NEUTRAL 或減碼
        if row["breadth_signal"] == "FRAGILE":
            # 即使原本要 BULL，也降級為 NEUTRAL
            return "NEUTRAL"

        if row["breadth_signal"] == "WEAK":
            return "BEAR"

        #  估值決策
        if row["final_return"] > 0.05 and row["macro_factor"] >= 1.0:
            return "BULL"
        elif row["final_return"] > 0:
            return "NEUTRAL"
        else:
            return "BEAR"

    df["signal"] = df.apply(get_signal, axis=1)

    df.to_csv(output_path, index=False)
    logging.info(f"   [Decision] Final signal saved to {output_path}")

if __name__ == "__main__":
    calc_final_signal_pipeline()