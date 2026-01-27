import pandas as pd
import os

def calc_final_signal_pipeline():
    print("   [Decision] Merging Macro, Market, and Breadth data...")

    try:
        macro = pd.read_csv("data/processed/macro_factor.csv")
        market = pd.read_csv("data/processed/market_return.csv")

        # 新增讀取 breadth
        breadth_path = "data/processed/breadth.csv"
        if os.path.exists(breadth_path):
            breadth = pd.read_csv(breadth_path)
            breadth["date"] = pd.to_datetime(breadth["date"])
            has_breadth = True
        else:
            print("⚠️ Warning: Breadth data not found. Skipping breadth check.")
            has_breadth = False

    except FileNotFoundError:
        print("Error: Missing files. Run Macro and Market steps first.")
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
        # 1. 宏觀風控
        if row["macro_factor"] < 0.8: # 嚴格一點
            return "BEAR"

        # 2. 技術面趨勢風控
        if not row["trend_signal"]:
            return "BEAR"

        # 3. [新增] 市場廣度風控
        # 如果大盤漲但廣度差 (Fragile)，禁止積極做多，強制轉為 NEUTRAL 或減碼
        if row["breadth_signal"] == "FRAGILE":
            # 即使原本要 BULL，也降級為 NEUTRAL
            return "NEUTRAL"

        if row["breadth_signal"] == "WEAK":
            return "BEAR"

        # 4. 估值決策
        if row["final_return"] > 0.05 and row["macro_factor"] >= 1.0:
            return "BULL"
        elif row["final_return"] > 0:
            return "NEUTRAL"
        else:
            return "BEAR"

    df["signal"] = df.apply(get_signal, axis=1)

    output_path = "data/processed/final_signal.csv"
    df.to_csv(output_path, index=False)
    print(f"   [Decision] Final signal saved to {output_path}")

if __name__ == "__main__":
    calc_final_signal_pipeline()