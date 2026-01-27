# src/macro/macro_factor_calc.py

import pandas as pd

def calc_macro_factor_logic(excess_liquidity, is_inverted):
    """
    純邏輯計算，與資料源解耦
    """
    if is_inverted or excess_liquidity < 0:
        return 0.3
    elif excess_liquidity > 0.02: # 2%
        return 1.0
    else:
        return 0.7

def calc_macro_factor_pipeline():
    """
    Pipeline 入口：讀取資料 -> 呼叫小模組 -> 存檔
    """
    print("   [Macro] Loading processed data...")
    # 讀取 Step 2 產生的 macro.csv
    df = pd.read_csv("data/processed/macro.csv")

    df["inverted"] = df["yield_spread"] < 0

    # 3. 計算 Macro Factor
    print("   [Macro] Calculating factors...")
    df["macro_factor"] = df.apply(
        lambda x: calc_macro_factor_logic(x["excess_liquidity"], x["inverted"]),
        axis=1
    )

    # 存檔
    output_path = "data/processed/macro_factor.csv"
    df.to_csv(output_path, index=False)
    print(f"   [Macro] Saved to {output_path}")

if __name__ == "__main__":
    calc_macro_factor_pipeline()