# src/macro/macro_factor_calc.py

import pandas as pd
import numpy as np
import os

# =========================================================
#  Part 1: 核心邏輯 (平衡獲利與避險版)
# =========================================================

def calc_macro_factor_logic(excess_liquidity, yield_spread, pmi=50):
    """
    平衡型宏觀邏輯：
    1. 利差在 0.2% 以上維持強勢，0.2% 以下才開始線性撤退。
    2. 加入 PMI 加分項，當經濟擴張時允許係數突破 1.0 (最高 1.3)。
    """

    # --- 1. 殖利率曲線得分 (精確化) ---
    if yield_spread < 0:
        # 倒掛期：極度保守
        spread_score = 0.5 + (yield_spread * 0.5)
    elif yield_spread < 0.2:
        # 警戒區 (0~0.2%)：快速反應，從 1.0 降至 0.6
        spread_score = 0.6 + (yield_spread / 0.2) * 0.4
    else:
        # 安全區：1.0x 基準
        spread_score = 1.0

    # --- 2. 流動性得分 ---
    # 改為較寬鬆的線性邏輯，流動性 > 0% 就不應大幅扣分
    if excess_liquidity > 0:
        liq_score = 0.9 + (excess_liquidity * 5) # 1% 時約 0.95, 2% 時為 1.0
    else:
        liq_score = 0.8 + (excess_liquidity * 10) # 負值時快速扣分

    # --- 3. 經濟擴張獎勵 (PMI) ---
    pmi_bonus = 0
    if pmi > 52:
        pmi_bonus = (pmi - 52) * 0.02 # PMI 60 時加 0.16
    elif pmi < 48:
        pmi_bonus = (pmi - 48) * 0.05 # PMI 40 時扣 0.4

    # --- 4. 綜合權衡 ---
    # 取兩者最小值作為底部，再疊加 PMI 獎勵
    base_score = min(spread_score, liq_score)
    final_factor = base_score + pmi_bonus

    return round(max(0.3, min(1.3, final_factor)), 2)


# =========================================================
#  Part 2: 兼容接口 (供 main.py Step 10 呼叫)
# =========================================================

def calculate_macro_factor(current_snapshot):
    """
    對接 main.py Step 10 的字典格式
    """
    # 1. 取得利差
    y10 = current_snapshot.get('10Y_Yield', 4.0)
    y2 = current_snapshot.get('2Y_Yield', 3.8)
    spread = y10 - y2

    # 2. 取得流動性 (若無則預設為 0.01)
    liq = current_snapshot.get('excess_liquidity', 0.01)

    # 3. 取得 PMI
    pmi = current_snapshot.get('PMI', 50)

    # 執行邏輯
    factor = calc_macro_factor_logic(liq, spread, pmi)

    # 產生風險標籤
    risks = []
    if spread < 0: risks.append("殖利率曲線倒掛")
    if pmi < 48: risks.append("製造業萎縮")
    if factor < 0.7: risks.append("宏觀綜合風險高")

    return factor, risks


# =========================================================
#  Part 3: Pipeline 流程
# =========================================================

def calc_macro_factor_pipeline():
    input_path = "data/processed/macro.csv"
    output_path = "data/processed/macro_factor.csv"

    if not os.path.exists(input_path):
        print(f"⚠️ [Macro] 找不到 {input_path}")
        return

    print("   [Macro] Loading data for historical calculation...")
    df = pd.read_csv(input_path)

    # 欄位預處理
    if "yield_spread" not in df.columns:
        if "DGS10" in df.columns and "DGS2" in df.columns:
            df["yield_spread"] = df["DGS10"] - df["DGS2"]
        else:
            df["yield_spread"] = 0.5

    if "excess_liquidity" not in df.columns:
        df["excess_liquidity"] = 0.01

    if "PMI" not in df.columns:
        df["PMI"] = 50

    df = df.ffill().fillna(0)

    # 執行批次計算
    df["macro_factor"] = df.apply(
        lambda x: calc_macro_factor_logic(x["excess_liquidity"], x["yield_spread"], x["PMI"]),
        axis=1
    )

    try:
        output_df = df[["date", "macro_factor"]]
        output_df.to_csv(output_path, index=False)
        print(f"    [Macro] 成功產生平衡型係數！已儲存至: {output_path}")
        print(f"    數據預覽 (最新 5 筆):\n{output_df.tail().to_string(index=False)}")
    except Exception as e:
        print(f"   ❌ 存檔失敗: {e}")

if __name__ == "__main__":
    calc_macro_factor_pipeline()