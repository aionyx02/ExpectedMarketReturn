# main.py

import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ==========================================
# Import Pipeline Modules
# ==========================================
from src.utils import fred_loader, macro_preprocess, future_mock
from src.macro import macro_factor_calc
from src.market import market_return_calc
from src.breadth import cap_vs_equal
from src.decision import signal_calc, report, backtest

def run_pipeline():
    # 設定目標日期
    target_date_str = datetime.now().strftime("%Y-%m-%d")

    print("==========================================")
    print(f" MVP Quant Pipeline: Scientific Trend Projection")
    print(f" Target Date : {target_date_str}")
    print("==========================================")

    # ------------------------------------------------------
    # Phase 1: Batch Processing (歷史數據與趨勢推演)
    # ------------------------------------------------------

    print("\n[Step 1] Fetching Real World Data...")
    fred_loader.update_all_fred()

    print("\n[Step 2] Preprocessing Macro Data...")
    macro_preprocess.load_macro_data()

    print("\n[Step 3] Calculating Historical Macro Factors...")
    macro_factor_calc.calc_macro_factor_pipeline()

    print("\n[Step 4] Calculating Historical Market Returns...")
    market_return_calc.calc_market_return_pipeline()

    print("\n[Step 4.5] Analyzing Market Breadth (Cap vs Equal)...")
    cap_vs_equal.calc_breadth_pipeline()

    print(f"\n[Step 5] Projecting Trend to {target_date_str}...")
    try:
        future_mock.mock_future_data(target_date_str=target_date_str)
    except AttributeError:
        print("⚠️ 警告：找不到 mock_future_data，跳過推算步驟。")

    print("\n[Step 6] Generating Final Signals...")
    signal_calc.calc_final_signal_pipeline()

    print("\n[Step 7] Analyzing Market Status...")
    report.generate_market_report()

    print("\n[Step 8] Running Backtest...")
    backtest.run_backtest()

    # ------------------------------------------------------
    # Phase 2: Real-Time Nowcasting & Actionable Advice
    # ------------------------------------------------------

    print("\n==========================================")
    print("[Step 10] Executing High-Frequency Nowcasting...")
    print("==========================================")

    try:
        # 1. 讀取與檢查數據
        macro_path = "data/processed/macro.csv"
        if not os.path.exists(macro_path):
            raise FileNotFoundError(f"找不到數據檔案: {macro_path}")

        df_macro = pd.read_csv(macro_path)
        # 數據健壯性檢查：處理缺失值
        if df_macro.isnull().values.any():
            print("⚠️ 檢測到數據缺失，執行自動填充 (ffill)...")
            df_macro.ffill(inplace=True)

        latest_row = df_macro.iloc[-1]
        prev_idx = max(0, len(df_macro) - 4)
        prev_3m_row = df_macro.iloc[prev_idx]

        current_snapshot = {
            '10Y_Yield': latest_row.get('10Y_Yield', latest_row.get('DGS10', 4.0)),
            '2Y_Yield': latest_row.get('2Y_Yield', latest_row.get('DGS2', 3.8)),
            'Jobless_Claims_4W_MA': latest_row.get('Jobless_Claims', latest_row.get('ICSA', 220000)),
            'Jobless_Claims_3M_Ago': prev_3m_row.get('Jobless_Claims', prev_3m_row.get('ICSA', 210000)),
            'PMI': latest_row.get('PMI', 50.0)
        }

        # 2. 計算當下宏觀係數
        nowcast_factor, risks = macro_factor_calc.calculate_macro_factor(current_snapshot)

        # 3. 取得估值與廣度資訊
        signal_path = "data/processed/final_signal.csv"
        raw_expected_return = 0.05
        breadth_status = "UNKNOWN"

        if os.path.exists(signal_path):
            df_signal = pd.read_csv(signal_path)
            latest_signal = df_signal.iloc[-1]
            raw_expected_return = latest_signal.get('expected_return', 0.05)
            breadth_status = latest_signal.get('breadth_signal', 'HEALTHY')

        # 4. 決策運算
        final_decision_return = raw_expected_return * nowcast_factor

        # 5. 輸出實戰診斷儀表板
        print(f"\n 數據基準日: {target_date_str}")
        print("-" * 50)
        print(f" 模型指標摘要:")
        print(f"   - 預期年化報酬: {raw_expected_return:.2%}")
        print(f"   - 宏觀風險修正: x{nowcast_factor:.2f}")
        print(f"   - 市場廣度狀態: {breadth_status}")
        print("-" * 50)
        print(f" 修正後預期回報: {final_decision_return:.2%}")

        # 6. 【核心】資產操作指令與槓桿建議
        print(f"\n 【推薦動作】")
        print("-" * 50)

        #leverage = 1.0
        #action = ""
        #reason = ""

        # 槓桿與操作邏輯判斷
        if final_decision_return <= 0:
            leverage = 0.0
            action = "🔴 避險/空手 (Risk Off)"
            reason = "模型預測為負報酬，大盤下行風險極高，建議撤離市場。"
        elif breadth_status == "FRAGILE":
            leverage = 0.5
            action = "🟡 減倉/避險 (Defensive)"
            reason = "偵測到『指標背離』：權值股獨強但廣度轉差，結構脆弱，建議部位減半。"
        elif final_decision_return > 0.08 and nowcast_factor >= 1.0 and breadth_status == "HEALTHY":
            leverage = 2.0
            action = "🟢 強力買進 (Aggressive Buy)"
            reason = "估值極度便宜且宏觀順風，建議開啟 2x 槓桿（如 SSO/UPRO）加速。"
        elif final_decision_return > 0.04 and nowcast_factor >= 0.9:
            leverage = 1.0
            action = "🔵 正常持有 (Neutral/Buy)"
            reason = "環境穩健但回報空間一般，建議 100% 現貨持倉（SPY/VOO），不開槓桿。"
        else:
            leverage = 0.8
            action = "🟡 謹慎持有 (Weak Buy)"
            reason = "雖有回報預期，但宏觀數據出現微弱逆風，建議稍微調低倉位。"

        print(f"指令動態：{action}")
        print(f"槓桿倍數：{leverage}x")
        print(f"建議配置：{int(leverage*100)}% 部位投資於 SPY/VOO，{int((1-min(leverage,1))*100)}% 留存現金")
        print(f"理由詳述：{reason}")
        print("-" * 50)

    except Exception as e:
        print(f"❌ Step 10 執行失敗: {e}")

    # Step 9: 畫圖 (視覺化)
    print("\n[Step 9] Visualizing Results...")
    visualize()
    print("\n✅ Pipeline Completed Successfully!")

def visualize():
    path = "data/processed/final_signal.csv"
    if not os.path.exists(path):
        print("No signal file found to plot.")
        return

    df = pd.read_csv(path, parse_dates=["date"])
    recent_date = df["date"].max() - pd.DateOffset(years=5)
    df_recent = df[df["date"] >= recent_date]

    if not df_recent.empty:
        fig, ax1 = plt.subplots(figsize=(14, 7))
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Macro Factor', color='tab:blue')
        ax1.plot(df_recent["date"], df_recent["macro_factor"], color='tab:blue', label='Macro Factor', alpha=0.8)
        ax1.axhline(y=1.0, color='gray', linestyle='--')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Final Return (Adjusted)', color='tab:orange')
        ax2.plot(df_recent["date"], df_recent["final_return"], color='tab:orange', label='Adjusted Return')
        ax2.fill_between(df_recent["date"], df_recent["final_return"], 0, where=(df_recent["final_return"] >= 0), color='tab:green', alpha=0.2)
        ax2.fill_between(df_recent["date"], df_recent["final_return"], 0, where=(df_recent["final_return"] < 0), color='tab:red', alpha=0.2)

        plt.title("MVP Quant Dashboard: Last 5 Years Projection")
        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_pipeline()