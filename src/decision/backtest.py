import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
from config.path import PathConfig
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
def run_backtest(path : str | None= None):

    if not os.path.exists(path):
        logging.error(" 錯誤：找不到數據文件，請先執行 main.py。")
        return

    logging.info(f" 正在進行 Phase 4 回測：動態槓桿 (Dynamic Leverage)...")

    #  讀取信號數據
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    #  計算大盤回報 (Benchmark Return)
    # 用 Close 價格計算
    df["pct_change"] = df["Close"].pct_change()

    # 定義策略邏輯
    # Shift 1: 用上個月的信號操作這個月
    df["signal_shifted"] = df["signal"].shift(1)

    risk_free_rate_annual = 0.03       # 無風險利率 (持有現金時賺的，年化 3%)
    borrowing_cost_annual = 0.05       # 借貸成本 (開槓桿要付的利息 + 耗損，年化 5%)

    rf_monthly = risk_free_rate_annual / 12
    borrow_cost_monthly = borrowing_cost_annual / 12

    def calculate_strategy_return(row):
        sig = row["signal_shifted"]
        market_ret = row["pct_change"]

        if pd.isna(sig):
            return 0

            #  動態槓桿邏輯
        if sig == "BULL":
            #  兩倍槓桿
            leverage = 2.0
            # 回報 = (市場漲跌 * 2) - (借那一半錢的利息成本)
            # 公式: Leverage * Return - (Leverage - 1) * Cost
            strat_ret = (market_ret * leverage) - ((leverage - 1) * borrow_cost_monthly)
            return strat_ret

        elif sig == "NEUTRAL":
            # 一倍槓桿
            return market_ret

        else: # BEAR
            # 空手
            # 持有現金賺無風險利息
            return rf_monthly

    df["strategy_return"] = df.apply(calculate_strategy_return, axis=1)

    # 計算淨值曲線
    # 假設初始資金 100
    df["benchmark_equity"] = (1 + df["pct_change"]).cumprod() * 100
    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod() * 100

    # 填補第一筆 NaN 為 100
    df.loc[0, "benchmark_equity"] = 100
    df.loc[0, "strategy_equity"] = 100

    #  計算績效指標
    total_ret_bench = (df["benchmark_equity"].iloc[-1] / 100) - 1
    total_ret_strat = (df["strategy_equity"].iloc[-1] / 100) - 1

    # 最大回撤
    def calc_max_drawdown(equity_series):
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        return drawdown.min()

    mdd_bench = calc_max_drawdown(df["benchmark_equity"])
    mdd_strat = calc_max_drawdown(df["strategy_equity"])

    # 夏普比率
    # 簡單年化處理
    if df["pct_change"].std() == 0: sharpe_bench = 0
    else: sharpe_bench = (df["pct_change"].mean() / df["pct_change"].std()) * (12**0.5)

    if df["strategy_return"].std() == 0: sharpe_strat = 0
    else: sharpe_strat = (df["strategy_return"].mean() / df["strategy_return"].std()) * (12**0.5)

    #  生成回測報告
    print("\n" + "="*50)
    print(f" 【回測：動態槓桿】")
    print("="*50)
    print(f"{'指標 (Metric)':<20} | {'大盤 (S&P 500)':<15} | {'MVP 2x (Strategy)':<15}")
    print("-" * 60)
    print(f"{'總報酬率 (Total Ret)':<20} | {total_ret_bench*100:6.2f}%          | {total_ret_strat*100:6.2f}%")
    print(f"{'最大回撤 (Max DD)':<20} | {mdd_bench*100:6.2f}%     | {mdd_strat*100:6.2f}%")
    print(f"{'夏普比率 (Sharpe)':<20} | {sharpe_bench:6.2f}            | {sharpe_strat:6.2f}")
    print("-" * 60)

    print("="*50 + "\n")

    # 7. 畫圖
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["benchmark_equity"], label="S&P 500 (1x)", color="gray", linestyle="--", alpha=0.6)
    plt.plot(df["date"], df["strategy_equity"], label="MVP Dynamic (0x-2x)", color="red", linewidth=2)

    plt.title(" Dynamic Leverage vs S&P 500", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Equity (Log Scale)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log") # 開啟對數座標
    plt.show()

if __name__ == "__main__":
    run_backtest(PathConfig.FINAL_SIGNAL_CSV)