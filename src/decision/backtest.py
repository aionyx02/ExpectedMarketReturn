import logging
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from config.path import PathConfig
from config.profile import get_runtime_profile


def _calc_max_drawdown(equity_series: pd.Series) -> float:
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    return float(drawdown.min())


def run_backtest(path: str | None = None):
    path = path or PathConfig.FINAL_SIGNAL_PARQUET
    if not os.path.exists(path):
        logging.error(f"[Backtest] Signal file not found: {path}")
        return False

    logging.info("[Backtest] Running dynamic leverage backtest...")
    profile = get_runtime_profile()
    backtest_cfg = profile.get("decision", {}).get("backtest", {})

    risk_free_rate_annual = float(backtest_cfg.get("risk_free_rate_annual", 0.03))
    borrowing_cost_annual = float(backtest_cfg.get("borrowing_cost_annual", 0.05))
    bull_leverage = float(backtest_cfg.get("bull_leverage", 2.0))
    neutral_leverage = float(backtest_cfg.get("neutral_leverage", 1.0))

    with tqdm(
        total=5,
        desc="Backtest",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {postfix}]",
    ) as pbar:
        pbar.set_postfix_str("Loading parquet...")
        df = pd.read_parquet(path)
        if df.empty:
            logging.error("[Backtest] Input data is empty.")
            return False

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        if df.empty:
            logging.error("[Backtest] No valid rows after date normalization.")
            return False
        pbar.update(1)
        time.sleep(0.2)

        pbar.set_postfix_str("Preparing returns...")
        if "Close" not in df.columns or "signal" not in df.columns:
            logging.error("[Backtest] Missing required columns: Close/signal.")
            return False

        df["pct_change"] = df["Close"].pct_change()
        df["signal_shifted"] = df["signal"].shift(1)

        rf_monthly = risk_free_rate_annual / 12
        borrow_cost_monthly = borrowing_cost_annual / 12

        def calculate_strategy_return(row: pd.Series) -> float:
            sig = row["signal_shifted"]
            market_ret = row["pct_change"]

            if pd.isna(sig):
                return 0.0
            if pd.isna(market_ret):
                market_ret = 0.0

            if sig == "BULL":
                leverage = bull_leverage
                return float((market_ret * leverage) - ((leverage - 1) * borrow_cost_monthly))
            if sig == "NEUTRAL":
                return float(market_ret * neutral_leverage)
            return float(rf_monthly)

        pbar.update(1)
        time.sleep(0.2)

        pbar.set_postfix_str("Simulating strategy...")
        df["strategy_return"] = [calculate_strategy_return(row) for _, row in df.iterrows()]
        pbar.update(1)
        time.sleep(0.2)

        pbar.set_postfix_str("Computing metrics...")
        df["benchmark_equity"] = (1 + df["pct_change"].fillna(0.0)).cumprod() * 100
        df["strategy_equity"] = (1 + df["strategy_return"]).cumprod() * 100
        df.loc[0, "benchmark_equity"] = 100
        df.loc[0, "strategy_equity"] = 100

        total_ret_bench = (df["benchmark_equity"].iloc[-1] / 100) - 1
        total_ret_strat = (df["strategy_equity"].iloc[-1] / 100) - 1
        mdd_bench = _calc_max_drawdown(df["benchmark_equity"])
        mdd_strat = _calc_max_drawdown(df["strategy_equity"])

        bench_std = df["pct_change"].std()
        strat_std = df["strategy_return"].std()
        sharpe_bench = 0.0 if bench_std == 0 else float((df["pct_change"].mean() / bench_std) * (12**0.5))
        sharpe_strat = 0.0 if strat_std == 0 else float((df["strategy_return"].mean() / strat_std) * (12**0.5))

        pbar.update(1)
        time.sleep(0.2)

        pbar.set_postfix_str("Rendering...")
        pbar.update(1)
        time.sleep(0.2)

    print("\n" + "=" * 50)
    print(" Dynamic Leverage Backtest")
    print("=" * 50)
    print(f"{'Metric':<22} | {'Benchmark':>12} | {'Strategy':>12}")
    print("-" * 52)
    print(f"{'Total Return':<22} | {total_ret_bench * 100:>11.2f}% | {total_ret_strat * 100:>11.2f}%")
    print(f"{'Max Drawdown':<22} | {mdd_bench * 100:>11.2f}% | {mdd_strat * 100:>11.2f}%")
    print(f"{'Sharpe':<22} | {sharpe_bench:>12.2f} | {sharpe_strat:>12.2f}")
    print("-" * 52)
    print("=" * 50 + "\n")

    try:
        plt.figure(figsize=(12, 6))
        plt.plot(
            df["date"],
            df["benchmark_equity"],
            label="S&P 500 (1x)",
            color="gray",
            linestyle="--",
            alpha=0.6,
        )
        plt.plot(
            df["date"],
            df["strategy_equity"],
            label="MVP Dynamic (0x-2x)",
            color="red",
            linewidth=2,
        )

        plt.title("Dynamic Leverage vs S&P 500", fontsize=14)
        plt.xlabel("Date")
        plt.ylabel("Equity (Log Scale)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale("log")
        plt.show()
    except Exception as exc:
        logging.warning(f"[Backtest] Plot skipped due to environment issue: {exc}")

    return True


if __name__ == "__main__":
    run_backtest(PathConfig.FINAL_SIGNAL_PARQUET)
