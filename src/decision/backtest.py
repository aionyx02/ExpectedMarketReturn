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


def _annualized_growth(equity_series: pd.Series, years: float) -> float:
    if equity_series.empty or years <= 0:
        return 0.0
    start = float(equity_series.iloc[0])
    end = float(equity_series.iloc[-1])
    if start <= 0 or end <= 0:
        return 0.0
    return float((end / start) ** (1 / years) - 1)


def _position_from_signal(
    signal: str | float | None,
    bull_leverage: float,
    neutral_leverage: float,
    bear_leverage: float,
) -> float:
    if signal == "BULL":
        return bull_leverage
    if signal == "NEUTRAL":
        return neutral_leverage
    return bear_leverage


def _annualized_sharpe(
    excess_return: pd.Series,
    periods_per_year: float,
) -> float:
    std = float(excess_return.std())
    if std == 0.0 or periods_per_year <= 0:
        return 0.0
    mean = float(excess_return.mean())
    return float((mean / std) * (periods_per_year**0.5))


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
    bear_leverage = float(backtest_cfg.get("bear_leverage", 0.0))
    transaction_cost_bps = float(backtest_cfg.get("transaction_cost_bps", 8.0))

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

        df["pct_change"] = df["Close"].pct_change().fillna(0.0)
        df["signal_shifted"] = df["signal"].shift(1)
        df["period_days"] = df["date"].diff().dt.days.fillna(0.0).clip(lower=0.0)
        df["period_years"] = df["period_days"] / 365.25

        df["rf_period"] = (1 + risk_free_rate_annual) ** df["period_years"] - 1
        df["borrow_period"] = (1 + borrowing_cost_annual) ** df["period_years"] - 1
        df["position"] = [
            _position_from_signal(sig, bull_leverage, neutral_leverage, bear_leverage)
            for sig in df["signal_shifted"]
        ]
        df["prev_position"] = df["position"].shift(1).fillna(bear_leverage)
        df["turnover"] = (df["position"] - df["prev_position"]).abs()
        transaction_cost_rate = transaction_cost_bps / 10000.0

        def calculate_strategy_return(row: pd.Series) -> float:
            market_ret = row["pct_change"]
            position = float(row["position"])
            rf_period = float(row["rf_period"])
            borrow_period = float(row["borrow_period"])
            turnover = float(row["turnover"])

            if position <= 1.0:
                gross_return = (position * market_ret) + ((1 - position) * rf_period)
            else:
                borrowed = position - 1.0
                gross_return = (position * market_ret) - (borrowed * borrow_period)

            net_return = gross_return - (turnover * transaction_cost_rate)
            return float(max(net_return, -0.999))

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

        total_days = float((df["date"].iloc[-1] - df["date"].iloc[0]).days)
        years = total_days / 365.25 if total_days > 0 else 0.0
        periods_per_year = (len(df) / years) if years > 0 else 0.0

        total_ret_bench = (df["benchmark_equity"].iloc[-1] / 100) - 1
        total_ret_strat = (df["strategy_equity"].iloc[-1] / 100) - 1
        cagr_bench = _annualized_growth(df["benchmark_equity"], years)
        cagr_strat = _annualized_growth(df["strategy_equity"], years)
        mdd_bench = _calc_max_drawdown(df["benchmark_equity"])
        mdd_strat = _calc_max_drawdown(df["strategy_equity"])

        benchmark_excess = df["pct_change"] - df["rf_period"]
        strategy_excess = df["strategy_return"] - df["rf_period"]
        sharpe_bench = _annualized_sharpe(benchmark_excess, periods_per_year)
        sharpe_strat = _annualized_sharpe(strategy_excess, periods_per_year)
        annual_turnover = float(df["turnover"].sum() / years) if years > 0 else 0.0
        total_cost = float((df["turnover"] * transaction_cost_rate).sum())

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
    print(f"{'CAGR':<22} | {cagr_bench * 100:>11.2f}% | {cagr_strat * 100:>11.2f}%")
    print(f"{'Max Drawdown':<22} | {mdd_bench * 100:>11.2f}% | {mdd_strat * 100:>11.2f}%")
    print(f"{'Sharpe':<22} | {sharpe_bench:>12.2f} | {sharpe_strat:>12.2f}")
    print("-" * 52)
    print(f"{'Sample Length':<22} | {years:>11.2f}y | {years:>11.2f}y")
    print(f"{'Annual Turnover':<22} | {'--':>12} | {annual_turnover:>11.2f}x")
    print(f"{'Txn Cost Assumption':<22} | {'--':>12} | {transaction_cost_bps:>10.1f} bps")
    print(f"{'Total Txn Cost':<22} | {'--':>12} | {total_cost * 100:>11.2f}%")
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
