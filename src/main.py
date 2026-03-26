import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from breadth import cap_vs_equal
from config.path import PathConfig
from config.profile import (
    get_runtime_profile,
    get_runtime_profile_path,
    initialize_runtime_profile,
)
from decision import backtest, report, signal_calc
from macro import macro_factor_calc
from market import market_return_calc
from utils import data_guard, fred_loader, future_mock, macro_preprocess

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


def _run_nowcast_advice(target_date_str: str):
    logging.info("[Step 10] Executing nowcast advice...")
    profile = get_runtime_profile()
    nowcast_cfg = profile.get("decision", {}).get("nowcast", {})
    breadth_cfg = profile.get("breadth", {})

    default_expected_return = float(nowcast_cfg.get("default_expected_return", 0.05))
    bear_leverage = float(nowcast_cfg.get("bear_leverage", 0.0))
    fragile_leverage = float(nowcast_cfg.get("fragile_leverage", 0.5))
    bull_leverage = float(nowcast_cfg.get("bull_leverage", 2.0))
    neutral_leverage = float(nowcast_cfg.get("neutral_leverage", 1.0))
    mixed_leverage = float(nowcast_cfg.get("mixed_leverage", 0.8))
    bull_return_threshold = float(nowcast_cfg.get("bull_return_threshold", 0.08))
    bull_nowcast_threshold = float(nowcast_cfg.get("bull_nowcast_threshold", 1.0))
    neutral_return_threshold = float(nowcast_cfg.get("neutral_return_threshold", 0.04))
    neutral_nowcast_threshold = float(nowcast_cfg.get("neutral_nowcast_threshold", 0.9))
    default_breadth = str(breadth_cfg.get("default_signal", "HEALTHY"))

    macro_path = PathConfig.MACRO_FACTOR_PARQUET
    if not os.path.exists(macro_path):
        logging.warning(f"[Nowcast] Missing macro factor file: {macro_path}")
        return

    df_macro = pd.read_parquet(macro_path)
    if df_macro.empty:
        logging.warning("[Nowcast] Macro factor file is empty.")
        return

    if df_macro.isnull().values.any():
        df_macro = df_macro.ffill()

    latest_row = df_macro.iloc[-1]
    prev_idx = max(0, len(df_macro) - 4)
    prev_3m_row = df_macro.iloc[prev_idx]

    current_snapshot = {
        "10Y_Yield": latest_row.get("10Y_Yield", latest_row.get("DGS10", 4.0)),
        "2Y_Yield": latest_row.get("2Y_Yield", latest_row.get("DGS2", 3.8)),
        "Jobless_Claims_4W_MA": latest_row.get(
            "Jobless_Claims", latest_row.get("ICSA", 220000)
        ),
        "Jobless_Claims_3M_Ago": prev_3m_row.get(
            "Jobless_Claims", prev_3m_row.get("ICSA", 210000)
        ),
        "PMI": latest_row.get("PMI", 50.0),
    }

    nowcast_factor, risks = macro_factor_calc.calculate_macro_factor(current_snapshot)

    signal_path = PathConfig.FINAL_SIGNAL_PARQUET
    raw_expected_return = default_expected_return
    breadth_status = "UNKNOWN"
    model_signal = "UNKNOWN"

    if os.path.exists(signal_path):
        df_signal = pd.read_parquet(signal_path)
        if not df_signal.empty:
            latest_signal = df_signal.iloc[-1]
            raw_expected_return = float(
                latest_signal.get("expected_return", default_expected_return)
            )
            breadth_status = str(latest_signal.get("breadth_signal", default_breadth))
            model_signal = str(latest_signal.get("signal", "UNKNOWN"))

    final_decision_return = raw_expected_return * nowcast_factor

    if model_signal == "BEAR" or breadth_status == "WEAK" or final_decision_return <= 0:
        leverage = bear_leverage
        action = "Defend (Risk Off)"
        reason = "BEAR/WEAK regime or non-positive adjusted return."
    elif breadth_status == "FRAGILE":
        leverage = fragile_leverage
        action = f"Defensive ({fragile_leverage:.1f}x)"
        reason = "Breadth is FRAGILE."
    elif (
        model_signal == "BULL"
        and final_decision_return > bull_return_threshold
        and nowcast_factor >= bull_nowcast_threshold
        and breadth_status == default_breadth
    ):
        leverage = bull_leverage
        action = f"Aggressive Buy ({bull_leverage:.1f}x)"
        reason = "BULL + healthy breadth + strong adjusted return."
    elif model_signal == "NEUTRAL" or (
        final_decision_return > neutral_return_threshold
        and nowcast_factor >= neutral_nowcast_threshold
    ):
        leverage = neutral_leverage
        action = f"Neutral/Buy ({neutral_leverage:.1f}x)"
        reason = "Neutral regime with stable expected return."
    else:
        leverage = mixed_leverage
        action = f"Weak Buy ({mixed_leverage:.1f}x)"
        reason = "Mixed conditions."

    print(f"\n Data Date: {target_date_str}")
    print("-" * 50)
    print(" Model Summary:")
    print(f"   - Raw Expected Return: {raw_expected_return:.2%}")
    print(f"   - Macro Nowcast Factor: x{nowcast_factor:.2f}")
    print(f"   - Breadth Status: {breadth_status}")
    print(f"   - Model Signal: {model_signal}")
    print("-" * 50)
    print(f" Adjusted Expected Return: {final_decision_return:.2%}")
    print("\n Action:")
    print("-" * 50)
    print(f"Action: {action}")
    print(f"Leverage: {leverage:.1f}x")
    print(
        f"Allocation: {int(leverage * 100)}% risk assets, "
        f"{int((1 - min(leverage, 1)) * 100)}% cash"
    )
    print(f"Reason: {reason}")
    if risks:
        print(f"Risk Flags: {', '.join(risks)}")
    print("-" * 50)


def visualize():
    path = PathConfig.FINAL_SIGNAL_PARQUET
    if not os.path.exists(path):
        logging.warning("No signal file found to plot.")
        return

    df = pd.read_parquet(path)
    if df.empty:
        logging.warning("Signal file is empty. Skip plotting.")
        return
    if "date" not in df.columns:
        logging.warning("Signal file has no date column. Skip plotting.")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        logging.warning("No valid date rows for plotting.")
        return

    recent_date = df["date"].max() - pd.DateOffset(years=5)
    df_recent = df[df["date"] >= recent_date]
    if df_recent.empty:
        return

    required = {"macro_factor", "final_return"}
    if not required.issubset(df_recent.columns):
        logging.warning("Missing columns for visualization. Skip plotting.")
        return

    try:
        fig, ax1 = plt.subplots(figsize=(14, 7))
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Macro Factor", color="tab:blue")
        ax1.plot(
            df_recent["date"],
            df_recent["macro_factor"],
            color="tab:blue",
            label="Macro Factor",
            alpha=0.8,
        )
        ax1.axhline(y=1.0, color="gray", linestyle="--")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Final Return (Adjusted)", color="tab:orange")
        ax2.plot(
            df_recent["date"],
            df_recent["final_return"],
            color="tab:orange",
            label="Adjusted Return",
        )
        ax2.fill_between(
            df_recent["date"],
            df_recent["final_return"],
            0,
            where=(df_recent["final_return"] >= 0),
            color="tab:green",
            alpha=0.2,
        )
        ax2.fill_between(
            df_recent["date"],
            df_recent["final_return"],
            0,
            where=(df_recent["final_return"] < 0),
            color="tab:red",
            alpha=0.2,
        )

        plt.title("MVP Quant Dashboard: Last 5 Years Projection")
        fig.tight_layout()
        plt.show()
    except Exception as exc:
        logging.warning(f"[Visualize] Plot skipped due to environment issue: {exc}")


def run_pipeline():
    profile = initialize_runtime_profile()
    profile_meta = profile.get("meta", {})
    profile_name = profile_meta.get("name", "default")
    future_cfg = profile.get("future_mock", {})

    target_date_str = datetime.now().strftime("%Y-%m-%d")
    PathConfig.ensure_dir()

    data_status = data_guard.bootstrap_local_storage()
    if data_status["repaired"]:
        logging.warning(
            f"[DataGuard] Repaired {len(data_status['repaired'])} parquet file(s)."
        )
    if data_status["failed"]:
        logging.warning(
            "[DataGuard] Some parquet files are unavailable; "
            "pipeline may require online fetch/regeneration."
        )

    print("==========================================")
    print(f" Target Date : {target_date_str}")
    print(f" Profile     : {profile_name} ({get_runtime_profile_path()})")
    print("==========================================")

    logging.info("[Step 1] Fetching real-world data...")
    fred_result = fred_loader.update_all_fred(output_dir=PathConfig.RAW_DATA_DIR)
    fred_status = str(fred_result.get("status", "failed"))
    if fred_status == "failed":
        logging.warning(
            "[Step 1] FRED data is unavailable (remote + cache). "
            "Pipeline may produce incomplete macro inputs."
        )

    logging.info("[Step 2] Preprocessing macro data...")
    macro_preprocess.load_macro_data(
        m2_parquet=PathConfig.M2_PARQUET,
        gdp_parquet=PathConfig.GDP_PARQUET,
        yield_10y_parquet=PathConfig.YIELD_10Y_PARQUET,
        yield_2y_parquet=PathConfig.YIELD_2Y_PARQUET,
    )

    logging.info("[Step 3] Calculating historical macro factors...")
    macro_factor_calc.calc_macro_factor_pipeline(
        input_path=PathConfig.MACRO_PARQUET,
        output_path=PathConfig.MACRO_FACTOR_PARQUET,
    )

    logging.info("[Step 4] Calculating historical market returns...")
    market_return_calc.calc_market_return_pipeline(
        output_path=PathConfig.MARKET_RETURN_PARQUET
    )

    logging.info("[Step 4.5] Analyzing market breadth...")
    breadth_updated = cap_vs_equal.calc_breadth_pipeline(
        output_path=PathConfig.BREADTH_PARQUET
    )
    if not breadth_updated:
        logging.warning(
            "[Step 4.5] Breadth update not completed; continue with local cached data."
        )

    future_enabled = bool(future_cfg.get("enabled", False))
    if future_enabled:
        logging.info(f"[Step 5] Projecting trend to {target_date_str}...")
        try:
            future_mock.mock_future_data(
                target_date_str=target_date_str,
                write_back=bool(future_cfg.get("write_back", False)),
            )
        except Exception as exc:
            logging.warning(f"[Step 5] Future projection skipped: {exc}")
    else:
        logging.info(
            "[Step 5] Future projection disabled by profile "
            "(prevents synthetic data from entering backtests)."
        )

    logging.info("[Step 6] Generating final signals...")
    signal_ok = signal_calc.calc_final_signal_pipeline(
        macro_path=PathConfig.MACRO_FACTOR_PARQUET,
        market_path=PathConfig.MARKET_RETURN_PARQUET,
        breadth_path=PathConfig.BREADTH_PARQUET,
        output_path=PathConfig.FINAL_SIGNAL_PARQUET,
    )
    if signal_ok is False:
        logging.warning("[Step 6] Final signal generation reported failure.")

    logging.info("[Step 7] Generating market report...")
    report.generate_market_report(PathConfig.FINAL_SIGNAL_PARQUET)

    logging.info("[Step 8] Running backtest...")
    backtest.run_backtest(PathConfig.FINAL_SIGNAL_PARQUET)

    _run_nowcast_advice(target_date_str=target_date_str)

    logging.info("[Step 9] Visualizing results...")
    visualize()
    logging.info("Pipeline completed.")


if __name__ == "__main__":
    run_pipeline()
