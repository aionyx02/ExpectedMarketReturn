import logging
import os
import time

import pandas as pd
from tqdm import tqdm

from config.path import PathConfig
from config.profile import get_runtime_profile


def generate_market_report(path: str | None = None):
    path = path or PathConfig.FINAL_SIGNAL_PARQUET
    if not os.path.exists(path):
        logging.error(f"[Report] Signal file not found: {path}")
        return False

    profile = get_runtime_profile()
    nowcast_cfg = profile.get("decision", {}).get("nowcast", {})
    bull_leverage = float(nowcast_cfg.get("bull_leverage", 2.0))
    neutral_leverage = float(nowcast_cfg.get("neutral_leverage", 1.0))
    bear_leverage = float(nowcast_cfg.get("bear_leverage", 0.0))

    with tqdm(total=3, desc="Market Report") as pbar:
        df = pd.read_parquet(path)
        if df.empty:
            logging.warning("[Report] Signal file is empty.")
            return False

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        if df.empty:
            logging.warning("[Report] No valid rows after date normalization.")
            return False

        time.sleep(0.2)
        pbar.update(1)
        pbar.set_postfix_str("Loaded")

        latest = df.iloc[-1]
        c_date = latest["date"].strftime("%Y-%m-%d")
        c_macro = float(latest.get("macro_factor", 0.0))
        c_ret = float(latest.get("final_return", 0.0)) * 100.0
        c_sig = str(latest.get("signal", "UNKNOWN"))

        time.sleep(0.2)
        pbar.update(1)
        pbar.set_postfix_str("Analyzed")

        time.sleep(0.2)
        pbar.update(1)
        pbar.set_postfix_str("Done")

    risk_label = "SAFE" if c_macro >= 1.0 else "RISKY"

    print("\n" + "=" * 60)
    print(" Market Diagnostic Report")
    print("=" * 60)
    print(f"Date: {c_date}")
    print(f"Macro Factor: {c_macro:.2f} ({risk_label})")
    print(f"Expected Return (Adjusted): {c_ret:.2f}%")
    print(f"Signal: {c_sig}")
    print("-" * 60)
    print("Suggested Exposure:")

    if c_sig == "BULL":
        print(f"  {bull_leverage:.1f}x (Aggressive)")
    elif c_sig == "NEUTRAL":
        print(f"  {neutral_leverage:.1f}x (Neutral)")
    else:
        print(f"  {bear_leverage:.1f}x (Defensive)")

    print("=" * 60 + "\n")
    return True


if __name__ == "__main__":
    generate_market_report(PathConfig.FINAL_SIGNAL_PARQUET)
