import logging
import os
import time

import pandas as pd
from tqdm import tqdm

from config.path import PathConfig
from config.profile import get_runtime_profile


def _normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").astype("datetime64[ns]")
    out = out.dropna(subset=["date"])
    return out


def _infer_signal(row: pd.Series, signal_cfg: dict) -> str:
    macro_bear_threshold = float(signal_cfg.get("macro_bear_threshold", 0.8))
    bull_final_return_threshold = float(
        signal_cfg.get("bull_final_return_threshold", 0.05)
    )
    bull_macro_threshold = float(signal_cfg.get("bull_macro_threshold", 1.0))
    neutral_final_return_threshold = float(
        signal_cfg.get("neutral_final_return_threshold", 0.0)
    )

    if row["macro_factor"] < macro_bear_threshold:
        return "BEAR"

    if not row["trend_signal"]:
        return "BEAR"

    if row["breadth_signal"] == "FRAGILE":
        return "NEUTRAL"

    if row["breadth_signal"] == "WEAK":
        return "BEAR"

    if (
        row["final_return"] > bull_final_return_threshold
        and row["macro_factor"] >= bull_macro_threshold
    ):
        return "BULL"

    if row["final_return"] > neutral_final_return_threshold:
        return "NEUTRAL"

    return "BEAR"


def calc_final_signal_pipeline(
    macro_path: str = PathConfig.MACRO_FACTOR_PARQUET,
    market_path: str = PathConfig.MARKET_RETURN_PARQUET,
    breadth_path: str = PathConfig.BREADTH_PARQUET,
    output_path: str = PathConfig.FINAL_SIGNAL_PARQUET,
):
    profile = get_runtime_profile()
    decision_signal_cfg = profile.get("decision", {}).get("signal", {})
    breadth_cfg = profile.get("breadth", {})

    expected_return_fallback = float(
        decision_signal_cfg.get("expected_return_fallback", 0.07)
    )
    default_breadth_signal = str(breadth_cfg.get("default_signal", "HEALTHY"))

    logging.info("[Decision] Merging Macro, Market, and Breadth data...")

    with tqdm(
        total=5,
        desc="Decision Pipeline",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {postfix}]",
    ) as pbar:
        pbar.set_postfix_str("Loading parquet files...")
        try:
            macro = pd.read_parquet(macro_path)
            market = pd.read_parquet(market_path)
        except FileNotFoundError:
            logging.error("Missing source files. Run Macro and Market steps first.")
            return False
        except Exception as exc:
            logging.error(f"Error while loading parquet files: {exc}")
            return False

        has_breadth = os.path.exists(breadth_path)
        breadth = None
        if has_breadth:
            try:
                breadth = pd.read_parquet(breadth_path)
            except Exception as exc:
                logging.warning(f"[Decision] Failed to read breadth data: {exc}")
                has_breadth = False

        time.sleep(0.3)
        pbar.update(1)

        pbar.set_postfix_str("Normalizing dates...")
        if "date" not in macro.columns or "date" not in market.columns:
            logging.error("Macro and Market data must include a 'date' column.")
            return False

        macro = _normalize_date_column(macro)
        market = _normalize_date_column(market)

        if has_breadth and breadth is not None:
            if "date" not in breadth.columns:
                logging.warning("[Decision] Breadth data has no 'date' column. Ignored.")
                has_breadth = False
            else:
                breadth = _normalize_date_column(breadth)

        time.sleep(0.3)
        pbar.update(1)

        pbar.set_postfix_str("Merging data (asof)...")
        macro = macro.sort_values("date")
        market = market.sort_values("date")
        df = pd.merge_asof(macro, market, on="date", direction="backward")

        if has_breadth and breadth is not None and not breadth.empty:
            breadth = breadth.sort_values("date")
            df = pd.merge_asof(
                df,
                breadth[["date", "breadth_signal"]],
                on="date",
                direction="backward",
            )
            df["breadth_signal"] = df["breadth_signal"].fillna(default_breadth_signal)
        else:
            df["breadth_signal"] = default_breadth_signal

        time.sleep(0.3)
        pbar.update(1)

        pbar.set_postfix_str("Calculating final signal...")
        if "macro_factor" not in df.columns:
            logging.error("Missing 'macro_factor' in merged data.")
            return False

        df["expected_return"] = df["expected_return"].fillna(expected_return_fallback)
        if "trend_signal" not in df.columns:
            df["trend_signal"] = True
        df["trend_signal"] = df["trend_signal"].fillna(True)

        df["final_return"] = df["expected_return"] * df["macro_factor"]
        df["signal"] = [
            _infer_signal(row, decision_signal_cfg) for _, row in df.iterrows()
        ]
        pbar.update(1)

        pbar.set_postfix_str("Saving parquet...")
        df.to_parquet(output_path, index=False)
        time.sleep(0.3)
        pbar.update(1)

    logging.info(f"[Decision] Final signal saved to {output_path}")
    return True


if __name__ == "__main__":
    calc_final_signal_pipeline()
