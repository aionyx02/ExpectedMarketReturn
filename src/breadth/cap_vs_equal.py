import logging
import os

import pandas as pd
import yfinance as yf

from config.path import PathConfig
from config.profile import get_runtime_profile


def breadth_signal_logic(cap_ret, equal_ret, threshold: float):
    if cap_ret > 0 and equal_ret > threshold:
        return "HEALTHY"
    if cap_ret > 0 and equal_ret <= threshold:
        return "FRAGILE"
    return "WEAK"


def _extract_close_series(raw_df: pd.DataFrame) -> pd.Series:
    if raw_df.empty:
        return pd.Series(dtype="float64")

    df = raw_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        close_mask = df.columns.get_level_values(0) == "Close"
        if close_mask.any():
            df = df.loc[:, close_mask]
            if isinstance(df, pd.DataFrame) and df.shape[1] >= 1:
                return df.iloc[:, 0]

    if "Close" in df.columns:
        close_col = df["Close"]
        if isinstance(close_col, pd.DataFrame):
            return close_col.iloc[:, 0]
        return close_col

    return pd.Series(dtype="float64")


def calc_breadth_pipeline(output_path=PathConfig.BREADTH_PARQUET):
    profile = get_runtime_profile()
    breadth_cfg = profile.get("breadth", {})
    download_cfg = breadth_cfg.get("download", {})

    cap_ticker = str(breadth_cfg.get("cap_ticker", "^GSPC"))
    equal_ticker = str(breadth_cfg.get("equal_ticker", "RSP"))
    period = str(download_cfg.get("period", "5y"))
    interval = str(download_cfg.get("interval", "1d"))
    lookback_days = int(breadth_cfg.get("lookback_days", 20))
    threshold = float(breadth_cfg.get("fragile_threshold", -0.01))

    logging.info("[Breadth] Fetching cap-weighted vs equal-weighted data...")

    try:
        cap_raw = yf.download(
            cap_ticker, period=period, interval=interval, progress=False
        )
        equal_raw = yf.download(
            equal_ticker, period=period, interval=interval, progress=False
        )
        df_cap = _extract_close_series(cap_raw)
        df_equal = _extract_close_series(equal_raw)
    except Exception as exc:
        logging.error(f"[Breadth] Download failed: {exc}")
        return False

    if df_cap.empty or df_equal.empty:
        logging.warning(
            "[Breadth] Download returned empty data. Keeping existing breadth file."
        )
        return False

    df = pd.DataFrame({"cap_price": df_cap, "equal_price": df_equal}).dropna()
    if df.empty:
        logging.warning(
            "[Breadth] Merged frame is empty after dropna. Keeping existing breadth file."
        )
        return False

    df["cap_ret_1m"] = df["cap_price"].pct_change(lookback_days)
    df["equal_ret_1m"] = df["equal_price"].pct_change(lookback_days)
    df["breadth_signal"] = df.apply(
        lambda row: breadth_signal_logic(
            row["cap_ret_1m"], row["equal_ret_1m"], threshold
        ),
        axis=1,
    )

    df.index.name = "date"
    df = df.reset_index()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    logging.info(f"[Breadth] Signal generated. Saved to {output_path}")

    latest = df.iloc[-1]
    logging.info(f"[Breadth] Running status ({latest['date'].strftime('%Y-%m-%d')})")
    logging.info(
        "[Breadth] Cap Return: "
        f"{latest['cap_ret_1m']:.2%} | Equal Return: {latest['equal_ret_1m']:.2%}"
    )
    logging.info(f"[Breadth] Signal: {latest['breadth_signal']}")
    return True


if __name__ == "__main__":
    calc_breadth_pipeline(output_path=PathConfig.BREADTH_PARQUET)
