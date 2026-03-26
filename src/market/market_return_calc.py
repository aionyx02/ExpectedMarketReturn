import logging

import pandas as pd
import yfinance as yf

from config.path import PathConfig
from config.profile import get_runtime_profile


def calc_market_return_pipeline(output_path=None):
    profile = get_runtime_profile()
    market_cfg = profile.get("market", {})
    download_cfg = market_cfg.get("download", {})

    ticker = str(market_cfg.get("index_ticker", "^GSPC"))
    period = str(download_cfg.get("period", "max"))
    interval = str(download_cfg.get("interval", "1mo"))
    bias_window = int(market_cfg.get("bias_ma_window", 24))
    trend_window = int(market_cfg.get("trend_ma_window", 10))
    base_return = float(market_cfg.get("base_return", 0.08))
    sensitivity = float(market_cfg.get("sensitivity", 0.2))
    expected_fallback = float(market_cfg.get("expected_return_fallback", base_return))

    logging.info(f"[Market] Fetching {ticker} data from yfinance...")

    try:
        market_df = yf.download(
            ticker, period=period, interval=interval, progress=False
        )
    except Exception as exc:
        logging.error(f"[Market] Download failed for {ticker}: {exc}")
        return False

    if market_df.empty:
        logging.error("[Market] Download returned an empty DataFrame.")
        return False

    if isinstance(market_df.columns, pd.MultiIndex):
        market_df.columns = [col[0] for col in list(market_df.columns)]

    cols = list(market_df.columns)
    if "Close" not in cols:
        if "Adj Close" in cols:
            market_df.rename(columns={"Adj Close": "Close"}, inplace=True)
        else:
            logging.error(f"[Market] Missing Close column. Available: {cols}")
            return False

    market_df = market_df[["Close"]].copy()

    if not isinstance(market_df.index, pd.DatetimeIndex):
        market_df.index = pd.to_datetime(market_df.index, errors="coerce")

    market_df = market_df.dropna(subset=["Close"])
    if market_df.empty:
        logging.error("[Market] No valid Close rows after normalization.")
        return False

    market_df.index = market_df.index.to_series().dt.to_period("M").dt.to_timestamp()
    market_df.index.name = "date"
    market_df.reset_index(inplace=True)

    market_df["ma_24"] = market_df["Close"].rolling(bias_window).mean()
    market_df["bias"] = (market_df["Close"] - market_df["ma_24"]) / market_df["ma_24"]
    market_df["expected_return"] = base_return - (market_df["bias"] * sensitivity)

    market_df["ma_10"] = market_df["Close"].rolling(trend_window).mean()
    market_df["trend_signal"] = market_df["Close"] > market_df["ma_10"]

    market_df["date"] = market_df["date"].dt.strftime("%Y-%m-%d")
    market_df["expected_return"] = market_df["expected_return"].fillna(expected_fallback)
    market_df["trend_signal"] = market_df["trend_signal"].fillna(True)

    output_df = market_df[["date", "Close", "expected_return", "trend_signal"]].copy()
    output_df.to_parquet(output_path, index=False)
    logging.info(f"[Market] Processed data saved to {output_path}")
    return True


if __name__ == "__main__":
    calc_market_return_pipeline(output_path=PathConfig.MARKET_RETURN_PARQUET)
