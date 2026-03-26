import logging
import os

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from config.path import PathConfig
from config.profile import get_runtime_profile


def mock_future_data(
    target_date_str,
    path_macro=PathConfig.MACRO_FACTOR_PARQUET,
    path_market=PathConfig.MARKET_RETURN_PARQUET,
):
    profile = get_runtime_profile()
    macro_cfg = profile.get("future_mock", {}).get("macro", {})
    market_cfg = profile.get("future_mock", {}).get("market", {})

    macro_target_mean = float(macro_cfg.get("target_mean", 1.0))
    macro_decay = float(macro_cfg.get("decay_rate", 0.1))
    market_long_term_growth = float(market_cfg.get("long_term_growth", 0.0058))
    market_noise_std = float(market_cfg.get("noise_std", 0.01))
    market_target_exp_ret = float(market_cfg.get("target_expected_return", 0.05))
    market_exp_ret_decay = float(market_cfg.get("expected_return_decay", 0.1))

    target_date: pd.Timestamp = pd.to_datetime(target_date_str)
    logging.info(
        "Projecting values with mean reversion up to "
        f"{target_date.strftime('%Y-%m')}..."
    )

    if os.path.exists(path_macro):
        df_macro = pd.read_parquet(path_macro)
        df_macro["date"] = pd.to_datetime(df_macro["date"], errors="coerce")
        df_macro = df_macro.dropna(subset=["date"])

        if not df_macro.empty:
            last_date = pd.Timestamp(df_macro["date"].max())
            if last_date < target_date:
                new_rows = []
                curr_date = last_date
                last_val = float(df_macro.iloc[-1]["macro_factor"])

                while curr_date < target_date:
                    curr_date += relativedelta(months=1)
                    if curr_date > target_date:
                        break

                    next_val = last_val + (macro_target_mean - last_val) * macro_decay

                    mock_row = df_macro.iloc[-1].copy()
                    mock_row["date"] = curr_date
                    mock_row["macro_factor"] = next_val
                    new_rows.append(mock_row)
                    last_val = next_val

                if new_rows:
                    df_macro = pd.concat([df_macro, pd.DataFrame(new_rows)], ignore_index=True)
                    df_macro.to_parquet(path_macro, index=False)
                    logging.info(
                        "Macro factor projected via mean-reversion "
                        f"(target: {macro_target_mean})."
                    )

    if os.path.exists(path_market):
        df_market = pd.read_parquet(path_market)
        df_market["date"] = pd.to_datetime(df_market["date"], errors="coerce")
        df_market = df_market.dropna(subset=["date"])

        if not df_market.empty:
            last_date = pd.Timestamp(df_market["date"].max())
            if last_date < target_date:
                if "Close" not in df_market.columns:
                    logging.error("[FutureMock] market_return.parquet has no Close column.")
                    return

                new_rows = []
                curr_date = last_date
                last_price = float(df_market.iloc[-1]["Close"])
                last_exp_ret = float(df_market.iloc[-1]["expected_return"])

                while curr_date < target_date:
                    curr_date += relativedelta(months=1)
                    if curr_date > target_date:
                        break

                    monthly_change = market_long_term_growth + np.random.normal(
                        0, market_noise_std
                    )
                    new_price = last_price * (1 + monthly_change)
                    new_exp_ret = last_exp_ret + (
                        market_target_exp_ret - last_exp_ret
                    ) * market_exp_ret_decay

                    mock_row = df_market.iloc[-1].copy()
                    mock_row["date"] = curr_date
                    mock_row["Close"] = new_price
                    mock_row["expected_return"] = new_exp_ret
                    mock_row["trend_signal"] = True

                    new_rows.append(mock_row)
                    last_price = new_price
                    last_exp_ret = new_exp_ret

                if new_rows:
                    df_market = pd.concat(
                        [df_market, pd.DataFrame(new_rows)], ignore_index=True
                    )
                    df_market.to_parquet(path_market, index=False)
                    logging.info("Market return series projected for missing months.")
