import logging

import pandas as pd

from config.path import PathConfig
from config.profile import get_runtime_profile


def _to_monthly_series(
    df: pd.DataFrame,
    value_col: str,
    month_rule: str,
    agg: str = "last",
) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns or value_col not in out.columns:
        raise ValueError(f"Missing required columns: date/{value_col}")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=["date", value_col]).sort_values("date")
    out = out.set_index("date")

    if agg == "mean":
        monthly = out[value_col].resample(month_rule).mean()
    else:
        monthly = out[value_col].resample(month_rule).last()

    return monthly.to_frame(name=value_col)


def _apply_release_lag(series: pd.Series, lag_months: int) -> pd.Series:
    if lag_months <= 0:
        return series
    return series.shift(lag_months)


def load_macro_data(
    m2_parquet=PathConfig.M2_PARQUET,
    gdp_parquet=PathConfig.GDP_PARQUET,
    yield_10y_parquet=PathConfig.YIELD_10Y_PARQUET,
    yield_2y_parquet=PathConfig.YIELD_2Y_PARQUET,
):
    profile = get_runtime_profile()
    preprocess_cfg = profile.get("macro", {}).get("preprocess", {})

    month_rule = str(preprocess_cfg.get("month_rule", "MS"))
    fill_method = str(preprocess_cfg.get("fill_method", "ffill"))
    m2_release_lag = int(preprocess_cfg.get("m2_release_lag_months", 1))
    gdp_release_lag = int(preprocess_cfg.get("gdp_release_lag_months", 2))
    yield_release_lag = int(preprocess_cfg.get("yield_release_lag_months", 0))

    m2 = pd.read_parquet(m2_parquet)
    gdp = pd.read_parquet(gdp_parquet)
    y10 = pd.read_parquet(yield_10y_parquet)
    y2 = pd.read_parquet(yield_2y_parquet)

    m2_m = _to_monthly_series(m2, "m2", month_rule=month_rule, agg="last")
    gdp_m = _to_monthly_series(gdp, "gdp", month_rule=month_rule, agg="last")
    y10_m = _to_monthly_series(y10, "yield_10y", month_rule=month_rule, agg="mean")
    y2_m = _to_monthly_series(y2, "yield_2y", month_rule=month_rule, agg="mean")

    df = pd.concat([m2_m, gdp_m, y10_m, y2_m], axis=1).sort_index()
    if fill_method == "ffill":
        df = df.ffill()
    elif fill_method == "bfill":
        df = df.bfill()

    df["m2"] = _apply_release_lag(df["m2"], m2_release_lag)
    df["gdp"] = _apply_release_lag(df["gdp"], gdp_release_lag)
    df["yield_10y"] = _apply_release_lag(df["yield_10y"], yield_release_lag)
    df["yield_2y"] = _apply_release_lag(df["yield_2y"], yield_release_lag)

    df["m2_yoy"] = df["m2"].pct_change(12)
    df["gdp_yoy"] = df["gdp"].pct_change(12)
    df["excess_liquidity"] = df["m2_yoy"] - df["gdp_yoy"]
    df["yield_spread"] = df["yield_10y"] - df["yield_2y"]

    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.dropna(subset=["yield_spread"]).reset_index().rename(
        columns={"index": "date"}
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()

    df.to_parquet(PathConfig.MACRO_PARQUET, index=False)
    logging.info(
        "[MacroPreprocess] Saved monthly-aligned macro panel with "
        f"{len(df)} rows (lags: m2={m2_release_lag}m, gdp={gdp_release_lag}m, "
        f"yield={yield_release_lag}m)."
    )
    return df


if __name__ == "__main__":
    load_macro_data(
        m2_parquet=PathConfig.M2_PARQUET,
        gdp_parquet=PathConfig.GDP_PARQUET,
        yield_10y_parquet=PathConfig.YIELD_10Y_PARQUET,
        yield_2y_parquet=PathConfig.YIELD_2Y_PARQUET,
    )
