import pandas as pd

from config.path import PathConfig


def load_macro_data(
    m2_parquet=PathConfig.M2_PARQUET,
    gdp_parquet=PathConfig.GDP_PARQUET,
    yield_10y_parquet=PathConfig.YIELD_10Y_PARQUET,
    yield_2y_parquet=PathConfig.YIELD_2Y_PARQUET,
):
    m2 = pd.read_parquet(m2_parquet)
    gdp = pd.read_parquet(gdp_parquet)
    y10 = pd.read_parquet(yield_10y_parquet)
    y2 = pd.read_parquet(yield_2y_parquet)

    df = m2.merge(gdp, on="date", how="inner")
    df = df.merge(y10, on="date", how="inner")
    df = df.merge(y2, on="date", how="inner")

    df["m2_yoy"] = df["m2"].pct_change(12)
    df["gdp_yoy"] = df["gdp"].pct_change(4)  # GDP usually quarterly

    df["excess_liquidity"] = df["m2_yoy"] - df["gdp_yoy"]
    df["yield_spread"] = df["yield_10y"] - df["yield_2y"]

    df.to_parquet(PathConfig.MACRO_PARQUET, index=False)
    return df


if __name__ == "__main__":
    load_macro_data(
        m2_parquet=PathConfig.M2_PARQUET,
        gdp_parquet=PathConfig.GDP_PARQUET,
        yield_10y_parquet=PathConfig.YIELD_10Y_PARQUET,
        yield_2y_parquet=PathConfig.YIELD_2Y_PARQUET,
    )
