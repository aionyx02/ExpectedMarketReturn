import pandas as pd
from config.path import PathConfig
def load_macro_data(m2_csv ="data/raw/fred/m2.csv",
                    gdp_csv = "data/raw/fred/gdp.csv",
                    yield_10y_csv = "data/raw/fred/yield_10y.csv",
                    yield_2y_csv = "data/raw/fred/yield_2y.csv"):
    m2 = pd.read_csv(m2_csv)
    gdp = pd.read_csv(gdp_csv)
    y10 = pd.read_csv(yield_10y_csv)
    y2 = pd.read_csv(yield_2y_csv)

    df = m2.merge(gdp, on="date", how="inner")
    df = df.merge(y10, on="date", how="inner")
    df = df.merge(y2, on="date", how="inner")

    df["m2_yoy"] = df["m2"].pct_change(12)
    df["gdp_yoy"] = df["gdp"].pct_change(4)  # GDP usually quarterly

    df["excess_liquidity"] = df["m2_yoy"] - df["gdp_yoy"]
    df["yield_spread"] = df["yield_10y"] - df["yield_2y"]

    df.to_csv("data/processed/macro.csv", index=False)
    return df

if __name__ == "__main__":
    load_macro_data(m2_csv= PathConfig.M2_CSV,
                    gdp_csv= PathConfig.GDP_CSV,
                    yield_10y_csv= PathConfig.YIELD_10Y_CSV,
                    yield_2y_csv= PathConfig.YIELD_2Y_CSV)
