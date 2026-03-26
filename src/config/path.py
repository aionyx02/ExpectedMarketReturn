from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


### data / processed : breadth.parquet , final_signal.parquet , macro.parquet , macro_factor.parquet , market_return.parquet
### data / raw / fred : gdp.parquet , m2.parquet , yield_2y.parquet , yield_10y.parquet
### data / raw : fred_raw.parquet


class PathConfig:
    DATA_DIR = ROOT_DIR / "data"  # data
    RAW_DATA_DIR = DATA_DIR / "raw"  # data/raw
    PROCESSED_DATA_DIR = DATA_DIR / "processed"  # data/processed
    DATA_RAW_FRED = RAW_DATA_DIR / "fred"  # data/raw/fred
    PROFILE_DIR = ROOT_DIR / "profiles"  # profiles

    SRC_DIR = ROOT_DIR / "src"  # src

    ### data / processed
    BREADTH_PARQUET = PROCESSED_DATA_DIR / "breadth.parquet"
    FINAL_SIGNAL_PARQUET = PROCESSED_DATA_DIR / "final_signal.parquet"
    MACRO_PARQUET = PROCESSED_DATA_DIR / "macro.parquet"
    MACRO_FACTOR_PARQUET = PROCESSED_DATA_DIR / "macro_factor.parquet"
    MARKET_RETURN_PARQUET = PROCESSED_DATA_DIR / "market_return.parquet"

    ### data / raw / fred
    FRED_RAW_PARQUET = DATA_RAW_FRED / "fred_raw.parquet"

    ### data / raw / fred
    GDP_PARQUET = DATA_RAW_FRED / "gdp.parquet"
    M2_PARQUET = DATA_RAW_FRED / "m2.parquet"
    YIELD_2Y_PARQUET = DATA_RAW_FRED / "yield_2y.parquet"
    YIELD_10Y_PARQUET = DATA_RAW_FRED / "yield_10y.parquet"

    @classmethod
    def ensure_dir(cls):
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATA_RAW_FRED.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROFILE_DIR.mkdir(parents=True, exist_ok=True)
