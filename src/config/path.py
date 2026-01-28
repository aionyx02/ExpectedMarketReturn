from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


### data / processed : breadth.csv , final_signal.csv , macro.csv , macro_factor.csv , market_return.csv
### data / raw / fred : gdp.csv , m2.csv , yield_2y.csv , yield_10y.csv
### data / raw : fred_raw.csv

class PathConfig:

    DATA_DIR = ROOT_DIR / "data"  # data
    RAW_DATA_DIR = DATA_DIR / "raw" # data/raw
    PROCESSED_DATA_DIR = DATA_DIR / "processed" # data/processed
    DATA_RAW_FRED = RAW_DATA_DIR / "fred" # data/raw/fred

    SRC_DIR = ROOT_DIR / "src" # src

    ### data / processed
    BREADTH_CSV = PROCESSED_DATA_DIR / "breadth.csv"
    FINAL_SIGNAL_CSV = PROCESSED_DATA_DIR / "final_signal.csv"
    MACRO_CSV = PROCESSED_DATA_DIR / "macro.csv"
    MACRO_FACTOR_CSV = PROCESSED_DATA_DIR / "macro_factor.csv"
    MARKET_RETURN_CSV = PROCESSED_DATA_DIR / "market_return.csv"

    ### data / raw
    FRED_RAW_CSV = RAW_DATA_DIR / "fred_raw.csv"

    ### data / raw / fred
    GDP_CSV = DATA_RAW_FRED / "gdp.csv"
    M2_CSV = DATA_RAW_FRED / "m2.csv"
    YIELD_2Y_CSV = DATA_RAW_FRED / "yield_2y.csv"
    YIELD_10Y_CSV = DATA_RAW_FRED / "yield_10y.csv"



    @classmethod
    def ensure_dir(cls):
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)