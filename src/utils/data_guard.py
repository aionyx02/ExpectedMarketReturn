import logging
from contextlib import suppress
from pathlib import Path

import pandas as pd

from config.path import PathConfig


PARQUET_FALLBACKS = (
    (PathConfig.GDP_PARQUET, (PathConfig.DATA_RAW_FRED / "gdp.csv",)),
    (PathConfig.M2_PARQUET, (PathConfig.DATA_RAW_FRED / "m2.csv",)),
    (PathConfig.YIELD_10Y_PARQUET, (PathConfig.DATA_RAW_FRED / "yield_10y.csv",)),
    (PathConfig.YIELD_2Y_PARQUET, (PathConfig.DATA_RAW_FRED / "yield_2y.csv",)),
    (
        PathConfig.FRED_RAW_PARQUET,
        (
            PathConfig.DATA_RAW_FRED / "fred_raw.csv",
            PathConfig.RAW_DATA_DIR / "fred_raw.csv",
        ),
    ),
    (
        PathConfig.MACRO_PARQUET,
        (PathConfig.PROCESSED_DATA_DIR / "macro.csv",),
    ),
    (
        PathConfig.MACRO_FACTOR_PARQUET,
        (PathConfig.PROCESSED_DATA_DIR / "macro_factor.csv",),
    ),
    (
        PathConfig.MARKET_RETURN_PARQUET,
        (PathConfig.PROCESSED_DATA_DIR / "market_return.csv",),
    ),
    (
        PathConfig.BREADTH_PARQUET,
        (PathConfig.PROCESSED_DATA_DIR / "breadth.csv",),
    ),
    (
        PathConfig.FINAL_SIGNAL_PARQUET,
        (PathConfig.PROCESSED_DATA_DIR / "final_signal.csv",),
    ),
)


def _is_valid_parquet(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        pd.read_parquet(path)
    except Exception:
        return False

    return True


def _repair_from_csv(parquet_path: Path, csv_candidates: tuple[Path, ...]) -> bool:
    for csv_path in csv_candidates:
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            logging.warning(
                f"[DataGuard] Failed to read fallback CSV {csv_path}: {exc}"
            )
            continue

        temp_path = parquet_path.with_suffix(f"{parquet_path.suffix}.tmp")
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            df.to_parquet(temp_path, index=False)
            temp_path.replace(parquet_path)
            logging.warning(
                f"[DataGuard] Repaired {parquet_path.name} from {csv_path.name}"
            )
            return True
        except Exception as exc:
            logging.warning(f"[DataGuard] Failed to repair {parquet_path}: {exc}")
            with suppress(FileNotFoundError):
                temp_path.unlink()

    return False


def bootstrap_local_storage():
    PathConfig.ensure_dir()

    healthy = []
    repaired = []
    failed = []

    for parquet_path, csv_candidates in PARQUET_FALLBACKS:
        if _is_valid_parquet(parquet_path):
            healthy.append(str(parquet_path))
            continue

        if _repair_from_csv(parquet_path, csv_candidates):
            repaired.append(str(parquet_path))
            continue

        failed.append(str(parquet_path))

    return {"healthy": healthy, "repaired": repaired, "failed": failed}
