import io
import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.exceptions import RequestException, Timeout

from config.path import PathConfig
from config.profile import get_runtime_profile

DEFAULT_SERIES_MAP = {
    "DGS10": "10Y_Yield",
    "DGS2": "2Y_Yield",
    "ICSA": "Jobless_Claims",
    "T10Y2Y": "Yield_Spread",
}

CONNECT_TIMEOUT_SEC = int(os.getenv("FRED_CONNECT_TIMEOUT", "3"))
READ_TIMEOUT_SEC = int(os.getenv("FRED_READ_TIMEOUT", "6"))
MAX_ATTEMPTS = int(os.getenv("FRED_MAX_ATTEMPTS", "1"))
RETRY_BACKOFF_SEC = float(os.getenv("FRED_RETRY_BACKOFF_SEC", "1.0"))
FAIL_FAST = os.getenv("FRED_FAIL_FAST", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _series_map() -> dict[str, str]:
    profile = get_runtime_profile()
    configured = profile.get("data", {}).get("fred", {}).get("series_map", {})
    if isinstance(configured, dict) and configured:
        return {str(k): str(v) for k, v in configured.items()}
    return DEFAULT_SERIES_MAP


def _fetch_policy() -> str:
    profile = get_runtime_profile()
    raw = str(
        profile.get("data", {})
        .get("fred", {})
        .get("fetch_policy", "remote_with_cache_fallback")
    ).strip()
    allowed = {"remote_with_cache_fallback", "cache_only", "remote_only"}
    if raw not in allowed:
        return "remote_with_cache_fallback"
    return raw


def _has_usable_cache(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        pd.read_parquet(path)
    except Exception:
        return False
    return True


def _cache_latest_date(path: Path) -> str | None:
    if not _has_usable_cache(path):
        return None
    try:
        df = pd.read_parquet(path, columns=["DATE"])
    except Exception:
        return None
    if "DATE" not in df.columns:
        return None
    latest = pd.to_datetime(df["DATE"], errors="coerce").max()
    if pd.isna(latest):
        return None
    return latest.strftime("%Y-%m-%d")


def _fetch_series(
    session: requests.Session, fred_code: str, col_name: str
) -> tuple[pd.DataFrame | None, str | None]:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_code}"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = session.get(url, timeout=(CONNECT_TIMEOUT_SEC, READ_TIMEOUT_SEC))
            response.raise_for_status()

            df = pd.read_csv(io.StringIO(response.text))

            if "observation_date" in df.columns:
                df = df.rename(columns={"observation_date": "DATE"})

            if "DATE" not in df.columns or fred_code not in df.columns:
                logging.error(
                    f"[FRED] Unexpected CSV format for {fred_code}. Columns: {list(df.columns)}"
                )
                return None, "format"

            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
            df = df.dropna(subset=["DATE"])
            df = df.set_index("DATE")
            df = df.rename(columns={fred_code: col_name})
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
            return df[[col_name]], None

        except Timeout as exc:
            logging.info(
                f"[FRED] Timeout for {fred_code} (attempt {attempt}/{MAX_ATTEMPTS}): {exc}"
            )
            err_kind = "timeout"
        except RequestException as exc:
            logging.info(
                f"[FRED] Request failed for {fred_code} (attempt {attempt}/{MAX_ATTEMPTS}): {exc}"
            )
            err_kind = "request"
        except Exception as exc:
            logging.error(f"[FRED] Parse failed for {fred_code}: {exc}")
            return None, "parse"

        if attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_BACKOFF_SEC * attempt)

    logging.info(
        f"[FRED] Failed to fetch {fred_code} after {MAX_ATTEMPTS} attempt(s)."
    )
    return None, err_kind if "err_kind" in locals() else "unknown"


def _status(
    status: str,
    output_path: Path,
    latest_date: str | None,
    reason: str,
    attempted_remote: bool,
) -> dict[str, Any]:
    return {
        "status": status,
        "path": str(output_path),
        "latest_date": latest_date,
        "reason": reason,
        "attempted_remote": attempted_remote,
    }


def update_all_fred(output_dir=PathConfig.RAW_DATA_DIR) -> dict[str, Any]:
    logging.info("[FRED] Downloading latest macro data...")

    fred_dir = Path(output_dir) / "fred"
    fred_dir.mkdir(parents=True, exist_ok=True)
    output_path = fred_dir / "fred_raw.parquet"
    cache_ok = _has_usable_cache(output_path)
    cache_latest = _cache_latest_date(output_path)
    fetch_policy = _fetch_policy()

    if fetch_policy == "cache_only":
        if cache_ok:
            return _status(
                "cache_used",
                output_path,
                cache_latest,
                "cache_only_policy",
                attempted_remote=False,
            )
        logging.error(
            "[FRED] cache_only policy enabled, but no usable local cache was found."
        )
        return _status(
            "failed",
            output_path,
            None,
            "cache_only_without_cache",
            attempted_remote=False,
        )

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
    )

    data_frames = []
    series_map = _series_map()

    try:
        for fred_code, col_name in series_map.items():
            logging.info(f"[FRED] Fetching {fred_code} -> {col_name}")
            try:
                series_df, err_kind = _fetch_series(session, fred_code, col_name)
            except KeyboardInterrupt:
                logging.info(
                    "[FRED] Download interrupted. Falling back to local cache when possible."
                )
                break

            if series_df is not None and not series_df.empty:
                data_frames.append(series_df)
                continue

            if FAIL_FAST and not data_frames and err_kind in {"timeout", "request"}:
                logging.info(
                    "[FRED] Fail-fast enabled: first network failure detected; "
                    "using cache fallback."
                )
                break

        if not data_frames:
            if cache_ok and fetch_policy != "remote_only":
                return _status(
                    "cache_used",
                    output_path,
                    cache_latest,
                    "remote_unavailable_use_cache",
                    attempted_remote=True,
                )
            logging.error(
                "[FRED] All downloads failed and no usable fallback cache is available."
            )
            return _status(
                "failed",
                output_path,
                None,
                "remote_failed_no_cache",
                attempted_remote=True,
            )

        df_merged = pd.concat(data_frames, axis=1, join="outer").sort_index()
        df_merged = df_merged.reset_index()

        if df_merged.empty:
            if cache_ok and fetch_policy != "remote_only":
                return _status(
                    "cache_used",
                    output_path,
                    cache_latest,
                    "remote_empty_use_cache",
                    attempted_remote=True,
                )
            logging.error("[FRED] Download result is empty and no local cache exists.")
            return _status(
                "failed",
                output_path,
                None,
                "remote_empty_no_cache",
                attempted_remote=True,
            )

        temp_path = output_path.with_suffix(".parquet.tmp")
        df_merged.to_parquet(temp_path, index=False)
        os.replace(temp_path, output_path)

        latest = pd.to_datetime(df_merged["DATE"], errors="coerce").max()
        latest_str = None if pd.isna(latest) else latest.strftime("%Y-%m-%d")
        return _status(
            "updated",
            output_path,
            latest_str,
            "remote_success",
            attempted_remote=True,
        )

    finally:
        session.close()


if __name__ == "__main__":
    update_all_fred()
