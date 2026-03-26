from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

from config.path import ROOT_DIR, PathConfig

DEFAULT_PROFILE_PATH = PathConfig.PROFILE_DIR / "global_default.yaml"

_RUNTIME_PROFILE: dict[str, Any] | None = None
_RUNTIME_PROFILE_PATH: Path | None = None


DEFAULT_PROFILE: dict[str, Any] = {
    "meta": {"name": "global_default", "version": 1},
    "data": {
        "fred": {
            "series_map": {
                "DGS10": "10Y_Yield",
                "DGS2": "2Y_Yield",
                "ICSA": "Jobless_Claims",
                "T10Y2Y": "Yield_Spread",
            }
        }
    },
    "market": {
        "index_ticker": "^GSPC",
        "download": {"period": "max", "interval": "1mo"},
        "bias_ma_window": 24,
        "trend_ma_window": 10,
        "base_return": 0.08,
        "sensitivity": 0.2,
        "expected_return_fallback": 0.08,
    },
    "breadth": {
        "cap_ticker": "^GSPC",
        "equal_ticker": "RSP",
        "download": {"period": "5y", "interval": "1d"},
        "lookback_days": 20,
        "fragile_threshold": -0.01,
        "default_signal": "HEALTHY",
    },
    "macro": {
        "preprocess": {
            "month_rule": "MS",
            "fill_method": "ffill",
            "m2_release_lag_months": 1,
            "gdp_release_lag_months": 2,
            "yield_release_lag_months": 0,
        },
        "spread": {
            "inversion_cutoff": 0.0,
            "flat_cutoff": 0.2,
            "inversion_base": 0.5,
            "inversion_slope": 0.5,
            "flat_base": 0.6,
            "flat_slope": 0.4,
            "healthy_score": 1.0,
        },
        "liquidity": {
            "positive_base": 0.9,
            "positive_slope": 5.0,
            "negative_base": 0.8,
            "negative_slope": 10.0,
        },
        "pmi": {
            "upper_threshold": 52.0,
            "lower_threshold": 48.0,
            "upper_bonus_slope": 0.02,
            "lower_bonus_slope": 0.05,
        },
        "factor_bounds": {"min": 0.3, "max": 1.3},
        "history_defaults": {
            "yield_spread": 0.5,
            "excess_liquidity": 0.01,
            "pmi": 50.0,
        },
    },
    "decision": {
        "signal": {
            "macro_bear_threshold": 0.8,
            "bull_final_return_threshold": 0.05,
            "bull_macro_threshold": 1.0,
            "neutral_final_return_threshold": 0.0,
            "expected_return_fallback": 0.07,
        },
        "backtest": {
            "risk_free_rate_annual": 0.03,
            "borrowing_cost_annual": 0.05,
            "bull_leverage": 2.0,
            "neutral_leverage": 1.0,
            "bear_leverage": 0.0,
            "transaction_cost_bps": 8.0,
        },
        "nowcast": {
            "default_expected_return": 0.05,
            "bear_leverage": 0.0,
            "fragile_leverage": 0.5,
            "bull_leverage": 2.0,
            "neutral_leverage": 1.0,
            "mixed_leverage": 0.8,
            "bull_return_threshold": 0.08,
            "bull_nowcast_threshold": 1.0,
            "neutral_return_threshold": 0.04,
            "neutral_nowcast_threshold": 0.9,
        },
    },
    "future_mock": {
        "enabled": False,
        "write_back": False,
        "macro": {"target_mean": 1.0, "decay_rate": 0.1},
        "market": {
            "long_term_growth": 0.0058,
            "noise_std": 0.01,
            "target_expected_return": 0.05,
            "expected_return_decay": 0.1,
        },
    },
}


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, Mapping)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _load_yaml_text(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
    except ModuleNotFoundError:
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "PyYAML is not installed. Please keep profile files in JSON-compatible "
                "YAML syntax or install PyYAML."
            ) from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Profile content must be a mapping.")
    return loaded


def _resolve_profile_path(profile_path: str | os.PathLike[str] | None = None) -> Path:
    requested = profile_path or os.getenv("EMR_PROFILE")
    if requested:
        candidate = Path(requested)
        if not candidate.is_absolute():
            candidate = (ROOT_DIR / candidate).resolve()
        return candidate
    return DEFAULT_PROFILE_PATH.resolve()


def load_profile(profile_path: str | os.PathLike[str] | None = None) -> tuple[dict[str, Any], Path]:
    resolved_path = _resolve_profile_path(profile_path)
    profile = copy.deepcopy(DEFAULT_PROFILE)

    if not resolved_path.exists():
        logging.warning(
            f"[Profile] Profile not found at {resolved_path}. Using built-in defaults."
        )
        return profile, resolved_path

    raw_text = resolved_path.read_text(encoding="utf-8")
    loaded = _load_yaml_text(raw_text)
    merged = _deep_merge(profile, loaded)
    return merged, resolved_path


def initialize_runtime_profile(
    profile_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    profile, resolved_path = load_profile(profile_path)
    set_runtime_profile(profile, resolved_path)
    return profile


def set_runtime_profile(
    profile: Mapping[str, Any], profile_path: str | os.PathLike[str] | None = None
) -> None:
    global _RUNTIME_PROFILE, _RUNTIME_PROFILE_PATH
    _RUNTIME_PROFILE = copy.deepcopy(dict(profile))
    if profile_path is None:
        _RUNTIME_PROFILE_PATH = _resolve_profile_path(None)
    else:
        _RUNTIME_PROFILE_PATH = Path(profile_path).resolve()


def get_runtime_profile() -> dict[str, Any]:
    global _RUNTIME_PROFILE
    if _RUNTIME_PROFILE is None:
        initialize_runtime_profile()
    return copy.deepcopy(_RUNTIME_PROFILE)


def get_runtime_profile_path() -> Path:
    global _RUNTIME_PROFILE_PATH
    if _RUNTIME_PROFILE_PATH is None:
        initialize_runtime_profile()
    return _RUNTIME_PROFILE_PATH


def reset_runtime_profile() -> None:
    global _RUNTIME_PROFILE, _RUNTIME_PROFILE_PATH
    _RUNTIME_PROFILE = None
    _RUNTIME_PROFILE_PATH = None
