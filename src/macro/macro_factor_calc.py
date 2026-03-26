import logging
import os

import pandas as pd

from config.path import PathConfig
from config.profile import get_runtime_profile


def _macro_cfg() -> dict:
    profile = get_runtime_profile()
    return profile.get("macro", {})


def calc_macro_factor_logic(excess_liquidity, yield_spread, pmi=50, macro_cfg=None):
    if macro_cfg is None:
        macro_cfg = _macro_cfg()
    spread_cfg = macro_cfg.get("spread", {})
    liquidity_cfg = macro_cfg.get("liquidity", {})
    pmi_cfg = macro_cfg.get("pmi", {})
    bound_cfg = macro_cfg.get("factor_bounds", {})

    inversion_cutoff = float(spread_cfg.get("inversion_cutoff", 0.0))
    flat_cutoff = float(spread_cfg.get("flat_cutoff", 0.2))
    inversion_base = float(spread_cfg.get("inversion_base", 0.5))
    inversion_slope = float(spread_cfg.get("inversion_slope", 0.5))
    flat_base = float(spread_cfg.get("flat_base", 0.6))
    flat_slope = float(spread_cfg.get("flat_slope", 0.4))
    healthy_score = float(spread_cfg.get("healthy_score", 1.0))

    if yield_spread < inversion_cutoff:
        spread_score = inversion_base + (yield_spread * inversion_slope)
    elif yield_spread < flat_cutoff:
        spread_score = flat_base + (yield_spread / flat_cutoff) * flat_slope
    else:
        spread_score = healthy_score

    positive_base = float(liquidity_cfg.get("positive_base", 0.9))
    positive_slope = float(liquidity_cfg.get("positive_slope", 5.0))
    negative_base = float(liquidity_cfg.get("negative_base", 0.8))
    negative_slope = float(liquidity_cfg.get("negative_slope", 10.0))

    if excess_liquidity > 0:
        liq_score = positive_base + (excess_liquidity * positive_slope)
    else:
        liq_score = negative_base + (excess_liquidity * negative_slope)

    upper_threshold = float(pmi_cfg.get("upper_threshold", 52.0))
    lower_threshold = float(pmi_cfg.get("lower_threshold", 48.0))
    upper_bonus_slope = float(pmi_cfg.get("upper_bonus_slope", 0.02))
    lower_bonus_slope = float(pmi_cfg.get("lower_bonus_slope", 0.05))

    pmi_bonus = 0.0
    if pmi > upper_threshold:
        pmi_bonus = (pmi - upper_threshold) * upper_bonus_slope
    elif pmi < lower_threshold:
        pmi_bonus = (pmi - lower_threshold) * lower_bonus_slope

    base_score = min(spread_score, liq_score)
    final_factor = base_score + pmi_bonus

    factor_min = float(bound_cfg.get("min", 0.3))
    factor_max = float(bound_cfg.get("max", 1.3))
    return round(max(factor_min, min(factor_max, final_factor)), 2)


def calculate_macro_factor(current_snapshot):
    macro_cfg = _macro_cfg()
    y10 = current_snapshot.get("10Y_Yield", current_snapshot.get("DGS10", 4.0))
    y2 = current_snapshot.get("2Y_Yield", current_snapshot.get("DGS2", 3.8))
    spread = y10 - y2
    liq = current_snapshot.get("excess_liquidity", 0.01)
    pmi = current_snapshot.get("PMI", 50)

    factor = calc_macro_factor_logic(liq, spread, pmi, macro_cfg=macro_cfg)
    risks = []
    if spread < 0:
        risks.append("Yield inversion")
    if pmi < 48:
        risks.append("Weak PMI")
    if factor < 0.7:
        risks.append("Low macro factor")

    return factor, risks


def calc_macro_factor_pipeline(input_path=None, output_path=None):
    if not os.path.exists(input_path):
        logging.warning(f"[Macro] Missing source file: {input_path}")
        return False

    logging.info("[Macro] Loading data for historical calculation...")
    df = pd.read_parquet(input_path)

    macro_cfg = _macro_cfg()
    history_defaults = macro_cfg.get("history_defaults", {})
    yield_spread_default = float(history_defaults.get("yield_spread", 0.5))
    excess_liquidity_default = float(history_defaults.get("excess_liquidity", 0.01))
    pmi_default = float(history_defaults.get("pmi", 50.0))

    if "yield_spread" not in df.columns:
        if "DGS10" in df.columns and "DGS2" in df.columns:
            df["yield_spread"] = df["DGS10"] - df["DGS2"]
        elif "yield_10y" in df.columns and "yield_2y" in df.columns:
            df["yield_spread"] = df["yield_10y"] - df["yield_2y"]
        else:
            df["yield_spread"] = yield_spread_default

    if "excess_liquidity" not in df.columns:
        df["excess_liquidity"] = excess_liquidity_default

    if "PMI" not in df.columns:
        df["PMI"] = pmi_default

    df = df.ffill().fillna(0)

    df["macro_factor"] = df.apply(
        lambda row: calc_macro_factor_logic(
            row["excess_liquidity"],
            row["yield_spread"],
            row["PMI"],
            macro_cfg=macro_cfg,
        ),
        axis=1,
    )

    output_df = df[["date", "macro_factor"]]
    output_df.to_parquet(output_path, index=False)
    logging.info(f"[Macro] Macro factor saved to: {output_path}")
    logging.info(f"[Macro] Preview (last 5 rows):\n{output_df.tail().to_string(index=False)}")
    return True


if __name__ == "__main__":
    calc_macro_factor_pipeline(
        input_path=PathConfig.MACRO_PARQUET,
        output_path=PathConfig.MACRO_FACTOR_PARQUET,
    )
