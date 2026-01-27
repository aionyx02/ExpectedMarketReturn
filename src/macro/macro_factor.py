# src/macro/factor.py
def calc_macro_factor(excess_liquidity, inverted):
    if inverted or excess_liquidity < 0:
        return 0.3
    elif excess_liquidity > 2:
        return 1.0
    else:
        return 0.7
