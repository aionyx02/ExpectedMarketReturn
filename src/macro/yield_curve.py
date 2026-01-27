# src/macro/yield_curve.py.py
def is_yield_curve_inverted(yield_10y, yield_2y):
    return (yield_10y - yield_2y) < 0
