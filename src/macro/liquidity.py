# src/macro/liquidity.py
def calc_excess_liquidity(m2_yoy, nominal_gdp_yoy):
    return m2_yoy - nominal_gdp_yoy
