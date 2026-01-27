# src/macro/rates.py
def rate_trend(current_10y, ma_3m_10y):
    if current_10y > ma_3m_10y:
        return "up"
    elif current_10y < ma_3m_10y:
        return "down"
    else:
        return "flat"
