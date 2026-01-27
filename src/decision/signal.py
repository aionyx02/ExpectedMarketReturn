# src/decision/signal.py
def final_signal(expected_return, macro_factor, breadth):
    final_return = expected_return * macro_factor

    if final_return > 0.05 and macro_factor >= 1.0 and breadth == "healthy":
        return "BULL", final_return
    elif final_return > 0:
        return "NEUTRAL", final_return
    else:
        return "BEAR", final_return
