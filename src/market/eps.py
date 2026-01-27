# src/market/eps.py
def forward_eps_growth(eps_now, eps_forward):
    return (eps_forward - eps_now) / eps_now
