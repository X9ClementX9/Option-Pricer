import math
import pandas as pd
import streamlit as st

# ------- Normal Distribution ---------
def norm_cdf(x: float) -> float:
    # Φ(x) via erf (standard library)
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x: float) -> float:
    # φ(x)
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


# -------- Black-Scholes ---------
def bs_d1_d2(S, K, r, q, sigma, T):
    # d1, d2 standard
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None, None
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2

def bs_price(option_type, S, K, r, q, sigma, T):
    d1, d2 = bs_d1_d2(S, K, r, q, sigma, T)
    if d1 is None:
        return float("nan")
    Nd1 = norm_cdf(d1)
    Nd2 = norm_cdf(d2)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    if option_type == "Call":
        return S * disc_q * Nd1 - K * disc_r * Nd2
    else:  # Put
        return K * disc_r * norm_cdf(-d2) - S * disc_q * norm_cdf(-d1)

def bs_greeks(option_type, S, K, r, q, sigma, T):
    d1, d2 = bs_d1_d2(S, K, r, q, sigma, T)
    if d1 is None:
        return {k: float("nan") for k in ["Delta", "Gamma", "Vega", "Theta", "Rho"]}

    phi_d1 = norm_pdf(d1)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    sqrtT = math.sqrt(T)

    # Delta
    if option_type == "Call":
        delta = disc_q * norm_cdf(d1)
    else:
        delta = disc_q * (norm_cdf(d1) - 1.0)

    # Gamma (même pour call/put)
    gamma = (disc_q * phi_d1) / (S * sigma * sqrtT)

    # Vega (par +1.00 de vol en décimal, ex: 0.20 -> 20%)
    vega = S * disc_q * phi_d1 * sqrtT

    # Theta (par an). Si tu veux "par jour", divise par 365.
    if option_type == "Call":
        theta = (
            - (S * disc_q * phi_d1 * sigma) / (2.0 * sqrtT)
            - r * K * disc_r * norm_cdf(d2)
            + q * S * disc_q * norm_cdf(d1)
        )
    else:
        theta = (
            - (S * disc_q * phi_d1 * sigma) / (2.0 * sqrtT)
            + r * K * disc_r * norm_cdf(-d2)
            - q * S * disc_q * norm_cdf(-d1)
        )

    # Rho (par +1.00 de taux en décimal, ex: 0.05 -> 5%)
    if option_type == "Call":
        rho = K * T * disc_r * norm_cdf(d2)
    else:
        rho = -K * T * disc_r * norm_cdf(-d2)

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Theta": theta,
        "Rho": rho
    }

# -------- Digital (cash-or-nothing) Options ---------
def bs_price_digital(option_type, S, K, r, q, sigma, T):
    d1, d2 = bs_d1_d2(S, K, r, q, sigma, T)
    if d1 is None:
        return float("nan")
    disc_r = math.exp(-r * T)
    if option_type == "Call":
        return disc_r * norm_cdf(d2)
    else:  # Put
        return disc_r * norm_cdf(-d2)

def bs_greeks_digital(option_type, S, K, r, q, sigma, T):
    """Numerical greeks for digital options via finite differences."""
    h_s = max(S * 1e-4, 1e-6)
    h_v = 1e-4
    h_r = 1e-5
    h_t = 1.0 / 365.0

    p0 = bs_price_digital(option_type, S, K, r, q, sigma, T)
    p_up = bs_price_digital(option_type, S + h_s, K, r, q, sigma, T)
    p_dn = bs_price_digital(option_type, S - h_s, K, r, q, sigma, T)

    delta = (p_up - p_dn) / (2 * h_s)
    gamma = (p_up - 2 * p0 + p_dn) / (h_s ** 2)

    vega = (bs_price_digital(option_type, S, K, r, q, sigma + h_v, T)
            - bs_price_digital(option_type, S, K, r, q, sigma - h_v, T)) / (2 * h_v)

    if T > h_t:
        theta = (bs_price_digital(option_type, S, K, r, q, sigma, T - h_t) - p0) / h_t
    else:
        theta = 0.0

    rho = (bs_price_digital(option_type, S, K, r + h_r, q, sigma, T)
           - bs_price_digital(option_type, S, K, r - h_r, q, sigma, T)) / (2 * h_r)

    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}

def payoff_digital(product: str, S_T: float, K: float) -> float:
    if product == "Call":
        return 1.0 if S_T > K else 0.0
    else:  # Put
        return 1.0 if S_T < K else 0.0

# -------- Barrier Options (Reiner-Rubinstein 1991) ---------
def barrier_price(option_type, barrier_type, S, K, H, r, q, sigma, T):
    """
    European barrier option price (continuous monitoring).
    option_type: 'Call' or 'Put'
    barrier_type: 'DI', 'DO', 'UI', 'UO'
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0 or H <= 0:
        return float("nan")

    # Barrier already breached
    if barrier_type.startswith("D") and S <= H:
        return bs_price(option_type, S, K, r, q, sigma, T) if "I" in barrier_type else 0.0
    if barrier_type.startswith("U") and S >= H:
        return bs_price(option_type, S, K, r, q, sigma, T) if "I" in barrier_type else 0.0

    b = r - q
    vol_sqrt_T = sigma * math.sqrt(T)
    mu = (b - 0.5 * sigma ** 2) / sigma ** 2
    lam = 1.0 + mu  # (b + sigma^2/2) / sigma^2

    phi = 1.0 if option_type == "Call" else -1.0
    eta = 1.0 if barrier_type.startswith("D") else -1.0

    HS = H / S
    HS_2lam = HS ** (2 * lam)
    HS_2mu = HS ** (2 * mu)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    x1 = math.log(S / K) / vol_sqrt_T + lam * vol_sqrt_T
    x2 = math.log(S / H) / vol_sqrt_T + lam * vol_sqrt_T
    y1 = math.log(H ** 2 / (S * K)) / vol_sqrt_T + lam * vol_sqrt_T
    y2 = math.log(H / S) / vol_sqrt_T + lam * vol_sqrt_T

    A = phi * (S * disc_q * norm_cdf(phi * x1) - K * disc_r * norm_cdf(phi * (x1 - vol_sqrt_T)))
    B = phi * (S * disc_q * norm_cdf(phi * x2) - K * disc_r * norm_cdf(phi * (x2 - vol_sqrt_T)))
    C = phi * (S * disc_q * HS_2lam * norm_cdf(eta * y1) - K * disc_r * HS_2mu * norm_cdf(eta * (y1 - vol_sqrt_T)))
    D = phi * (S * disc_q * HS_2lam * norm_cdf(eta * y2) - K * disc_r * HS_2mu * norm_cdf(eta * (y2 - vol_sqrt_T)))

    is_call = option_type == "Call"

    if barrier_type == "DI":
        return C if (is_call and K >= H) else (A - B + D) if is_call else (B - C + D) if K >= H else A
    elif barrier_type == "DO":
        return (A - C) if (is_call and K >= H) else (B - D) if is_call else (A - B + C - D) if K >= H else 0.0
    elif barrier_type == "UI":
        return A if (is_call and K >= H) else (B - C + D) if is_call else C if K >= H else (A - B + D)
    elif barrier_type == "UO":
        return 0.0 if (is_call and K >= H) else (A - B + C - D) if is_call else (A - C) if K >= H else (B - D)
    return float("nan")

def barrier_greeks(option_type, barrier_type, S, K, H, r, q, sigma, T):
    """Numerical greeks for barrier options via finite differences."""
    h_s = max(S * 1e-4, 1e-6)
    h_v = 1e-4
    h_r = 1e-5
    h_t = 1.0 / 365.0

    p0 = barrier_price(option_type, barrier_type, S, K, H, r, q, sigma, T)
    p_up = barrier_price(option_type, barrier_type, S + h_s, K, H, r, q, sigma, T)
    p_dn = barrier_price(option_type, barrier_type, S - h_s, K, H, r, q, sigma, T)

    delta = (p_up - p_dn) / (2 * h_s)
    gamma = (p_up - 2 * p0 + p_dn) / (h_s ** 2)

    vega = (barrier_price(option_type, barrier_type, S, K, H, r, q, sigma + h_v, T)
            - barrier_price(option_type, barrier_type, S, K, H, r, q, sigma - h_v, T)) / (2 * h_v)

    if T > h_t:
        theta = (barrier_price(option_type, barrier_type, S, K, H, r, q, sigma, T - h_t) - p0) / h_t
    else:
        theta = 0.0

    rho = (barrier_price(option_type, barrier_type, S, K, H, r + h_r, q, sigma, T)
           - barrier_price(option_type, barrier_type, S, K, H, r - h_r, q, sigma, T)) / (2 * h_r)

    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}

def payoff_barrier(option_type, barrier_type, S_T, K, H):
    """Approximate payoff at maturity (path-dependent, shown as best estimate)."""
    vanilla = payoff_vanilla(option_type, S_T, K)
    if barrier_type == "DI":
        return vanilla if S_T <= H else 0.0
    elif barrier_type == "DO":
        return 0.0 if S_T <= H else vanilla
    elif barrier_type == "UI":
        return vanilla if S_T >= H else 0.0
    elif barrier_type == "UO":
        return 0.0 if S_T >= H else vanilla
    return 0.0

def payoff_vanilla(product: str, S_T: float, K: float) -> float:
    if product == "Call":
        return max(S_T - K, 0.0)
    else:  # Put
        return max(K - S_T, 0.0)

def portfolio_payoff(legs: list[dict], Ss: list[float]) -> list[float]:
    out = []
    for S_T in Ss:
        total = 0.0
        for leg in legs:
            sign = 1.0 if leg["side"] == "Long" else -1.0
            qty = float(leg.get("qty", 1.0))
            K = float(leg["K"])
            product = leg["product"]
            style = leg.get("style", "vanilla")
            if style == "digital":
                total += sign * qty * payoff_digital(product, S_T, K)
            elif style == "barrier":
                total += sign * qty * payoff_barrier(product, leg["barrier_type"], S_T, K, leg["H"])
            else:
                total += sign * qty * payoff_vanilla(product, S_T, K)
        out.append(total)
    return out

def portfolio_price(legs: list[dict], S: float, r: float, q: float, sigma: float) -> float:
    total = 0.0
    for leg in legs:
        sign = 1.0 if leg["side"] == "Long" else -1.0
        qty = float(leg.get("qty", 1.0))
        K = float(leg["K"])
        T = float(leg["T"])
        product = leg["product"]
        style = leg.get("style", "vanilla")
        if style == "digital":
            total += sign * qty * bs_price_digital(product, S, K, r, q, sigma, T)
        elif style == "barrier":
            total += sign * qty * barrier_price(product, leg["barrier_type"], S, K, leg["H"], r, q, sigma, T)
        else:
            total += sign * qty * bs_price(product, S, K, r, q, sigma, T)
    return total

def portfolio_greeks(legs: list[dict], S: float, r: float, q: float, sigma: float) -> dict:
    totals = {"Delta": 0.0, "Gamma": 0.0, "Vega": 0.0, "Theta": 0.0, "Rho": 0.0}
    for leg in legs:
        sign = 1.0 if leg["side"] == "Long" else -1.0
        qty = float(leg.get("qty", 1.0))
        K = float(leg["K"])
        T = float(leg["T"])
        product = leg["product"]
        style = leg.get("style", "vanilla")
        if style == "digital":
            g = bs_greeks_digital(product, S, K, r, q, sigma, T)
        elif style == "barrier":
            g = barrier_greeks(product, leg["barrier_type"], S, K, leg["H"], r, q, sigma, T)
        else:
            g = bs_greeks(product, S, K, r, q, sigma, T)
        for k in totals:
            totals[k] += sign * qty * g[k]
    return totals

def build_portfolio_profile_vs_spot(legs: list[dict], Ss: list[float], r: float, q: float, sigma: float) -> pd.DataFrame:
    rows = []
    for S_i in Ss:
        price_i = portfolio_price(legs, S_i, r, q, sigma)
        g = portfolio_greeks(legs, S_i, r, q, sigma)
        rows.append({
            "S": S_i,
            "Premium": price_i,
            "Delta": g["Delta"],
            "Gamma": g["Gamma"],
            "Vega": g["Vega"],
            "Theta": g["Theta"],
            "Rho": g["Rho"],
        })
    return pd.DataFrame(rows).set_index("S").sort_index()

#------- Payoff & Profiles ---------
def build_spot_grid(S, s_min=1.0, s_max=300.0, width=0.6, n_pts=200):
    """
    Builds a spot grid around current S:
    [S*(1-width), S*(1+width)] clipped to [s_min, s_max].
    """
    S_min = max(s_min, S * (1.0 - width))
    S_max = min(s_max, S * (1.0 + width))
    if n_pts < 2:
        return [S_min, S_max]
    return [S_min + i * (S_max - S_min) / (n_pts - 1) for i in range(n_pts)]

def payoff_at_maturity(product, Ss, K, sign=1.0):
    out = []
    for S_T in Ss:
        if product == "Call":
            p = max(S_T - K, 0.0)
        else:  # Put
            p = max(K - S_T, 0.0)
        out.append(sign * p)
    return out

def build_greeks_profile_vs_spot(product, Ss, K, r, q, sigma, T, sign=1.0):
    rows = []
    for S_i in Ss:
        price_i = bs_price(product, S_i, K, r, q, sigma, T)
        g = bs_greeks(product, S_i, K, r, q, sigma, T)
        rows.append({
            "S": S_i,
            "Premium": sign * price_i,
            "Delta":   sign * g["Delta"],
            "Gamma":   sign * g["Gamma"],
            "Vega":    sign * g["Vega"],
            "Theta":   sign * g["Theta"],
            "Rho":     sign * g["Rho"],
        })

    df = pd.DataFrame(rows).set_index("S").sort_index()
    return df

# ------- Options Strategies ---------
def build_strategy_legs(strategy: str, K: dict, T: dict, H: float = None) -> list[dict]:

    s = strategy

    # Spreads Call
    if s == "Bull Call Spread":
        return [
            {"product": "Call", "side": "Long",  "qty": 1, "K": K["K1"], "T": T["T"]},
            {"product": "Call", "side": "Short", "qty": 1, "K": K["K2"], "T": T["T"]},
        ]
    if s == "Bear Call Spread":
        return [
            {"product": "Call", "side": "Short", "qty": 1, "K": K["K1"], "T": T["T"]},
            {"product": "Call", "side": "Long",  "qty": 1, "K": K["K2"], "T": T["T"]},
        ]

    # Spreads Put
    if s == "Bull Put Spread":
        return [
            {"product": "Put",  "side": "Long",  "qty": 1, "K": K["K1"], "T": T["T"]},
            {"product": "Put",  "side": "Short", "qty": 1, "K": K["K2"], "T": T["T"]},
        ]
    if s == "Bear Put Spread":
        return [
            {"product": "Put",  "side": "Short", "qty": 1, "K": K["K1"], "T": T["T"]},
            {"product": "Put",  "side": "Long",  "qty": 1, "K": K["K2"], "T": T["T"]},
        ]

    # Straddle / Strangle
    if s == "Straddle":
        return [
            {"product": "Call", "side": "Long", "qty": 1, "K": K["K"],  "T": T["T"]},
            {"product": "Put",  "side": "Long", "qty": 1, "K": K["K"],  "T": T["T"]},
        ]
    if s == "Strangle":
        return [
            {"product": "Put",  "side": "Long", "qty": 1, "K": K["K1"], "T": T["T"]},
            {"product": "Call", "side": "Long", "qty": 1, "K": K["K2"], "T": T["T"]},
        ]

    # Strip / Strap (ratios)
    if s == "Strip":
        return [
            {"product": "Call", "side": "Long", "qty": 1, "K": K["K"], "T": T["T"]},
            {"product": "Put",  "side": "Long", "qty": 2, "K": K["K"], "T": T["T"]},
        ]
    if s == "Strap":
        return [
            {"product": "Call", "side": "Long", "qty": 2, "K": K["K"], "T": T["T"]},
            {"product": "Put",  "side": "Long", "qty": 1, "K": K["K"], "T": T["T"]},
        ]

    # Butterfly / Condor (Call)
    if s == "Butterfly":
        return [
            {"product": "Call", "side": "Long",  "qty": 1, "K": K["K1"], "T": T["T"]},
            {"product": "Call", "side": "Short", "qty": 2, "K": K["K2"], "T": T["T"]},
            {"product": "Call", "side": "Long",  "qty": 1, "K": K["K3"], "T": T["T"]},
        ]
    if s == "Condor":
        return [
            {"product": "Call", "side": "Long",  "qty": 1, "K": K["K1"], "T": T["T"]},
            {"product": "Call", "side": "Short", "qty": 1, "K": K["K2"], "T": T["T"]},
            {"product": "Call", "side": "Short", "qty": 1, "K": K["K3"], "T": T["T"]},
            {"product": "Call", "side": "Long",  "qty": 1, "K": K["K4"], "T": T["T"]},
        ]

    # Digital (cash-or-nothing)
    if s == "Digital Call":
        return [{"product": "Call", "side": "Long", "qty": 1, "K": K["K"], "T": T["T"], "style": "digital"}]
    if s == "Digital Put":
        return [{"product": "Put", "side": "Long", "qty": 1, "K": K["K"], "T": T["T"], "style": "digital"}]

    # Barrier options
    _barrier_map = {
        "Put Down In":  ("Put",  "DI"),
        "Put Down Out": ("Put",  "DO"),
        "Call Up In":   ("Call", "UI"),
        "Call Up Out":  ("Call", "UO"),
    }
    if s in _barrier_map:
        opt, bt = _barrier_map[s]
        return [{"product": opt, "side": "Long", "qty": 1, "K": K["K"], "T": T["T"],
                 "style": "barrier", "barrier_type": bt, "H": H}]

    raise ValueError(f"Unknown strategy: {strategy}")


# ------- Streamlit  ---------
def K_input(name, default):
    return st.sidebar.number_input(name, min_value=1.0, max_value=300.0, value=float(default), step=0.5)