import streamlit as st
import pandas as pd
from helper import *

st.set_page_config(page_title="Option Pricer", layout="wide")
st.title("Option Pricer - B&S")

# =========================
# Sidebar: Mode
# =========================
st.sidebar.header("Mode")
mode = st.sidebar.selectbox("Choose mode", ["Single option", "Classic Strategy", "Exotic Strategy"], index=0)

# Single option
if mode == "Single option":
    st.sidebar.header("Product to Price")
    product = st.sidebar.selectbox("Choose your product", ["Call", "Put"], index=0)

    position = st.sidebar.selectbox("Position", ["Long", "Short"], index=0)
    sign = 1.0 if position == "Long" else -1.0

    K = st.sidebar.slider("Strike (K)", 1.0, 300.0, 100.0, 0.5)

# Classic Strategy
elif mode == "Classic Strategy":
    st.sidebar.header("Classic Strategy")
    strategy = st.sidebar.selectbox(
        "Choose strategy",
        [
            "Bull Call Spread", "Bear Call Spread",
            "Bull Put Spread", "Bear Put Spread",
            "Butterfly", "Condor",
            "Straddle", "Strangle",
            "Strip", "Strap",
        ],
        index=0
    )

    position = st.sidebar.selectbox("Position", ["Long", "Short"], index=0)
    global_sign = 1.0 if position == "Long" else -1.0

    # Strikes
    K_dict = {}

    if strategy in ["Bull Call Spread", "Bear Call Spread", "Bull Put Spread", "Bear Put Spread", "Strangle"]:
        K_dict["K1"] = K_input("K1", 90.0)
        K_dict["K2"] = K_input("K2", 110.0)
        if K_dict["K2"] <= K_dict["K1"]:
            st.sidebar.error("Need K2 > K1")
            st.stop()

    elif strategy == "Butterfly":
        K_dict["K1"] = K_input("K1", 90.0)
        K_dict["K2"] = K_input("K2 (middle)", 100.0)
        K_dict["K3"] = K_input("K3", 110.0)
        if not (K_dict["K1"] < K_dict["K2"] < K_dict["K3"]):
            st.sidebar.error("Need K1 < K2 < K3")
            st.stop()

    elif strategy == "Condor":
        K_dict["K1"] = K_input("K1", 90.0)
        K_dict["K2"] = K_input("K2", 95.0)
        K_dict["K3"] = K_input("K3", 105.0)
        K_dict["K4"] = K_input("K4", 110.0)
        if not (K_dict["K1"] < K_dict["K2"] < K_dict["K3"] < K_dict["K4"]):
            st.sidebar.error("Need K1 < K2 < K3 < K4")
            st.stop()

    else:
        K_dict["K"] = K_input("K", 100.0)

# Exotic Strategy
else:
    st.sidebar.header("Exotic Strategy")
    strategy = st.sidebar.selectbox(
        "Choose strategy",
        [
            "Digital Call", "Digital Put",
            "Put Down In", "Put Down Out",
            "Call Up In", "Call Up Out",
        ],
        index=0
    )

    position = st.sidebar.selectbox("Position", ["Long", "Short"], index=0)
    global_sign = 1.0 if position == "Long" else -1.0

    K_dict = {}
    K_dict["K"] = K_input("K", 100.0)

    # Barrier level
    H_val = None
    barrier_strategies = ["Put Down In", "Put Down Out", "Call Up In", "Call Up Out"]
    if strategy in barrier_strategies:
        default_H = 80.0 if "Down" in strategy else 120.0
        H_val = K_input("Barrier (H)", default_H)

# =========================
# Sidebar: Common parameters (used in both modes)
# =========================
st.sidebar.header("Parameters")

S = st.sidebar.slider("Spot (S)", 1.0, 300.0, 100.0, 0.5)

T_days = st.sidebar.slider("Maturity (days)", 1, 1095, 365, 1)
T = T_days / 365.0
if mode in ("Classic Strategy", "Exotic Strategy"):
    T_dict = {"T": T}

sigma_pct = st.sidebar.slider("Volatility in %", 1.0, 100.0, 15.0, 0.5)
sigma = sigma_pct / 100.0

r_pct = st.sidebar.slider("Risk free rate in %", -5.0, 20.0, 2.0, 0.1)
r = r_pct / 100.0

q_pct = st.sidebar.slider("Dividend yield in %", 0.0, 20.0, 0.0, 0.1)
q = q_pct / 100.0

# =========================
# MODE 1: Single option
# =========================
if mode == "Single option":

    # ----- Single price -----
    price = sign * bs_price(product, S, K, r, q, sigma, T)
    st.subheader(f"{position} {product} Price: {price:,.2f}$")

    # ----- Profiles vs Spot -----
    Ss = build_spot_grid(S, s_min=1.0, s_max=300.0, width=0.6, n_pts=200)

    df_greeks = build_greeks_profile_vs_spot(
        product=product, Ss=Ss, K=K, r=r, q=q, sigma=sigma, T=T, sign=sign
    )

    payoff = payoff_at_maturity(product=product, Ss=Ss, K=K, sign=sign)
    df_premium_payoff = pd.DataFrame(
        {"Premium": df_greeks["Premium"].values, "Payoff": payoff},
        index=Ss
    )
    df_premium_payoff.index.name = "Spot"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Premium and Payoff Profile")
        st.line_chart(df_premium_payoff, use_container_width=True)

        st.subheader("Delta Profile")
        st.line_chart(df_greeks[["Delta"]], use_container_width=True)

        st.subheader("Theta Profile")
        st.line_chart(df_greeks[["Theta"]], use_container_width=True)

    with col2:
        st.subheader("Vega Profile")
        st.line_chart(df_greeks[["Vega"]], use_container_width=True)

        st.subheader("Gamma Profile")
        st.line_chart(df_greeks[["Gamma"]], use_container_width=True)

        st.subheader("Rho Profile")
        st.line_chart(df_greeks[["Rho"]], use_container_width=True)

# =========================
# MODE 2 & 3: Strategy (Classic & Exotic)
# =========================
else:

    # Build legs
    H_param = H_val if mode == "Exotic Strategy" else None
    legs = build_strategy_legs(strategy, K=K_dict, T=T_dict, H=H_param)

    # Price + profile
    price = global_sign * portfolio_price(legs, S, r, q, sigma)
    st.subheader(f"{position} {strategy} Price: {price:,.2f}$")

    Ss = build_spot_grid(S, s_min=1.0, s_max=300.0, width=0.6, n_pts=200)
    df = build_portfolio_profile_vs_spot(legs=legs, Ss=Ss, r=r, q=q, sigma=sigma)

    # --- Payoff at maturity (Strategy) ---
    payoff = portfolio_payoff(legs=legs, Ss=Ss)

    df_premium_payoff = pd.DataFrame(
        {"Premium": df["Premium"].values, "Payoff": payoff},
        index=Ss
    )
    df_premium_payoff.index.name = "Spot"

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Premium and Payoff Profile (Strategy)")
        st.line_chart(df_premium_payoff, use_container_width=True)

        st.subheader("Delta Profile (Strategy)")
        st.line_chart(df[["Delta"]], use_container_width=True)

        st.subheader("Theta Profile (Strategy)")
        st.line_chart(df[["Theta"]], use_container_width=True)

    with col2:
        st.subheader("Vega Profile (Strategy)")
        st.line_chart(df[["Vega"]], use_container_width=True)

        st.subheader("Gamma Profile (Strategy)")
        st.line_chart(df[["Gamma"]], use_container_width=True)

        st.subheader("Rho Profile (Strategy)")
        st.line_chart(df[["Rho"]], use_container_width=True)