# app.py
from __future__ import annotations

from dataclasses import replace
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from model import (Inputs, SKU, ltv_5yr_device_consumables, mix_delta_ltv_and_roi,
                    mix_heatmap, device_tradeoff_heatmap,
                    discount_factors, retention_with_durability, devices_per_user_per_year,
                    annual_packs, demand_with_elasticity)

# Optional optimizer (size ladder)
try:
    from scipy.optimize import minimize
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

st.set_page_config(page_title="Juicero Juice Press — LTV Calculator", layout="wide")

# Global CSS: yellow italic for all scenario-related labels
st.markdown(
    """
    <style>
    .scenario-text {
        color: #e6b800;
        font-style: italic;
    }
    .scenario-label {
        color: #e6b800;
        font-style: italic;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Juicero Juice Press — LTV Calculator")
st.caption("Juicero Juice Press Analysis: device + consumable bag lifetime value modeling across single SKU, mix, heatmaps, ladder optimization, and DCF scenarios.")


def copy_mix_single_from(mah: float, price_per_mah: float, cogs: float) -> None:
    st.session_state["mix_single_mah"] = mah
    st.session_state["mix_single_price_per_mah"] = price_per_mah
    st.session_state["mix_single_cogs"] = cogs


with st.sidebar:
    st.header("Global Settings")
    years = st.slider("Horizon (years)", 3, 10, 5)
    discount_rate = st.number_input("Discount rate (annual)", value=0.15, step=0.01, format="%.2f")
    cac = st.number_input("CAC", value=30.0, step=1.0)

    st.divider()
    st.header("Usage")
    mah_per_month = st.number_input("mL per month", value=60.0, step=1.0)

    st.divider()
    st.header("Device Duration Scenario")
    baseline_sd = st.number_input("Baseline service duration (years)", value=1.0, step=0.1, format="%.2f")
    st.markdown('<p class="scenario-text" style="font-size:0.875rem;margin-bottom:0.25rem;">Scenario service duration (years)</p>', unsafe_allow_html=True)
    scenario_sd = st.number_input("Scenario service duration (years)", value=1.2, step=0.1, format="%.2f", label_visibility="collapsed")
    lift_per_year = st.number_input("Retention lift per +1 year durability", value=0.05, step=0.01, format="%.2f")
    st.markdown('<p class="scenario-text" style="font-size:0.875rem;margin-bottom:0.25rem;">Δ Juice Device COGS (scenario)</p>', unsafe_allow_html=True)
    delta_device_cogs = st.number_input("Δ Juice Device COGS (scenario)", value=0.0, step=0.5, format="%.2f", label_visibility="collapsed")

    st.divider()
    st.header("Juice Device Economics")
    device_asp = st.number_input("Juice Device ASP", value=12.0, step=0.5, format="%.2f")
    device_cogs = st.number_input("Juice Device COGS (baseline)", value=8.0, step=0.5, format="%.2f")
    device_other = st.number_input("Juice Device other costs", value=0.0, step=0.5, format="%.2f")

    st.divider()
    st.header("Complexity")
    annual_complexity_cost = st.number_input("Annual complexity cost (multi-SKU)", value=250000.0, step=25000.0)

    st.divider()
    st.header("Retention (probability active)")
    r1 = st.slider("Year 1 retention", 0.0, 1.0, 0.70, 0.01)
    r2 = st.slider("Year 2 retention (12-month)", 0.0, 1.0, 0.50, 0.01)
    r3 = st.slider("Year 3 retention", 0.0, 1.0, 0.40, 0.01)
    r4 = st.slider("Year 4 retention", 0.0, 1.0, 0.32, 0.01)
    r5 = st.slider("Year 5 retention", 0.0, 1.0, 0.25, 0.01)

    st.divider()
    st.header("Demand (ladder optimizer)")
    elasticity = st.number_input("Elasticity (mL vs blended price/mL)", value=-2.00, step=0.05, format="%.2f")


inp = Inputs(
    years=years,
    discount_rate=discount_rate,
    mah_per_month=mah_per_month,
    retention=(r1, r2, r3, r4, r5),
    baseline_service_years=baseline_sd,
    retention_lift_per_year=lift_per_year,
    device_asp=device_asp,
    device_cogs_baseline=device_cogs,
    device_other_costs=device_other,
    delta_device_cogs=delta_device_cogs,
    scenario_service_years=scenario_sd,
    cac=cac,
    annual_complexity_cost=annual_complexity_cost,
    elasticity=elasticity,
)

if "saved_scenarios" not in st.session_state:
    st.session_state["saved_scenarios"] = []

tabs = st.tabs(["Single SKU", "Consumable Mix (2–3)", "Mix Heatmaps", "Size Ladder Optimizer", "DCF LTV Analysis", "Investment Strategy Graphs"])

# ---------------- Single SKU ----------------
with tabs[0]:
    st.subheader("Single SKU Inputs")

    c1, c2, c3 = st.columns(3)
    with c1:
        mah_per_pack_single = st.number_input("Single SKU mL per juice bag", value=2.0, step=0.1, format="%.2f")
    with c2:
        retail_price_per_mah_single = st.number_input("Single SKU retail price per mL", value=3.00, step=0.05, format="%.2f")
    with c3:
        cogs_per_pack_single = st.number_input("Single SKU COGS per juice bag", value=1.55, step=0.05, format="%.2f")
    price_per_pack_single = mah_per_pack_single * retail_price_per_mah_single
    st.caption(f"Computed Single SKU price per juice bag: ${price_per_pack_single:,.2f}")

    single_sku = SKU("Single", mah_per_pack_single, price_per_pack_single, cogs_per_pack_single)

    # Price elasticity toggle
    sku_apply_elasticity = st.checkbox("Include price elasticity in LTV", value=False, key="sku_elasticity_toggle")

    sku_inp = inp
    if sku_apply_elasticity:
        sku_ref_price_per_ml = st.number_input(
            "Reference price per mL (baseline demand)",
            value=float(retail_price_per_mah_single),
            step=0.05,
            format="%.2f",
            key="sku_ref_price_per_ml",
        )
        observed_price_per_ml = price_per_pack_single / mah_per_pack_single if mah_per_pack_single else 0.0
        adj_mah_per_month = demand_with_elasticity(
            inp.mah_per_month, observed_price_per_ml, sku_ref_price_per_ml, inp.elasticity,
        )
        sku_inp = Inputs(**{**sku_inp.__dict__, "mah_per_month": float(adj_mah_per_month)})
        st.caption(
            f"Elasticity {inp.elasticity:.2f} · price/mL ${observed_price_per_ml:.2f} vs "
            f"ref ${sku_ref_price_per_ml:.2f} → adjusted mL/month: {adj_mah_per_month:.1f} "
            f"(base: {inp.mah_per_month:.1f})"
        )
    # Retention lift toggle
    sku_apply_retention_lift = st.checkbox("Include retention lift from durability or increased desirability", value=True, key="sku_retention_lift_toggle")
    if not sku_apply_retention_lift:
        sku_inp = Inputs(**{**sku_inp.__dict__, "retention_lift_per_year": 0.0})

    res_base = ltv_5yr_device_consumables(
        sku_inp,
        mah_per_pack=single_sku.mah_per_pack,
        price_per_pack=single_sku.price_per_pack,
        cogs_per_pack=single_sku.cogs_per_pack,
        service_years=baseline_sd,
        delta_device_cogs=0.0,
    )
    res_scn = ltv_5yr_device_consumables(
        sku_inp,
        mah_per_pack=single_sku.mah_per_pack,
        price_per_pack=single_sku.price_per_pack,
        cogs_per_pack=single_sku.cogs_per_pack,
        service_years=scenario_sd,
        delta_device_cogs=delta_device_cogs,
    )

    # Juice bags sold per device
    bags_per_device_per_year = annual_packs(sku_inp.mah_per_month, mah_per_pack_single)
    dpuy_single = devices_per_user_per_year(scenario_sd)
    bags_per_device = bags_per_device_per_year / dpuy_single if dpuy_single > 0 else float("inf")
    dpuy_base = devices_per_user_per_year(baseline_sd)
    bags_per_device_base = bags_per_device_per_year / dpuy_base if dpuy_base > 0 else float("inf")

    # Helper for delta display
    def _delta_html(val: float, fmt: str = ",.0f", prefix: str = "", suffix: str = "") -> str:
        arrow = "▲" if val >= 0 else "▼"
        color = "#09ab3b" if val >= 0 else "#ff2b2b"
        sign = "+" if val >= 0 else "-"
        return f'<p style="font-size:0.875rem;color:{color};margin:0;">{arrow} {sign}{prefix}{abs(val):{fmt}}{suffix}</p>'

    # Compute all deltas
    d_ltv = res_scn["ltv"] - res_base["ltv"]
    d_margin = (res_scn["avg_margin_pct"] - res_base["avg_margin_pct"]) * 100
    d_ret = (res_scn["retention_12mo"] - res_base["retention_12mo"]) * 100
    d_ltv_cac = res_scn["ltv_cac"] - res_base["ltv_cac"]
    d_bags = bags_per_device - bags_per_device_base

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("LTV (baseline)", f"${res_base['ltv']:,.0f}")
    with k2:
        st.markdown('<p class="scenario-text" style="font-size:0.875rem;margin-bottom:0;">LTV (scenario)</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:1.75rem;font-weight:700;margin:0;">${res_scn["ltv"]:,.0f}</p>'
            + _delta_html(d_ltv, prefix="$"),
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown('<p class="scenario-text" style="font-size:0.875rem;margin-bottom:0;">Avg Margin % (scenario)</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:1.75rem;font-weight:700;margin:0;">{res_scn["avg_margin_pct"]*100:.1f}%</p>'
            + _delta_html(d_margin, fmt=".2f", suffix=" pp"),
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown('<p class="scenario-text" style="font-size:0.875rem;margin-bottom:0;">12-mo retention (scenario)</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:1.75rem;font-weight:700;margin:0;">{res_scn["retention_12mo"]*100:.1f}%</p>'
            + _delta_html(d_ret, fmt=".1f", suffix=" pp"),
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown('<p class="scenario-text" style="font-size:0.875rem;margin-bottom:0;">LTV:CAC (scenario)</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:1.75rem;font-weight:700;margin:0;">{res_scn["ltv_cac"]:.2f}</p>'
            + _delta_html(d_ltv_cac, fmt=".2f"),
            unsafe_allow_html=True,
        )
    with k6:
        st.markdown('<p class="scenario-text" style="font-size:0.875rem;margin-bottom:0;">Bags sold / device (scenario)</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:1.75rem;font-weight:700;margin:0;">{bags_per_device:,.0f}</p>'
            + _delta_html(d_bags),
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<p class="scenario-text"><b>Δ Device COGS:</b> ${delta_device_cogs:,.2f} &nbsp;&nbsp;|&nbsp;&nbsp; '
        f'<b>Δ Service Duration:</b> {scenario_sd - baseline_sd:+.2f} yr '
        f'(baseline {baseline_sd:.2f} → scenario {scenario_sd:.2f})</p>',
        unsafe_allow_html=True,
    )

    st.caption("Payback year is the earliest year where cumulative discounted profit becomes ≥ 0. (Shown as NaN if never reaches ≥0.)")
    payback_val_scn = res_scn["payback_year"]
    payback_val_base = res_base["payback_year"]
    payback_display = f'{payback_val_scn:.0f}' if not np.isnan(payback_val_scn) else 'NaN'
    if not np.isnan(payback_val_scn) and not np.isnan(payback_val_base):
        d_payback = payback_val_scn - payback_val_base
        # For payback, lower is better, so flip the arrow color
        pb_arrow = "▼" if d_payback <= 0 else "▲"
        pb_color = "#09ab3b" if d_payback <= 0 else "#ff2b2b"
        pb_sign = "+" if d_payback >= 0 else ""
        payback_delta_html = f' <span style="color:{pb_color};">{pb_arrow} {pb_sign}{d_payback:.0f} yr</span>'
    else:
        payback_delta_html = ""
    st.markdown(
        f'<p class="scenario-text"><b>Payback year (scenario):</b> {payback_display}{payback_delta_html}</p>',
        unsafe_allow_html=True,
    )

    # ---- Bags per device vs LTV chart ----
    st.divider()
    st.markdown("##### Bags Sold per Device vs LTV")
    sd_range = np.arange(0.5, 5.1, 0.1)
    chart_ltvs = []
    chart_bags = []
    for sd_val in sd_range:
        r = ltv_5yr_device_consumables(
            sku_inp,
            mah_per_pack=single_sku.mah_per_pack,
            price_per_pack=single_sku.price_per_pack,
            cogs_per_pack=single_sku.cogs_per_pack,
            service_years=float(sd_val),
            delta_device_cogs=delta_device_cogs,
        )
        dpuy_val = devices_per_user_per_year(float(sd_val))
        bpd = bags_per_device_per_year / dpuy_val if dpuy_val > 0 else 0.0
        chart_bags.append(bpd)
        chart_ltvs.append(r["ltv"])

    fig_bpd, ax_bpd = plt.subplots(figsize=(8, 4))
    ax_bpd.plot(chart_bags, chart_ltvs, linewidth=2)
    # Mark current scenario point with LTV label
    ax_bpd.scatter([bags_per_device], [res_scn["ltv"]], color="#e6b800", s=100, zorder=5, label="Current scenario")
    ax_bpd.annotate(
        f'LTV ${res_scn["ltv"]:,.0f}\n{bags_per_device:,.0f} bags/device',
        xy=(bags_per_device, res_scn["ltv"]),
        xytext=(15, 15),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        color="#e6b800",
        arrowprops=dict(arrowstyle="->", color="#e6b800", lw=1.5),
    )
    ax_bpd.set_xlabel("Bags sold per device (over device lifetime)")
    ax_bpd.set_ylabel("LTV ($)")
    ax_bpd.set_title("Bags Sold per Device vs LTV (varying service duration)")
    ax_bpd.legend()
    ax_bpd.grid(True, alpha=0.3)
    st.pyplot(fig_bpd)
    plt.close(fig_bpd)

    # ---- Save scenario ----
    st.divider()
    save_cols = st.columns([3, 1])
    with save_cols[0]:
        scenario_label = st.text_input(
            "Scenario label",
            value=f"Scenario {len(st.session_state['saved_scenarios']) + 1}",
            key="scenario_label_input",
        )
    with save_cols[1]:
        st.markdown("<br>", unsafe_allow_html=True)
        save_clicked = st.button("Save current scenario")

    if save_clicked:
        st.session_state["saved_scenarios"].append({
            "label": scenario_label,
            "ltv_base": res_base["ltv"],
            "ltv_scn": res_scn["ltv"],
            "delta_ltv": res_scn["ltv"] - res_base["ltv"],
            "avg_margin_pct": res_scn["avg_margin_pct"],
            "retention_12mo": res_scn["retention_12mo"],
            "ltv_cac": res_scn["ltv_cac"],
            "payback_year": res_scn["payback_year"],
            "bags_per_device": bags_per_device,
            "delta_device_cogs": delta_device_cogs,
            "delta_service_dur": scenario_sd - baseline_sd,
            "scenario_sd": scenario_sd,
            "baseline_sd": baseline_sd,
            "mL_per_bag": mah_per_pack_single,
            "price_per_bag": price_per_pack_single,
            "cogs_per_bag": cogs_per_pack_single,
        })
        st.rerun()

    # ---- Saved scenarios table ----
    if st.session_state["saved_scenarios"]:
        st.divider()
        st.markdown("### Saved Scenarios")

        # Build editable dataframe
        saved = st.session_state["saved_scenarios"]
        saved_df = pd.DataFrame({
            "Label": [s["label"] for s in saved],
            "Δ Device COGS": [f"${s['delta_device_cogs']:,.2f}" for s in saved],
            "Δ Service Dur (yr)": [f"{s['delta_service_dur']:+.2f}" for s in saved],
            "Service Dur (yr)": [f"{s['scenario_sd']:.2f}" for s in saved],
            "LTV (base)": [f"${s['ltv_base']:,.0f}" for s in saved],
            "LTV (scenario)": [f"${s['ltv_scn']:,.0f}" for s in saved],
            "ΔLTV": [f"${s['delta_ltv']:,.0f}" for s in saved],
            "Margin %": [f"{s['avg_margin_pct']*100:.1f}%" for s in saved],
            "12-mo Ret": [f"{s['retention_12mo']*100:.1f}%" for s in saved],
            "LTV:CAC": [f"{s['ltv_cac']:.2f}" for s in saved],
            "Payback Yr": [f"{s['payback_year']:.0f}" if not np.isnan(s['payback_year']) else "N/A" for s in saved],
            "Bags/Device": [f"{s['bags_per_device']:,.0f}" for s in saved],
            "mL/Bag": [f"{s['mL_per_bag']:.2f}" for s in saved],
            "$/Bag": [f"${s['price_per_bag']:,.2f}" for s in saved],
        })

        # Render with styled labels via HTML
        label_html = " ".join(
            f'<span class="scenario-label">{s["label"]}</span>&nbsp;&nbsp;'
            for s in saved
        )
        st.markdown(f"Scenarios: {label_html}", unsafe_allow_html=True)

        # Editable labels
        edited_df = st.data_editor(
            saved_df,
            use_container_width=True,
            hide_index=True,
            disabled=[c for c in saved_df.columns if c != "Label"],
            key="saved_scenarios_editor",
        )

        # Sync edited labels back
        if edited_df is not None:
            for i, row in edited_df.iterrows():
                if i < len(st.session_state["saved_scenarios"]):
                    st.session_state["saved_scenarios"][i]["label"] = row["Label"]

        # Clear button
        if st.button("Clear all saved scenarios"):
            st.session_state["saved_scenarios"] = []
            st.rerun()


# ---------------- Mix ----------------
with tabs[1]:
    st.subheader("Consumable Mix (2–3 sizes)")

    st.caption("Shares are % of total volume (mL). Set one share to 0 to model 2 sizes.")

    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlock"]:has(#mix-single-baseline-marker) {
            background-color: transparent;
            border: 1px solid #d9d9d9;
            border-radius: 8px;
            padding: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    with cols[1]:
        st.markdown("### Small")
        s_mah = st.number_input("Small mL/juice bag", value=1.5, step=0.1, format="%.2f")
        s_price_per_mah = st.number_input("Small retail price per mL", value=3.00, step=0.05, format="%.2f")
        s_price = s_mah * s_price_per_mah
        st.caption(f"Computed Small price/juice bag: ${s_price:,.2f}")
        s_cogs = st.number_input("Small cogs/juice bag", value=1.30, step=0.05, format="%.2f")
        s_share = st.slider("Small share", 0.0, 1.0, 0.40, 0.01)

    with cols[2]:
        st.markdown("### Medium")
        m_mah = st.number_input("Medium mL/juice bag", value=2.0, step=0.1, format="%.2f")
        m_price_per_mah = st.number_input("Medium retail price per mL", value=3.00, step=0.05, format="%.2f")
        m_price = m_mah * m_price_per_mah
        st.caption(f"Computed Medium price/juice bag: ${m_price:,.2f}")
        m_cogs = st.number_input("Medium cogs/juice bag", value=1.55, step=0.05, format="%.2f")
        m_share = st.slider("Medium share", 0.0, 1.0, 0.40, 0.01)

    with cols[3]:
        st.markdown("### Large")
        l_mah = st.number_input("Large mL/juice bag", value=3.0, step=0.1, format="%.2f")
        l_price_per_mah = st.number_input("Large retail price per mL", value=2.83, step=0.05, format="%.2f")
        l_price = l_mah * l_price_per_mah
        st.caption(f"Computed Large price/juice bag: ${l_price:,.2f}")
        l_cogs = st.number_input("Large cogs/juice bag", value=2.10, step=0.05, format="%.2f")
        l_share = st.slider("Large share", 0.0, 1.0, 0.20, 0.01)

    with cols[0]:
        st.markdown('<span id="mix-single-baseline-marker"></span>', unsafe_allow_html=True)
        st.markdown("### Single Size SKU")
        st.caption("Baseline for Mix Delta (Mix LTV - Single Size SKU LTV)")

    with cols[0]:
        single_mah = st.number_input("Single size mL/juice bag", value=2.0, step=0.1, format="%.2f", key="mix_single_mah")
        single_price_per_mah = st.number_input("Single size retail price per mL", value=3.0, step=0.05, format="%.2f", key="mix_single_price_per_mah")
        single_price = single_mah * single_price_per_mah
        st.caption(f"Computed Single size price/juice bag: ${single_price:,.2f}")
        single_cogs = st.number_input("Single size COGS/juice bag", value=1.55, step=0.05, format="%.2f", key="mix_single_cogs")
        st.button(
            "Copy from Small",
            key="mix_copy_from_small",
            on_click=copy_mix_single_from,
            args=(s_mah, s_price_per_mah, s_cogs),
        )
        st.button(
            "Copy from Medium",
            key="mix_copy_from_medium",
            on_click=copy_mix_single_from,
            args=(m_mah, m_price_per_mah, m_cogs),
        )
        st.button(
            "Copy from Large",
            key="mix_copy_from_large",
            on_click=copy_mix_single_from,
            args=(l_mah, l_price_per_mah, l_cogs),
        )

    st.markdown("### Mix Comparison Assumptions")
    mix_assumption_cols = st.columns(3)
    with mix_assumption_cols[0]:
        mix_service_years = st.number_input(
            "Mix scenario service duration (years)",
            value=float(scenario_sd),
            step=0.1,
            format="%.2f",
            key="mix_service_years",
        )
    with mix_assumption_cols[1]:
        mix_delta_device_cogs = st.number_input(
            "Mix scenario Δ juice device COGS",
            value=float(delta_device_cogs),
            step=0.5,
            format="%.2f",
            key="mix_delta_device_cogs",
        )
    with mix_assumption_cols[2]:
        mix_annual_complexity_cost = st.number_input(
            "Mix annual complexity cost",
            value=float(annual_complexity_cost),
            step=25000.0,
            format="%.0f",
            key="mix_annual_complexity_cost",
        )

    share_sum = s_share + m_share + l_share
    if abs(share_sum - 1.0) > 1e-6:
        st.warning(f"Shares sum to {share_sum:.2f}. For best interpretation, make them sum to 1.0 (100%). The model will normalize internally.")

    single_sku = SKU("Single", single_mah, single_price, single_cogs)

    skus = [
        SKU("Small", s_mah, s_price, s_cogs),
        SKU("Medium", m_mah, m_price, m_cogs),
        SKU("Large", l_mah, l_price, l_cogs),
    ]
    shares = [s_share, m_share, l_share]
    mix_inp = replace(inp, annual_complexity_cost=mix_annual_complexity_cost)

    comp = mix_delta_ltv_and_roi(
        inp=mix_inp,
        single=single_sku,
        mix_skus=skus,
        shares=shares,
        service_years=mix_service_years,
        delta_device_cogs=mix_delta_device_cogs,
    )

    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlock"]:has(#mix-comparison-output-marker) {
            background-color: transparent;
            border: 1px solid #d9d9d9;
            border-radius: 8px;
            padding: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<span id="mix-comparison-output-marker"></span>', unsafe_allow_html=True)
        st.markdown("### Comparison Output")
        delta_ltv = comp["delta_ltv"]
        delta_ltv_display = f"+${abs(delta_ltv):,.0f}" if delta_ltv >= 0 else f"-${abs(delta_ltv):,.0f}"
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Single SKU LTV", f"${comp['single_ltv']:,.0f}")
        k2.metric("Mix LTV", f"${comp['mix_ltv']:,.0f}", delta=delta_ltv_display, delta_color="normal")
        k3.metric("PV complexity cost", f"${comp['pv_complexity_cost']:,.0f}")
        k4.metric("Net value", f"${comp['net_value']:,.0f}")
        k5.metric("ROI (ΔLTV / PV cost)", f"{comp['roi']:.2f}")

        st.write({
            "Breakeven annual complexity cost": f"${comp['breakeven_annual_complexity_cost']:,.0f}/yr",
            "Mix margin per mL": f"${comp['mix_margin_per_mL']:.3f}/mL",
            "Mix avg margin %": f"{comp['mix_avg_margin_pct']*100:.1f}%",
        })


# ---------------- Mix Heatmaps ----------------
with tabs[2]:
    st.subheader("Mix Heatmaps")

    out = st.selectbox("Heatmap output", ["ΔLTV", "Net Value", "Blended Margin per mL", "Blended Price per mL"])
    step_pct = st.select_slider("Grid step", options=[0.05, 0.10, 0.20], value=0.10)
    output_key = {
        "ΔLTV": "delta_ltv",
        "Net Value": "net_value",
        "Blended Margin per mL": "margin_per_mL",
        "Blended Price per mL": "price_per_mL",
    }[out]

    # Re-use SKU inputs from tab 1 via session state is messy; use defaults for now,
    # but Streamlit keeps widget values across tabs by label, so this reads user-edited values.
    # Construct again
    single_mah_hm = st.session_state.get("Single SKU mL per juice bag", 2.0)
    single_price_per_mah_hm = st.session_state.get("Single SKU retail price per mL", 3.0)
    single_sku = SKU("Single",
                     single_mah_hm,
                     single_mah_hm * single_price_per_mah_hm,
                     st.session_state.get("Single SKU COGS per juice bag", 1.55))

    skus = [
        SKU("Small",
            st.session_state.get("Small mL/juice bag", 1.5),
            st.session_state.get("Small mL/juice bag", 1.5) * st.session_state.get("Small retail price per mL", 3.0),
            st.session_state.get("Small cogs/juice bag", 1.3)),
        SKU("Medium",
            st.session_state.get("Medium mL/juice bag", 2.0),
            st.session_state.get("Medium mL/juice bag", 2.0) * st.session_state.get("Medium retail price per mL", 3.0),
            st.session_state.get("Medium cogs/juice bag", 1.55)),
        SKU("Large",
            st.session_state.get("Large mL/juice bag", 3.0),
            st.session_state.get("Large mL/juice bag", 3.0) * st.session_state.get("Large retail price per mL", 2.83),
            st.session_state.get("Large cogs/juice bag", 2.1)),
    ]

    s_axis, l_axis, grid = mix_heatmap(
        inp=inp,
        single=single_sku,
        skus=skus,
        step=step_pct,
        service_years=scenario_sd,
        delta_device_cogs=delta_device_cogs,
        output=output_key,
    )

    fig, ax = plt.subplots()
    im = ax.imshow(grid, origin="lower", aspect="auto")
    ax.set_xticks(range(len(l_axis)))
    ax.set_xticklabels([f"{int(x*100)}%" for x in l_axis], rotation=45, ha="right")
    ax.set_yticks(range(len(s_axis)))
    ax.set_yticklabels([f"{int(x*100)}%" for x in s_axis])
    ax.set_xlabel("Large share of volume")
    ax.set_ylabel("Small share of volume")
    ax.set_title(out)

    cbar = plt.colorbar(im, ax=ax)
    st.pyplot(fig)

    st.caption("Cells where Medium share would be negative are infeasible and shown as blank/NaN.")

    st.divider()
    st.subheader("Delta Juice Device COGS vs Delta Service Duration Tradeoff")
    st.caption("Visualizes cost/benefit tradeoffs using deltas from the baseline service duration.")

    tradeoff_output_label = st.selectbox(
        "Tradeoff heatmap output",
        ["ΔLTV vs baseline", "Scenario LTV"],
        key="tradeoff_output_label",
    )
    tradeoff_output = {
        "ΔLTV vs baseline": "delta_ltv",
        "Scenario LTV": "scenario_ltv",
    }[tradeoff_output_label]

    t1, t2, t3 = st.columns(3)
    with t1:
        delta_service_min = st.number_input("Δ Service duration min (years)", value=-0.4, step=0.1, format="%.2f")
        delta_service_max = st.number_input("Δ Service duration max (years)", value=1.0, step=0.1, format="%.2f")
        delta_service_step = st.select_slider("Δ Service duration step", options=[0.05, 0.10, 0.20], value=0.10)
    with t2:
        cogs_min = st.number_input("Δ COGS min", value=-1.0, step=0.5, format="%.2f")
        cogs_max = st.number_input("Δ COGS max", value=4.0, step=0.5, format="%.2f")
        cogs_step = st.select_slider("Δ COGS step", options=[0.10, 0.25, 0.50], value=0.25)
    with t3:
        baseline_service_tradeoff = st.number_input(
            "Baseline service duration (years)",
            value=float(baseline_sd),
            step=0.1,
            format="%.2f",
            key="tradeoff_baseline_service",
        )

    if delta_service_max <= delta_service_min:
        st.warning("Δ service duration max must be greater than min.")
    elif cogs_max <= cogs_min:
        st.warning("Δ COGS max must be greater than min.")
    else:
        delta_service_axis = np.arange(delta_service_min, delta_service_max + 1e-9, delta_service_step)
        service_axis = baseline_service_tradeoff + delta_service_axis
        cogs_axis = np.arange(cogs_min, cogs_max + 1e-9, cogs_step)

        tradeoff_single_mah = st.session_state.get("Single SKU mL per juice bag", 2.0)
        tradeoff_single_price_per_mah = st.session_state.get("Single SKU retail price per mL", 3.0)
        tradeoff_single = SKU(
            "Single",
            tradeoff_single_mah,
            tradeoff_single_mah * tradeoff_single_price_per_mah,
            st.session_state.get("Single SKU COGS per juice bag", 1.55),
        )

        tradeoff_grid = device_tradeoff_heatmap(
            inp=inp,
            sku=tradeoff_single,
            service_axis=service_axis,
            delta_cogs_axis=cogs_axis,
            baseline_service_years=baseline_service_tradeoff,
            baseline_delta_device_cogs=0.0,
            output=tradeoff_output,
        )

        fig2, ax2 = plt.subplots()
        cmap = "RdYlGn" if tradeoff_output == "delta_ltv" else "viridis"
        im2 = ax2.imshow(tradeoff_grid, origin="lower", aspect="auto", cmap=cmap)
        ax2.set_xticks(range(len(cogs_axis)))
        ax2.set_xticklabels([f"{x:.2f}" for x in cogs_axis], rotation=45, ha="right")
        ax2.set_yticks(range(len(delta_service_axis)))
        ax2.set_yticklabels([f"{x:.2f}" for x in delta_service_axis])
        ax2.set_xlabel("Δ COGS")
        ax2.set_ylabel("Δ Service duration (years)")
        ax2.set_title(tradeoff_output_label)
        plt.colorbar(im2, ax=ax2)
        st.pyplot(fig2)


# ---------------- Size Ladder Optimizer ----------------
with tabs[3]:
    st.subheader("Size Ladder Optimizer (price per mL by size)")

    if not SCIPY_OK:
        st.error("scipy is not installed. Run: pip install scipy")
        st.stop()

    st.caption("Optimizes price per mL for 2 or 3 sizes to maximize 5-year discounted LTV under elasticity.")

    size_order = ["Small", "Medium", "Large"]
    size_count = st.radio("Number of sizes to optimize", [2, 3], horizontal=True, key="opt_size_count")
    if size_count == 2:
        selected_sizes = st.multiselect(
            "Choose 2 sizes",
            options=size_order,
            default=["Small", "Large"],
            key="opt_selected_sizes",
        )
        if len(selected_sizes) != 2:
            st.warning("Select exactly 2 sizes to run optimization.")
            selected_sizes = []
    else:
        selected_sizes = size_order

    selected_sizes = [s for s in size_order if s in selected_sizes]

    sku_map = {
        "Small": SKU(
            "Small",
            st.session_state.get("Small mL/juice bag", 1.5),
            st.session_state.get("Small mL/juice bag", 1.5) * st.session_state.get("Small retail price per mL", 3.0),
            st.session_state.get("Small cogs/juice bag", 1.3),
        ),
        "Medium": SKU(
            "Medium",
            st.session_state.get("Medium mL/juice bag", 2.0),
            st.session_state.get("Medium mL/juice bag", 2.0) * st.session_state.get("Medium retail price per mL", 3.0),
            st.session_state.get("Medium cogs/juice bag", 1.55),
        ),
        "Large": SKU(
            "Large",
            st.session_state.get("Large mL/juice bag", 3.0),
            st.session_state.get("Large mL/juice bag", 3.0) * st.session_state.get("Large retail price per mL", 2.83),
            st.session_state.get("Large cogs/juice bag", 2.1),
        ),
    }
    share_map = {
        "Small": st.session_state.get("Small share", 0.40),
        "Medium": st.session_state.get("Medium share", 0.40),
        "Large": st.session_state.get("Large share", 0.20),
    }

    skus = [sku_map[s] for s in selected_sizes]
    shares = np.array([share_map[s] for s in selected_sizes], dtype=float) if selected_sizes else np.array([])
    if len(shares) > 0:
        if shares.sum() <= 0:
            shares = np.ones(len(shares), dtype=float) / len(shares)
        else:
            shares = shares / shares.sum()
        st.caption("Optimizer uses normalized shares for selected sizes: " + ", ".join([f"{s}={w:.2f}" for s, w in zip(selected_sizes, shares)]))

    price_cols = st.columns(max(1, len(selected_sizes)))
    p0_vals = []
    for i, size_name in enumerate(selected_sizes):
        default_price = float(st.session_state.get(f"{size_name} retail price per mL", 3.0))
        with price_cols[i]:
            p0_vals.append(
                st.number_input(
                    f"Start price/mL {size_name}",
                    value=default_price,
                    step=0.10,
                    key=f"opt_start_price_{size_name.lower()}",
                )
            )

    monotonic_text = " ≥ ".join(selected_sizes) if selected_sizes else "Small ≥ Medium ≥ Large"
    monotonic = st.checkbox(f"Enforce ladder: {monotonic_text}", value=True)
    pmin = st.number_input("Min price/mL", value=1.50, step=0.10)
    pmax = st.number_input("Max price/mL", value=6.00, step=0.10)

    from model import ladder_objective_ltv

    run = st.button("Run optimization")
    if run and selected_sizes:
        x0 = np.array(p0_vals, dtype=float)
        bounds = [(pmin, pmax)] * len(selected_sizes)

        def objective(x: np.ndarray) -> float:
            # Maximize LTV -> minimize negative
            return -ladder_objective_ltv(inp, skus, shares.tolist(), x, scenario_sd, delta_device_cogs)

        cons = []
        if monotonic:
            for i in range(len(selected_sizes) - 1):
                cons.append({"type": "ineq", "fun": lambda x, idx=i: x[idx] - x[idx + 1]})

        res = minimize(objective, x0, bounds=bounds, constraints=cons, method="SLSQP")
        if not res.success:
            st.warning(f"Optimizer did not fully converge: {res.message}")

        x = res.x
        best_ltv = -res.fun

        out = {f"Price/mL {size_name}": float(price) for size_name, price in zip(selected_sizes, x)}
        out["Max 5y discounted LTV"] = float(best_ltv)

        st.success("Optimization complete.")
        st.write(out)


# ---------------- DCF LTV Analysis ----------------
with tabs[4]:
    st.subheader("DCF LTV Analysis — Year-by-Year Breakdown")
    st.caption("Uses Single SKU inputs from Tab 1 and scenario settings from the sidebar.")

    # Reuse Single SKU inputs (already computed above as module-level variables)
    dcf_mah_per_pack = mah_per_pack_single
    dcf_price_per_pack = price_per_pack_single
    dcf_cogs_per_pack = cogs_per_pack_single

    # Price elasticity toggle
    dcf_e1, dcf_e2 = st.columns(2)
    with dcf_e1:
        dcf_apply_elasticity = st.checkbox("Apply price elasticity to demand", value=False, key="dcf_elasticity_toggle")
    with dcf_e2:
        dcf_ref_price_per_mah = st.number_input(
            "Reference price per mL (baseline demand)",
            value=float(retail_price_per_mah_single),
            step=0.05,
            format="%.2f",
            key="dcf_ref_price_per_mah",
            disabled=not dcf_apply_elasticity,
        )

    # Compute intermediate arrays
    dcf_years = inp.years
    dcf_pv = discount_factors(dcf_years, inp.discount_rate)
    dcf_base_ret = np.array(inp.retention[:dcf_years], dtype=float)
    dcf_ret = retention_with_durability(dcf_base_ret, inp.baseline_service_years, scenario_sd, inp.retention_lift_per_year)

    # Adjust mL/month for price elasticity
    dcf_observed_price_per_mah = dcf_price_per_pack / dcf_mah_per_pack if dcf_mah_per_pack else 0.0
    if dcf_apply_elasticity:
        dcf_mah_per_month = demand_with_elasticity(
            inp.mah_per_month, dcf_observed_price_per_mah, dcf_ref_price_per_mah, inp.elasticity,
        )
    else:
        dcf_mah_per_month = inp.mah_per_month

    dcf_packs_y = annual_packs(dcf_mah_per_month, dcf_mah_per_pack)
    dcf_dpuy = devices_per_user_per_year(scenario_sd)

    if dcf_apply_elasticity:
        st.caption(
            f"Elasticity {inp.elasticity:.2f} · observed price/mL ${dcf_observed_price_per_mah:.2f} vs "
            f"reference ${dcf_ref_price_per_mah:.2f} → adjusted mL/month: {dcf_mah_per_month:.1f} "
            f"(base: {inp.mah_per_month:.1f})"
        )

    # Per-year calculations
    dcf_rows = []
    dcf_cum_pv_profit = 0.0
    for yr in range(dcf_years):
        r = float(dcf_ret[yr])
        df = float(dcf_pv[yr])

        # Juice bag (consumable)
        bag_rev = dcf_packs_y * dcf_price_per_pack * r
        bag_cogs = dcf_packs_y * dcf_cogs_per_pack * r
        bag_gp = bag_rev - bag_cogs

        # Juice device
        dev_rev = inp.device_asp * dcf_dpuy * r
        dev_cogs = (inp.device_cogs_baseline + delta_device_cogs + inp.device_other_costs) * dcf_dpuy * r
        dev_gp = dev_rev - dev_cogs

        total_rev = bag_rev + dev_rev
        total_profit = bag_gp + dev_gp
        pv_profit = total_profit * df
        dcf_cum_pv_profit += pv_profit

        dcf_rows.append({
            "Year": yr + 1,
            "Retention": f"{r:.1%}",
            "Bag Revenue": f"${bag_rev:,.2f}",
            "Bag COGS": f"${bag_cogs:,.2f}",
            "Bag Gross Profit": f"${bag_gp:,.2f}",
            "Device Revenue": f"${dev_rev:,.2f}",
            "Device COGS": f"${dev_cogs:,.2f}",
            "Device Gross Profit": f"${dev_gp:,.2f}",
            "Total Revenue": f"${total_rev:,.2f}",
            "Total Profit": f"${total_profit:,.2f}",
            "Discount Factor": f"{df:.4f}",
            "PV Profit": f"${pv_profit:,.2f}",
            "Cumulative PV Profit": f"${dcf_cum_pv_profit:,.2f}",
        })

    dcf_df = pd.DataFrame(dcf_rows)
    st.dataframe(dcf_df, use_container_width=True, hide_index=True)

    # Summary metrics
    st.markdown("### Summary Metrics")
    total_ltv = dcf_cum_pv_profit
    ltv_cac_ratio = total_ltv / inp.cac if inp.cac else float("nan")

    # Payback year: first year where cumulative PV profit >= 0
    cum_check = 0.0
    payback_yr = None
    for yr in range(dcf_years):
        r = float(dcf_ret[yr])
        df = float(dcf_pv[yr])
        bag_gp = dcf_packs_y * (dcf_price_per_pack - dcf_cogs_per_pack) * r
        dev_gp = (inp.device_asp - inp.device_cogs_baseline - delta_device_cogs - inp.device_other_costs) * dcf_dpuy * r
        cum_check += (bag_gp + dev_gp) * df
        if cum_check >= 0 and payback_yr is None:
            payback_yr = yr + 1

    # Average margin %
    total_pv_rev = 0.0
    for yr in range(dcf_years):
        r = float(dcf_ret[yr])
        df = float(dcf_pv[yr])
        rev = (dcf_packs_y * dcf_price_per_pack + inp.device_asp * dcf_dpuy) * r
        total_pv_rev += rev * df
    avg_margin = total_ltv / total_pv_rev if total_pv_rev else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("5-Year PV Profit (LTV)", f"${total_ltv:,.2f}")
    m2.metric("LTV:CAC Ratio", f"{ltv_cac_ratio:.2f}")
    m3.metric("Payback Year", f"{payback_yr}" if payback_yr else "N/A")
    m4.metric("Average Margin %", f"{avg_margin * 100:.1f}%")


# ---------------- Investment Strategy Graphs ----------------
with tabs[5]:
    st.subheader("Investment Strategy Graphs")
    st.caption("Visualizations showing how investing in device durability (higher COGS) impacts key business metrics. All graphs use Single SKU inputs and current sidebar settings.")

    # Helper: save figure to PNG bytes for download
    def _fig_to_png(fig) -> bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        return buf.getvalue()

    # Shared sweep ranges — start low enough to cover short baseline durations
    _sd_range = np.arange(max(0.1, min(baseline_sd, 0.5)), 5.1, 0.1)
    _dcogs_range = np.arange(0.0, 6.1, 0.25)

    # Pre-compute Single SKU reference objects
    _sku_ref = SKU("Single", mah_per_pack_single, price_per_pack_single, cogs_per_pack_single)
    _bags_per_year = annual_packs(sku_inp.mah_per_month, mah_per_pack_single)

    # ================================================================
    # 1. Average Profit Margin % — Baseline vs Scenario
    # ================================================================
    st.divider()
    st.markdown("##### 1. Average Profit Margin % — Baseline vs Scenario")
    st.markdown(
        "Compares the average profit margin between baseline and scenario as service duration varies. "
        "The baseline holds device COGS constant while the scenario includes your Δ COGS investment — "
        "when the scenario line stays close to or above the baseline, the margin impact of the investment is minimal "
        "relative to the LTV gains from improved retention."
    )
    margins_base_sd = []
    margins_scn_sd = []
    for sd in _sd_range:
        r_bl = ltv_5yr_device_consumables(
            sku_inp, mah_per_pack=_sku_ref.mah_per_pack,
            price_per_pack=_sku_ref.price_per_pack, cogs_per_pack=_sku_ref.cogs_per_pack,
            service_years=float(sd), delta_device_cogs=0.0,
        )
        r_sc = ltv_5yr_device_consumables(
            sku_inp, mah_per_pack=_sku_ref.mah_per_pack,
            price_per_pack=_sku_ref.price_per_pack, cogs_per_pack=_sku_ref.cogs_per_pack,
            service_years=float(sd), delta_device_cogs=delta_device_cogs,
        )
        margins_base_sd.append(r_bl["avg_margin_pct"] * 100)
        margins_scn_sd.append(r_sc["avg_margin_pct"] * 100)

    # Compute exact dot values directly (avoid index lookup errors when baseline/scenario fall outside sweep)
    _g1_bl = ltv_5yr_device_consumables(
        sku_inp, mah_per_pack=_sku_ref.mah_per_pack,
        price_per_pack=_sku_ref.price_per_pack, cogs_per_pack=_sku_ref.cogs_per_pack,
        service_years=baseline_sd, delta_device_cogs=0.0,
    )["avg_margin_pct"] * 100
    _g1_sc = ltv_5yr_device_consumables(
        sku_inp, mah_per_pack=_sku_ref.mah_per_pack,
        price_per_pack=_sku_ref.price_per_pack, cogs_per_pack=_sku_ref.cogs_per_pack,
        service_years=scenario_sd, delta_device_cogs=delta_device_cogs,
    )["avg_margin_pct"] * 100

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(_sd_range, margins_base_sd, linewidth=2, color="#1f77b4", label="Baseline (Δ COGS $0)")
    ax1.plot(_sd_range, margins_scn_sd, linewidth=2, color="#e6b800", label=f"Scenario (Δ COGS ${delta_device_cogs:.2f})")
    ax1.fill_between(_sd_range, margins_base_sd, margins_scn_sd, alpha=0.12, color="#e6b800")
    ax1.scatter([scenario_sd], [_g1_sc], color="#e6b800", s=80, zorder=5)
    ax1.scatter([baseline_sd], [_g1_bl], color="#1f77b4", s=80, zorder=5)
    ax1.annotate(f'Scenario {_g1_sc:.1f}%', xy=(scenario_sd, _g1_sc),
                 xytext=(15, -20), textcoords="offset points", fontsize=9, fontweight="bold", color="#e6b800",
                 arrowprops=dict(arrowstyle="->", color="#e6b800", lw=1.5))
    ax1.annotate(f'Baseline {_g1_bl:.1f}%', xy=(baseline_sd, _g1_bl),
                 xytext=(15, 10), textcoords="offset points", fontsize=9, fontweight="bold", color="#1f77b4",
                 arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5))
    ax1.set_xlabel("Service Duration (years)")
    ax1.set_ylabel("Average Profit Margin (%)")
    ax1.set_title("Average Profit Margin: Baseline vs Scenario")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    c1a, c1b = st.columns([10, 1])
    with c1a:
        st.pyplot(fig1)
    with c1b:
        st.download_button("PNG", _fig_to_png(fig1), "margin_baseline_vs_scenario.png", "image/png", key="dl1")
    plt.close(fig1)

    # ================================================================
    # 2. Profit Margin Delta Heatmap — Δ COGS vs Δ Service Duration
    # ================================================================
    st.divider()
    st.markdown("##### 2. Profit Margin Delta Heatmap — Δ Device COGS vs Δ Service Duration")
    st.markdown(
        "Maps the change in average profit margin (vs baseline) across combinations of additional device cost "
        "and service duration improvement. White = no change, green = margin improvement, red = margin erosion — "
        "use this to find the sweet spot where durability gains justify the cost increase without sacrificing margin."
    )
    _dsd_range = np.arange(0.0, 2.1, 0.1)
    _dcogs_heat = np.arange(0.0, 5.1, 0.25)
    _margin_base_ref = ltv_5yr_device_consumables(
        sku_inp, mah_per_pack=_sku_ref.mah_per_pack,
        price_per_pack=_sku_ref.price_per_pack, cogs_per_pack=_sku_ref.cogs_per_pack,
        service_years=baseline_sd, delta_device_cogs=0.0,
    )["avg_margin_pct"] * 100
    margin_delta_grid = np.zeros((len(_dsd_range), len(_dcogs_heat)))
    for i, dsd in enumerate(_dsd_range):
        for j, dc in enumerate(_dcogs_heat):
            r = ltv_5yr_device_consumables(
                sku_inp, mah_per_pack=_sku_ref.mah_per_pack,
                price_per_pack=_sku_ref.price_per_pack, cogs_per_pack=_sku_ref.cogs_per_pack,
                service_years=baseline_sd + float(dsd), delta_device_cogs=float(dc),
            )
            margin_delta_grid[i, j] = r["avg_margin_pct"] * 100 - _margin_base_ref

    _vmax = max(abs(margin_delta_grid.min()), abs(margin_delta_grid.max()), 0.1)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    im = ax2.imshow(margin_delta_grid, aspect="auto", origin="lower",
                    cmap="RdYlGn", vmin=-_vmax, vmax=_vmax,
                    extent=[_dcogs_heat[0], _dcogs_heat[-1], _dsd_range[0], _dsd_range[-1]])
    ax2.set_xlabel("Δ Device COGS ($)")
    ax2.set_ylabel("Δ Service Duration (years)")
    ax2.set_title("Profit Margin Delta vs Baseline (pp)")
    fig2.colorbar(im, ax=ax2, label="Margin Δ (pp)")
    cur_dsd = scenario_sd - baseline_sd
    _scn_margin_delta = ltv_5yr_device_consumables(
        sku_inp, mah_per_pack=_sku_ref.mah_per_pack,
        price_per_pack=_sku_ref.price_per_pack, cogs_per_pack=_sku_ref.cogs_per_pack,
        service_years=scenario_sd, delta_device_cogs=delta_device_cogs,
    )["avg_margin_pct"] * 100 - _margin_base_ref
    ax2.scatter([0.0], [0.0], color="#1f77b4", s=120, zorder=5, edgecolors="black", linewidths=1.5)
    ax2.annotate(f"Baseline\n0.0 pp", xy=(0.0, 0.0),
                 xytext=(10, -15), textcoords="offset points", fontsize=9, fontweight="bold", color="#1f77b4",
                 arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5))
    ax2.scatter([delta_device_cogs], [cur_dsd], color="#e6b800", s=120, zorder=5, edgecolors="black", linewidths=1.5)
    ax2.annotate(f"Scenario\n{_scn_margin_delta:+.2f} pp", xy=(delta_device_cogs, cur_dsd),
                 xytext=(10, 10), textcoords="offset points", fontsize=9, fontweight="bold", color="#e6b800",
                 arrowprops=dict(arrowstyle="->", color="#e6b800", lw=1.5))
    fig2.tight_layout()
    c2a, c2b = st.columns([10, 1])
    with c2a:
        st.pyplot(fig2)
    with c2b:
        st.download_button("PNG", _fig_to_png(fig2), "margin_heatmap.png", "image/png", key="dl2")
    plt.close(fig2)

    # ================================================================
    # 3. LTV vs Delta Device COGS
    # ================================================================
    st.divider()
    st.markdown("##### 3. LTV vs Delta Device COGS")
    st.markdown(
        "Shows how LTV changes as you invest more in device COGS. "
        "If the LTV curve rises faster than the cost increase, the investment pays for itself — "
        "each dollar spent on better components yields more than a dollar back in lifetime profit."
    )
    ltvs_dcogs = []
    for dc in _dcogs_range:
        r = ltv_5yr_device_consumables(
            sku_inp, mah_per_pack=_sku_ref.mah_per_pack,
            price_per_pack=_sku_ref.price_per_pack, cogs_per_pack=_sku_ref.cogs_per_pack,
            service_years=scenario_sd, delta_device_cogs=float(dc),
        )
        ltvs_dcogs.append(r["ltv"])

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(_dcogs_range, ltvs_dcogs, linewidth=2, color="#1f77b4")
    ax3.axvline(delta_device_cogs, color="#e6b800", linestyle="--", alpha=0.8, label=f"Current Δ COGS ${delta_device_cogs:.2f}")
    cur_ltv_idx = min(range(len(_dcogs_range)), key=lambda i: abs(_dcogs_range[i] - delta_device_cogs))
    ax3.scatter([delta_device_cogs], [ltvs_dcogs[cur_ltv_idx]], color="#e6b800", s=80, zorder=5)
    ax3.annotate(f'LTV ${ltvs_dcogs[cur_ltv_idx]:,.0f}', xy=(delta_device_cogs, ltvs_dcogs[cur_ltv_idx]),
                 xytext=(15, 10), textcoords="offset points", fontsize=9, fontweight="bold", color="#e6b800",
                 arrowprops=dict(arrowstyle="->", color="#e6b800", lw=1.5))
    ax3.set_xlabel("Δ Device COGS ($)")
    ax3.set_ylabel("LTV ($)")
    ax3.set_title("LTV vs Additional Device COGS Investment")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    c3a, c3b = st.columns([10, 1])
    with c3a:
        st.pyplot(fig3)
    with c3b:
        st.download_button("PNG", _fig_to_png(fig3), "ltv_vs_delta_cogs.png", "image/png", key="dl3")
    plt.close(fig3)

    # ================================================================
    # 4. LTV:CAC Ratio vs Service Duration
    # ================================================================
    st.divider()
    st.markdown("##### 4. LTV:CAC Ratio vs Service Duration")
    st.markdown(
        "Shows how the LTV-to-CAC ratio improves as devices last longer. "
        "A ratio above 3x is the typical benchmark for healthy unit economics — "
        "this chart helps identify the service duration needed to hit that target."
    )
    ltv_cac_sd = []
    for sd in _sd_range:
        r = ltv_5yr_device_consumables(
            sku_inp, mah_per_pack=_sku_ref.mah_per_pack,
            price_per_pack=_sku_ref.price_per_pack, cogs_per_pack=_sku_ref.cogs_per_pack,
            service_years=float(sd), delta_device_cogs=delta_device_cogs,
        )
        ltv_cac_sd.append(r["ltv_cac"])

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(_sd_range, ltv_cac_sd, linewidth=2, color="#2ca02c")
    ax2.axhline(3.0, color="#ff7f0e", linestyle=":", alpha=0.7, label="3x benchmark")
    ax2.axvline(scenario_sd, color="#e6b800", linestyle="--", alpha=0.8, label=f"Current {scenario_sd:.1f}yr")
    _g4_cac = ltv_5yr_device_consumables(
        sku_inp, mah_per_pack=_sku_ref.mah_per_pack,
        price_per_pack=_sku_ref.price_per_pack, cogs_per_pack=_sku_ref.cogs_per_pack,
        service_years=scenario_sd, delta_device_cogs=delta_device_cogs,
    )["ltv_cac"]
    ax2.scatter([scenario_sd], [_g4_cac], color="#e6b800", s=80, zorder=5)
    ax2.annotate(f'{_g4_cac:.2f}x', xy=(scenario_sd, _g4_cac),
                 xytext=(15, 10), textcoords="offset points", fontsize=9, fontweight="bold", color="#e6b800",
                 arrowprops=dict(arrowstyle="->", color="#e6b800", lw=1.5))
    ax2.set_xlabel("Service Duration (years)")
    ax2.set_ylabel("LTV:CAC Ratio")
    ax2.set_title("LTV:CAC Ratio vs Device Service Duration")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    c2a, c2b = st.columns([10, 1])
    with c2a:
        st.pyplot(fig2)
    with c2b:
        st.download_button("PNG", _fig_to_png(fig2), "ltv_cac_vs_service_duration.png", "image/png", key="dl4")
    plt.close(fig2)

    # ================================================================
    # 5. Payback Period vs Service Duration
    # ================================================================
    st.divider()
    st.markdown("##### 5. Payback Period vs Service Duration")
    st.markdown(
        "Shows how quickly you recoup customer acquisition costs as devices last longer. "
        "A shorter payback period means lower risk and faster return on investment — "
        "critical for cash flow planning and investor confidence."
    )
    payback_sd = []
    for sd in _sd_range:
        r = ltv_5yr_device_consumables(
            sku_inp, mah_per_pack=_sku_ref.mah_per_pack,
            price_per_pack=_sku_ref.price_per_pack, cogs_per_pack=_sku_ref.cogs_per_pack,
            service_years=float(sd), delta_device_cogs=delta_device_cogs,
        )
        payback_sd.append(r["payback_year"])

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    # Filter out NaN for plotting
    valid_sd = [s for s, p in zip(_sd_range, payback_sd) if not np.isnan(p)]
    valid_pb = [p for p in payback_sd if not np.isnan(p)]
    ax3.plot(valid_sd, valid_pb, linewidth=2, color="#d62728", marker="o", markersize=3)
    cur_pb = res_scn["payback_year"]
    if not np.isnan(cur_pb):
        ax3.scatter([scenario_sd], [cur_pb], color="#e6b800", s=80, zorder=5)
        ax3.annotate(f'Year {cur_pb:.0f}', xy=(scenario_sd, cur_pb),
                     xytext=(15, 10), textcoords="offset points", fontsize=9, fontweight="bold", color="#e6b800",
                     arrowprops=dict(arrowstyle="->", color="#e6b800", lw=1.5))
    ax3.set_xlabel("Service Duration (years)")
    ax3.set_ylabel("Payback Year")
    ax3.set_title("Payback Period vs Device Service Duration")
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    c3a, c3b = st.columns([10, 1])
    with c3a:
        st.pyplot(fig3)
    with c3b:
        st.download_button("PNG", _fig_to_png(fig3), "payback_vs_service_duration.png", "image/png", key="dl5")
    plt.close(fig3)

    # ================================================================
    # 6. Cumulative Profit Waterfall — Baseline vs Scenario
    # ================================================================
    st.divider()
    st.markdown("##### 6. Cumulative Discounted Profit — Baseline vs Scenario")
    st.markdown(
        "Compares year-by-year cumulative discounted profit between your baseline and scenario. "
        "The scenario overtaking the baseline in later years shows where the durability investment pays off — "
        "the area between the curves represents the value created by the investment."
    )
    # Compute year-by-year cumulative PV profit for baseline and scenario
    _base_ret = np.array(sku_inp.retention[:sku_inp.years], dtype=float)
    _scn_ret = retention_with_durability(_base_ret, sku_inp.baseline_service_years, scenario_sd, sku_inp.retention_lift_per_year)
    _bl_ret = retention_with_durability(_base_ret, sku_inp.baseline_service_years, baseline_sd, sku_inp.retention_lift_per_year)
    _pv = discount_factors(sku_inp.years, sku_inp.discount_rate)
    _dpuy_scn = devices_per_user_per_year(scenario_sd)
    _dpuy_bl = devices_per_user_per_year(baseline_sd)
    _dev_margin_scn = device_asp - (device_cogs + delta_device_cogs) - device_other
    _dev_margin_bl = device_asp - device_cogs - device_other
    _cons_profit = _bags_per_year * (price_per_pack_single - cogs_per_pack_single)

    cum_base = []
    cum_scn = []
    cb = 0.0
    cs = 0.0
    yr_labels = []
    for yr in range(sku_inp.years):
        yr_labels.append(f"Year {yr+1}")
        prof_bl = (_cons_profit + _dev_margin_bl * _dpuy_bl) * float(_bl_ret[yr]) * float(_pv[yr])
        prof_sc = (_cons_profit + _dev_margin_scn * _dpuy_scn) * float(_scn_ret[yr]) * float(_pv[yr])
        cb += prof_bl
        cs += prof_sc
        cum_base.append(cb)
        cum_scn.append(cs)

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    x_pos = np.arange(len(yr_labels))
    width = 0.35
    ax4.bar(x_pos - width/2, cum_base, width, label="Baseline", color="#1f77b4", alpha=0.8)
    ax4.bar(x_pos + width/2, cum_scn, width, label="Scenario", color="#e6b800", alpha=0.8)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(yr_labels)
    ax4.set_ylabel("Cumulative PV Profit ($)")
    ax4.set_title("Cumulative Discounted Profit: Baseline vs Scenario")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")
    fig4.tight_layout()
    c4a, c4b = st.columns([10, 1])
    with c4a:
        st.pyplot(fig4)
    with c4b:
        st.download_button("PNG", _fig_to_png(fig4), "cumulative_profit_waterfall.png", "image/png", key="dl6")
    plt.close(fig4)

    # ================================================================
    # 7. Retention Curves — Baseline vs Scenario
    # ================================================================
    st.divider()
    st.markdown("##### 7. Retention Curves — Baseline vs Scenario")
    st.markdown(
        "Shows how retention probability diverges between baseline and scenario over 5 years. "
        "The gap between curves represents additional users kept active by the durability investment — "
        "each retained user continues generating consumable revenue year after year."
    )
    fig5, ax5 = plt.subplots(figsize=(8, 4))
    yr_nums = np.arange(1, sku_inp.years + 1)
    ax5.plot(yr_nums, _bl_ret, linewidth=2, color="#1f77b4", marker="o", label="Baseline")
    ax5.plot(yr_nums, _scn_ret, linewidth=2, color="#e6b800", marker="s", label="Scenario")
    ax5.fill_between(yr_nums, _bl_ret, _scn_ret, alpha=0.15, color="#e6b800")
    for i in range(len(yr_nums)):
        diff = float(_scn_ret[i] - _bl_ret[i])
        if abs(diff) > 0.001:
            mid = (float(_scn_ret[i]) + float(_bl_ret[i])) / 2
            ax5.annotate(f'+{diff:.1%}', xy=(yr_nums[i], mid), fontsize=8, ha="center",
                         color="#e6b800", fontweight="bold")
    ax5.set_xlabel("Year")
    ax5.set_ylabel("Retention Probability")
    ax5.set_title("Retention Curves: Baseline vs Scenario")
    ax5.set_ylim(0, 1.05)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    fig5.tight_layout()
    c5a, c5b = st.columns([10, 1])
    with c5a:
        st.pyplot(fig5)
    with c5b:
        st.download_button("PNG", _fig_to_png(fig5), "retention_curves.png", "image/png", key="dl7")
    plt.close(fig5)

    # ================================================================
    # 8. Total Devices Sold per User vs Service Duration
    # ================================================================
    st.divider()
    st.markdown("##### 8. Devices Sold per User vs Service Duration & LTV")
    st.markdown(
        "Shows the trade-off between device sales volume and LTV as devices last longer. "
        "Longer-lasting devices mean fewer replacements per user (lower device revenue), "
        "but higher retention drives more consumable sales — this chart shows whether the "
        "net effect is positive for total lifetime profit."
    )
    devs_per_user = []
    ltvs_for_devs = []
    for sd in _sd_range:
        dpuy_val = devices_per_user_per_year(float(sd))
        # Total devices over horizon weighted by retention
        _tmp_ret = retention_with_durability(
            np.array(sku_inp.retention[:sku_inp.years], dtype=float),
            sku_inp.baseline_service_years, float(sd), sku_inp.retention_lift_per_year,
        )
        total_devs = float(np.sum(_tmp_ret * dpuy_val))
        devs_per_user.append(total_devs)
        r = ltv_5yr_device_consumables(
            sku_inp, mah_per_pack=_sku_ref.mah_per_pack,
            price_per_pack=_sku_ref.price_per_pack, cogs_per_pack=_sku_ref.cogs_per_pack,
            service_years=float(sd), delta_device_cogs=delta_device_cogs,
        )
        ltvs_for_devs.append(r["ltv"])

    fig7, ax7a = plt.subplots(figsize=(8, 4))
    color_devs = "#1f77b4"
    color_ltv = "#2ca02c"
    ax7a.set_xlabel("Service Duration (years)")
    ax7a.set_ylabel("Total Devices per User (5yr)", color=color_devs)
    ax7a.plot(_sd_range, devs_per_user, linewidth=2, color=color_devs, label="Devices/user")
    ax7a.tick_params(axis="y", labelcolor=color_devs)
    ax7b = ax7a.twinx()
    ax7b.set_ylabel("LTV ($)", color=color_ltv)
    ax7b.plot(_sd_range, ltvs_for_devs, linewidth=2, color=color_ltv, linestyle="--", label="LTV")
    ax7b.tick_params(axis="y", labelcolor=color_ltv)
    # Compute baseline/scenario dots directly to avoid sweep index errors
    _g8_bl_ret = retention_with_durability(
        np.array(sku_inp.retention[:sku_inp.years], dtype=float),
        sku_inp.baseline_service_years, baseline_sd, sku_inp.retention_lift_per_year,
    )
    _g8_bl_dpuy = devices_per_user_per_year(baseline_sd)
    _g8_bl_devs = float(np.sum(_g8_bl_ret * _g8_bl_dpuy))
    _g8_bl_ltv = ltv_5yr_device_consumables(
        sku_inp, mah_per_pack=_sku_ref.mah_per_pack,
        price_per_pack=_sku_ref.price_per_pack, cogs_per_pack=_sku_ref.cogs_per_pack,
        service_years=baseline_sd, delta_device_cogs=0.0,
    )["ltv"]
    _g8_sc_ret = retention_with_durability(
        np.array(sku_inp.retention[:sku_inp.years], dtype=float),
        sku_inp.baseline_service_years, scenario_sd, sku_inp.retention_lift_per_year,
    )
    _g8_sc_dpuy = devices_per_user_per_year(scenario_sd)
    _g8_sc_devs = float(np.sum(_g8_sc_ret * _g8_sc_dpuy))
    _g8_sc_ltv = ltv_5yr_device_consumables(
        sku_inp, mah_per_pack=_sku_ref.mah_per_pack,
        price_per_pack=_sku_ref.price_per_pack, cogs_per_pack=_sku_ref.cogs_per_pack,
        service_years=scenario_sd, delta_device_cogs=delta_device_cogs,
    )["ltv"]
    # Baseline point
    ax7a.scatter([baseline_sd], [_g8_bl_devs], color="#1f77b4", s=80, zorder=5, edgecolors="black", linewidths=1)
    ax7a.annotate(f'Baseline\n{_g8_bl_devs:.1f} devices', xy=(baseline_sd, _g8_bl_devs),
                  xytext=(-70, -20), textcoords="offset points", fontsize=8, fontweight="bold", color="#1f77b4",
                  arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5))
    ax7b.scatter([baseline_sd], [_g8_bl_ltv], color="#1f77b4", s=80, zorder=5, marker="D", edgecolors="black", linewidths=1)
    ax7b.annotate(f'LTV ${_g8_bl_ltv:,.0f}', xy=(baseline_sd, _g8_bl_ltv),
                  xytext=(-70, 15), textcoords="offset points", fontsize=8, fontweight="bold", color="#1f77b4",
                  arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5))
    # Scenario point
    ax7a.scatter([scenario_sd], [_g8_sc_devs], color="#e6b800", s=80, zorder=5, edgecolors="black", linewidths=1)
    ax7a.annotate(f'Scenario\n{_g8_sc_devs:.1f} devices', xy=(scenario_sd, _g8_sc_devs),
                  xytext=(15, -20), textcoords="offset points", fontsize=8, fontweight="bold", color="#e6b800",
                  arrowprops=dict(arrowstyle="->", color="#e6b800", lw=1.5))
    ax7b.scatter([scenario_sd], [_g8_sc_ltv], color="#e6b800", s=80, zorder=5, marker="D", edgecolors="black", linewidths=1)
    ax7b.annotate(f'LTV ${_g8_sc_ltv:,.0f}', xy=(scenario_sd, _g8_sc_ltv),
                  xytext=(15, 15), textcoords="offset points", fontsize=8, fontweight="bold", color="#e6b800",
                  arrowprops=dict(arrowstyle="->", color="#e6b800", lw=1.5))
    ax7a.set_title("Devices per User vs LTV (varying service duration)")
    lines_a, labels_a = ax7a.get_legend_handles_labels()
    lines_b, labels_b = ax7b.get_legend_handles_labels()
    ax7a.legend(lines_a + lines_b, labels_a + labels_b, loc="center right")
    ax7a.grid(True, alpha=0.3)
    fig7.tight_layout()
    c7a, c7b = st.columns([10, 1])
    with c7a:
        st.pyplot(fig7)
    with c7b:
        st.download_button("PNG", _fig_to_png(fig7), "devices_per_user_vs_ltv.png", "image/png", key="dl8")
    plt.close(fig7)

