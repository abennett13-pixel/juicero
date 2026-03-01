# model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class Inputs:
    # Horizon / discount
    years: int = 5
    discount_rate: float = 0.15  # annual

    # Usage (mL-based)
    mah_per_month: float = 60.0

    # Retention curve (Year 1..5)
    # Interpreted as probability still active in each year
    retention: Tuple[float, float, float, float, float] = (0.70, 0.50, 0.40, 0.32, 0.25)

    # Durability / service duration
    baseline_service_years: float = 1.0

    # Sensitivity: retention lift per +1 year service duration (applied to all years)
    # Spreadsheet durability-only heatmap used a linear lift.
    retention_lift_per_year: float = 0.05

    # Juice device economics
    device_asp: float = 12.0
    device_cogs_baseline: float = 8.0
    device_other_costs: float = 0.0  # shipping, warranty accrual, etc.
    delta_device_cogs: float = 0.0   # scenario: + cost investment

    # Scenario service duration
    scenario_service_years: float = 1.0

    # Single SKU consumable economics
    mah_per_pack_single: float = 2.0
    price_per_pack_single: float = 6.00
    cogs_per_pack_single: float = 1.55

    # CAC
    cac: float = 30.0

    # Mix / complexity
    annual_complexity_cost: float = 250_000.0

    # Size ladder / demand
    elasticity: float = -2.00  # constant elasticity on blended price per mL


@dataclass
class SKU:
    name: str
    mah_per_pack: float
    price_per_pack: float
    cogs_per_pack: float

    @property
    def margin_per_pack(self) -> float:
        return self.price_per_pack - self.cogs_per_pack

    @property
    def price_per_ml(self) -> float:
        return self.price_per_pack / self.mah_per_pack if self.mah_per_pack else 0.0

    @property
    def cogs_per_ml(self) -> float:
        return self.cogs_per_pack / self.mah_per_pack if self.mah_per_pack else 0.0

    @property
    def margin_per_ml(self) -> float:
        return self.margin_per_pack / self.mah_per_pack if self.mah_per_pack else 0.0


def discount_factors(years: int, r: float) -> np.ndarray:
    # Year index 1..years
    t = np.arange(1, years + 1)
    return 1.0 / ((1.0 + r) ** t)


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, np.maximum(0.0, x))


def retention_with_durability(base_ret: np.ndarray, baseline_sd: float, sd: float, lift_per_year: float) -> np.ndarray:
    lift = lift_per_year * (sd - baseline_sd)
    return clamp01(base_ret + lift)


def devices_per_user_per_year(service_years: float) -> float:
    return 0.0 if service_years <= 0 else 1.0 / service_years


def annual_packs(mah_per_month: float, mah_per_pack: float) -> float:
    if mah_per_pack <= 0:
        return 0.0
    return (mah_per_month / mah_per_pack) * 12.0


def demand_with_elasticity(
    base_mah_per_month: float,
    observed_price_per_ml: float,
    reference_price_per_ml: float,
    elasticity: float,
) -> float:
    if base_mah_per_month <= 0:
        return 0.0
    if observed_price_per_ml <= 0 or reference_price_per_ml <= 0:
        return base_mah_per_month
    return base_mah_per_month * (observed_price_per_ml / reference_price_per_ml) ** elasticity


def blended_from_mix(skus: List[SKU], shares: List[float]) -> Dict[str, float]:
    """
    shares = volume share in mL (should sum to 1). We use it as weights for mL-based blending.
    For pack-level blending we use the same weights as in spreadsheet (approximation):
      blended_ml_per_pack = sum(share_i * ml_per_pack_i)
      blended_price_per_pack = sum(share_i * price_per_pack_i)
      blended_cogs_per_pack = sum(share_i * cogs_per_pack_i)
    """
    w = np.array(shares, dtype=float)
    if w.sum() == 0:
        w = np.ones(len(skus)) / len(skus)
    w = w / w.sum()

    mah = np.array([s.mah_per_pack for s in skus], dtype=float)
    p = np.array([s.price_per_pack for s in skus], dtype=float)
    c = np.array([s.cogs_per_pack for s in skus], dtype=float)

    blended_mah = float(np.sum(w * mah))
    blended_p = float(np.sum(w * p))
    blended_c = float(np.sum(w * c))
    blended_margin = blended_p - blended_c

    return {
        "blended_ml_per_pack": blended_mah,
        "blended_price_per_pack": blended_p,
        "blended_cogs_per_pack": blended_c,
        "blended_margin_per_pack": blended_margin,
        "blended_price_per_mL": (blended_p / blended_mah) if blended_mah else 0.0,
        "blended_margin_per_mL": (blended_margin / blended_mah) if blended_mah else 0.0,
    }


def ltv_5yr_device_consumables(
    inp: Inputs,
    mah_per_pack: float,
    price_per_pack: float,
    cogs_per_pack: float,
    service_years: float,
    delta_device_cogs: float = 0.0,
    durability_only_retention: bool = True,
) -> Dict[str, float]:
    """
    Computes discounted 5-year totals for revenue, profit, and LTV contribution.

    durability_only_retention=True means retention lift only comes from service duration (no direct delight lift from cogs).
    """
    years = inp.years
    pv = discount_factors(years, inp.discount_rate)

    base_ret = np.array(inp.retention[:years], dtype=float)
    ret = retention_with_durability(base_ret, inp.baseline_service_years, service_years, inp.retention_lift_per_year)

    packs_y = annual_packs(inp.mah_per_month, mah_per_pack)

    # Consumables
    cons_rev_y = packs_y * price_per_pack
    cons_profit_y = packs_y * (price_per_pack - cogs_per_pack)

    # Juice device
    dev_margin = inp.device_asp - (inp.device_cogs_baseline + delta_device_cogs) - inp.device_other_costs
    dev_rev_per_device = inp.device_asp
    dpuy = devices_per_user_per_year(service_years)
    dev_rev_y = dev_rev_per_device * dpuy
    dev_profit_y = dev_margin * dpuy

    # Totals per year with retention
    rev_y = (cons_rev_y + dev_rev_y) * ret
    prof_y = (cons_profit_y + dev_profit_y) * ret

    pv_rev = float(np.sum(pv * rev_y))
    pv_prof = float(np.sum(pv * prof_y))

    # LTV definition in this model = PV profit (gross contribution before CAC)
    ltv = pv_prof

    # Payback: earliest year where cumulative discounted profit >= 0 (or >= device loss threshold)
    cum = np.cumsum(pv * prof_y)
    payback_year = next((i + 1 for i, v in enumerate(cum) if v >= 0), None)

    return {
        "pv_revenue": pv_rev,
        "pv_profit": pv_prof,
        "ltv": ltv,
        "avg_margin_pct": (pv_prof / pv_rev) if pv_rev else 0.0,
        "payback_year": float(payback_year) if payback_year is not None else np.nan,
        "retention_12mo": float(ret[1]) if years >= 2 else float(ret[-1]),
        "ltv_cac": (ltv / inp.cac) if inp.cac else np.nan,
    }


def mix_delta_ltv_and_roi(
    inp: Inputs,
    single: SKU,
    mix_skus: List[SKU],
    shares: List[float],
    service_years: float,
    delta_device_cogs: float,
    apply_price_elasticity: bool = False,
    reference_price_per_ml: Optional[float] = None,
) -> Dict[str, float]:
    # Single baseline uses single SKU economics
    b = blended_from_mix(mix_skus, shares)

    single_inp = inp
    mix_inp = inp
    if apply_price_elasticity:
        ref_price = reference_price_per_ml if reference_price_per_ml is not None else single.price_per_ml
        single_mah_per_month = demand_with_elasticity(
            inp.mah_per_month,
            single.price_per_ml,
            ref_price,
            inp.elasticity,
        )
        mix_mah_per_month = demand_with_elasticity(
            inp.mah_per_month,
            b["blended_price_per_mL"],
            ref_price,
            inp.elasticity,
        )
        single_inp = Inputs(**{**inp.__dict__, "mah_per_month": float(single_mah_per_month)})
        mix_inp = Inputs(**{**inp.__dict__, "mah_per_month": float(mix_mah_per_month)})

    single_res = ltv_5yr_device_consumables(
        single_inp,
        mah_per_pack=single.mah_per_pack,
        price_per_pack=single.price_per_pack,
        cogs_per_pack=single.cogs_per_pack,
        service_years=service_years,
        delta_device_cogs=delta_device_cogs,
    )

    mix_res = ltv_5yr_device_consumables(
        mix_inp,
        mah_per_pack=b["blended_ml_per_pack"],
        price_per_pack=b["blended_price_per_pack"],
        cogs_per_pack=b["blended_cogs_per_pack"],
        service_years=service_years,
        delta_device_cogs=delta_device_cogs,
    )

    pv_cost = inp.annual_complexity_cost * float(np.sum(discount_factors(inp.years, inp.discount_rate)))
    delta_ltv = mix_res["ltv"] - single_res["ltv"]
    net_value = delta_ltv - pv_cost

    return {
        "single_ltv": single_res["ltv"],
        "mix_ltv": mix_res["ltv"],
        "delta_ltv": delta_ltv,
        "pv_complexity_cost": pv_cost,
        "net_value": net_value,
        "roi": (delta_ltv / pv_cost) if pv_cost else np.nan,
        "breakeven_annual_complexity_cost": (delta_ltv / float(np.sum(discount_factors(inp.years, inp.discount_rate)))) if inp.years else np.nan,
        "mix_avg_margin_pct": mix_res["avg_margin_pct"],
        "mix_margin_per_mL": b["blended_margin_per_mL"],
        "mix_price_per_mL": b["blended_price_per_mL"],
        "single_ml_per_month": float(single_inp.mah_per_month),
        "mix_ml_per_month": float(mix_inp.mah_per_month),
    }


def mix_heatmap(
    inp: Inputs,
    single: SKU,
    skus: List[SKU],  # [Small, Medium, Large]
    step: float = 0.10,
    service_years: Optional[float] = None,
    delta_device_cogs: float = 0.0,
    output: str = "delta_ltv",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Grid over small share (rows) and large share (cols), medium = 1 - small - large.
    Returns: small_axis, large_axis, grid (NaN for infeasible)
    output in {"delta_ltv", "margin_per_mL", "price_per_mL", "net_value"}
    """
    if service_years is None:
        service_years = inp.scenario_service_years

    axis = np.arange(0.0, 1.0 + 1e-9, step)
    grid = np.full((len(axis), len(axis)), np.nan)

    pv_cost = inp.annual_complexity_cost * float(np.sum(discount_factors(inp.years, inp.discount_rate)))

    for i, s in enumerate(axis):
        for j, l in enumerate(axis):
            m = 1.0 - s - l
            if m < -1e-9:
                continue
            shares = [s, max(0.0, m), l]
            res = mix_delta_ltv_and_roi(
                inp=inp,
                single=single,
                mix_skus=skus,
                shares=shares,
                service_years=service_years,
                delta_device_cogs=delta_device_cogs,
            )
            if output == "delta_ltv":
                grid[i, j] = res["delta_ltv"]
            elif output == "net_value":
                grid[i, j] = res["delta_ltv"] - pv_cost
            elif output == "margin_per_mL":
                grid[i, j] = res["mix_margin_per_mL"]
            elif output == "price_per_mL":
                grid[i, j] = res["mix_price_per_mL"]
            else:
                raise ValueError(f"Unknown output: {output}")

    return axis, axis, grid


def device_tradeoff_heatmap(
    inp: Inputs,
    sku: SKU,
    service_axis: np.ndarray,
    delta_cogs_axis: np.ndarray,
    baseline_service_years: Optional[float] = None,
    baseline_delta_device_cogs: float = 0.0,
    output: str = "delta_ltv",
) -> np.ndarray:
    """
    Grid over service duration (rows) and delta device COGS (cols).
    output in {"delta_ltv", "scenario_ltv"}
    """
    if baseline_service_years is None:
        baseline_service_years = inp.baseline_service_years

    base = ltv_5yr_device_consumables(
        inp,
        mah_per_pack=sku.mah_per_pack,
        price_per_pack=sku.price_per_pack,
        cogs_per_pack=sku.cogs_per_pack,
        service_years=baseline_service_years,
        delta_device_cogs=baseline_delta_device_cogs,
    )["ltv"]

    grid = np.full((len(service_axis), len(delta_cogs_axis)), np.nan)
    for i, service_years in enumerate(service_axis):
        for j, delta_cogs in enumerate(delta_cogs_axis):
            scenario = ltv_5yr_device_consumables(
                inp,
                mah_per_pack=sku.mah_per_pack,
                price_per_pack=sku.price_per_pack,
                cogs_per_pack=sku.cogs_per_pack,
                service_years=float(service_years),
                delta_device_cogs=float(delta_cogs),
            )["ltv"]
            if output == "scenario_ltv":
                grid[i, j] = scenario
            elif output == "delta_ltv":
                grid[i, j] = scenario - base
            else:
                raise ValueError(f"Unknown output: {output}")
    return grid


def ladder_objective_ltv(
    inp: Inputs,
    skus: List[SKU],          # [Small, Medium, Large]
    shares: List[float],      # volume shares (mL) summing to 1
    price_per_ml_vec: np.ndarray,  # [p_small_mL, p_med_mL, p_large_mL]
    service_years: float,
    delta_device_cogs: float,
) -> float:
    # Apply ladder prices to compute implied price/pack and margin/pack
    adj = []
    for sku, pmah in zip(skus, price_per_ml_vec):
        price_pack = float(pmah) * sku.mah_per_pack
        adj.append(SKU(sku.name, sku.mah_per_pack, price_pack, sku.cogs_per_pack))

    b = blended_from_mix(adj, shares)

    # Demand: adjust mL/month by elasticity relative to reference (single SKU)
    ref_price_mah = inp.price_per_pack_single / inp.mah_per_pack_single if inp.mah_per_pack_single else 0.0
    blended_price_mah = b["blended_price_per_mL"]
    if ref_price_mah > 0 and blended_price_mah > 0:
        mah_month = inp.mah_per_month * (blended_price_mah / ref_price_mah) ** inp.elasticity
    else:
        mah_month = inp.mah_per_month

    # Compute LTV with adjusted usage
    tmp = Inputs(**{**inp.__dict__, "mah_per_month": float(mah_month)})
    res = ltv_5yr_device_consumables(
        tmp,
        mah_per_pack=b["blended_ml_per_pack"],
        price_per_pack=b["blended_price_per_pack"],
        cogs_per_pack=b["blended_cogs_per_pack"],
        service_years=service_years,
        delta_device_cogs=delta_device_cogs,
    )
    return res["ltv"]
