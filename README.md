# Juicero Juice Press — LTV Calculator

A Streamlit app for modeling the lifetime value (LTV) of a "device + consumable"
business — like a Juicero juice press (durable device) sold alongside juice bags
(consumables). It lets you explore how pricing, device durability, retention, and
consumable mix decisions affect discounted profit over a multi-year horizon.

Two files make up the whole app:

- **`model.py`** — pure calculation functions (no UI). This is the financial model.
- **`app.py`** — the Streamlit UI: sidebar inputs, seven tabs of analysis, and all charts.

## Running it

```bash
pip install streamlit numpy pandas matplotlib scipy
streamlit run app.py
```

This opens the app at `http://localhost:8501`. `scipy` is optional — everything
works without it except the Size Ladder Optimizer tab, which needs
`scipy.optimize.minimize`.

## The core idea

Every "customer" adopts one or more **devices** over time and buys **consumable
bags** to use with them. The model projects revenue and profit for both the device
and its consumables year-by-year, discounts each year back to present value (DCF),
and sums a multi-year horizon (3–10 years, default 5) into a single LTV number.

```
LTV = Σ (year 1..N) [ (consumable profit + device profit) × retention(year) × discount_factor(year) ]
```

- **Consumable profit** in a year = (bags bought per year) × (price − COGS per bag).
  Bags bought per year is derived from mL consumed per month ÷ mL per bag × 12.
- **Device profit** in a year = (device ASP − device COGS − other costs) × (devices
  bought per user that year). Devices bought per year = `1 / service_years` — a
  device that lasts 2 years implies a customer buys a new one every other year, so
  device revenue/profit is "smeared" across years at that rate.
- **Retention** is the probability a customer is still active in a given year. It's
  a 5-value curve (Year 1–5) set in the sidebar, optionally boosted by a
  **durability lift**: longer service duration than the baseline durability adds
  `retention_lift_per_year × (service_years − baseline_service_years)` to every
  year's retention (clamped to [0, 1]).
- **Discounting** uses a standard DCF factor `1 / (1 + discount_rate)^year`.

`ltv_5yr_device_consumables()` in `model.py` is the single function that computes
all of this for one SKU/scenario and returns LTV, average margin %, 12-month
retention, payback year, and LTV:CAC.

## Sidebar — global inputs (apply everywhere)

- **Single SKU** — mL per bag, retail price per mL, COGS per bag. Defines the
  default consumable economics used throughout the app.
- **Usage** — mL consumed per month per customer; drives how many bags/year they buy.
- **Device Duration Scenario** — baseline vs. scenario "bags per device" (how many
  bags a customer buys before their device needs replacing). This is converted to
  **service duration in years** (`bags_per_device / annual_packs`) since the model
  works in years internally. Also sets the retention lift per +1 year of durability
  and the scenario's Δ device COGS (the cost of investing in a more durable device).
- **Juice Device Economics** — device ASP, baseline COGS, other costs (shipping,
  warranty, etc.).
- **Complexity** — annual cost of supporting multiple SKUs (used by the Mix tabs).
- **Retention** — Year 1–5 retention sliders (probability still active).
- **Demand (ladder optimizer)** — price elasticity of mL demand vs. blended price/mL.
- **Global Settings** — horizon length, annual discount rate, CAC.

Changing any of these instantly recalculates every tab (Streamlit reruns the whole
script top-to-bottom on each input change).

## Tabs

### 1. Single SKU
The main scenario view for one consumable SKU. Shows headline KPIs (baseline vs.
scenario LTV, margin %, 12-month retention, LTV:CAC, bags sold/device, payback
year) with deltas, then a set of charts, all driven by the same baseline/scenario
comparison (no extra device COGS vs. your Δ COGS investment):

- **Average Profit Margin % vs Service Duration** — margin curve across 0–2 years
  of service duration, with dots marking your current baseline and scenario.
- **Long-Term Value vs Service Duration** — same idea, LTV ($) on the y-axis.
- **Average Profit Margin % vs Consumables per Durable** and **Long-Term Value vs
  Consumables per Durable** — the same two curves, but with the x-axis expressed in
  bags-per-device ("Consumables per Durable") instead of years, since
  `bags_per_device = service_years × annual_packs` is a direct linear conversion.
  Dot labels show the bags-per-device count alongside the metric value.
- **Bags Sold per Device vs LTV** — LTV as service duration (expressed in bags/device)
  varies from 0.5 to 5 years.
- **Save current scenario** — snapshots the current baseline/scenario KPIs into a
  table (`st.session_state["saved_scenarios"]`) so you can compare multiple
  configurations side-by-side. Labels are editable in place; scenarios can be cleared.

### 2. Consumable Mix (2–3)
Model a 3-size ladder (Small/Medium/Large) sold as a mix, and compare it to a
single-SKU baseline. You set mL/bag, price/mL, and COGS/bag for each size plus its
share of total volume (shares are normalized to sum to 1 internally). Output:
Single SKU LTV vs. Mix LTV, the annual complexity cost of running multiple SKUs
(discounted over the horizon), net value (ΔLTV − complexity cost), ROI, and the
breakeven annual complexity cost at which the mix stops paying for itself.

### 3. Mix Heatmaps
Two heatmaps:

- **Small/Large share grid** — sweeps small-volume-share (rows) × large-volume-share
  (columns), with medium implied as the remainder, showing ΔLTV, net value, blended
  margin/mL, or blended price/mL at each mix point. Infeasible cells (medium share
  would go negative) are blank.
- **Δ Device COGS vs Δ Service Duration tradeoff** — sweeps a device durability
  investment (Δ COGS) against a change in service duration, showing either ΔLTV vs.
  baseline or absolute scenario LTV at each combination. This is the "is it worth
  spending more on a longer-lasting device?" chart.

### 4. Size Ladder Optimizer
Uses `scipy.optimize.minimize` (SLSQP) to solve for the price/mL of 2 or 3 sizes
that maximizes 5-year discounted LTV, given each size's mix share, elasticity of
demand, and optional monotonic pricing constraint (Small ≥ Medium ≥ Large price/mL).

### 5. DCF LTV Analysis
A year-by-year breakdown table for the Single SKU scenario: bag revenue/profit,
device revenue/profit, retention-weighted totals, discount factor, and cumulative
discounted profit for each year of the horizon — the full arithmetic behind the
single LTV number shown on Tab 1.

### 6. Investment Strategy Graphs
Nine charts built specifically to answer "is investing in a more durable device
worth it?" using the Single SKU inputs and current sidebar scenario:

1. Average Profit Margin % — Baseline vs Scenario
2. Profit Margin Delta Heatmap — Δ Device COGS vs Δ Service Duration
3. LTV vs Δ Device COGS
4. LTV:CAC Ratio vs Service Duration
5. Payback Period vs Service Duration
6. Cumulative Discounted Profit — Baseline vs Scenario
7. Retention Curves — Baseline vs Scenario
8. Devices Sold per User vs Service Duration & LTV
9. Total Cost of Ownership (TCO) Over Time — Cheap vs Premium Device

### 7. Compare Scenarios
Two fully independent Single SKU / Usage / Device Duration / Device Economics
configurations (A and B), sharing the sidebar's retention curves, discount rate,
CAC, complexity cost, and elasticity. Renders LTV vs Service Duration and Average
Profit Margin % vs Service Duration curves for A and B on the same axes so you can
compare two strategies directly.

## Key modeling assumptions worth knowing

- **LTV = present value of gross profit** (device + consumables), not revenue, and
  is computed *before* CAC — LTV:CAC is reported separately as a ratio.
- **Retention only shifts with service duration**, not directly with device COGS —
  spending more on a device only helps retention/LTV insofar as it buys more years
  of service duration (`durability_only_retention=True` in `model.py`).
- **Payback year** is the first year where cumulative discounted profit ≥ 0; shown
  as N/A if the horizon ends before that happens.
- Mix and heatmap tabs read Single-SKU/size-ladder values out of
  `st.session_state` by widget label, so editing those widgets on one tab updates
  the values reused on other tabs.
