import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

print("Working directory:", os.getcwd())


# ============================================================
# AUTO-DETECT DATE COLUMN
# ============================================================
def find_date_column(df):
  for col in df.columns:
    c = col.lower().strip()
    if "occur" in c or "date" in c or "time" in c or "timestamp" in c:
      return col
  raise ValueError("No date-like column found.")


# ============================================================
# SPE ARPS HYPERBOLIC DECLINE MODEL
# ============================================================
def hyperbolic_arps(t, qi, Di, b):
  b = np.maximum(b, 1e-5)
  return qi / np.power(1.0 + b * Di * t, 1.0 / b)


# ============================================================
# BUILD MONTHLY DATA (UNIFIED AVERAGE DAILY RATES)
# ============================================================
def build_monthly_rates(df):
  df = df.copy()
  date_col = find_date_column(df)
  df[date_col] = pd.to_datetime(df[date_col])
  df["month"] = df[date_col].dt.to_period("M").dt.to_timestamp()

  df["oil_rate"] = df["oil_rate"].fillna(0)
  df["gas_rate"] = df["gas_rate"].fillna(0)

  df["gas_boe"] = df["gas_rate"] / 6.0
  df["boe_rate"] = df["oil_rate"] + df["gas_boe"]

  monthly = (
      df.groupby(["well_id", "site_id", "month"])
      .agg({
          "boe_rate": "mean",
          "operating_cost": "mean",
          "temperature": "mean",
          "pressure": "mean",
      })
      .reset_index()
      .sort_values(["well_id", "month"])
  )

  return monthly


# ============================================================
# LOAD IoT ANOMALY CSV (IF AVAILABLE)
# ============================================================
def load_iot_anomalies(csv_path="well_sensor_anomalies.csv"):
  if not os.path.exists(csv_path):
    return pd.DataFrame(columns=["well_id", "iot_reliability_percent"])

  anomalies = pd.read_csv(csv_path)
  anomalies["combined_total_anomalies"] = pd.to_numeric(
      anomalies["combined_total_anomalies"], errors="coerce"
  ).fillna(0)
  anomalies["total_readings"] = pd.to_numeric(
      anomalies["total_readings"], errors="coerce"
  ).fillna(1)

  agg_anomalies = (
      anomalies.groupby("well_id")
      .agg({"combined_total_anomalies": "sum", "total_readings": "sum"})
      .reset_index()
  )

  agg_anomalies["iot_reliability_percent"] = (
      1.0
      - (
          agg_anomalies["combined_total_anomalies"]
          / agg_anomalies["total_readings"]
      )
  ) * 100.0
  agg_anomalies["iot_reliability_percent"] = agg_anomalies[
      "iot_reliability_percent"
  ].clip(0, 100)

  return agg_anomalies


# ============================================================
# TOP 5 WELLS BY BOE IN LAST 6 MONTHS
# ============================================================
def get_top5_last6_boe(monthly_df):
  last_month = monthly_df["month"].max()
  cutoff = last_month - pd.DateOffset(months=6)

  last6 = monthly_df[monthly_df["month"] >= cutoff]

  last6_boe = (
      last6.groupby("well_id")["boe_rate"]
      .mean()
      .reset_index()
      .sort_values("boe_rate", ascending=False)
  )

  top5 = last6_boe.head(5)["well_id"].tolist()
  return top5, last6_boe


def save_top5_report(last6_boe, outdir="Top5_Last6_Months"):
  os.makedirs(outdir, exist_ok=True)
  last6_boe.head(5).to_csv(
      os.path.join(outdir, "top5_wells_last_6_months.csv"), index=False
  )
  last6_boe.to_csv(
      os.path.join(outdir, "all_wells_last_6_months_ranked.csv"), index=False
  )


# ============================================================
# FIT DECLINE + SPE PRMS BALANCED COMPOUNDING MONTE CARLO & ECON LIMIT
# ============================================================
def fit_and_forecast_all_wells(
    monthly_df, anomalies_df, boe_price=60.0, net_revenue_interest=0.80
):
  results = {}
  all_wells = monthly_df["well_id"].unique()

  for well in all_wells:
    w = monthly_df[monthly_df["well_id"] == well].sort_values("month")
    if len(w) < 6:
      continue

    first_date = w["month"].min()
    t_years = ((w["month"] - first_date).dt.days.values / 365.0).astype(float)
    q = w["boe_rate"].values

    qi0, Di0, b0 = q[0], 0.8, 1.0

    try:
      popt, _ = curve_fit(
          hyperbolic_arps,
          t_years,
          q,
          p0=[qi0, Di0, b0],
          bounds=([0.0, 0.0, 1e-5], [np.inf, 10.0, 5.0]),
          maxfev=20000,
      )
      qi, Di, b = popt
    except Exception:
      qi, Di, b = q[0], 0.5, 1.0

    future_months = pd.date_range(
        w["month"].max() + pd.offsets.MonthBegin(1), periods=60, freq="MS"
    )
    t_future_years = ((future_months - first_date).days / 365.0).astype(float)

    # Base Deterministic Forecast (2P Median Expectation)
    q_future = hyperbolic_arps(t_future_years, qi, Di, b)
    q_future = np.clip(q_future, 0, None)

    forecast_df = pd.DataFrame(
        {
            "well_id": well,
            "month": future_months,
            "boe_rate_forecast": q_future,
        }
    )

    # --- Operational Risk Adjustment ---
    risk_score = 0.0
    temp = w["temperature"].mean()
    pressure = w["pressure"].mean()

    if temp < 60 or temp > 302:
      risk_score += 0.10
    if pressure < 200 or pressure > 10000:
      risk_score += 0.10

    well_iot = (
        anomalies_df[anomalies_df["well_id"] == well]
        if not anomalies_df.empty
        else pd.DataFrame()
    )
    iot_rel = (
        well_iot["iot_reliability_percent"].values[0]
        if not well_iot.empty and "iot_reliability_percent" in well_iot.columns
        else 100.0
    )

    if iot_rel < 95.0:
      risk_score += (95.0 - iot_rel) / 150.0

    # --- SPE PRMS Standard Economic Limit Calculation ---
    avg_op_cost_monthly = w["operating_cost"].mean()
    if np.isnan(avg_op_cost_monthly) or avg_op_cost_monthly <= 0:
      avg_op_cost_monthly = 9000.0

    avg_op_cost_daily = avg_op_cost_monthly / 30.44
    risk_cost_multiplier = 1.0 + (risk_score * 0.15) + max(
        0.0, (100.0 - iot_rel) / 400.0
    )
    adjusted_op_cost_daily = avg_op_cost_daily * risk_cost_multiplier

    effective_net_price = boe_price * net_revenue_interest
    calculated_econ_limit = (
        adjusted_op_cost_daily / effective_net_price
        if effective_net_price > 0
        else 6.0
    )

    is_horizontal = "H" in str(well).upper() or "HZ" in str(well).upper()
    regional_min_floor = 5.0 if is_horizontal else 2.5
    econ_limit = max(calculated_econ_limit, regional_min_floor)

    # --- SPE PRMS Balanced Compounding Monte Carlo Simulation ---
    np.random.seed(42)
    n_sims = 500
    sim_matrix = np.zeros((n_sims, len(future_months)))

    qi_sims = np.random.normal(q_future[0], q_future[0] * 0.01, n_sims)
    Di_sims = np.random.normal(Di, Di * 0.15, n_sims)
    b_sims = np.random.normal(b, 0.05, n_sims)

    dt = 1.0 / 12.0
    annual_volatility = 0.09  # Balanced, realistic compounding volatility

    for i in range(n_sims):
      sim_curve = hyperbolic_arps(
          t_future_years - t_future_years[0],
          qi_sims[i],
          max(Di_sims[i], 0.01),
          max(b_sims[i], 0.01),
      )

      shocks = np.random.normal(0, np.sqrt(dt), len(future_months))
      cum_shocks = np.cumsum(shocks)
      time_indices = np.arange(len(future_months))
      gradual_scale = np.sqrt(time_indices) / np.sqrt(len(future_months) - 1)

      compounding_factor = np.exp(
          annual_volatility * cum_shocks * gradual_scale
          - 0.5
          * (annual_volatility**2)
          * (time_indices * dt)
          * (gradual_scale**2)
      )

      sim_curve_expanded = sim_curve * compounding_factor
      sim_matrix[i, :] = np.clip(sim_curve_expanded, 0, None)

    # SPE PRMS Reserves Mapping:
    forecast_df["boe_3p"] = np.percentile(
        sim_matrix, 90, axis=0
    )  # P10 / Optimistic
    forecast_df["boe_2p"] = np.percentile(
        sim_matrix, 50, axis=0
    )  # P50 / Base Median
    forecast_df["boe_1p"] = np.percentile(
        sim_matrix, 10, axis=0
    )  # P90 / Conservative

    results[well] = {
        "history": w,
        "params": (qi, Di, b),
        "forecast": forecast_df,
        "risk_score": risk_score,
        "iot_reliability": iot_rel,
        "econ_limit": econ_limit,
    }

  return results


# ============================================================
# PLOT WELLS WITH SPE PRMS RESERVES ENVELOPE
# ============================================================
def plot_all_wells_prms(results, outdir="wells_curves_prms"):
  os.makedirs(outdir, exist_ok=True)

  for well, data in results.items():
    w = data["history"]
    qi, Di, b = data["params"]
    f = data["forecast"]
    iot_rel = data["iot_reliability"]
    econ_limit = data["econ_limit"]

    first_date = w["month"].min()
    t_years = ((w["month"] - first_date).dt.days.values / 365.0).astype(float)
    q_fit = np.clip(hyperbolic_arps(t_years, qi, Di, b), 0, None)

    plt.figure(figsize=(12, 6))
    plt.plot(w["month"], w["boe_rate"], "o-", label="Actual BOE/day")
    plt.plot(w["month"], q_fit, "--", label="Arps Fitted Decline")
    plt.plot(
        f["month"], f["boe_2p"], "r--", label="2P Forecast (P50 Base Case)"
    )
    plt.fill_between(
        f["month"],
        f["boe_1p"],
        f["boe_3p"],
        color="purple",
        alpha=0.25,
        label=f"1P - 3P Reserves Envelope (P90-P10)",
    )
    plt.axhline(
        y=econ_limit,
        color="g",
        linestyle="-",
        linewidth=2,
        label=f"SPE Econ Limit ({econ_limit:.1f} BOE/d)",
    )

    plt.title(
        f"Well Forecast — {well} | SPE PRMS Balanced Compounding Reserves &"
        " Econ Limit"
    )
    plt.xlabel("Month")
    plt.ylabel("BOE/day")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{well}_prms_forecast.png"))
    plt.close()


def plot_specific_wells_prms(
    results, target_wells, outdir="Top5_Curves_PRMS"
):
  os.makedirs(outdir, exist_ok=True)

  for well in target_wells:
    if well not in results:
      continue

    data = results[well]
    w = data["history"]
    qi, Di, b = data["params"]
    f = data["forecast"]
    econ_limit = data["econ_limit"]

    first_date = w["month"].min()
    t_years = ((w["month"] - first_date).dt.days.values / 365.0).astype(float)
    q_fit = np.clip(hyperbolic_arps(t_years, qi, Di, b), 0, None)

    plt.figure(figsize=(12, 6))
    plt.plot(w["month"], w["boe_rate"], "o-", label="Actual BOE/day")
    plt.plot(w["month"], q_fit, "--", label="Arps Fitted Decline")
    plt.plot(
        f["month"], f["boe_2p"], "r--", label="2P Forecast (P50 Base Case)"
    )
    plt.fill_between(
        f["month"],
        f["boe_1p"],
        f["boe_3p"],
        color="purple",
        alpha=0.25,
        label="1P - 3P Reserves Envelope (P90-P10)",
    )
    plt.axhline(
        y=econ_limit,
        color="g",
        linestyle="-",
        linewidth=2,
        label=f"SPE Econ Limit ({econ_limit:.1f} BOE/d)",
    )

    plt.title(
        f"Top 5 Well Forecast — {well} | SPE PRMS Balanced Compounding"
        " Framework"
    )
    plt.xlabel("Month")
    plt.ylabel("BOE/day")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{well}_prms_forecast.png"))
    plt.close()


# ============================================================
# RUN PIPELINE
# ============================================================
if __name__ == "__main__":
  df = pd.read_csv("cleaned_unified_master.csv")
  anomalies_df = load_iot_anomalies("well_sensor_anomalies.csv")

  monthly = build_monthly_rates(df)

  top5, last6_boe_table = get_top5_last6_boe(monthly)
  save_top5_report(last6_boe_table, outdir="Top5_Last6_Months")

  results = fit_and_forecast_all_wells(
      monthly, anomalies_df, boe_price=60.0, net_revenue_interest=0.80
  )

  plot_all_wells_prms(results, outdir="wells_curves_prms")
  plot_specific_wells_prms(results, top5, outdir="Top5_Curves_PRMS")

  print("\nSPE PRMS Balanced Compounding pipeline executed successfully!")
  print("- Curves saved to 'wells_curves_prms' and 'Top5_Curves_PRMS'")