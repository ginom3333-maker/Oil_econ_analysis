import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
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
# HYPERBOLIC ARPS MODEL
# ============================================================

def hyperbolic_arps(t, qi, Di, b):
    # Guard against b near zero to prevent ZeroDivisionError
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

    # NOTE: If gas_rate is logged in SCF/day (instead of MSCF/day), 
    # change 6.0 to 6000.0 below to prevent 1000x gas inflation.
    df["gas_boe"] = df["gas_rate"] / 6.0
    df["boe_rate"] = df["oil_rate"] + df["gas_boe"]

    # Calculate average daily rates per month per well
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
# LOAD IoT ANOMALY CSV
# ============================================================

def load_iot_anomalies(csv_path="well_sensor_anomalies.csv"):
    anomalies = pd.read_csv(csv_path)

    anomalies["iot_reliability_percent"] = (
        1.0 - (anomalies["combined_total_anomalies"] / anomalies["total_readings"])
    ) * 100.0

    return anomalies

# ============================================================
# TOP 5 WELLS BY BOE IN LAST 6 MONTHS
# ============================================================

def get_top5_last6_boe(monthly_df):
    last_month = monthly_df["month"].max()
    cutoff = last_month - pd.offsets.MonthBegin(6)

    last6 = monthly_df[monthly_df["month"] >= cutoff]

    last6_boe = (
        last6.groupby("well_id")["boe_rate"]
        .mean()
        .reset_index()
        .sort_values("boe_rate", ascending=False)
    )

    top5 = last6_boe.head(5)["well_id"].tolist()
    return top5, last6_boe

# ============================================================
# FIT DECLINE + FORECAST 5 YEARS FOR TOP 5 WELLS
# ============================================================

def fit_and_forecast_top5(monthly_df, top5):
    results = {}

    for well in top5:
        w = monthly_df[monthly_df["well_id"] == well].sort_values("month")
        if len(w) < 6:
            continue

        first_date = w["month"].min()
        t_years = ((w["month"] - first_date).dt.days.values / 365.0).astype(float)
        q = w["boe_rate"].values

        qi0, Di0, b0 = q[0], 0.8, 1.0

        # Set lower bound for b to 1e-5 to prevent divide-by-zero
        popt, _ = curve_fit(
            hyperbolic_arps,
            t_years,
            q,
            p0=[qi0, Di0, b0],
            bounds=([0.0, 0.0, 1e-5], [np.inf, 10.0, 5.0]),
            maxfev=20000
        )
        qi, Di, b = popt

        future_months = pd.date_range(
            w["month"].max() + pd.offsets.MonthBegin(1),
            periods=60,
            freq="MS"
        )
        
        # FIXED BUG: Accessing .days directly on DatetimeIndex subtraction
        t_future_years = ((future_months - first_date).days / 365.0).values.astype(float)

        q_future = hyperbolic_arps(t_future_years, qi, Di, b)
        q_future = np.array(q_future, dtype=float)
        q_future[q_future < 0] = 0

        forecast_df = pd.DataFrame({
            "well_id": well,
            "month": future_months,
            "boe_rate_forecast": q_future
        })

        # Risk score based on temp & pressure
        risk_score = 0.0
        temp = w["temperature"].mean()
        pressure = w["pressure"].mean()

        if temp < 60: risk_score += 0.25
        if temp > 302: risk_score += 0.25
        if pressure < 200: risk_score += 0.25
        if pressure > 10000: risk_score += 0.25

        qi_risk = qi * (1.0 - 0.10 * risk_score)
        Di_risk = Di * (1.0 + risk_score)

        q_future_risk = hyperbolic_arps(t_future_years, qi_risk, Di_risk, b)
        q_future_risk = np.array(q_future_risk, dtype=float)
        q_future_risk[q_future_risk < 0] = 0

        forecast_df["boe_rate_risk"] = q_future_risk

        results[well] = {
            "history": w,
            "params": (qi, Di, b),
            "params_risk": (qi_risk, Di_risk, b),
            "forecast": forecast_df,
            "risk_score": risk_score
        }

    return results

# ============================================================
# PLOT TOP 5 WELLS
# ============================================================

def plot_top5(results, outdir="top5_last6_boe"):
    os.makedirs(outdir, exist_ok=True)

    for well, data in results.items():
        w = data["history"]
        qi, Di, b = data["params"]
        f = data["forecast"]

        first_date = w["month"].min()
        t_years = ((w["month"] - first_date).dt.days.values / 365.0).astype(float)
        q_fit = hyperbolic_arps(t_years, qi, Di, b)
        q_fit = np.array(q_fit, dtype=float)
        q_fit[q_fit < 0] = 0

        plt.figure(figsize=(12, 6))
        plt.plot(w["month"], w["boe_rate"], "o-", label="Actual BOE/day")
        plt.plot(w["month"], q_fit, "--", label="Fitted Decline")
        plt.plot(f["month"], f["boe_rate_forecast"], "r--", label="Forecast")
        plt.plot(f["month"], f["boe_rate_risk"], "m--", label="Risk Forecast")
        plt.title(f"Top 5 BOE — {well}")
        plt.xlabel("Month")
        plt.ylabel("BOE/day")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{well}_forecast.png"))
        plt.close()

# ============================================================
# SITE-LEVEL PRODUCTION EFFICIENCY (FIXED)
# ============================================================

def compute_site_efficiency(monthly_df, anomalies_df):
    """
    Computes site performance based on average daily BOE rates per well
    to avoid 10x inflation caused by summing daily rates.
    """
    # 1. Average monthly BOE rate per well across its active months
    well_boe_avg = (
        monthly_df.groupby(["site_id", "well_id"])["boe_rate"]
        .mean()
        .reset_index()
        .rename(columns={"boe_rate": "boe_avg_per_well"})
    )

    # 2. Site BOE average = mean of its wells' average daily rates
    site_boe_avg = (
        well_boe_avg.groupby("site_id")["boe_avg_per_well"]
        .mean()
        .reset_index()
        .rename(columns={"boe_avg_per_well": "boe_avg"})
    )

    # 3. Merge IoT reliability per well
    merged_iot = well_boe_avg.merge(
        anomalies_df[["well_id", "iot_reliability_percent"]],
        on="well_id",
        how="left"
    )

    # 4. Site IoT reliability = average reliability of site's wells
    site_iot = (
        merged_iot.groupby("site_id")["iot_reliability_percent"]
        .mean()
        .reset_index()
        .rename(columns={"iot_reliability_percent": "iot_percent"})
    )

    # 5. Combine BOE + IoT
    site_stats = site_boe_avg.merge(site_iot, on="site_id")

    # Final score weighting
    site_stats["score"] = site_stats["boe_avg"] * (site_stats["iot_percent"] / 100.0)

    return site_stats.sort_values("score", ascending=False)

# ============================================================
# SAVE SITE OUTPUT
# ============================================================

def save_site_efficiency_outputs(site_eff, monthly_df, anomalies_df, outdir="Sites_High_Efficiency"):
    os.makedirs(outdir, exist_ok=True)

    site_eff.to_csv(f"{outdir}/site_efficiency_ranked.csv", index=False)

    for _, row in site_eff.iterrows():
        site_id = row["site_id"]
        site_dir = os.path.join(outdir, site_id)
        os.makedirs(site_dir, exist_ok=True)

        row.to_frame().T.to_csv(f"{site_dir}/site_summary.csv", index=False)

        wells = monthly_df[monthly_df["site_id"] == site_id]["well_id"].unique()
        pd.DataFrame({"well_id": wells}).to_csv(f"{site_dir}/wells_in_site.csv", index=False)

        site_anomalies = anomalies_df[anomalies_df["well_id"].isin(wells)]
        site_anomalies.to_csv(f"{site_dir}/iot_reliability_by_well.csv", index=False)

        site_monthly = monthly_df[monthly_df["site_id"] == site_id]
        site_monthly.to_csv(f"{site_dir}/monthly_boe_by_well.csv", index=False)

# ============================================================
# SAVE IoT RELIABILITY FILE
# ============================================================

def save_iot_reliability_file(anomalies_df, outdir="IoT_Reliability"):
    os.makedirs(outdir, exist_ok=True)
    anomalies_df.to_csv(f"{outdir}/iot_reliability_percent.csv", index=False)

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    df = pd.read_csv("cleaned_unified_master.csv")
    anomalies_df = load_iot_anomalies("well_sensor_anomalies.csv")

    # Build monthly rates first to serve as single source of truth
    monthly = build_monthly_rates(df)

    # Pass pre-aggregated monthly DataFrame to site efficiency calculation
    site_eff = compute_site_efficiency(monthly, anomalies_df)
    save_site_efficiency_outputs(site_eff, monthly, anomalies_df)

    save_iot_reliability_file(anomalies_df)

    top5, last6_boe_table = get_top5_last6_boe(monthly)
    results = fit_and_forecast_top5(monthly, top5)
    plot_top5(results)

    print("\nDone: IoT reliability + monthly BOE averages + top 5 forecasts.")