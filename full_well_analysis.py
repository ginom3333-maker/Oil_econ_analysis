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
# LOAD IoT ANOMALY CSV
# ============================================================

def load_iot_anomalies(csv_path="well_sensor_anomalies.csv"):
    anomalies = pd.read_csv(csv_path)
    
    anomalies["combined_total_anomalies"] = pd.to_numeric(anomalies["combined_total_anomalies"], errors="coerce").fillna(0)
    anomalies["total_readings"] = pd.to_numeric(anomalies["total_readings"], errors="coerce").fillna(1)

    agg_anomalies = (
        anomalies.groupby("well_id")
        .agg({
            "combined_total_anomalies": "sum",
            "total_readings": "sum"
        })
        .reset_index()
    )

    agg_anomalies["iot_reliability_percent"] = (
        1.0 - (agg_anomalies["combined_total_anomalies"] / agg_anomalies["total_readings"])
    ) * 100.0

    agg_anomalies["iot_reliability_percent"] = agg_anomalies["iot_reliability_percent"].clip(0, 100)

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
    last6_boe.head(5).to_csv(os.path.join(outdir, "top5_wells_last_6_months.csv"), index=False)
    last6_boe.to_csv(os.path.join(outdir, "all_wells_last_6_months_ranked.csv"), index=False)

# ============================================================
# FIT DECLINE + FORECAST FOR ALL WELLS (DAILY OPEX & NW TEXAS FLOORS)
# ============================================================

def fit_and_forecast_all_wells(monthly_df, anomalies_df, boe_price=60.0):
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
        
        t_future_years = ((future_months - first_date).days / 365.0).astype(float)

        q_future = hyperbolic_arps(t_future_years, qi, Di, b)
        q_future = np.array(q_future, dtype=float)
        q_future[q_future < 0] = 0

        forecast_df = pd.DataFrame({
            "well_id": well,
            "month": future_months,
            "boe_rate_forecast": q_future
        })

        # --- Adjusted Lighter Risk Scoring Weights ---
        risk_score = 0.0
        temp = w["temperature"].mean()
        pressure = w["pressure"].mean()

        if temp < 60: risk_score += 0.10
        if temp > 302: risk_score += 0.10
        if pressure < 200: risk_score += 0.10
        if pressure > 10000: risk_score += 0.10

        well_iot = anomalies_df[anomalies_df["well_id"] == well]
        iot_rel = well_iot["iot_reliability_percent"].values[0] if not well_iot.empty else 100.0
        
        if iot_rel < 95.0:
            risk_score += (95.0 - iot_rel) / 150.0

        Di_risk = Di * (1.0 + risk_score)

        t_fc = ((future_months - future_months[0]).days / 365.0).astype(float)
        q_future_risk = hyperbolic_arps(t_fc, q_future[0], Di_risk, b)
        q_future_risk = np.array(q_future_risk, dtype=float)
        q_future_risk[q_future_risk < 0] = 0

        forecast_df["boe_rate_risk"] = q_future_risk

        # --- Integrated Economic Limit (Converted Monthly to Daily OPEX) ---
        avg_op_cost_monthly = w["operating_cost"].mean()
        if np.isnan(avg_op_cost_monthly) or avg_op_cost_monthly <= 0:
            avg_op_cost_monthly = 9000.0  # Default fallback monthly OPEX (~$300/day)

        avg_op_cost_daily = avg_op_cost_monthly / 30.44

        risk_cost_multiplier = 1.0 + (risk_score * 0.15) + max(0.0, (100.0 - iot_rel) / 500.0)
        adjusted_op_cost_daily = avg_op_cost_daily * risk_cost_multiplier
        calculated_econ_limit = adjusted_op_cost_daily / boe_price if boe_price > 0 else 5.0

        # Northwest Texas industry baseline floors (5 BOE/d for horizontal, 2.5 BOE/d for vertical)
        is_horizontal = "H" in str(well).upper() or "HZ" in str(well).upper()
        regional_min_floor = 5.0 if is_horizontal else 2.5  
        econ_limit = max(calculated_econ_limit, regional_min_floor)

        results[well] = {
            "history": w,
            "params": (qi, Di, b),
            "params_risk": (q_future[0], Di_risk, b),
            "forecast": forecast_df,
            "risk_score": risk_score,
            "iot_reliability": iot_rel,
            "econ_limit": econ_limit
        }

    return results

# ============================================================
# PLOT ALL WELLS
# ============================================================

def plot_all_wells(results, outdir="wells_curves"):
    os.makedirs(outdir, exist_ok=True)

    for well, data in results.items():
        w = data["history"]
        qi, Di, b = data["params"]
        f = data["forecast"]
        iot_rel = data["iot_reliability"]
        risk_score = data["risk_score"]
        econ_limit = data["econ_limit"]

        first_date = w["month"].min()
        t_years = ((w["month"] - first_date).dt.days.values / 365.0).astype(float)
        q_fit = hyperbolic_arps(t_years, qi, Di, b)
        q_fit = np.array(q_fit, dtype=float)
        q_fit[q_fit < 0] = 0

        plt.figure(figsize=(12, 6))
        plt.plot(w["month"], w["boe_rate"], "o-", label="Actual BOE/day")
        plt.plot(w["month"], q_fit, "--", label="Fitted Decline")
        plt.plot(f["month"], f["boe_rate_forecast"], "r--", label="Forecast")
        plt.plot(f["month"], f["boe_rate_risk"], "m--", label=f"Risk Forecast (IoT Rel: {iot_rel:.1f}%, Risk: {risk_score:.2f})")
        
        plt.axhline(y=econ_limit, color="g", linestyle="-", linewidth=2, label=f"NW Texas Econ Limit ({econ_limit:.1f} BOE/d)")

        plt.title(f"Well Forecast — {well} (IoT Rel: {iot_rel:.1f}% | Econ Limit: {econ_limit:.1f} BOE/d)")
        plt.xlabel("Month")
        plt.ylabel("BOE/day")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{well}_forecast.png"))
        plt.close()

# ============================================================
# PLOT SPECIFIC WELLS (E.G., TOP 5) TO SEPARATE FOLDER
# ============================================================

def plot_specific_wells(results, target_wells, outdir="Top5_Curves"):
    os.makedirs(outdir, exist_ok=True)

    for well in target_wells:
        if well not in results:
            continue
        
        data = results[well]
        w = data["history"]
        qi, Di, b = data["params"]
        f = data["forecast"]
        iot_rel = data["iot_reliability"]
        risk_score = data["risk_score"]
        econ_limit = data["econ_limit"]

        first_date = w["month"].min()
        t_years = ((w["month"] - first_date).dt.days.values / 365.0).astype(float)
        q_fit = hyperbolic_arps(t_years, qi, Di, b)
        q_fit = np.array(q_fit, dtype=float)
        q_fit[q_fit < 0] = 0

        plt.figure(figsize=(12, 6))
        plt.plot(w["month"], w["boe_rate"], "o-", label="Actual BOE/day")
        plt.plot(w["month"], q_fit, "--", label="Fitted Decline")
        plt.plot(f["month"], f["boe_rate_forecast"], "r--", label="Forecast")
        plt.plot(f["month"], f["boe_rate_risk"], "m--", label=f"Risk Forecast (IoT Rel: {iot_rel:.1f}%, Risk: {risk_score:.2f})")
        
        plt.axhline(y=econ_limit, color="g", linestyle="-", linewidth=2, label=f"NW Texas Econ Limit ({econ_limit:.1f} BOE/d)")

        plt.title(f"Top 5 Well Forecast — {well} (IoT Rel: {iot_rel:.1f}% | Econ Limit: {econ_limit:.1f} BOE/d)")
        plt.xlabel("Month")
        plt.ylabel("BOE/day")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{well}_forecast.png"))
        plt.close()

# ============================================================
# SITE-LEVEL PRODUCTION EFFICIENCY
# ============================================================

def compute_site_efficiency(monthly_df, anomalies_df):
    well_boe_avg = (
        monthly_df.groupby(["site_id", "well_id"])["boe_rate"]
        .mean()
        .reset_index()
        .rename(columns={"boe_rate": "boe_avg_per_well"})
    )

    merged_iot = well_boe_avg.merge(
        anomalies_df[["well_id", "iot_reliability_percent"]],
        on="well_id",
        how="left"
    ).fillna({"iot_reliability_percent": 100.0})

    merged_iot["expected_well_boe"] = (
        merged_iot["boe_avg_per_well"] * (merged_iot["iot_reliability_percent"] / 100.0)
    )

    site_stats = (
        merged_iot.groupby("site_id")
        .agg(
            total_site_boe=("boe_avg_per_well", "sum"),
            well_count=("well_id", "count"),
            boe_avg_per_well=("boe_avg_per_well", "mean"),
            iot_percent=("iot_reliability_percent", "mean"),
            score=("expected_well_boe", "mean")
        )
        .reset_index()
    )

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

    monthly = build_monthly_rates(df)

    site_eff = compute_site_efficiency(monthly, anomalies_df)
    save_site_efficiency_outputs(site_eff, monthly, anomalies_df)

    save_iot_reliability_file(anomalies_df)

    # Get top 5 wells from the last 6 months and save tables
    top5, last6_boe_table = get_top5_last6_boe(monthly)
    save_top5_report(last6_boe_table, outdir="Top5_Last6_Months")

    # Fit and forecast all wells
    results = fit_and_forecast_all_wells(monthly, anomalies_df, boe_price=60.0)
    
    # Save all well curves to general folder
    plot_all_wells(results, outdir="wells_curves")
    
    # Save ONLY the top 5 well curves into their own separate folder
    plot_specific_wells(results, top5, outdir="Top5_Curves")

    print("\nDone: Top 5 curves saved to 'Top5_Curves', reports to 'Top5_Last6_Months', and all well curves saved to 'wells_curves'.")