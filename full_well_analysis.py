import pandas as pd
import numpy as np
import os
import shutil
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
    return qi / np.power(1 + b * Di * t, 1.0 / b)

# ============================================================
# BUILD MONTHLY DATA
# ============================================================

def build_monthly_rates(df):
    date_col = find_date_column(df)
    df[date_col] = pd.to_datetime(df[date_col])

    df["month"] = df[date_col].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby(["well_id", "month"])
        .agg({
            "oil_rate": "mean",          # bbl/day
            "operating_cost": "mean",    # $/month per well
            "revenue": "mean"            # $/month
        })
        .reset_index()
        .sort_values(["well_id", "month"])
    )

    return monthly

# ============================================================
# FIRST / LAST 6 MONTHS SUMMARY
# ============================================================

def compute_first_last_6_months(monthly_df):
    summaries = []

    for well in monthly_df["well_id"].unique():
        w = monthly_df[monthly_df["well_id"] == well].sort_values("month")

        first6 = w.head(6)
        last6 = w.tail(6)

        summary = {
            "well_id": well,

            "first6_start": first6["month"].min(),
            "first6_end": first6["month"].max(),
            "first6_oil_min": first6["oil_rate"].min(),
            "first6_oil_max": first6["oil_rate"].max(),
            "first6_oil_avg": first6["oil_rate"].mean(),

            "last6_start": last6["month"].min(),
            "last6_end": last6["month"].max(),
            "last6_oil_min": last6["oil_rate"].min(),
            "last6_oil_max": last6["oil_rate"].max(),
            "last6_oil_avg": last6["oil_rate"].mean(),
        }

        summaries.append(summary)

    return pd.DataFrame(summaries)

# ============================================================
# FIT HYPERBOLIC ARPS PER WELL
# ============================================================

def fit_hyperbolic_per_well(monthly_df):
    params = []
    forecasts = []

    for well in monthly_df["well_id"].unique():
        w = monthly_df[monthly_df["well_id"] == well].sort_values("month")

        if len(w) < 6:
            continue

        # Use actual time difference in years
        first_date = w["month"].min()
        t_years = ((w["month"] - first_date).dt.days.values / 365.0).astype(float)

        q = w["oil_rate"].values

        qi0 = q[0]
        Di0 = 0.8
        b0 = 1.0

        try:
            popt, pcov = curve_fit(
                hyperbolic_arps,
                t_years,
                q,
                p0=[qi0, Di0, b0],
                bounds=([0.0, 0.0, 0.0], [np.inf, 10.0, 5.0]),
                maxfev=20000
            )
            qi, Di, b = popt
        except Exception:
            continue

        params.append({
            "well_id": well,
            "qi": qi,
            "Di": Di,
            "b": b
        })

        # Forecast using same time basis
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
            "oil_rate_forecast": q_future
        })

        forecasts.append(forecast_df)

    params_df = pd.DataFrame(params)
    forecast_all = pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame()

    return params_df, forecast_all

# ============================================================
# DELETE OLD OUTPUT FOLDERS
# ============================================================

def delete_old_outputs():
    folders = [
        "well_decline_data",
        "top5_producers",
        "at_risk_wells"
    ]
    for f in folders:
        if os.path.exists(f):
            shutil.rmtree(f)

# ============================================================
# DECLINE CURVE GRAPHING (Economic limit ≈ 8–10 bbl/day)
# ============================================================

def plot_decline_curve(well, monthly_df, params_df, forecast_df, outdir):
    os.makedirs(outdir, exist_ok=True)

    w = monthly_df[monthly_df["well_id"] == well].sort_values("month")
    p = params_df[params_df["well_id"] == well]

    if p.empty:
        return

    qi = p["qi"].values[0]
    Di = p["Di"].values[0]
    b  = p["b"].values[0]

    # --- Economic limit using MONTHLY operating cost and assumed oil price ---
    op_cost_monthly = w["operating_cost"].mean()   # $/month
    op_cost_daily = op_cost_monthly / 30.0         # $/day

    oil_price = 75.0                               # $/bbl (adjustable)
    economic_limit = op_cost_daily / oil_price     # bbl/day

    # Historical fitted curve
    first_date = w["month"].min()
    t_years = ((w["month"] - first_date).dt.days.values / 365.0).astype(float)

    q_fit = hyperbolic_arps(t_years, qi, Di, b)
    q_fit = np.array(q_fit, dtype=float)
    q_fit[q_fit < 0] = 0

    # Forecast
    f = forecast_df[forecast_df["well_id"] == well]
    if not f.empty:
        q_future = np.array(f["oil_rate_forecast"].values, dtype=float)
        q_future[q_future < 0] = 0

    plt.figure(figsize=(12, 6))

    plt.plot(w["month"], w["oil_rate"], "o-", label="Actual Monthly Oil Rate")
    plt.plot(w["month"], q_fit, "--", label="Fitted Arps Curve")

    if not f.empty:
        plt.plot(f["month"], q_future, "r--", label="Forecast (5 years)")

    # Economic limit line
    plt.axhline(economic_limit, color="red", linestyle="--", linewidth=2,
                label=f"Economic Limit ({economic_limit:.2f} bbl/day)")

    # Shutdown marker
    if not f.empty:
        below_limit = f[q_future < economic_limit]
        if not below_limit.empty:
            shutdown_month = below_limit.iloc[0]["month"]
            plt.axvline(shutdown_month, color="red", linestyle=":",
                        label=f"Unprofitable after {shutdown_month.date()}")

    plt.title(f"Decline Curve — {well}")
    plt.xlabel("Month")
    plt.ylabel("Oil Rate (bbl/day)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(outdir, f"{well}_decline_curve.png"))
    plt.close()

# ============================================================
# SAVE PER-WELL FILES + TOP5 + AT-RISK + GRAPHS
# ============================================================

def save_per_well_files(first_last_df, params_df, forecast_df, monthly_df):
    delete_old_outputs()

    os.makedirs("well_decline_data", exist_ok=True)
    os.makedirs("top5_producers", exist_ok=True)
    os.makedirs("at_risk_wells", exist_ok=True)

    avg_oil = (
        monthly_df.groupby("well_id")["oil_rate"]
        .mean()
        .reset_index()
        .sort_values("oil_rate", ascending=False)
    )

    top5 = avg_oil.head(5)["well_id"].tolist()

    at_risk = []
    for _, row in first_last_df.iterrows():
        if row["first6_oil_avg"] > 0:
            decline_ratio = row["last6_oil_avg"] / row["first6_oil_avg"]
            if decline_ratio < 0.20:
                at_risk.append(row["well_id"])

    wells = sorted(set(first_last_df["well_id"]) | set(params_df["well_id"]))

    for well in wells:
        fl = first_last_df[first_last_df["well_id"] == well]
        p = params_df[params_df["well_id"] == well]
        f = forecast_df[forecast_df["well_id"] == well]

        base = fl.copy()

        if not p.empty:
            base["qi"] = p["qi"].values[0]
            base["Di"] = p["Di"].values[0]
            base["b"] = p["b"].values[0]
        else:
            base["qi"] = np.nan
            base["Di"] = np.nan
            base["b"] = np.nan

        well_dir = f"well_decline_data/{well}"
        os.makedirs(well_dir, exist_ok=True)

        base.to_csv(f"{well_dir}/{well}_summary.csv", index=False)
        if not f.empty:
            f.to_csv(f"{well_dir}/{well}_forecast.csv", index=False)

        plot_decline_curve(well, monthly_df, params_df, forecast_df, well_dir)

        if well in top5:
            top_dir = f"top5_producers/{well}"
            os.makedirs(top_dir, exist_ok=True)

            base.to_csv(f"{top_dir}/{well}_summary.csv", index=False)
            if not f.empty:
                f.to_csv(f"{top_dir}/{well}_forecast.csv", index=False)

            plot_decline_curve(well, monthly_df, params_df, forecast_df, top_dir)

        if well in at_risk:
            risk_dir = f"at_risk_wells/{well}"
            os.makedirs(risk_dir, exist_ok=True)

            base.to_csv(f"{risk_dir}/{well}_summary.csv", index=False)
            if not f.empty:
                f.to_csv(f"{risk_dir}/{well}_forecast.csv", index=False)

            plot_decline_curve(well, monthly_df, params_df, forecast_df, risk_dir)

    print("Saved per-well summary, top5 producers, at-risk wells, and graphs.")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    csv_path = "cleaned_unified_master.csv"

    df = pd.read_csv(csv_path)

    monthly = build_monthly_rates(df)
    first_last = compute_first_last_6_months(monthly)
    params, forecast = fit_hyperbolic_per_well(monthly)

    save_per_well_files(first_last, params, forecast, monthly)

    print("Done.")




