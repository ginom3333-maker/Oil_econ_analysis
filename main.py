import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# Helpers
# -----------------------------

def clean_currency(value):
    """Remove USD prefix and commas."""
    if isinstance(value, str):
        return float(value.replace("USD", "").replace(",", "").strip())
    return float(value)

def calculate_elr(row):
    """Economic Limit Rate = operatingCost / costPerBarrel."""
    if row['costPerBarrel'] <= 0:
        return None
    return row['operatingCost'] / row['costPerBarrel']

def find_cutoff_date(well_df):
    """
    Economic cutoff date = first date where production < ELR.
    If production is missing, return None.
    """
    if 'production' not in well_df.columns:
        return None

    for _, row in well_df.iterrows():
        if row['production'] < row['ELR']:
            return row['period']
    return None

def generate_well_report(well_df):
    """Create a per-well economic summary."""
    wellId = well_df['wellId'].iloc[0]

    report = {
        "wellId": wellId,
        "firstDate": well_df['period'].min(),
        "lastDate": well_df['period'].max(),
        "totalRevenue": well_df['revenue'].sum(),
        "totalOperatingCost": well_df['operatingCost'].sum(),
        "totalProfit": well_df['profit'].sum(),
        "averageNetback": well_df['costPerBarrel'].mean(),
        "averageELR": well_df['ELR'].mean(),
        "economicCutoffDate": find_cutoff_date(well_df)
    }

    return report

# -----------------------------
# Graphing Function
# -----------------------------

def plot_well(well_df, output_folder="well_graphs"):
    """Generate a graph for each well."""
    os.makedirs(output_folder, exist_ok=True)

    wellId = well_df['wellId'].iloc[0]

    plt.figure(figsize=(12, 6))
    plt.plot(well_df['period'], well_df['revenue'], label='Revenue', marker='o')
    plt.plot(well_df['period'], well_df['operatingCost'], label='Operating Cost', marker='o')
    plt.plot(well_df['period'], well_df['profit'], label='Profit', marker='o')
    plt.plot(well_df['period'], well_df['ELR'], label='ELR (Economic Limit Rate)', linestyle='--')

    # Cutoff marker
    cutoff = find_cutoff_date(well_df)
    if cutoff:
        plt.axvline(cutoff, color='red', linestyle='--', label='Economic Cutoff')

    plt.title(f"Well {wellId} — Economic Performance Over Time")
    plt.xlabel("Period")
    plt.ylabel("USD")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{output_folder}/{wellId}_economic_graph.png")
    plt.close()

# -----------------------------
# Main Function
# -----------------------------

def analyze_financials(csv_path):
    df = pd.read_csv(csv_path)

    # Clean fields
    df['revenue'] = df['revenue'].apply(clean_currency)
    df['operatingCost'] = df['operatingCost'].astype(float)
    df['profit'] = df['profit'].astype(float)
    df['costPerBarrel'] = df['costPerBarrel'].astype(float)

    # Convert period to datetime
    df['period'] = pd.to_datetime(df['period'])

    # Compute ELR
    df['ELR'] = df.apply(calculate_elr, axis=1)

    # Group by wellId
    well_groups = df.groupby('wellId')

    # Build per-well reports + graphs
    reports = []
    for wellId, well_df in well_groups:
        well_df = well_df.sort_values('period')
        report = generate_well_report(well_df)
        reports.append(report)

        # Generate graph
        plot_well(well_df)

    return df, pd.DataFrame(reports)

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    detailed, reports = analyze_financials("data/financial_estacado.csv")
    print("Detailed per-period data:")
    print(detailed)
    print("\nPer-well economic reports:")
    print(reports)


