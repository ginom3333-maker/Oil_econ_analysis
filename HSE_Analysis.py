import os
import pandas as pd


def export_all_site_safety_scores(
    output_csv="site_hse_scores.csv", file_path="cleaned_unified_master.csv"
):
  if not os.path.exists(file_path):
    print(f"Error: '{file_path}' was not found in the current directory.")
    return

  df = pd.read_csv(file_path)

  # Determine grouping identifier (use 'site_id' if present, otherwise map/fallback from 'well_id')
  group_col = "site_id" if "site_id" in df.columns else "well_id"
  if group_col not in df.columns:
    print("Error: Neither 'site_id' nor 'well_id' column found in the dataset.")
    return

  # If grouping by site but only well_id exists, create a site mapping
  if group_col == "site_id" and "site_id" not in df.columns:
    df["site_id"] = df["well_id"]

  def get_incident_points(description):
    if not description or pd.isna(description):
      return 0
    text = str(description).lower()

    if any(
        kw in text
        for kw in ["serious", "hospital", "h2s exposure", "uncontrolled", "blowout"]
    ):
      return 16
    elif any(
        kw in text
        for kw in [
            "lost-time",
            "lti",
            "strained",
            "off work",
            "days away",
            "off-pad",
            "reached soil",
            "off-location",
        ]
    ):
      return 8
    elif any(
        kw in text
        for kw in [
            "recordable",
            "stitches",
            "laceration",
            "fire",
            "near miss",
            "close call",
            "extinguished",
        ]
    ):
      return 4
    elif any(
        kw in text
        for kw in [
            "spill",
            "overflow",
            "crude",
            "exceedance",
            "flare",
            "permit",
            "citation",
            "trrc",
            "signage",
            "produced water",
            "pw line",
        ]
    ):
      return 2
    elif any(
        kw in text
        for kw in [
            "slip",
            "trip",
            "fall",
            "ice",
            "vehicle",
            "truck",
            "fender bender",
            "bumped",
            "minor",
        ]
    ):
      return 1

    return 0

  results = []
  unique_sites = df[group_col].unique()

  for site_id in unique_sites:
    site_rows = df[df[group_col] == site_id]
    total_penalty = 0

    for _, row in site_rows.iterrows():
      desc = row.get("incident_description", "")
      total_penalty += get_incident_points(desc)

    final_score = max(0, 100 - total_penalty)
    results.append({"site_id": site_id, "hse_score": final_score})

  result_df = pd.DataFrame(results)
  result_df.to_csv(output_csv, index=False)
  print(
      f"Successfully generated and saved '{output_csv}' with"
      f" {len(unique_sites)} sites."
  )


if __name__ == "__main__":
  export_all_site_safety_scores()