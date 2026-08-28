import os
import pandas as pd


def export_site_hse_scores(
    output_csv="site_hse_scores.csv", file_path="cleaned_unified_master.csv"
):
  if not os.path.exists(file_path):
    print(f"Error: '{file_path}' was not found in the current directory.")
    return

  df = pd.read_csv(file_path)

  if "well_id" not in df.columns:
    print("Error: 'well_id' column missing from the dataset.")
    return

  # Ensure 'site_id' exists; if not present, map/fallback from 'well_id'
  if "site_id" not in df.columns:
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

  # 1. Calculate individual score for each well starting from 100 with penalties
  well_scores = []
  unique_wells = df["well_id"].unique()

  for well_id in unique_wells:
    well_rows = df[df["well_id"] == well_id]
    total_penalty = 0

    for _, row in well_rows.iterrows():
      desc = row.get("incident_description", "")
      total_penalty += get_incident_points(desc)

    well_score = max(0, 100 - total_penalty)
    site_id = well_rows["site_id"].iloc[0]
    well_scores.append(
        {"well_id": well_id, "site_id": site_id, "well_score": well_score}
    )

  well_df = pd.DataFrame(well_scores)

  # 2. Average the well scores grouped by site, and assign that score to the site
  site_grouped = well_df.groupby("site_id")["well_score"].mean().reset_index()
  site_grouped.rename(columns={"well_score": "hse_score"}, inplace=True)
  site_grouped["hse_score"] = site_grouped["hse_score"].round(2)

  # 3. Export site scores to CSV
  site_grouped.to_csv(output_csv, index=False)
  print(
      f"Successfully generated and saved '{output_csv}' with site-averaged"
      f" scores for {len(site_grouped)} sites."
  )


if __name__ == "__main__":
  export_site_hse_scores()