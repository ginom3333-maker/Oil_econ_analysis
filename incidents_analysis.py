import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------

CATEGORY_WEIGHTS = {
    "Safety": 0.40,
    "Process saftey": 0.30,   # note spelling from your data
    "Process safety": 0.30,
    "Environment": 0.20,
    "Enviromental": 0.20,     # if misspelled in data
    "Regulatory": 0.10
}

KEYWORDS = ["flare", "exceedance", "spill", "failure", "violation", "shutdown"]

# -----------------------------
# HSE INDEX
# -----------------------------

def compute_hse_index(df):
    df['incidentWeight'] = df['category'].apply(
        lambda c: CATEGORY_WEIGHTS.get(c, 0.10)
    )

    hse = df.groupby('wellId')['incidentWeight'].sum().reset_index()
    hse.rename(columns={'incidentWeight': 'HSEIndex'}, inplace=True)

    max_index = hse['HSEIndex'].max()
    if max_index > 0:
        hse['HSEIndexNormalized'] = hse['HSEIndex'] / max_index
    else:
        hse['HSEIndexNormalized'] = 0.0

    return hse

# -----------------------------
# RECURRENCE SCORING
# -----------------------------

def time_cluster_score(well_df):
    well_df = well_df.sort_values('occurredOn')
    dates = well_df['occurredOn'].tolist()

    clusters = 0
    for i in range(1, len(dates)):
        delta = (dates[i] - dates[i-1]).days
        if delta <= 30:
            clusters += 1
    return clusters

def keyword_recurrence(well_df):
    text = " ".join(well_df['incidentDescription'].astype(str).str.lower())
    return sum(text.count(k) for k in KEYWORDS)

def recurrence_index(well_df):
    # category recurrence: max count of any single category
    cat_score = well_df.groupby('category').size().max()

    # time clustering: number of close‑in‑time pairs (<= 30 days)
    time_score = time_cluster_score(well_df)

    # keyword recurrence: repeated failure patterns
    keyword_score = keyword_recurrence(well_df)

    return (0.6 * cat_score) + (0.3 * time_score) + (0.1 * keyword_score)

def compute_recurrence(df):
    rec = df.groupby('wellId').apply(recurrence_index).reset_index(name='RecurrenceIndex')

    max_rec = rec['RecurrenceIndex'].max()
    if max_rec > 0:
        rec['RecurrenceNormalized'] = rec['RecurrenceIndex'] / max_rec
    else:
        rec['RecurrenceNormalized'] = 0.0

    return rec

# -----------------------------
# MAIN ENTRYPOINT
# -----------------------------

def analyze_incidents(json_path):
    df = pd.read_json(json_path)

    # parse dates
    df['occurredOn'] = pd.to_datetime(df['occurredOn'])

    hse = compute_hse_index(df)
    rec = compute_recurrence(df)

    # merge HSE + recurrence per well
    result = pd.merge(hse, rec, on='wellId', how='outer')

    return result

# -----------------------------
# EXAMPLE USAGE
# -----------------------------

if __name__ == "__main__":
    per_well = analyze_incidents("data/hse_incidents.json")
