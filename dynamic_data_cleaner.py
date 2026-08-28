"""
ConocoPhillips Innovation Challenge 2026 - Master Ingestion, Dynamic Cleaning & Interpolation Pipeline
Author: Asset Rationalization Team
Description: Dynamically determines file-level date format (YYYY-MM-DD vs YYYY-DD-MM)
             using dual month-bound (1-12) and 1-year window validation (2025-08-27 to 2026-08-26), 
             normalizes schemas, cleans numeric/string fields, relationally merges asset datasets, 
             imputes missing daily operational entries per well, and performs linear endpoint interpolation.
"""

import json
import re
import pandas as pd
import numpy as np

# Strict 1-Year Window Bounds (August 27, 2025 through August 26, 2026)
MIN_VALID_DATE = pd.Timestamp("2025-08-27 00:00:00", tz="UTC")
MAX_VALID_DATE = pd.Timestamp("2026-08-26 23:59:59", tz="UTC")


# ==============================================================================
# 1. SPECIALIZED TRANSFORMERS & NORMALIZERS
# ==============================================================================

def normalize_column_name(col_name: str) -> str:
    """Standardizes arbitrary header names into clean snake_case and aligns aliases."""
    if pd.isna(col_name) or col_name is None:
        return "unnamed_column"
    s = str(col_name).strip()
    s = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', s)
    s = re.sub(r'[^a-zA-Z0-9]', '_', s)
    s = re.sub(r'_+', '_', s).lower().strip('_')
    
    alias_map = {
        'well_id': 'well_id', 'wellid': 'well_id', 'well': 'well_id',
        'site_code': 'site_id', 'siteid': 'site_id', 'site_name': 'site_name', 
        'location_name': 'site_name', 'latitude': 'lat', 'longitude': 'lon',
        'contact': 'manager', 'contact_number': 'phone', 'contact_email': 'email',
        'business_unit': 'region', 'occured_on': 'occurred_on',
        'datetime': 'datetime', 'period': 'period', 'timestamp': 'datetime'
    }
    return alias_map.get(s, s)


def standardize_company(val) -> str:
    """Normalizes company alias variations ('EE', 'estacado', etc.) to 'Estacado Energy'."""
    if pd.isna(val) or val is None or str(val).strip().lower() in ['', 'none', 'nan', 'null']:
        return np.nan
    s = str(val).strip().upper()
    if 'ESTACADO' in s or s == 'EE':
        return 'Estacado Energy'
    if 'CONOCO' in s or s == 'COP':
        return 'ConocoPhillips'
    return str(val).strip().title()


def standardize_well_id(val) -> str:
    """Normalizes pure numbers and prefixed strings (23 -> 'W0023', 'W0023' -> 'W0023')."""
    if pd.isna(val) or val is None or str(val).strip().lower() in ['', 'none', 'nan', 'null']:
        return np.nan
    s = str(val).strip().upper()
    if s.isdigit():
        return f"W{int(s):04d}"
    if s.startswith('W') and s[1:].isdigit():
        return f"W{int(s[1:]):04d}"
    return s


def standardize_site_id(val) -> str:
    """Normalizes site codes (e.g. lowercase 's' to uppercase 'S001')."""
    if pd.isna(val) or val is None or str(val).strip().lower() in ['', 'none', 'nan', 'null']:
        return np.nan
    return str(val).strip().upper()


def clean_phone_number(val) -> str:
    """Standardizes contact numbers formatted with dots or dashes to format: +1-XXX-XXX-XXXX."""
    if pd.isna(val) or val is None or str(val).strip().lower() in ['', 'none', 'nan', 'null']:
        return np.nan
    digits = re.sub(r'\D', '', str(val))
    if len(digits) == 10:
        return f"+1-{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    elif len(digits) == 11 and digits.startswith('1'):
        return f"+1-{digits[1:4]}-{digits[4:7]}-{digits[7:]}"
    return str(val).strip()


def clean_currency_or_numeric(val):
    """
    Cleans numeric values formatted as strings (e.g., 'USD 35,649.59' -> 35649.59, '142.5 bbls' -> 142.5).
    Strips trailing text and non-numeric characters automatically.
    """
    if pd.isna(val) or val is None or str(val).strip().lower() in ['', 'none', 'nan', 'null']:
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).replace(',', '').strip()
    match = re.search(r'[-+]?\d*\.\d+|\d+', s)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return np.nan
    return np.nan


def infer_date_format_combined(series: pd.Series) -> bool:
    """
    Determines date orientation (YYYY-MM-DD vs YYYY-DD-MM) by evaluating which layout 
    satisfies BOTH valid month boundaries (1-12) AND the 2025-08-27 to 2026-08-26 window constraint.
    Returns True for YYYY-DD-MM (dayfirst), False for YYYY-MM-DD.
    """
    clean_series = series.dropna().astype(str).str.strip()
    if clean_series.empty:
        return False

    # 1. Test YYYY-MM-DD (dayfirst=False)
    parsed_ymd = pd.to_datetime(clean_series, errors='coerce', dayfirst=False, utc=True)
    valid_ymd_mask = (
        parsed_ymd.notna() & 
        (parsed_ymd.dt.month.between(1, 12)) & 
        (parsed_ymd >= MIN_VALID_DATE) &
        (parsed_ymd <= MAX_VALID_DATE)
    )
    ymd_score = valid_ymd_mask.sum()

    # 2. Test YYYY-DD-MM (dayfirst=True)
    parsed_ydm = pd.to_datetime(clean_series, errors='coerce', dayfirst=True, utc=True)
    valid_ydm_mask = (
        parsed_ydm.notna() & 
        (parsed_ydm.dt.month.between(1, 12)) & 
        (parsed_ydm >= MIN_VALID_DATE) &
        (parsed_ydm <= MAX_VALID_DATE)
    )
    ydm_score = valid_ydm_mask.sum()

    return ydm_score > ymd_score


def parse_datetime_flexible(val, dayfirst: bool = False):
    """
    Parses date inputs using the file's inferred dayfirst setting, enforcing
    month boundaries and restricting dates strictly between 2025-08-27 and 2026-08-26.
    """
    if pd.isna(val) or val is None or str(val).strip().lower() in ['', 'none', 'nan', 'null']:
        return np.nan
    
    dt = None

    # Handle Epoch Timestamps
    if isinstance(val, (int, float)) or (isinstance(val, str) and str(val).strip().isdigit()):
        num_val = float(val)
        if num_val > 1e11:  # Milliseconds
            dt = pd.to_datetime(num_val, unit='ms', utc=True)
        elif num_val > 1e8:  # Seconds
            dt = pd.to_datetime(num_val, unit='s', utc=True)

    # Parse String Date
    if dt is None:
        try:
            dt = pd.to_datetime(val, errors='coerce', dayfirst=dayfirst, utc=True)
        except Exception:
            return np.nan

    # Validate against strict 1-year window & Month Bounds
    if pd.notna(dt):
        if not (MIN_VALID_DATE <= dt <= MAX_VALID_DATE) or not (1 <= dt.month <= 12):
            return np.nan
        
        if dt.time() == pd.Timestamp('00:00:00').time():
            return dt.strftime('%Y-%m-%d')
        return dt.strftime('%Y-%m-%d %H:%M:%S')

    return np.nan


def convert_gas_to_mcf(val):
    """Converts gas volumes from MMBtu to MCF (1 MMBtu ≈ 1.037 MCF) if specified in string."""
    if pd.isna(val) or val is None or str(val).strip().lower() in ['', 'none', 'nan', 'null']:
        return np.nan
    s = str(val).strip().lower()
    num = clean_currency_or_numeric(s)
    if pd.isna(num):
        return np.nan
    
    if 'mmbtu' in s:
        return round(num * 1.037, 4)
    return num


def handle_shut_in_temperature(val):
    """Handles 'SHUT_IN' string values in temperature fields by returning NaN."""
    if pd.isna(val) or val is None:
        return np.nan
    s = str(val).strip().upper()
    if 'SHUT' in s or 'IN' in s or s in ['', 'NONE', 'NAN', 'NULL']:
        return np.nan
    return clean_currency_or_numeric(val)


# ==============================================================================
# 2. DYNAMIC FILE CLEANING ENGINE
# ==============================================================================

def clean_file(filepath: str) -> pd.DataFrame:
    """Loads and runs dataset-specific transformations dynamically across formats."""
    df = None
    
    if filepath.endswith('.csv'):
        for enc in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'utf-16']:
            try:
                df = pd.read_csv(filepath, encoding=enc, on_bad_lines='skip')
                if not df.empty:
                    break
            except Exception:
                continue
    elif filepath.endswith('.json'):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                    df = pd.DataFrame(v)
                    break
            if df is None:
                df = pd.json_normalize(data)
        elif isinstance(data, list):
            df = pd.DataFrame(data)

    if df is None or df.empty:
        raise ValueError(f"Unable to parse dataset: {filepath}")

    # 1. Normalize Header Columns
    df.columns = [normalize_column_name(c) for c in df.columns]

    # 2. Trim string whitespace
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # 3. Standardize Keys & Entities
    if 'company' in df.columns:
        df['company'] = df['company'].apply(standardize_company)

    if 'well_id' in df.columns:
        df['well_id'] = df['well_id'].apply(standardize_well_id)

    if 'site_id' in df.columns:
        df['site_id'] = df['site_id'].apply(standardize_site_id)

    # 4. Standardize Dates via Combined Dual Validation
    for date_col in ['datetime', 'period', 'occurred_on', 'occured_on']:
        if date_col in df.columns:
            is_day_first = infer_date_format_combined(df[date_col])
            df[date_col] = df[date_col].apply(lambda x: parse_datetime_flexible(x, dayfirst=is_day_first))

    # 5. Handle Long-to-Wide Sensor Reading Formatting
    if 'sensor_type' in df.columns and 'value' in df.columns:
        df['value'] = df['value'].apply(clean_currency_or_numeric)
        
        pivot_cols = [c for c in ['company', 'well_id', 'datetime'] if c in df.columns]
        df = df.pivot_table(
            index=pivot_cols,
            columns='sensor_type',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        df.columns.name = None
        df.columns = [normalize_column_name(c) for c in df.columns]

    # 6. Clean Phone & Metadata Text
    if 'phone' in df.columns:
        df['phone'] = df['phone'].apply(clean_phone_number)

    if 'region' in df.columns:
        df['region'] = df['region'].apply(lambda x: str(x).strip().title() if pd.notna(x) else np.nan)

    # 7. Clean Financial, Oil Rates, Sensor Metrics & Volumes
    numeric_columns = [
        'revenue', 'operating_cost', 'profit', 'cost_per_barrel', 
        'oil_rate', 'water_rate', 'total_liquid_rate', 
        'tubing_pressure', 'casing_pressure', 'line_pressure', 'gas_oil_ratio', 
        'allocation_oil_volume', 'allocation_water_volume',
        'temperature', 'pressure', 'vibration', 'flow_rate'
    ]
    for fin_col in numeric_columns:
        if fin_col in df.columns:
            df[fin_col] = df[fin_col].apply(clean_currency_or_numeric)

    # Convert Gas Volume/Rates
    for gas_col in ['gas_rate', 'allocation_gas_volume']:
        if gas_col in df.columns:
            df[gas_col] = df[gas_col].apply(convert_gas_to_mcf)

    # Temperature SHUT_IN handling
    if 'flowing_temperature' in df.columns:
        df['flowing_temperature'] = df['flowing_temperature'].apply(handle_shut_in_temperature)

    # 8. Per-file O(N log N) Chronological Sort
    for date_col in ['datetime', 'period', 'occurred_on']:
        if date_col in df.columns:
            sort_keys = [c for c in ['well_id', date_col] if c in df.columns]
            df = df.sort_values(by=sort_keys, ascending=True, kind='quicksort').reset_index(drop=True)
            break

    return df


# ==============================================================================
# 3. DAILY TIMELINE IMPUTATION & LINEAR INTERPOLATION
# ==============================================================================

def impute_missing_days_and_interpolate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures every well has a continuous daily record from 2025-08-27 to 2026-08-26.
    Missing dates are inserted, numeric values are linearly interpolated (averaged across endpoints),
    and static text/metadata fields are forward and backward filled.
    """
    if 'well_id' not in df.columns or 'datetime' not in df.columns:
        return df

    # Convert datetime strings to pure date objects for complete daily reindexing
    df['date_only'] = pd.to_datetime(df['datetime']).dt.date
    
    # Complete 365-day timeline range
    full_date_range = pd.date_range(start=MIN_VALID_DATE, end=MAX_VALID_DATE, freq='D').date
    
    interpolated_frames = []
    
    # Differentiate numeric columns (for interpolation) vs metadata columns (for ffill/bfill)
    ignored_cols = ['well_id', 'datetime', 'date_only']
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ignored_cols]
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in ignored_cols]

    for well, group in df.groupby('well_id'):
        # Deduplicate intra-day rows to keep clean 1-row-per-day baseline
        group_unique = group.drop_duplicates(subset=['date_only'], keep='first').set_index('date_only')
        
        # Reindex to full daily range (creates NaNs for missing gap dates)
        reindexed = group_unique.reindex(full_date_range)
        reindexed['well_id'] = well
        reindexed['datetime'] = reindexed.index.astype(str)
        
        # 1. Linear Endpoint Interpolation for Numeric Operational & Financial Metrics
        if numeric_cols:
            reindexed[numeric_cols] = reindexed[numeric_cols].interpolate(method='linear', limit_direction='both')
        
        # 2. Forward/Backward Fill for Static Identifiers & Text Attributes
        if categorical_cols:
            reindexed[categorical_cols] = reindexed[categorical_cols].ffill().bfill()
            
        interpolated_frames.append(reindexed.reset_index(drop=True))

    full_df = pd.concat(interpolated_frames, ignore_index=True)
    if 'date_only' in full_df.columns:
        full_df = full_df.drop(columns=['date_only'])
        
    return full_df


# ==============================================================================
# 4. MASTER RELATIONAL PIPELINE
# ==============================================================================

def build_unified_master_pipeline(file_paths: list) -> pd.DataFrame:
    """Ingests dynamic file paths, cleans each individually, merges cleanly, and imputes timelines."""
    cleaned_frames = {}
    site_meta_frames = []
    
    print("--- PHASE 1: INDIVIDUAL FILE CLEANING, DATE VALIDATION & SORTING ---")
    for path in file_paths:
        try:
            cleaned_df = clean_file(path)
            cleaned_frames[path] = cleaned_df
            print(f"[SUCCESS] Cleaned {path} | Shape: {cleaned_df.shape}")
            
            if 'site_id' in cleaned_df.columns and ('lat' in cleaned_df.columns or 'site_name' in cleaned_df.columns):
                site_meta_frames.append(cleaned_df)
        except Exception as e:
            print(f"[ERROR] Failed to process {path}: {e}")

    # Consolidate site metadata cleanly
    if site_meta_frames:
        df_sites_unified = pd.concat(site_meta_frames, ignore_index=True)
        valid_sites = df_sites_unified.dropna(subset=['site_id']).drop_duplicates(subset=['site_id'])
        missing_sites = df_sites_unified[df_sites_unified['site_id'].isna()]
        df_sites_unified = pd.concat([valid_sites, missing_sites], ignore_index=True)
    else:
        df_sites_unified = pd.DataFrame()

    print("\n--- PHASE 2: RELATIONAL MERGE INTO MASTER DATAFRAME ---")
    
    # Establish base dataset
    baseline_path = None
    master_df = None
    for path, df in cleaned_frames.items():
        if 'well_id' in df.columns and any(c in df.columns for c in ['rate', 'volume', 'temperature', 'pressure', 'flow_rate']):
            master_df = df.copy()
            baseline_path = path
            print(f"Set Baseline Dataset: {path}")
            break

    if master_df is None:
        baseline_path = list(cleaned_frames.keys())[0]
        master_df = cleaned_frames[baseline_path].copy()

    # Dynamic safe relational merges
    for path, df in cleaned_frames.items():
        if path == baseline_path:
            continue

        if 'site_name' in df.columns and not df_sites_unified.empty and 'well_id' not in df.columns:
            continue

        if 'well_id' in df.columns:
            merge_keys = ['well_id']
            if 'datetime' in master_df.columns and 'datetime' in df.columns:
                merge_keys.append('datetime')

            cols_to_use = [c for c in df.columns if c in merge_keys or c not in master_df.columns]
            if len(cols_to_use) > len(merge_keys):
                df_to_merge = df[cols_to_use].copy()
                
                # Prevent M:N row explosion
                if df_to_merge.duplicated(subset=merge_keys).any():
                    df_to_merge = df_to_merge.drop_duplicates(subset=merge_keys, keep='first')

                master_df = master_df.merge(df_to_merge, on=merge_keys, how='outer' if 'datetime' in merge_keys else 'left')
                print(f"Merged {path} via {merge_keys}")

    # Join site metadata via site_id
    if not df_sites_unified.empty and 'site_id' in master_df.columns:
        cols_to_use = [c for c in df_sites_unified.columns if c == 'site_id' or c not in master_df.columns]
        if len(cols_to_use) > 1:
            site_lookup = df_sites_unified[cols_to_use].dropna(subset=['site_id']).drop_duplicates(subset=['site_id'])
            master_df = master_df.merge(site_lookup, on='site_id', how='left')
            print("Merged Unified Site Metadata via 'site_id'")

    # Standardize status and category fields
    for status_col in ['status', 'category']:
        if status_col in master_df.columns:
            master_df[status_col] = master_df[status_col].apply(
                lambda x: str(x).strip().lower() 
                if pd.notna(x) and str(x).strip().lower() not in ['', 'none', 'nan', 'null'] 
                else np.nan
            )

    # Drop unparsed empty date rows before timeline expansion
    if 'datetime' in master_df.columns:
        master_df = master_df.dropna(subset=['datetime']).reset_index(drop=True)

    print("\n--- PHASE 3: DAILY TIMELINE IMPUTATION & LINEAR INTERPOLATION ---")
    master_df = impute_missing_days_and_interpolate(master_df)
    print("Inserted missing daily records and performed linear endpoint interpolation.")

    # Final Master O(N log N) Chronological Sort
    sort_cols = [c for c in ['well_id', 'datetime'] if c in master_df.columns]
    if sort_cols:
        master_df = master_df.sort_values(by=sort_cols, ascending=True, kind='quicksort').reset_index(drop=True)
        print(f"Sorted Master Dataset by {sort_cols} [O(N log N)]")

    return master_df


if __name__ == "__main__":
    input_files = [
        'production.csv',
        'sites_conocophillips_preliminary.csv',
        'sites_estacado.csv',
        'well_site_map.csv',
        'financial_estacado.json',
        'hse_incidents.json',
        'sensor_reading.csv'
    ]
    
    master_dataset = build_unified_master_pipeline(input_files)
    master_dataset.to_csv('cleaned_unified_master.csv', index=False)
    master_dataset.to_json('cleaned_unified_master.json', orient='records', indent=2)
    
    print("\n[PIPELINE COMPLETE] Single Master Dataframe successfully created, imputed & chronologically sorted!")
    print(f"Final Shape: {master_dataset.shape}")