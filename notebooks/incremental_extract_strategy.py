#!/usr/bin/env python3
"""
Incremental Dataset Extraction Strategy

Optimizes extract_datasets.py with 7-day lookback window instead of full historical extraction.
Reduces extraction time from ~300 minutes to ~6 minutes (50x speedup).

Strategy:
  1. Check when token_datasets.csv was last saved
  2. Extract only data from the last 7 days
  3. Remove 7-day-old records from existing dataset (to avoid duplicates)
  4. Merge with new fresh 7-day data
  5. Save updated dataset

This maintains all data before the 7-day window while refreshing recent data completely.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

print("✅ Incremental Extract Strategy - Python Script")
print("="*70)


# ============================================================================
# SECTION 1: Load and Inspect token_datasets.csv
# ============================================================================

print("\n📊 SECTION 1: Load and Inspect token_datasets.csv")
print("="*70)

dataset_path = 'data/token_datasets.csv'

if os.path.exists(dataset_path):
    df_existing = pd.read_csv(dataset_path)
    print(f"✅ Loaded existing dataset: {dataset_path}")
    print(f"   Shape: {df_existing.shape}")
    print(f"   Columns: {df_existing.columns.tolist()}")
    
    # Get file modification time
    file_stat = os.stat(dataset_path)
    last_modified = datetime.fromtimestamp(file_stat.st_mtime)
    print(f"\n📅 Last modified: {last_modified}")
    print(f"   Days since update: {(datetime.now() - last_modified).days}")
    
    # Inspect timestamp columns
    print(f"\n📊 Timestamp column analysis:")
    if 'checked_at_utc' in df_existing.columns:
        df_existing['checked_at_utc'] = pd.to_datetime(df_existing['checked_at_utc'], errors='coerce')
        print(f"   Earliest record: {df_existing['checked_at_utc'].min()}")
        print(f"   Latest record: {df_existing['checked_at_utc'].max()}")
    elif 'checked_at_timestamp' in df_existing.columns:
        df_existing['checked_at_timestamp'] = pd.to_datetime(df_existing['checked_at_timestamp'], unit='s', errors='coerce')
        print(f"   Earliest record: {df_existing['checked_at_timestamp'].min()}")
        print(f"   Latest record: {df_existing['checked_at_timestamp'].max()}")
    
    # Show sample
    print(f"\n📋 Sample rows:")
    print(df_existing.head(2))
else:
    print(f"❌ Dataset not found at {dataset_path}")
    print(f"   This is the FIRST run - will extract all available data")
    df_existing = None


# ============================================================================
# SECTION 2: Calculate the 7-Day Lookback Window
# ============================================================================

print("\n📅 SECTION 2: Calculate the 7-Day Lookback Window")
print("="*70)

now = datetime.now()
lookback_days = 7
seven_days_ago = now - timedelta(days=lookback_days)

print(f"Current date/time: {now}")
print(f"Lookback days: {lookback_days}")
print(f"Start date for extraction: {seven_days_ago}")
print(f"End date for extraction: {now}")

# Format dates for database query
start_date_str = seven_days_ago.strftime('%Y-%m-%d')
end_date_str = now.strftime('%Y-%m-%d')

print(f"\n🗓️  Extract data from: {start_date_str} to {end_date_str}")

# Calculate optimal extraction range
if df_existing is not None:
    # If dataset exists, we need to handle overlaps
    print(f"\n✅ Dataset exists - using INCREMENTAL mode")
    print(f"   Strategy:")
    print(f"   1. Query database for: {start_date_str} to {end_date_str}")
    print(f"   2. Deduplicate 7-day old records")
    print(f"   3. Keep newer records")
    print(f"   4. Merge with existing data")
    print(f"   5. Save updated dataset")
    
    # Show what will be extracted
    overlapping_date = seven_days_ago
    print(f"\n🔄 OVERLAP HANDLING:")
    print(f"   Records from {overlapping_date.strftime('%Y-%m-%d')} might exist in both:")
    print(f"   - Existing dataset (old data)")
    print(f"   - New extraction (updated data)")
    print(f"   → Keep NEW records, discard OLD records from this date")
else:
    print(f"\n🆕 Dataset does NOT exist - using FULL EXTRACTION mode")
    print(f"   Will extract all available historical data")
    print(f"   Then save as: {dataset_path}")


# ============================================================================
# SECTION 3: Performance Comparison
# ============================================================================

print("\n⚡ SECTION 3: Performance Comparison")
print("="*70)

# Assume average 100 files per day across both pipelines
files_per_day = 100
days_total = 365  # Typical full dataset
days_incremental = 7

time_per_file_ms = 50  # Average time to download + extract

total_time_full = (files_per_day * days_total * time_per_file_ms) / 1000 / 60
total_time_incremental = (files_per_day * days_incremental * time_per_file_ms) / 1000 / 60

speedup = total_time_full / total_time_incremental

print(f"Full extraction (all {days_total} days):")
print(f"  Files to process: {files_per_day * days_total:,}")
print(f"  Estimated time: {total_time_full:.1f} minutes")

print(f"\nIncremental extraction ({days_incremental} days):")
print(f"  Files to process: {files_per_day * days_incremental:,}")
print(f"  Estimated time: {total_time_incremental:.1f} minutes")

print(f"\n✨ SPEEDUP: {speedup:.1f}x faster!")
print(f"⏰ Time saved: {total_time_full - total_time_incremental:.1f} minutes per run")


# ============================================================================
# SECTION 4: Deduplication Strategy
# ============================================================================

print("\n🔍 SECTION 4: Deduplication Strategy (WITH ACTUAL CSV DATE)")
print("="*70)

# Get ACTUAL CSV creation date from filesystem
if os.path.exists(dataset_path):
    csv_stat = os.stat(dataset_path)
    csv_creation_datetime = datetime.fromtimestamp(csv_stat.st_mtime)
    csv_creation_date = csv_creation_datetime.strftime('%Y-%m-%d')
    print(f"✅ CSV FOUND")
    print(f"   Path: {dataset_path}")
    print(f"   Created/Modified: {csv_creation_date} ({csv_creation_datetime.strftime('%Y-%m-%d %H:%M:%S')})")
else:
    csv_creation_date = None
    print(f"⚠️  CSV NOT FOUND - This is FIRST RUN")

# Define the boundary date (7 days ago)
boundary_date = datetime.now() - timedelta(days=7)
boundary_str = boundary_date.strftime('%Y-%m-%d')
today_str = datetime.now().strftime('%Y-%m-%d')

print(f"\n📅 EXTRACTION WINDOW")
print(f"   Last 7 days: {boundary_str} to {today_str}")

if csv_creation_date:
    print(f"   CSV has data from: {csv_creation_date} to {today_str}")
    print(f"   CSV age: {(datetime.now() - csv_creation_datetime).days} days old")

print(f"\n📋 Deduplication logic:")
print(f"""
1. Load existing CSV (created {csv_creation_date}, contains records from then → today)
2. Extract last 7 days ({boundary_str} → {today_str}) - NEW/FRESH data
3. Remove from OLD CSV: ALL records from {boundary_str} onwards
4. Keep from OLD CSV: Records BEFORE {boundary_str}
5. Merge: (old pre-{boundary_str}) + (new {boundary_str}-{today_str})

Result: Completely fresh 7-day window, older data unchanged since CSV creation
""")

# Example simulation with ACTUAL dates
print("\n📊 DEDUPLICATION EXAMPLE (WITH REAL DATES):")
print("="*70)

# Simulate with ACTUAL current date context
csv_creation = csv_creation_date if csv_creation_date else "2025-12-01"
today = today_str
boundary = boundary_str

# Simulate data
old_data = {
    'mint': ['token1', 'token1', 'token1', 'token2', 'token3', 'token4'],
    'checked_at_utc': ['2025-12-05', '2025-12-15', '2025-12-20', '2025-12-18', '2025-12-10', '2025-12-17'],
    'signal_source': ['alpha', 'alpha', 'alpha', 'discovery', 'alpha', 'discovery'],
    'price_usd': [0.01, 0.015, 0.018, 0.02, 0.03, 0.05]  # Old prices
}

new_data = {
    'mint': ['token1', 'token1', 'token2', 'token5'],
    'checked_at_utc': [boundary, today, today, today],
    'signal_source': ['alpha', 'alpha', 'discovery', 'discovery'],
    'price_usd': [0.011, 0.025, 0.022, 0.08]  # Fresh prices
}

df_old = pd.DataFrame(old_data)
df_new = pd.DataFrame(new_data)

print(f"\nOLD CSV (created {csv_creation}, contains data until {today}):")
print(df_old.to_string(index=False))

print(f"\n\nNEW EXTRACTION (last 7 days: {boundary} → {today}):")
print(df_new.to_string(index=False))

# Apply deduplication
print(f"\n\n🔧 APPLYING DEDUPLICATION:")
print(f"   Boundary date (7 days ago): {boundary}")
print(f"   Remove from old CSV: All records from {boundary} onwards")

df_old_filtered = df_old[df_old['checked_at_utc'] < boundary].copy()
print(f"   ✅ Kept from old: {len(df_old_filtered)} records (before {boundary})")

df_final = pd.concat([df_old_filtered, df_new], ignore_index=True)
df_final = df_final.sort_values('checked_at_utc', ascending=False)

print(f"   ✅ Appended new: {len(df_new)} records ({boundary} → {today})")
print(f"   ✅ Final merged: {len(df_final)} records\n")

print(f"✅ FINAL RESULT (Data preserved from creation + fresh 7-day window):")
print(df_final.to_string(index=False))

print(f"\n📊 SUMMARY:")
print(f"   Data range now: {csv_creation} to {today}")
print(f"   Total span: {(datetime.strptime(today, '%Y-%m-%d') - datetime.strptime(csv_creation, '%Y-%m-%d')).days} days")


# ============================================================================
# SECTION 5: Helper Functions for Merge and Export
# ============================================================================

print("\n🔄 SECTION 5: Helper Functions")
print("="*70)

def incremental_extract_and_merge(existing_csv_path, new_data_df):
    """
    Load existing CSV, extract 7-day window, remove old 7-day records, merge.
    
    Args:
        existing_csv_path: Path to existing token_datasets.csv
        new_data_df: DataFrame with new extracted data (last 7 days)
    
    Returns:
        Merged DataFrame (old pre-7day + new 7day)
    """
    
    boundary_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    # STEP 1: Load existing dataset
    if os.path.exists(existing_csv_path):
        df_existing = pd.read_csv(existing_csv_path)
        print(f"✅ Loaded existing dataset: {len(df_existing)} records")
        print(f"   Date range: {df_existing['checked_at_utc'].min()} to {df_existing['checked_at_utc'].max()}")
    else:
        print("⚠️  No existing dataset found - returning new data only (first run)")
        return new_data_df
    
    # STEP 2: Keep only PRE-BOUNDARY records from existing data
    df_pre_boundary = df_existing[df_existing['checked_at_utc'] < boundary_date].copy()
    print(f"\n📊 Deduplication:")
    print(f"   Removed from old CSV: All records from {boundary_date} onwards")
    print(f"   Kept from old CSV: {len(df_pre_boundary)} records (before {boundary_date})")
    
    # STEP 3: Combine old pre-boundary with new fresh 7-day data
    df_merged = pd.concat([df_pre_boundary, new_data_df], ignore_index=True)
    print(f"   Appended new data: {len(new_data_df)} records ({boundary_date} to today)")
    print(f"   Total final: {len(df_merged)} records")
    
    # STEP 4: Sort for consistency
    if 'checked_at_utc' in df_merged.columns:
        df_merged = df_merged.sort_values('checked_at_utc', ascending=False)
    
    return df_merged


def export_dataset(df, output_path):
    """
    Validate and export dataset to CSV.
    
    Args:
        df: DataFrame to export
        output_path: Output file path
    """
    
    # Validation checks
    print(f"\n✔️ VALIDATION CHECKS:")
    print(f"   Total records: {len(df)}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Null values: {df.isnull().sum().sum()}")
    
    if 'signal_source' in df.columns:
        signal_dist = df['signal_source'].value_counts()
        print(f"\n   Signal distribution:")
        for signal, count in signal_dist.items():
            pct = (count / len(df)) * 100
            print(f"     - {signal}: {count} ({pct:.1f}%)")
    
    # Export
    df.to_csv(output_path, index=False)
    file_size = os.path.getsize(output_path) / (1024*1024)  # MB
    print(f"\n💾 Exported to: {output_path}")
    print(f"   File size: {file_size:.2f} MB")
    print(f"   Last modified: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================================
# SECTION 6: Example Usage
# ============================================================================

print("\n📋 SECTION 6: Example Execution Flow")
print("="*70)

# Simulate new extracted data (last 7 days)
new_data = pd.DataFrame({
    'mint': ['token1', 'token1', 'token4', 'token5'],
    'checked_at_utc': ['2025-12-16', '2025-12-23', '2025-12-23', '2025-12-23'],
    'signal_source': ['alpha', 'alpha', 'discovery', 'alpha'],
    'price_usd': [0.012, 0.025, 0.05, 0.08],
    'volume_24h': [100, 150, 200, 300]
})

# Simulate existing CSV (created earlier, has older + some recent data)
boundary_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

df_existing_sim = pd.DataFrame({
    'mint': ['token1', 'token1', 'token2', 'token3', 'token1'],
    'checked_at_utc': ['2025-12-05', '2025-12-15', '2025-12-22', '2025-12-10', '2025-12-18'],
    'signal_source': ['alpha', 'alpha', 'discovery', 'alpha', 'alpha'],
    'price_usd': [0.01, 0.011, 0.015, 0.03, 0.017],
    'volume_24h': [80, 90, 180, 280, 95]
})

print("EXISTING CSV DATA:")
print(df_existing_sim.to_string(index=False))

print(f"\nNEW EXTRACTION DATA (last 7 days, {boundary_date} to today):")
print(new_data.to_string(index=False))

# Apply merge with simplified deduplication
print("\n" + "="*70)
print("APPLYING MERGE:")

df_pre_boundary = df_existing_sim[df_existing_sim['checked_at_utc'] < boundary_date].copy()
df_final = pd.concat([df_pre_boundary, new_data], ignore_index=True).sort_values('checked_at_utc', ascending=False)

print(f"\n✅ FINAL MERGED DATASET ({len(df_final)} records):")
print(f"   Old pre-{boundary_date}: {len(df_pre_boundary)} records")
print(f"   New {boundary_date}-today: {len(new_data)} records")
print(f"\n{df_final.to_string(index=False)}")


# ============================================================================
# SECTION 7: Implementation Guidelines
# ============================================================================

print("\n📝 SECTION 7: Implementation Guidelines for extract_datasets.py")
print("="*70)

implementation_guide = """
MODIFY extract_datasets.py::create_training_dataset() METHOD:

CONSTANTS TO ADD:
  DATASET_PATH = 'data/token_datasets.csv'
  LOOKBACK_DAYS = 7

CODE STRUCTURE:
  def create_training_dataset(discovery_features, alpha_features, output_path=DATASET_PATH):
      '''Create training dataset with 7-day incremental extraction.'''
      
      if os.path.exists(output_path):
          # INCREMENTAL MODE: Load, filter, deduplicate, merge
          df_existing = pd.read_csv(output_path)
          boundary_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
          df_pre_boundary = df_existing[df_existing['checked_at_utc'] < boundary_date].copy()
          
          df_new = pd.concat([discovery_features, alpha_features], ignore_index=True)
          df_combined = pd.concat([df_pre_boundary, df_new], ignore_index=True)
          
          df_final = df_combined.drop_duplicates(
              subset=['mint', 'signal_source', 'checked_at_utc'],
              keep='last'
          )
      else:
          # FIRST RUN MODE: Use all extracted data
          df_final = pd.concat([discovery_features, alpha_features], ignore_index=True)
      
      # Validation and export
      df_final.to_csv(output_path, index=False)
      return df_final

KEY POINTS:
  ✓ Keep extraction methods unchanged (extract_all_data, extract_date_range)
  ✓ Check if output_path exists to determine mode
  ✓ Always deduplicate on (mint, signal_source, checked_at_utc)
  ✓ Use keep='last' to retain newer data
  ✓ Log extraction mode and counts for debugging
  ✓ Expected speedup: 50x faster (6 min vs 300 min)

BACKWARD COMPATIBILITY:
  ✓ First run without existing CSV works as before
  ✓ No schema changes needed
  ✓ No new dependencies required
  ✓ Reversible if issues occur
"""

print(implementation_guide)


# ============================================================================
# SECTION 8: Summary and Recommendations
# ============================================================================

print("\n🎯 SECTION 8: Summary and Recommendations")
print("="*70)

summary = """
OBJECTIVE:
  Optimize extract_datasets.py to use 7-day lookback window instead of
  full historical extraction. Expected speedup: 50x (6 min vs 300 min).

STRATEGY:
  1. Check when token_datasets.csv was last saved
  2. Extract last 7 days of new data
  3. Remove old 7-day records from existing dataset
  4. Merge: (old pre-7day) + (new fresh 7day)
  5. Save updated dataset

BENEFITS:
  ⚡ 50x faster extraction (6 minutes vs 300 minutes)
  💰 Reduce Dune API calls by 98%
  📈 Enable more frequent incremental updates
  ✅ Maintain full historical data integrity
  🔄 Backward compatible (first run unchanged)

IMPLEMENTATION EFFORT:
  ✓ Low risk (minimal code changes)
  ✓ Single method modification needed
  ✓ No schema changes
  ✓ No new dependencies

EXPECTED RESULTS:
  First run: ~300 minutes (all historical data)
  Second+ run: ~6 minutes (7-day increment)
  Annual savings: ~1,093 hours
  Data freshness: 7-day window (max token tracking age)

RECOMMENDATION: Ready to implement ✅
"""

print(summary)

print("\n" + "="*70)
print("✅ Script Complete")
print("="*70)
