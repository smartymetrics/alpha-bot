#!/usr/bin/env python3
"""
Script to patch train_model.py with age-aware features and fixed temporal split
"""

import re

# Read the original file
with open('train_model.py', 'r', encoding='utf-8') as f:
    content = f.read()

# ====================================================================================
# PATCH 1: Add age-aware features to TIME FEATURES section
# ====================================================================================

# Find and replace the TIME FEATURES section
old_time_features = """# === TIME FEATURES ===
df['token_age_hours'] = df['token_age_at_signal_seconds'] / 3600

# Token freshness (single binary flag instead of multiple overlapping)
df['is_very_fresh'] = (df['token_age_hours'] < 4).astype(int)
df['freshness_score'] = np.exp(-df['token_age_hours'] / 6)"""

new_time_features = """# === TIME FEATURES (AGE-AWARE) ===
df['token_age_hours'] = df['token_age_at_signal_seconds' / 3600

# Age-based tracking windows (core feature for time-aware predictions)
df['is_young_token'] = (df['token_age_hours'] < 12).astype(int)  # <12h = 24h window, ≥12h = 168h window
df['tracking_window_hours'] = df['is_young_token'].map({1: 24, 0: 168})

# Token freshness (single binary flag instead of multiple overlapping)
df['is_very_fresh'] = (df['token_age_hours'] < 4).astype(int)
df['freshness_score'] = np.exp(-df['token_age_hours'] / 6)"""

content = content.replace(old_time_features, new_time_features)

# ====================================================================================
# PATCH 2: Add age-interaction features after TIME-BASED PATTERNS
# ====================================================================================

old_time_patterns = """# === TIME-BASED PATTERNS ===
print("🕐 Creating time-based features...")

df['peak_hours'] = ((df['time_of_day_utc'] >= 14) & (df['time_of_day_utc'] <= 20)).astype(int)
df['off_peak_day'] = (df['is_weekend_utc'] | df['is_public_holiday_any']).astype(int)

print("✅ Feature engineering complete!")"""

new_time_patterns = """# === TIME-BASED PATTERNS ===
print("🕐 Creating time-based features...")

df['peak_hours'] = ((df['time_of_day_utc'] >= 14) & (df['time_of_day_utc'] <= 20)).astype(int)
df['off_peak_day'] = (df['is_weekend_utc'] | df['is_public_holiday_any']).astype(int)

# === AGE-BASED INTERACTION FEATURES ===
print("🔗 Creating age-interaction features...")

# Momentum behaves differently for young vs old tokens
df['young_x_momentum'] = df['is_young_token'] * df['price_change_h24_pct']
df['old_x_volume'] = (1 - df['is_young_token']) * df['log_volume']

# Liquidity importance varies by age
df['young_x_liquidity'] = df['is_young_token'] * df['log_liquidity']

# Holder concentration risk varies by age
df['young_x_concentration'] = df['is_young_token'] * df['top_10_holders_pct']

print("✅ Feature engineering complete!")"""

content = content.replace(old_time_patterns, new_time_patterns)

# ====================================================================================
# PATCH 3: Add new interaction features to DERIVED_FEATURES list
# ====================================================================================

old_derived_list = """# Streamlined derived features
DERIVED_FEATURES = [
    # Time
    'token_age_hours',
    'is_very_fresh',
    'freshness_score',
    
    # Insider (single score instead of multiple overlapping)
    'insider_supply_pct',
    'insider_risk_score',
    
    # Liquidity (log only, no raw duplicates)
    'log_liquidity',
    'log_volume',
    'log_fdv',
    'log_lp_locked',
    'liquidity_depth_pct',
    'lp_lock_quality',
    
    # Holder
    'holder_quality_score',
    'whale_dominated',
    'concentration_risk',
    
    # Momentum
    'strong_uptrend',
    'weak_momentum',
    'volume_quality',
    
    # Time patterns
    'peak_hours',
    'off_peak_day',
    
    # Anomaly
    'anomaly_score',
    'is_anomaly'
]"""

new_derived_list = """# Streamlined derived features
DERIVED_FEATURES = [
    # Time (age-aware)
    'token_age_hours',
    'is_very_fresh',
    'freshness_score',
    'is_young_token',
    'tracking_window_hours',
    
    # Insider (single score instead of multiple overlapping)
    'insider_supply_pct',
    'insider_risk_score',
    
    # Liquidity (log only, no raw duplicates)
    'log_liquidity',
    'log_volume',
    'log_fdv',
    'log_lp_locked',
    'liquidity_depth_pct',
    'lp_lock_quality',
    
    # Holder
    'holder_quality_score',
    'whale_dominated',
    'concentration_risk',
    
    # Momentum
    'strong_uptrend',
    'weak_momentum',
    'volume_quality',
    
    # Time patterns
    'peak_hours',
    'off_peak_day',
    
    # Age-based interactions
    'young_x_momentum',
    'old_x_volume',
    'young_x_liquidity',
    'young_x_concentration',
    
    # Anomaly
    'anomaly_score',
    'is_anomaly'
]"""

content = content.replace(old_derived_list, new_derived_list)

# ====================================================================================
# PATCH 4: Fix temporal split to use signal timestamp instead of token age
# ====================================================================================

old_temporal_split = """# Sort by token age to create temporal split
df = df.sort_values('token_age_at_signal_seconds').reset_index(drop=True)

# Use last 20% as test set (most recent tokens)
split_idx = int(len(df) * 0.8)
train_df = df[:split_idx].copy()
test_df = df[split_idx:].copy()

print(f"✅ Temporal Split:")
print(f"   Train: {len(train_df)} samples | Win rate: {train_df['target'].mean()*100:.2f}%")
print(f"   Test:  {len(test_df)} samples  | Win rate: {test_df['target'].mean()*100:.2f}%")
print(f"   Train age range: {train_df['token_age_hours'].min():.1f}h to {train_df['token_age_hours'].max():.1f}h")
print(f"   Test age range:  {test_df['token_age_hours'].min():.1f}h to {test_df[ token_age_hours'].max():.1f}h")"""

new_temporal_split = """# Sort by SIGNAL TIMESTAMP for proper temporal validation (not token age!)
# This ensures we test on recent signals, not ancient tokens
df = df.sort_values('checked_at_timestamp').reset_index(drop=True)

# Use last 20% as test set (most RECENT SIGNALS)
split_idx = int(len(df) * 0.8)
train_df = df[:split_idx].copy()
test_df = df[split_idx:].copy()

print(f"✅ Temporal Split (by signal timestamp):")
print(f"   Train: {len(train_df)} samples | Win rate: {train_df['target'].mean()*100:.2f}%")
print(f"   Test:  {len(test_df)} samples  | Win rate: {test_df['target'].mean()*100:.2f}%")
print(f"   Train age range: {train_df['token_age_hours'].min():.1f}h to {train_df['token_age_hours'].max():.1f}h")
print(f"   Test age range:  {test_df['token_age_hours'].min():.1f}h to {test_df['token_age_hours'].max():.1f}h")

# Age-stratified metrics
print(f"\\n📊 Age Distribution:")
train_young = (train_df['is_young_token'] == 1).sum()
train_old = (train_df['is_young_token'] == 0).sum()
test_young = (test_df['is_young_token'] == 1).sum()
test_old = (test_df['is_young_token'] == 0).sum()

print(f"   Train: {train_young} young (<12h), {train_old} old (≥12h)")
print(f"   Test:  {test_young} young (<12h), {test_old} old (≥12h)")

if test_young > 0:
    test_young_win =  test_df[test_df['is_young_token'] == 1]['target'].mean()
    print(f"   Young token win rate (test): {test_young_win*100:.1f}%")
if test_old > 0:
    test_old_win = test_df[test_df['is_young_token'] == 0]['target'].mean()
    print(f"   Old token win rate (test): {test_old_win*100:.1f}%")"""

content = content.replace(old_temporal_split, new_temporal_split)

# Write modified content
with open('train_model.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Successfully patched train_model.py")
print("   - Added age-aware features (is_young_token, tracking_window_hours)")
print("   - Added age-interaction features")
print("   - Fixed temporal split to use checked_at_timestamp")
print("   - Added age-stratified validation metrics")
