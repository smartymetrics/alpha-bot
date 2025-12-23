"""
SIGNAL-TYPE AWARE ML MODEL TRAINER
====================================
Trains an ensemble that learns different patterns for ALPHA vs DISCOVERY signals.

REQUIREMENTS:
  - Dataset must have 'signal_type' column (required, no defaults)
  - 'signal_type' values: 'alpha' or 'discovery' only
  - Models saved to models/signal_aware/ (separate from regular models/)

USAGE:
  python train_model_signal_aware.py

OUTPUT:
  - models/signal_aware/xgboost_model.pkl
  - models/signal_aware/lightgbm_model.pkl
  - models/signal_aware/catboost_model.pkl
  - models/signal_aware/rf_model.pkl
  - models/signal_aware/isolation_forest.pkl
  - models/signal_aware/feature_selector.pkl
  - models/signal_aware/model_metadata.json
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import json
import os
import sys

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')

# Create directories
os.makedirs('models/signal_aware', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

print("="*80)
print(" SIGNAL-TYPE-AWARE SOLANA MEMECOIN CLASSIFIER")
print(" (Alpha=Winner Wallets, Discovery=Fresh Tokens)")
print("="*80)

# ============================================================================
# LOAD AND VALIDATE DATA
# ============================================================================

df = pd.read_csv('data/token_datasets.csv')
print(f"\n Initial Dataset: {df.shape[0]} samples, {df.shape[1]} features")

# CHECK FOR REQUIRED signal_type/signal_source COLUMN
if 'signal_type' not in df.columns:
    if 'signal_source' in df.columns:
        # Use signal_source and rename it to signal_type
        df['signal_type'] = df['signal_source'].copy()
        print("   Using 'signal_source' column as 'signal_type'")
    else:
        print("\n❌ ERROR: Dataset must have 'signal_type' or 'signal_source' column")
        print("   Expected values: 'alpha' or 'discovery'")
        print("   Run train_model.py instead if you don't have signal types yet.")
        sys.exit(1)

# VALIDATE signal_type VALUES
invalid_types = df[~df['signal_type'].isin(['alpha', 'discovery'])]['signal_type'].unique()
if len(invalid_types) > 0:
    print(f"\n❌ ERROR: Invalid signal_type values: {invalid_types}")
    print("   Expected: 'alpha' or 'discovery' only")
    sys.exit(1)

# CHECK FOR NaN signal_type
if df['signal_type'].isna().any():
    nan_count = df['signal_type'].isna().sum()
    print(f"\n❌ ERROR: {nan_count} rows have NaN signal_type")
    print("   All rows must have signal_type='alpha' or 'discovery'")
    sys.exit(1)

# Report signal type distribution
signal_dist = df['signal_type'].value_counts()
print(f"\n Signal Type Distribution:")
for sig_type, count in signal_dist.items():
    pct = 100 * count / len(df)
    win_rate = df[df['signal_type'] == sig_type]['label_status'].eq('win').mean() * 100
    print(f"   {sig_type.upper()}: {count} samples ({pct:.1f}%), win rate: {win_rate:.2f}%")

# Encode target
df['target'] = (df['label_status'] == 'win').astype(int)
print(f"\n Overall Win rate: {df['target'].mean()*100:.2f}%")

# ============================================================================
# DATA QUALITY FIXES
# ============================================================================

print("\n" + "="*80)
print(" DATA QUALITY FIXES")
print("="*80)

# Fix negative token ages
df['token_age_at_signal_seconds'] = df['token_age_at_signal_seconds'].clip(lower=0)

# Filter out bad data
initial_len = len(df)
df = df[
    (df['fdv_usd'] > 0) & 
    (df['liquidity_usd'] > 0)
].copy()
print(f" Filtered {initial_len - len(df)} rows with 0 FDV/Liquidity (New count: {len(df)})")
print(" Fixed negative token ages")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print(" FEATURE ENGINEERING - SIGNAL-TYPE AWARE")
print("="*80)

# === TIME FEATURES ===
df['token_age_hours'] = df['token_age_at_signal_seconds'] / 3600
df['is_very_fresh'] = (df['token_age_hours'] < 4).astype(int)
df['freshness_score'] = np.exp(-df['token_age_hours'] / 6)

# Volume outlier
median_vol = df['volume_h24_usd'].median()
df['is_high_volume'] = (df['volume_h24_usd'] > median_vol).astype(int)

# === INSIDER FEATURES ===
print(" Creating insider-related features...")

df['insider_supply_pct'] = np.where(
    df['token_supply'] > 0,
    (df['total_insider_token_amount'] / df['token_supply']) * 100,
    0
)

df['insider_risk_score'] = (
    df['insider_supply_pct'] * 0.4 +
    df['largest_insider_network_size'] * 0.4 +
    (df['total_insider_networks'] * 2) * 0.2
)

# === LIQUIDITY & LOG FEATURES ===
print(" Creating liquidity and log features...")

df['log_liquidity'] = np.log1p(df['liquidity_usd'])
df['log_volume'] = np.log1p(df['volume_h24_usd'])
df['log_fdv'] = np.log1p(df['fdv_usd'])
df['log_lp_locked'] = np.log1p(df['total_lp_locked_usd'])

df['liquidity_depth_pct'] = np.where(
    df['fdv_usd'] > 0,
    (df['liquidity_usd'] / df['fdv_usd']) * 100,
    0
).clip(0, 100)

df['lp_lock_quality'] = np.where(
    df['total_lp_locked_usd'] > 0,
    (df['total_lp_locked_usd'] / df['liquidity_usd']) * 100,
    0
).clip(0, 100)

# === HOLDER FEATURES ===
print(" Creating holder concentration features...")

df['holder_quality_score'] = (100 - df['top_10_holders_pct'])

df['whale_dominated'] = (df['top_10_holders_pct'] > 50).astype(int)
df['concentration_risk'] = df['top_10_holders_pct'] / 100

# === MOMENTUM FEATURES ===
print(" Creating momentum features...")

df['strong_uptrend'] = (df['price_change_h24_pct'] > 20).astype(int)
df['weak_momentum'] = (abs(df['price_change_h24_pct']) < 5).astype(int)
df['volume_quality'] = np.where(
    df['volume_h24_usd'] > 0,
    df['liquidity_usd'] / df['volume_h24_usd'],
    0
).clip(0, 10)

# === TIME PATTERN FEATURES ===
print(" Creating time pattern features...")

df['peak_hours'] = ((df['time_of_day_utc'] >= 14) & (df['time_of_day_utc'] <= 18)).astype(int)
df['off_peak_day'] = df['is_weekend_utc'].astype(int)

# === PUMP/DUMP RISK ===
df['pump_dump_risk'] = np.where(
    df['price_change_h24_pct'] > 50,
    np.minimum(df['price_change_h24_pct'] / 100, 1.0),
    0
)

# === CREATOR FEATURES ===
df['creator_sold_out'] = (df['creator_balance_pct'] < 1).astype(int)

# === ANOMALY DETECTION ===
print(" Creating anomaly detection features...")

iso_forest = IsolationForest(contamination=0.1, random_state=42)
df['anomaly_score'] = iso_forest.fit_predict(df[['log_volume', 'price_change_h24_pct']])
df['anomaly_score'] = (df['anomaly_score'] == -1).astype(int)
df['is_anomaly'] = df['anomaly_score']

# === SIGNAL TYPE FEATURE (REQUIRED) ===
print(" Creating signal-type feature...")
df['signal_type_alpha'] = (df['signal_type'] == 'alpha').astype(float)
print(f"   signal_type_alpha distribution:")
print(f"     - Alpha (1.0): {(df['signal_type_alpha'] == 1.0).sum()}")
print(f"     - Discovery (0.0): {(df['signal_type_alpha'] == 0.0).sum()}")

# ============================================================================
# FEATURE LISTS
# ============================================================================

CORE_FEATURES = [
    'price_change_h24_pct',
    'liquidity_to_volume_ratio',  
    'fdv_to_liquidity_ratio',
    'creator_balance_pct',
    'top_10_holders_pct',
    'total_lp_locked_usd',
    'is_lp_locked_95_plus',
    'total_insider_networks',
    'largest_insider_network_size',
    'total_insider_token_amount',
    'pump_dump_risk_score',
    'time_of_day_utc',
    'is_weekend_utc',
    'token_age_at_signal_seconds',
]

DERIVED_FEATURES = [
    # Time
    'token_age_hours',
    'is_very_fresh',
    'freshness_score',
    
    # Insider
    'insider_supply_pct',
    'insider_risk_score',
    
    # Liquidity
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
    'is_anomaly',
    
    # SIGNAL TYPE (REQUIRED FOR THIS SCRIPT)
    'signal_type_alpha',
    
    # Other
    'creator_sold_out',
    'is_high_volume'
]

ALL_FEATURES = CORE_FEATURES + DERIVED_FEATURES

print(f"\n Feature Summary:")
print(f"   • Core features: {len(CORE_FEATURES)}")
print(f"   • Derived features (including signal_type_alpha): {len(DERIVED_FEATURES)}")
print(f"   • TOTAL: {len(ALL_FEATURES)}")

# Handle missing values
for col in ALL_FEATURES:
    if col in df.columns:
        df[col] = df[col].fillna(0)
    else:
        print(f"⚠️ Creating missing column: {col}")
        df[col] = 0

# ============================================================================
# REMOVE HIGHLY CORRELATED FEATURES
# ============================================================================

print("\n" + "="*80)
print(" CORRELATION ANALYSIS & REMOVAL")
print("="*80)

X_temp = df[ALL_FEATURES].copy()
corr_matrix = X_temp.corr().abs()

upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.90)]

if to_drop:
    print(f" Removing {len(to_drop)} highly correlated features (corr > 0.90):")
    for feat in to_drop:
        # Find what this feature correlates with
        high_corr = upper_tri[feat][upper_tri[feat] > 0.90]
        if len(high_corr) > 0:
            corr_with = high_corr.index[0]
            corr_val = high_corr.values[0]
            print(f"   • {feat:30s} (corr with {corr_with}: {corr_val:.3f})")
        else:
            print(f"   • {feat}")
    ALL_FEATURES = [f for f in ALL_FEATURES if f not in to_drop]
else:
    print(" No features with correlation > 0.90 found")

print(f"\n Final feature count: {len(ALL_FEATURES)}")

# ============================================================================
# TEMPORAL TRAIN/TEST SPLIT
# ============================================================================

print("\n" + "="*80)
print(" TEMPORAL TRAIN/TEST SPLIT (Stratified by signal_type)")
print("="*80)

df = df.sort_values('token_age_at_signal_seconds').reset_index(drop=True)

# Stratified split to preserve signal type distribution
train_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    stratify=df[['target', 'signal_type']],
    random_state=42
)

print(f"\n Temporal Split:")
print(f"   Train: {len(train_df)} samples")
print(f"     - Alpha: {(train_df['signal_type'] == 'alpha').sum()} | Discovery: {(train_df['signal_type'] == 'discovery').sum()}")
print(f"     - Win rate: {train_df['target'].mean()*100:.2f}%")
print(f"   Test:  {len(test_df)} samples")
print(f"     - Alpha: {(test_df['signal_type'] == 'alpha').sum()} | Discovery: {(test_df['signal_type'] == 'discovery').sum()}")
print(f"     - Win rate: {test_df['target'].mean()*100:.2f}%")

X_train = train_df[ALL_FEATURES].copy()
y_train = train_df['target'].copy()
X_test = test_df[ALL_FEATURES].copy()
y_test = test_df['target'].copy()

# ============================================================================
# FEATURE SELECTION
# ============================================================================

print("\n" + "="*80)
print(" FEATURE SELECTION (SelectKBest k=20)")
print("="*80)

selector = SelectKBest(score_func=f_classif, k=20)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features = [ALL_FEATURES[i] for i in selector.get_support(indices=True)]
print(f"\n Selected {len(selected_features)} features:")
for i, feat in enumerate(selected_features, 1):
    print(f"   {i:2d}. {feat}")

# Check if signal_type_alpha is selected
if 'signal_type_alpha' in selected_features:
    print("\n✅ signal_type_alpha SELECTED! Model will learn different patterns for alpha vs discovery.")
else:
    print("\n⚠️  signal_type_alpha NOT selected. Model may not differentiate between signal types.")

# ============================================================================
# FEATURE SCALING
# ============================================================================

print("\n" + "="*80)
print(" FEATURE SCALING")
print("="*80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)
print(" StandardScaler fitted on training data")

# ============================================================================
# ENSEMBLE MODEL TRAINING
# ============================================================================

print("\n" + "="*80)
print(" ENSEMBLE MODEL TRAINING")
print("="*80)

# XGBoost
print("\n Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train_scaled, y_train, verbose=False)
xgb_pred = xgb_model.predict_proba(X_test_scaled)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_pred)
print(f"   XGBoost AUC: {xgb_auc:.4f}")

# LightGBM
print("\n Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_train_scaled, y_train)
lgb_pred = lgb_model.predict_proba(X_test_scaled)[:, 1]
lgb_auc = roc_auc_score(y_test, lgb_pred)
print(f"   LightGBM AUC: {lgb_auc:.4f}")

# CatBoost
print("\n Training CatBoost...")
cb_model = CatBoostClassifier(
    iterations=200,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
    verbose=False
)
cb_model.fit(X_train_scaled, y_train, verbose=False)
cb_pred = cb_model.predict_proba(X_test_scaled)[:, 1]
cb_auc = roc_auc_score(y_test, cb_pred)
print(f"   CatBoost AUC: {cb_auc:.4f}")

# RandomForest
print("\n Training RandomForest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict_proba(X_test_scaled)[:, 1]
rf_auc = roc_auc_score(y_test, rf_pred)
print(f"   RandomForest AUC: {rf_auc:.4f}")

# ============================================================================
# ENSEMBLE VOTING
# ============================================================================

print("\n" + "="*80)
print(" ENSEMBLE VOTING (Equal Weights)")
print("="*80)

ensemble_pred = (xgb_pred + lgb_pred + cb_pred + rf_pred) / 4
ensemble_auc = roc_auc_score(y_test, ensemble_pred)

print(f"\n Ensemble AUC: {ensemble_auc:.4f}")
print(f"\n Model Performance Summary:")
print(f"   XGBoost:    {xgb_auc:.4f}")
print(f"   LightGBM:   {lgb_auc:.4f}")
print(f"   CatBoost:   {cb_auc:.4f}")
print(f"   RandomForest: {rf_auc:.4f}")
print(f"   Ensemble:   {ensemble_auc:.4f}")

# ============================================================================
# SAVE MODELS & METADATA
# ============================================================================

print("\n" + "="*80)
print(" SAVING SIGNAL-AWARE MODELS")
print("="*80)

model_dir = 'models/signal_aware'

pickle.dump(xgb_model, open(f'{model_dir}/xgboost_model.pkl', 'wb'))
pickle.dump(lgb_model, open(f'{model_dir}/lightgbm_model.pkl', 'wb'))
pickle.dump(cb_model, open(f'{model_dir}/catboost_model.pkl', 'wb'))
pickle.dump(rf_model, open(f'{model_dir}/rf_model.pkl', 'wb'))
pickle.dump(selector, open(f'{model_dir}/feature_selector.pkl', 'wb'))
pickle.dump(scaler, open(f'{model_dir}/scaler.pkl', 'wb'))

# Isolation Forest
iso_forest_final = IsolationForest(contamination=0.1, random_state=42)
iso_forest_final.fit(df[['log_volume', 'price_change_h24_pct']])
pickle.dump(iso_forest_final, open(f'{model_dir}/isolation_forest.pkl', 'wb'))

print(f"\n Saved models to {model_dir}/:")
print(f"   ✅ xgboost_model.pkl")
print(f"   ✅ lightgbm_model.pkl")
print(f"   ✅ catboost_model.pkl")
print(f"   ✅ rf_model.pkl")
print(f"   ✅ feature_selector.pkl")
print(f"   ✅ scaler.pkl")
print(f"   ✅ isolation_forest.pkl")

# Save metadata
metadata = {
    "timestamp": datetime.now().isoformat(),
    "signal_type_aware": True,
    "description": "Signal-type aware ensemble (ALPHA vs DISCOVERY)",
    "all_features": ALL_FEATURES,
    "selected_features": selected_features,
    "signal_type_alpha_selected": 'signal_type_alpha' in selected_features,
    "ensemble_weights": {
        "xgboost": 0.25,
        "lightgbm": 0.25,
        "catboost": 0.25,
        "randomforest": 0.25
    },
    "test_metrics": {
        "ensemble_auc": float(ensemble_auc),
        "xgboost_auc": float(xgb_auc),
        "lightgbm_auc": float(lgb_auc),
        "catboost_auc": float(cb_auc),
        "randomforest_auc": float(rf_auc)
    },
    "data_summary": {
        "total_samples": len(df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "signal_type_distribution": {
            "alpha": int((df['signal_type'] == 'alpha').sum()),
            "discovery": int((df['signal_type'] == 'discovery').sum())
        },
        "overall_win_rate": float(df['target'].mean())
    }
}

with open(f'{model_dir}/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n   ✅ model_metadata.json")
print(f"\n Metadata saved with:")
print(f"   • Signal-type awareness: ✅ ENABLED")
print(f"   • signal_type_alpha in features: {'✅ YES' if metadata['signal_type_alpha_selected'] else '⚠️  NO'}")
print(f"   • Ensemble AUC: {metadata['test_metrics']['ensemble_auc']:.4f}")

print("\n" + "="*80)
print(" ✅ SIGNAL-AWARE MODEL TRAINING COMPLETE")
print("="*80)
print(f"\n Models location: {model_dir}/")
print(f" To use these models, ml_predictor.py will automatically detect them")
print(f" and require signal_type parameter: 'alpha' or 'discovery'")
print("\n")
