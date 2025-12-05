import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import json
import os

# ML libraries
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

print("="*80)
print("🚀 IMPROVED SOLANA MEMECOIN CLASSIFIER - 6-MODEL ENSEMBLE + ANOMALY DETECTION")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

df = pd.read_csv('data/token_datasets.csv')
print(f"\n📊 Initial Dataset: {df.shape[0]} samples, {df.shape[1]} features")

# Encode target
df['target'] = (df['label_status'] == 'win').astype(int)
print(f"🎯 Overall Win rate: {df['target'].mean()*100:.2f}%")

# ============================================================================
# FIX DATA QUALITY ISSUES
# ============================================================================

print("\n" + "="*80)
print("🔧 DATA QUALITY FIXES")
print("="*80)

# Fix negative token ages
df['token_age_at_signal_seconds'] = df['token_age_at_signal_seconds'].clip(lower=0)
print("✅ Fixed negative token ages")

# ============================================================================
# STREAMLINED FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("🔧 FEATURE ENGINEERING - REDUCED CORRELATION")
print("="*80)

# === TIME FEATURES ===
df['token_age_hours'] = df['token_age_at_signal_seconds'] / 3600

# Token freshness (single binary flag instead of multiple overlapping)
df['is_very_fresh'] = (df['token_age_hours'] < 4).astype(int)
df['freshness_score'] = np.exp(-df['token_age_hours'] / 6)

# === INSIDER FEATURES (Simplified) ===
print("📊 Creating insider-related features...")

df['insider_supply_pct'] = np.where(
    df['token_supply'] > 0,
    (df['total_insider_token_amount'] / df['token_supply']) * 100,
    0
)

# Single composite insider risk (not multiple overlapping versions)
df['insider_risk_score'] = (
    df['insider_supply_pct'] * 0.4 +
    df['largest_insider_network_size'] * 0.4 +
    (df['total_insider_networks'] * 2) * 0.2
)

# === LIQUIDITY & VOLUME FEATURES (Log only, remove raw duplicates) ===
print("💧 Creating liquidity/volume features...")

df['log_liquidity'] = np.log1p(df['liquidity_usd'])
df['log_volume'] = np.log1p(df['volume_h24_usd'])
df['log_fdv'] = np.log1p(df['fdv_usd'])
df['log_lp_locked'] = np.log1p(df['total_lp_locked_usd'])

# Liquidity depth
df['liquidity_depth_pct'] = np.where(
    df['fdv_usd'] > 0,
    (df['liquidity_usd'] / df['fdv_usd']) * 100,
    0
)

# LP lock quality
df['lp_lock_quality'] = np.where(
    df['liquidity_usd'] > 0,
    (df['total_lp_locked_usd'] / df['liquidity_usd']) * 100,
    0
)

# liquidity to volume
df['liquidity_to_volume_ratio'] = np.where(
    df['volume_h24_usd'] > 0,
    df['liquidity_usd'] / df['volume_h24_usd'],
    0
)

# === HOLDER CONCENTRATION (Simplified) ===
print("👥 Creating holder concentration features...")

df['holder_quality_score'] = 100 - df['top_10_holders_pct']
df['whale_dominated'] = (df['top_10_holders_pct'] > 70).astype(int)

# Single concentration risk metric
df['concentration_risk'] = (
    df['top_10_holders_pct'] * 0.6 +
    df['creator_balance_pct'] * 0.4
)

# === MOMENTUM FEATURES ===
print("📈 Creating momentum features...")

df['strong_uptrend'] = (df['price_change_h24_pct'] > 50).astype(int)
df['weak_momentum'] = (df['price_change_h24_pct'] < 10).astype(int)

# Volume quality
df['volume_quality'] = np.where(
    abs(df['price_change_h24_pct']) > 0,
    df['volume_h24_usd'] / (1 + abs(df['price_change_h24_pct'])),
    df['volume_h24_usd']
)

# === TIME-BASED PATTERNS ===
print("🕐 Creating time-based features...")

df['peak_hours'] = ((df['time_of_day_utc'] >= 14) & (df['time_of_day_utc'] <= 20)).astype(int)
df['off_peak_day'] = (df['is_weekend_utc'] | df['is_public_holiday_any']).astype(int)

print("✅ Feature engineering complete!")

# ============================================================================
# ANOMALY DETECTION (ISOLATION FOREST)
# ============================================================================

print("\n" + "="*80)
print("👻 ANOMALY DETECTION (ISOLATION FOREST)")
print("="*80)

# Features to use for anomaly detection
anomaly_features = [
    'log_liquidity', 'log_volume', 'price_change_h24_pct', 
    'top_10_holders_pct', 'creator_balance_pct'
]

# Fill NaNs for anomaly detection
X_anomaly = df[anomaly_features].fillna(0)

# Train Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(X_anomaly)

# Generate anomaly scores (lower is more anomalous)
# We invert it so higher = more anomalous/rare
df['anomaly_score'] = -iso_forest.score_samples(X_anomaly)
df['is_anomaly'] = iso_forest.predict(X_anomaly)
df['is_anomaly'] = (df['is_anomaly'] == -1).astype(int)

print(f"✅ Anomaly detection complete. Found {df['is_anomaly'].sum()} anomalies.")
print(f"   Mean anomaly score: {df['anomaly_score'].mean():.4f}")

# ============================================================================
# REDUCED FEATURE LIST (Remove Overlapping Features)
# ============================================================================

# Core features (keep only non-redundant ones)
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

# Streamlined derived features
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
]

ALL_FEATURES = CORE_FEATURES + DERIVED_FEATURES

print(f"\n📋 Feature Summary:")
print(f"   • Core features: {len(CORE_FEATURES)}")
print(f"   • Derived features: {len(DERIVED_FEATURES)}")
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
print("🔍 CORRELATION ANALYSIS & REMOVAL")
print("="*80)

X_temp = df[ALL_FEATURES].copy()
corr_matrix = X_temp.corr().abs()

# Find features with correlation > 0.90
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.90)]

if to_drop:
    print(f"🔴 Removing {len(to_drop)} highly correlated features (corr > 0.90):")
    for feat in to_drop:
        print(f"   • {feat}")
    ALL_FEATURES = [f for f in ALL_FEATURES if f not in to_drop]
else:
    print("✅ No features with correlation > 0.90 found")

print(f"\n✅ Final feature count: {len(ALL_FEATURES)}")

# ============================================================================
# TEMPORAL TRAIN/TEST SPLIT (Fix Temporal Leakage)
# ============================================================================

print("\n" + "="*80)
print("📊 TEMPORAL TRAIN/TEST SPLIT")
print("="*80)

# Sort by token age to create temporal split
df = df.sort_values('token_age_at_signal_seconds').reset_index(drop=True)

# Use last 20% as test set (most recent tokens)
split_idx = int(len(df) * 0.8)
train_df = df[:split_idx].copy()
test_df = df[split_idx:].copy()

print(f"✅ Temporal Split:")
print(f"   Train: {len(train_df)} samples | Win rate: {train_df['target'].mean()*100:.2f}%")
print(f"   Test:  {len(test_df)} samples  | Win rate: {test_df['target'].mean()*100:.2f}%")
print(f"   Train age range: {train_df['token_age_hours'].min():.1f}h to {train_df['token_age_hours'].max():.1f}h")
print(f"   Test age range:  {test_df['token_age_hours'].min():.1f}h to {test_df['token_age_hours'].max():.1f}h")

X_train = train_df[ALL_FEATURES].copy()
y_train = train_df['target'].copy()
X_test = test_df[ALL_FEATURES].copy()
y_test = test_df['target'].copy()

# ============================================================================
# FEATURE SELECTION
# ============================================================================

print("\n" + "="*80)
print("🎯 FEATURE SELECTION")
print("="*80)

# Test different k values
k_values = [15, 20, 25]
best_k = None
best_cv_score = 0

print("\n📊 Testing different k values...")

for k in k_values:
    selector = SelectKBest(f_classif, k=min(k, len(ALL_FEATURES)))
    X_train_selected = selector.fit_transform(X_train, y_train)
    
    # Quick CV test with XGBoost
    temp_model = xgb.XGBClassifier(
        max_depth=3, 
        learning_rate=0.05, 
        n_estimators=50,
        min_child_weight=20,
        random_state=42, 
        eval_metric='auc'
    )
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in skf.split(X_train_selected, y_train):
        temp_model.fit(X_train_selected[train_idx], y_train.iloc[train_idx])
        score = roc_auc_score(
            y_train.iloc[val_idx],
            temp_model.predict_proba(X_train_selected[val_idx])[:, 1]
        )
        cv_scores.append(score)
    
    avg_score = np.mean(cv_scores)
    print(f"   k={k:2d} → CV AUC: {avg_score:.4f} ± {np.std(cv_scores):.4f}")
    
    if avg_score > best_cv_score:
        best_cv_score = avg_score
        best_k = k

print(f"\n✅ Best k value: {best_k} (CV AUC: {best_cv_score:.4f})")

# Apply best k
selector_f = SelectKBest(f_classif, k=best_k)
X_train_selected = selector_f.fit_transform(X_train, y_train)
X_test_selected = selector_f.transform(X_test)

# Get selected features
f_scores = selector_f.scores_
f_selected_mask = selector_f.get_support()
selected_features = [ALL_FEATURES[i] for i in range(len(ALL_FEATURES)) if f_selected_mask[i]]

print(f"\n📊 Top {best_k} Selected Features:")
feature_scores = list(zip(ALL_FEATURES, f_scores))
feature_scores.sort(key=lambda x: x[1] if not np.isnan(x[1]) else -1, reverse=True)

for i, (feat, score) in enumerate(feature_scores[:best_k], 1):
    selected = "✓" if feat in selected_features else " "
    score_str = f"{score:10.2f}" if not np.isnan(score) else "       nan"
    is_insider = "🔴" if 'insider' in feat.lower() else "  "
    is_derived = "🆕" if feat in DERIVED_FEATURES else "  "
    
    print(f"  {selected} {is_insider}{is_derived} {i:2d}. {feat:45s} | F-score: {score_str}")

# ============================================================================
# TRAIN ENSEMBLE MODELS (TUNED & CALIBRATED)
# ============================================================================

print("\n" + "="*80)
print("🚀 TRAINING 6-MODEL ENSEMBLE (CALIBRATED)")
print("="*80)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Class weight ratio: {scale_pos_weight:.2f}")

# 1. XGBoost (Calibrated)
print("\n🟦 Training XGBoost (Calibrated)...")
xgb_base = xgb.XGBClassifier(
    max_depth=4,               # Increased depth slightly
    learning_rate=0.05,
    n_estimators=100,          # Increased estimators
    min_child_weight=10,       # Reduced to allow more specific splits
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,             # Reduced regularization
    reg_lambda=1.0,            # Reduced regularization
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='auc'
)
xgb_model = CalibratedClassifierCV(xgb_base, method='isotonic', cv=3)
xgb_model.fit(X_train_selected, y_train)
xgb_test_proba = xgb_model.predict_proba(X_test_selected)[:, 1]
print(f"  Test AUC: {roc_auc_score(y_test, xgb_test_proba):.4f}")

# 2. LightGBM (Calibrated)
print("\n🟩 Training LightGBM (Calibrated)...")
lgb_base = lgb.LGBMClassifier(
    num_leaves=20,             # Increased
    max_depth=5,               # Increased
    learning_rate=0.05,
    n_estimators=100,
    min_child_samples=15,      # Reduced
    reg_alpha=0.5,             # Reduced
    reg_lambda=1.0,            # Reduced
    class_weight='balanced',
    random_state=42,
    verbose=-1
)
lgb_model = CalibratedClassifierCV(lgb_base, method='isotonic', cv=3)
lgb_model.fit(X_train_selected, y_train)
lgb_test_proba = lgb_model.predict_proba(X_test_selected)[:, 1]
print(f"  Test AUC: {roc_auc_score(y_test, lgb_test_proba):.4f}")

# 3. CatBoost (Calibrated)
print("\n🟨 Training CatBoost (Calibrated)...")
cat_base = CatBoostClassifier(
    iterations=100,
    depth=5,                   # Increased
    learning_rate=0.05,
    l2_leaf_reg=5,             # Reduced significantly (was 20)
    auto_class_weights='Balanced',
    random_state=42,
    verbose=False
)
cat_model = CalibratedClassifierCV(cat_base, method='isotonic', cv=3)
cat_model.fit(X_train_selected, y_train)
cat_test_proba = cat_model.predict_proba(X_test_selected)[:, 1]
print(f"  Test AUC: {roc_auc_score(y_test, cat_test_proba):.4f}")

# 4. Random Forest (New)
print("\n🌲 Training Random Forest (New)...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train_selected, y_train)
rf_test_proba = rf_model.predict_proba(X_test_selected)[:, 1]
print(f"  Test AUC: {roc_auc_score(y_test, rf_test_proba):.4f}")

# 5. Logistic Regression (New - Baseline)
print("\n📐 Training Logistic Regression (New)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

lr_model = LogisticRegression(
    C=0.1, 
    class_weight='balanced', 
    random_state=42
)
lr_model.fit(X_train_scaled, y_train)
lr_test_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
print(f"  Test AUC: {roc_auc_score(y_test, lr_test_proba):.4f}")

# 6. MLP Neural Network (New)
print("\n🧠 Training MLP Neural Network (New)...")
mlp_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=32,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)
mlp_model.fit(X_train_scaled, y_train)
mlp_test_proba = mlp_model.predict_proba(X_test_scaled)[:, 1]
print(f"  Test AUC: {roc_auc_score(y_test, mlp_test_proba):.4f}")

# ============================================================================
# ENSEMBLE TESTING
# ============================================================================

print("\n" + "="*80)
print("🎯 ENSEMBLE TESTING")
print("="*80)

# Define weights for 6 models
ensemble_configs = [
    ("Balanced 6-Model", [0.16, 0.16, 0.16, 0.16, 0.16, 0.20]),
    ("Boosters + RF + MLP", [0.2, 0.2, 0.2, 0.2, 0.0, 0.2]),
    ("RF + MLP Heavy", [0.1, 0.1, 0.1, 0.35, 0.0, 0.35]),
    ("Tree Heavy (No Linear/NN)", [0.25, 0.25, 0.25, 0.25, 0.0, 0.0]),
]

best_ensemble_name = None
best_ensemble_auc = 0
best_ensemble_weights = None

for name, weights in ensemble_configs:
    ensemble_proba = (
        weights[0] * xgb_test_proba +
        weights[1] * lgb_test_proba +
        weights[2] * cat_test_proba +
        weights[3] * rf_test_proba +
        weights[4] * lr_test_proba +
        weights[5] * mlp_test_proba
    )
    
    ensemble_auc = roc_auc_score(y_test, ensemble_proba)
    print(f"\n{name}:")
    print(f"  AUC: {ensemble_auc:.4f}")
    
    if ensemble_auc > best_ensemble_auc:
        best_ensemble_auc = ensemble_auc
        best_ensemble_name = name
        best_ensemble_weights = weights

print(f"\n✅ Best Ensemble: {best_ensemble_name}")
print(f"   Test AUC: {best_ensemble_auc:.4f}")

# ============================================================================
# SAVE MODELS
# ============================================================================

print("\n" + "="*80)
print("💾 SAVING MODELS")
print("="*80)

# Save all 6 models + extras
joblib.dump(xgb_model, 'models/xgboost_model.pkl')
joblib.dump(lgb_model, 'models/lightgbm_model.pkl')
joblib.dump(cat_model, 'models/catboost_model.pkl')
joblib.dump(rf_model, 'models/rf_model.pkl')
joblib.dump(lr_model, 'models/lr_model.pkl')
joblib.dump(mlp_model, 'models/mlp_model.pkl')
joblib.dump(iso_forest, 'models/isolation_forest.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(selector_f, 'models/feature_selector.pkl')

print("✅ Models saved")

metadata = {
    'model_version': '4.0_neural_anomaly',
    'model_type': '6-Model Ensemble',
    'ensemble_weights': {
        'xgboost': float(best_ensemble_weights[0]),
        'lightgbm': float(best_ensemble_weights[1]),
        'catboost': float(best_ensemble_weights[2]),
        'random_forest': float(best_ensemble_weights[3]),
        'logistic_regression': float(best_ensemble_weights[4]),
        'mlp_neural_network': float(best_ensemble_weights[5])
    },
    'trained_at': datetime.now().isoformat(),
    'performance': {
        'test_auc': float(best_ensemble_auc),
    },
    'selected_features': selected_features,
    'all_features': ALL_FEATURES,
    'anomaly_detection': {
        'model': 'IsolationForest',
        'contamination': 0.1,
        'features': anomaly_features
    }
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✅ Metadata saved")

# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

print("\n" + "="*80)
print("🎯 THRESHOLD OPTIMIZATION")
print("="*80)

final_proba = (
    best_ensemble_weights[0] * xgb_test_proba +
    best_ensemble_weights[1] * lgb_test_proba +
    best_ensemble_weights[2] * cat_test_proba +
    best_ensemble_weights[3] * rf_test_proba +
    best_ensemble_weights[4] * lr_test_proba +
    best_ensemble_weights[5] * mlp_test_proba
)

print(f"\n{'Threshold':<12} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10}")
print("-" * 54)

thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]

for thresh in thresholds:
    pred = (final_proba >= thresh).astype(int)
    acc = (pred == y_test).mean()
    
    tp = ((pred == 1) & (y_test == 1)).sum()
    fp = ((pred == 1) & (y_test == 0)).sum()
    fn = ((pred == 0) & (y_test == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{thresh:<12.2f} {acc:<10.3f} {precision:<12.3f} {recall:<10.3f} {f1:<10.3f}")

print("\n" + "="*80)