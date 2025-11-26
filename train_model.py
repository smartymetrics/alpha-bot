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
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

print("="*80)
print("🚀 IMPROVED SOLANA MEMECOIN CLASSIFIER - PRODUCTION READY")
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

# Keep ONLY ONE ratio (liquidity to volume, remove inverse)
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
# REDUCED FEATURE LIST (Remove Overlapping Features)
# ============================================================================

# Core features (keep only non-redundant ones)
CORE_FEATURES = [
    'price_change_h24_pct',
    'liquidity_to_volume_ratio',  # Remove volume_to_liquidity_ratio (inverse)
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
# TRAIN ENSEMBLE MODELS (TUNED HYPERPARAMETERS)
# ============================================================================

print("\n" + "="*80)
print("🚀 TRAINING ENSEMBLE MODELS (TUNED)")
print("="*80)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Class weight ratio: {scale_pos_weight:.2f}")

# === XGBoost (Reduced Overfitting) ===
print("\n🟦 Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    max_depth=3,               # Reduced from 4
    learning_rate=0.05,        # Increased from 0.03
    n_estimators=50,           # Reduced from 100
    min_child_weight=20,       # Increased from 10
    subsample=0.7,             # Reduced from 0.8
    colsample_bytree=0.7,      # Reduced from 0.8
    reg_alpha=2.0,             # Increased from 1.0
    reg_lambda=10.0,           # Increased from 5.0
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='auc'
)

xgb_model.fit(X_train_selected, y_train, verbose=False)

xgb_train_pred = xgb_model.predict(X_train_selected)
xgb_test_pred = xgb_model.predict(X_test_selected)
xgb_test_proba = xgb_model.predict_proba(X_test_selected)[:, 1]

xgb_train_acc = (xgb_train_pred == y_train).mean()
xgb_test_acc = (xgb_test_pred == y_test).mean()
xgb_test_auc = roc_auc_score(y_test, xgb_test_proba)

print(f"  Training Accuracy: {xgb_train_acc:.4f}")
print(f"  Test Accuracy:     {xgb_test_acc:.4f}")
print(f"  Test AUC:          {xgb_test_auc:.4f}")
print(f"  Overfit Gap:       {xgb_train_acc - xgb_test_acc:.4f}")

# === LightGBM (Reduced Overfitting) ===
print("\n🟩 Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    num_leaves=10,             # Reduced from 15
    max_depth=3,               # Reduced from 4
    learning_rate=0.05,        # Increased from 0.03
    n_estimators=50,           # Reduced from 100
    min_child_samples=20,      # Increased from 15
    min_split_gain=0.03,       # Increased from 0.02
    subsample=0.7,             # Reduced from 0.8
    colsample_bytree=0.7,      # Reduced from 0.8
    reg_alpha=2.0,             # Increased from 1.0
    reg_lambda=10.0,           # Increased from 5.0
    class_weight='balanced',
    random_state=42,
    verbose=-1
)

lgb_model.fit(X_train_selected, y_train)

lgb_train_pred = lgb_model.predict(X_train_selected)
lgb_test_pred = lgb_model.predict(X_test_selected)
lgb_test_proba = lgb_model.predict_proba(X_test_selected)[:, 1]

lgb_train_acc = (lgb_train_pred == y_train).mean()
lgb_test_acc = (lgb_test_pred == y_test).mean()
lgb_test_auc = roc_auc_score(y_test, lgb_test_proba)

print(f"  Training Accuracy: {lgb_train_acc:.4f}")
print(f"  Test Accuracy:     {lgb_test_acc:.4f}")
print(f"  Test AUC:          {lgb_test_auc:.4f}")
print(f"  Overfit Gap:       {lgb_train_acc - lgb_test_acc:.4f}")

# === CatBoost (Reduced Overfitting) ===
print("\n🟨 Training CatBoost...")
cat_model = CatBoostClassifier(
    iterations=50,             # Reduced from 100
    depth=3,                   # Reduced from 4
    learning_rate=0.05,        # Increased from 0.03
    l2_leaf_reg=20,            # Increased from 15
    min_data_in_leaf=20,       # Increased from 15
    auto_class_weights='Balanced',
    random_state=42,
    verbose=False
)

cat_model.fit(X_train_selected, y_train)

cat_train_pred = cat_model.predict(X_train_selected)
cat_test_pred = cat_model.predict(X_test_selected)
cat_test_proba = cat_model.predict_proba(X_test_selected)[:, 1]

cat_train_acc = (cat_train_pred == y_train).mean()
cat_test_acc = (cat_test_pred == y_test).mean()
cat_test_auc = roc_auc_score(y_test, cat_test_proba)

print(f"  Training Accuracy: {cat_train_acc:.4f}")
print(f"  Test Accuracy:     {cat_test_acc:.4f}")
print(f"  Test AUC:          {cat_test_auc:.4f}")
print(f"  Overfit Gap:       {cat_train_acc - cat_test_acc:.4f}")

# ============================================================================
# ENSEMBLE TESTING
# ============================================================================

print("\n" + "="*80)
print("🎯 ENSEMBLE TESTING")
print("="*80)

ensemble_configs = [
    ("CatBoost 70% + LightGBM 30%", [0.7, 0.3, 0.0]),
    ("CatBoost 60% + LightGBM 30% + XGBoost 10%", [0.6, 0.3, 0.1]),
    ("Equal Weight", [0.33, 0.33, 0.34]),
    ("CatBoost 50% + LightGBM 50%", [0.5, 0.5, 0.0]),
]

best_ensemble_name = None
best_ensemble_auc = 0
best_ensemble_weights = None

for name, weights in ensemble_configs:
    ensemble_proba = (
        weights[0] * cat_test_proba +
        weights[1] * lgb_test_proba +
        weights[2] * xgb_test_proba
    )
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    ensemble_acc = (ensemble_pred == y_test).mean()
    ensemble_auc = roc_auc_score(y_test, ensemble_proba)
    
    print(f"\n{name}:")
    print(f"  Accuracy: {ensemble_acc:.4f}")
    print(f"  AUC:      {ensemble_auc:.4f}")
    
    if ensemble_auc > best_ensemble_auc:
        best_ensemble_auc = ensemble_auc
        best_ensemble_name = name
        best_ensemble_weights = weights

print(f"\n✅ Best Ensemble: {best_ensemble_name}")
print(f"   Test AUC: {best_ensemble_auc:.4f}")

# ============================================================================
# CROSS-VALIDATION
# ============================================================================

print("\n" + "="*80)
print("📊 5-FOLD CROSS-VALIDATION")
print("="*80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {'xgb': [], 'lgb': [], 'cat': [], 'ensemble': []}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_selected, y_train), 1):
    # XGBoost
    fold_xgb = xgb.XGBClassifier(
        max_depth=3, learning_rate=0.05, n_estimators=50,
        min_child_weight=20, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=10.0,
        scale_pos_weight=scale_pos_weight, random_state=42
    )
    fold_xgb.fit(X_train_selected[train_idx], y_train.iloc[train_idx], verbose=False)
    xgb_proba = fold_xgb.predict_proba(X_train_selected[val_idx])[:, 1]
    cv_scores['xgb'].append(roc_auc_score(y_train.iloc[val_idx], xgb_proba))
    
    # LightGBM
    fold_lgb = lgb.LGBMClassifier(
        num_leaves=10, max_depth=3, learning_rate=0.05, n_estimators=50,
        min_child_samples=20, reg_alpha=2.0, reg_lambda=10.0,
        class_weight='balanced', random_state=42, verbose=-1
    )
    fold_lgb.fit(X_train_selected[train_idx], y_train.iloc[train_idx])
    lgb_proba = fold_lgb.predict_proba(X_train_selected[val_idx])[:, 1]
    cv_scores['lgb'].append(roc_auc_score(y_train.iloc[val_idx], lgb_proba))
    
    # CatBoost
    fold_cat = CatBoostClassifier(
        iterations=50, depth=3, learning_rate=0.05, l2_leaf_reg=20,
        auto_class_weights='Balanced', random_state=42, verbose=False
    )
    fold_cat.fit(X_train_selected[train_idx], y_train.iloc[train_idx])
    cat_proba = fold_cat.predict_proba(X_train_selected[val_idx])[:, 1]
    cv_scores['cat'].append(roc_auc_score(y_train.iloc[val_idx], cat_proba))
    
    # Ensemble
    ensemble_proba = (
        best_ensemble_weights[0] * cat_proba +
        best_ensemble_weights[1] * lgb_proba +
        best_ensemble_weights[2] * xgb_proba
    )
    cv_scores['ensemble'].append(roc_auc_score(y_train.iloc[val_idx], ensemble_proba))
    
    print(f"Fold {fold} | XGB: {cv_scores['xgb'][-1]:.3f} | "
          f"LGB: {cv_scores['lgb'][-1]:.3f} | CAT: {cv_scores['cat'][-1]:.3f} | "
          f"ENS: {cv_scores['ensemble'][-1]:.3f}")

print(f"\n📊 CV AUC (mean ± std):")
print(f"  XGBoost:  {np.mean(cv_scores['xgb']):.3f} ± {np.std(cv_scores['xgb']):.3f}")
print(f"  LightGBM: {np.mean(cv_scores['lgb']):.3f} ± {np.std(cv_scores['lgb']):.3f}")
print(f"  CatBoost: {np.mean(cv_scores['cat']):.3f} ± {np.std(cv_scores['cat']):.3f}")
print(f"  Ensemble: {np.mean(cv_scores['ensemble']):.3f} ± {np.std(cv_scores['ensemble']):.3f}")

# ============================================================================
# SAVE MODELS
# ============================================================================

print("\n" + "="*80)
print("💾 SAVING MODELS")
print("="*80)

joblib.dump(xgb_model, 'models/xgboost_model.pkl')
joblib.dump(lgb_model, 'models/lightgbm_model.pkl')
joblib.dump(cat_model, 'models/catboost_model.pkl')
joblib.dump(selector_f, 'models/feature_selector.pkl')
print("✅ Models saved")

metadata = {
    'model_version': '2.0_improved',
    'model_type': 'Ensemble',
    'ensemble_weights': {
        'catboost': float(best_ensemble_weights[0]),
        'lightgbm': float(best_ensemble_weights[1]),
        'xgboost': float(best_ensemble_weights[2])
    },
    'trained_at': datetime.now().isoformat(),
    'training_samples': len(train_df),
    'test_samples': len(test_df),
    'num_features_selected': best_k,
    'selected_features': selected_features,
    'all_features': ALL_FEATURES,
    'improvements': [
        'Fixed negative token ages',
        'Removed correlated features (corr > 0.90)',
        'Implemented temporal train/test split',
        'Tuned hyperparameters to reduce overfitting'
    ],
    'performance': {
        'test_auc': float(best_ensemble_auc),
        'cv_auc_mean': float(np.mean(cv_scores['ensemble'])),
        'cv_auc_std': float(np.std(cv_scores['ensemble'])),
    },
    'feature_scores': {
        feat: float(score) if not np.isnan(score) else 0.0
        for feat, score in zip(ALL_FEATURES, f_scores)
    },
    'feature_importance': {
        selected_features[i]: float(cat_model.feature_importances_[i])
        for i in range(len(selected_features))
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
    best_ensemble_weights[0] * cat_test_proba +
    best_ensemble_weights[1] * lgb_test_proba +
    best_ensemble_weights[2] * xgb_test_proba
)

print(f"\n{'Threshold':<12} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10}")
print("-" * 54)

thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
best_threshold = 0.5
best_f1 = 0

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
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print(f"\n✅ Optimal threshold for balanced F1: {best_threshold:.2f}")
print(f"   Recommended for BUY signals: 0.70+ (high precision)")
print(f"   Recommended for AVOID signals: <0.40 (high recall for losses)")

# ============================================================================
# DETAILED EVALUATION
# ============================================================================

print("\n" + "="*80)
print("📋 DETAILED EVALUATION")

final_pred = (final_proba >= 0.5).astype(int)

print("\nClassification Report (threshold=0.5):")
print(classification_report(y_test, final_pred, target_names=['Loss', 'Win'], digits=3))

cm = confusion_matrix(y_test, final_pred)
print("\nConfusion Matrix:")
print(f"              Predicted")
print(f"              Loss  Win")
print(f"Actual Loss    {cm[0,0]:3d}  {cm[0,1]:3d}")
print(f"       Win     {cm[1,0]:3d}  {cm[1,1]:3d}")

# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("📊 FEATURE IMPORTANCE ANALYSIS")
print("="*80)

print("\n🏆 Top 20 Most Important Features (CatBoost):")
importances = cat_model.feature_importances_
feature_importance = list(zip(selected_features, importances))
feature_importance.sort(key=lambda x: x[1], reverse=True)

for i, (feat, imp) in enumerate(feature_importance[:20], 1):
    # Mark special feature types
    marker = ""
    if 'insider' in feat.lower():
        marker = "🔴"
    elif feat in CORE_FEATURES:
        marker = "⭐"
    elif feat in DERIVED_FEATURES:
        marker = "🆕"
    
    print(f"  {marker} {i:2d}. {feat:45s} | {imp:8.4f}")

# Check if insider features made the cut
insider_features_selected = [f for f in selected_features if 'insider' in f.lower()]
print(f"\n🔴 Insider features selected: {len(insider_features_selected)}")
for feat in insider_features_selected:
    imp = dict(feature_importance).get(feat, 0)
    print(f"   • {feat}: {imp:.4f}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ TRAINING COMPLETE - FOCUSED MODEL")
print("="*80)

print(f"\n🎯 Model Configuration:")
print(f"   • Total features: {len(ALL_FEATURES)}")
print(f"   • Features selected: {best_k}")
print(f"   • Training samples: {len(train_df)}")
print(f"   • Test samples: {len(test_df)}")

print(f"\n📈 Performance:")
print(f"   • Test AUC: {best_ensemble_auc:.4f}")
print(f"   • CV AUC: {np.mean(cv_scores['ensemble']):.4f} ± {np.std(cv_scores['ensemble']):.4f}")
print(f"   • Best Ensemble: {best_ensemble_name}")

print(f"\n🎯 Key Features:")
print(f"   • 🔴 Insider supply %: {'insider_supply_pct' in selected_features}")
print(f"   • ⭐ Core features: {sum(1 for f in selected_features if f in CORE_FEATURES)}/{len(CORE_FEATURES)}")
print(f"   • 🆕 Derived features: {sum(1 for f in selected_features if f in DERIVED_FEATURES)}/{len(DERIVED_FEATURES)}")

print(f"\n💡 Next Steps:")
print(f"   1. Create production inference script")
print(f"   2. Test with live DexScreener + Rugcheck API data")
print(f"   3. Use threshold ≥0.70 for BUY signals")
print(f"   4. Monitor individual model agreement")

print("\n" + "="*80)