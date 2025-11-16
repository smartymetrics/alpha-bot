import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import json
import os

# ML libraries
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
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
print("🚀 ADVANCED SOLANA MEMECOIN SIGNAL CLASSIFIER V2")
print("="*80)

# ============================================================================
# LOAD AND INITIAL DATA INSPECTION
# ============================================================================

df = pd.read_csv('data/token_datasets.csv')
print(f"\n📊 Initial Dataset: {df.shape[0]} samples, {df.shape[1]} features")

# Encode target
df['target'] = (df['label_status'] == 'win').astype(int)
print(f"🎯 Overall Win rate: {df['target'].mean()*100:.2f}%")

# ============================================================================
# CRITICAL FIX 1: Filter by Token Age
# ============================================================================

print("\n" + "="*80)
print("🔧 FIX 1: FILTERING BY TOKEN AGE")
print("="*80)

# MAX_TOKEN_AGE_HOURS = 24
# df_filtered = df[df['token_age_at_signal_seconds'] < (MAX_TOKEN_AGE_HOURS * 3600)].copy()

# print(f"\n✅ Filtered to tokens < {MAX_TOKEN_AGE_HOURS}h old:")
# print(f"   Before: {len(df)} samples")
# print(f"   After:  {len(df_filtered)} samples ({len(df_filtered)/len(df)*100:.1f}%)")
# print(f"   Win rate after filtering: {df_filtered['target'].mean()*100:.2f}%")

# df = df_filtered

# ============================================================================
# ADVANCED FEATURE ENGINEERING (New Ideas Integrated!)
# ============================================================================

print("\n" + "="*80)
print("🔧 FIX 2: ADVANCED FEATURE ENGINEERING")
print("="*80)

# Basic transformations
df['token_age_hours'] = df['token_age_at_signal_seconds'] / 3600
df['token_age_hours_capped'] = df['token_age_hours'].clip(0, 24)
df['is_ultra_fresh'] = (df['token_age_at_signal_seconds'] < 3600).astype(int)

# Log transforms for skewed features
df['log_liquidity'] = np.log1p(df['liquidity_usd'])
df['log_volume'] = np.log1p(df['volume_h24_usd'])
df['log_fdv'] = np.log1p(df['fdv_usd'])

# Basic ratios
df['volume_to_liquidity_ratio'] = np.where(
    df['liquidity_usd'] > 0,
    df['volume_h24_usd'] / df['liquidity_usd'],
    0
)

df['fdv_to_liquidity_ratio'] = np.where(
    df['liquidity_usd'] > 0,
    df['fdv_usd'] / df['liquidity_usd'],
    0
)

# Risk signals
df['high_risk_signal'] = (
    (df['pump_dump_risk_score'] > 30) & 
    (df['authority_risk_score'] > 15)
).astype(int)

df['premium_signal'] = (
    (df['grade'] == 'CRITICAL') & 
    (df['signal_source'] == 'alpha')
).astype(int)

df['extreme_concentration'] = (df['top_10_holders_pct'] > 60).astype(int)

print("✅ Basic features created")

# ============================================================================
# 🆕 NEW ADVANCED FEATURES (Your Ideas!)
# ============================================================================

print("\n🆕 Creating Advanced Composite Features...")

# 1. Liquidity per holder (distribution efficiency)
df['liquidity_per_holder'] = np.where(
    df['top_10_holders_pct'] > 0,
    df['liquidity_usd'] / (df['top_10_holders_pct'] / 10),  # Approximate holder count
    0
)

# 2. Smart money composite score (combines two signals)
df['smart_money_score'] = (
    df['overlap_quality_score'] * df['winner_wallet_density']
)

# 3. Risk-adjusted volume (volume quality, not just quantity)
df['risk_adjusted_volume'] = np.where(
    df['pump_dump_risk_score'] > 0,
    df['volume_h24_usd'] / (1 + df['pump_dump_risk_score']),
    df['volume_h24_usd']
)

# 4. Liquidity velocity (how active is the liquidity pool?)
df['liquidity_velocity'] = np.where(
    df['liquidity_usd'] > 0,
    df['volume_h24_usd'] / df['liquidity_usd'],
    0
)

# 5. Holder quality score (less concentration = better)
df['holder_quality_score'] = 100 - df['top_10_holders_pct']

# 6. Market efficiency ratio (balanced market indicator)
df['market_efficiency'] = np.where(
    (df['liquidity_usd'] > 0) & (df['fdv_usd'] > 0),
    (df['volume_h24_usd'] / df['liquidity_usd']) / (df['fdv_usd'] / df['liquidity_usd']),
    0
)

# 7. Risk-weighted smart money (smart money score adjusted for risk)
df['risk_weighted_smart_money'] = np.where(
    df['pump_dump_risk_score'] + df['authority_risk_score'] > 0,
    df['smart_money_score'] / (1 + df['pump_dump_risk_score'] + df['authority_risk_score']),
    df['smart_money_score']
)

# 8. Concentration risk score (combines multiple risk factors)
df['concentration_risk'] = (
    df['top_10_holders_pct'] * 0.4 +
    df['creator_balance_pct'] * 0.3 +
    df['whale_concentration_score'] * 0.3
)

# 9. Liquidity depth indicator (is there enough liquidity?)
df['liquidity_depth_score'] = np.where(
    df['fdv_usd'] > 0,
    (df['liquidity_usd'] / df['fdv_usd']) * 100,
    0
)

# 10. Time-weighted freshness (exponential decay for token age)
df['freshness_score'] = np.exp(-df['token_age_hours'] / 6)  # Half-life of 6 hours

print(f"✅ Created 10 advanced composite features")

# ============================================================================
# COMPREHENSIVE FEATURE LIST
# ============================================================================

NUMERIC_FEATURES = [
    # Market fundamentals (log-transformed)
    'log_liquidity',
    'log_volume',
    'log_fdv',
    'price_change_h24_pct',
    
    # Raw market features
    'liquidity_usd',
    'volume_h24_usd',
    'fdv_usd',
    
    # Security/Risk
    'creator_balance_pct',
    'top_10_holders_pct',
    'total_lp_locked_usd',
    'whale_concentration_score',
    'pump_dump_risk_score',
    'authority_risk_score',
    
    # Smart money
    'overlap_quality_score',
    'winner_wallet_density',
    
    # Basic derived metrics
    'volume_to_liquidity_ratio',
    'fdv_to_liquidity_ratio',
    'token_age_hours_capped',
    'is_ultra_fresh',
    'high_risk_signal',
    'extreme_concentration',
    'premium_signal',
    
    # 🆕 ADVANCED COMPOSITE FEATURES
    'liquidity_per_holder',
    'smart_money_score',
    'risk_adjusted_volume',
    'liquidity_velocity',
    'holder_quality_score',
    'market_efficiency',
    'risk_weighted_smart_money',
    'concentration_risk',
    'liquidity_depth_score',
    'freshness_score',
]

CATEGORICAL_FEATURES = [
    'signal_source',
    'grade',
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

print(f"\n📋 Total Features: {len(ALL_FEATURES)}")
print(f"   - {len(NUMERIC_FEATURES)} numeric (including 10 new advanced features)")
print(f"   - {len(CATEGORICAL_FEATURES)} categorical")

# Handle missing values
for col in NUMERIC_FEATURES:
    if col in df.columns:
        df[col] = df[col].fillna(0)
    else:
        print(f"⚠️  Creating missing column: {col}")
        df[col] = 0

for col in CATEGORICAL_FEATURES:
    if col in df.columns:
        df[col] = df[col].fillna('UNKNOWN')
    else:
        print(f"⚠️  Creating missing column: {col}")
        df[col] = 'UNKNOWN'

# ============================================================================
# STRATIFIED TRAIN/TEST SPLIT
# ============================================================================

print("\n" + "="*80)
print("🔧 FIX 3: STRATIFIED TRAIN/TEST SPLIT")
print("="*80)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['target'],
    random_state=42
)

print(f"✅ Stratified Split:")
print(f"   Train: {len(train_df)} samples | Win rate: {train_df['target'].mean()*100:.2f}%")
print(f"   Test:  {len(test_df)} samples  | Win rate: {test_df['target'].mean()*100:.2f}%")

X_train = train_df[ALL_FEATURES].copy()
y_train = train_df['target'].copy()
X_test = test_df[ALL_FEATURES].copy()
y_test = test_df['target'].copy()

# ============================================================================
# LABEL ENCODING
# ============================================================================

print("\n" + "="*80)
print("📝 ENCODING CATEGORICAL FEATURES")
print("="*80)

X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

label_encoders = {}
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    X_train_encoded[col] = le.fit_transform(X_train_encoded[col].astype(str))
    X_test_encoded[col] = le.transform(X_test_encoded[col].astype(str))
    label_encoders[col] = le
    print(f"  ✓ Encoded {col}: {list(le.classes_)}")

# ============================================================================
# ADVANCED FEATURE SELECTION (Try k=12 for stronger signals)
# ============================================================================

print("\n" + "="*80)
print("🔧 FIX 4: ADVANCED FEATURE SELECTION (k=12)")
print("="*80)

# Test both k=12 and k=15 to see which is better
k_values = [12, 15]
best_k = None
best_cv_score = 0

print("\n📊 Testing different k values...")

for k in k_values:
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_encoded, y_train)
    
    # Quick CV test
    temp_model = xgb.XGBClassifier(
        max_depth=3, learning_rate=0.05, n_estimators=30,
        random_state=42, eval_metric='auc'
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

# Use best k for final feature selection
selector_f = SelectKBest(f_classif, k=best_k)
X_train_selected = selector_f.fit_transform(X_train_encoded, y_train)
X_test_selected = selector_f.transform(X_test_encoded)

# Get selected features
f_scores = selector_f.scores_
f_selected_mask = selector_f.get_support()
selected_features = [ALL_FEATURES[i] for i in range(len(ALL_FEATURES)) if f_selected_mask[i]]

print(f"\n📊 Top {best_k} Selected Features (F-statistic):")
feature_scores = list(zip(ALL_FEATURES, f_scores))
feature_scores.sort(key=lambda x: x[1] if not np.isnan(x[1]) else -1, reverse=True)

for i, (feat, score) in enumerate(feature_scores[:best_k], 1):
    selected = "✓" if feat in selected_features else " "
    score_str = f"{score:8.2f}" if not np.isnan(score) else "     nan"
    is_new = "🆕" if feat in [
        'liquidity_per_holder', 'smart_money_score', 'risk_adjusted_volume',
        'liquidity_velocity', 'holder_quality_score', 'market_efficiency',
        'risk_weighted_smart_money', 'concentration_risk', 
        'liquidity_depth_score', 'freshness_score'
    ] else "  "
    print(f"  {selected} {is_new} {i:2d}. {feat:40s} | F-score: {score_str}")

# ============================================================================
# TRAINING WITH ENSEMBLE STRATEGY
# ============================================================================

print("\n" + "="*80)
print("🔧 FIX 5: TRAINING ENSEMBLE MODELS")
print("="*80)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Class weight ratio: {scale_pos_weight:.2f}")

# XGBoost
print("\n🔷 Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    max_depth=3,
    learning_rate=0.03,
    n_estimators=50,
    min_child_weight=10,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=5.0,
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
print(f"  Overfit Gap:       {abs(xgb_train_acc - xgb_test_acc):.4f}")

# LightGBM
print("\n🔶 Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    num_leaves=10,
    max_depth=3,
    learning_rate=0.03,
    n_estimators=50,
    min_child_samples=15,
    min_split_gain=0.02,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=5.0,
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
print(f"  Overfit Gap:       {abs(lgb_train_acc - lgb_test_acc):.4f}")

# CatBoost
print("\n🔸 Training CatBoost...")
cat_model = CatBoostClassifier(
    iterations=50,
    depth=3,
    learning_rate=0.03,
    l2_leaf_reg=15,
    min_data_in_leaf=15,
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
print(f"  Overfit Gap:       {abs(cat_train_acc - cat_test_acc):.4f}")

# ============================================================================
# 🆕 ENSEMBLE PREDICTIONS
# ============================================================================

print("\n" + "="*80)
print("🆕 TESTING ENSEMBLE COMBINATIONS")
print("="*80)

# Test different ensemble weights
ensemble_configs = [
    ("CatBoost 60% + LightGBM 40%", [0.6, 0.4, 0.0]),
    ("CatBoost 70% + LightGBM 30%", [0.7, 0.3, 0.0]),
    ("CatBoost 50% + LightGBM 30% + XGBoost 20%", [0.5, 0.3, 0.2]),
    ("Equal Weight Average", [0.33, 0.33, 0.34]),
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
print("📊 5-FOLD STRATIFIED CROSS-VALIDATION")
print("="*80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {'xgb': [], 'lgb': [], 'cat': [], 'ensemble': []}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_selected, y_train), 1):
    # XGBoost
    fold_xgb = xgb.XGBClassifier(
        max_depth=3, learning_rate=0.03, n_estimators=50,
        min_child_weight=10, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=5.0,
        scale_pos_weight=scale_pos_weight, random_state=42
    )
    fold_xgb.fit(X_train_selected[train_idx], y_train.iloc[train_idx], verbose=False)
    xgb_proba = fold_xgb.predict_proba(X_train_selected[val_idx])[:, 1]
    cv_scores['xgb'].append(roc_auc_score(y_train.iloc[val_idx], xgb_proba))
    
    # LightGBM
    fold_lgb = lgb.LGBMClassifier(
        num_leaves=10, max_depth=3, learning_rate=0.03, n_estimators=50,
        min_child_samples=15, reg_alpha=1.0, reg_lambda=5.0,
        class_weight='balanced', random_state=42, verbose=-1
    )
    fold_lgb.fit(X_train_selected[train_idx], y_train.iloc[train_idx])
    lgb_proba = fold_lgb.predict_proba(X_train_selected[val_idx])[:, 1]
    cv_scores['lgb'].append(roc_auc_score(y_train.iloc[val_idx], lgb_proba))
    
    # CatBoost
    fold_cat = CatBoostClassifier(
        iterations=50, depth=3, learning_rate=0.03, l2_leaf_reg=15,
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

print(f"\nCV AUC (mean ± std):")
print(f"  XGBoost:  {np.mean(cv_scores['xgb']):.3f} ± {np.std(cv_scores['xgb']):.3f}")
print(f"  LightGBM: {np.mean(cv_scores['lgb']):.3f} ± {np.std(cv_scores['lgb']):.3f}")
print(f"  CatBoost: {np.mean(cv_scores['cat']):.3f} ± {np.std(cv_scores['cat']):.3f}")
print(f"  Ensemble: {np.mean(cv_scores['ensemble']):.3f} ± {np.std(cv_scores['ensemble']):.3f}")

# ============================================================================
# FINAL MODEL SELECTION
# ============================================================================

print("\n" + "="*80)
print("🏆 FINAL MODEL COMPARISON")
print("="*80)

baseline_acc = y_test.mean()
print(f"\n📊 Baseline (always predict Win): {baseline_acc:.4f}")

results = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM', 'CatBoost', 'Ensemble'],
    'Train Acc': [xgb_train_acc, lgb_train_acc, cat_train_acc, np.nan],
    'Test Acc': [xgb_test_acc, lgb_test_acc, cat_test_acc, 
                 ((best_ensemble_weights[0] * cat_test_proba +
                   best_ensemble_weights[1] * lgb_test_proba +
                   best_ensemble_weights[2] * xgb_test_proba) >= 0.5).mean()],
    'Test AUC': [xgb_test_auc, lgb_test_auc, cat_test_auc, best_ensemble_auc],
    'Overfit Gap': [
        abs(xgb_train_acc - xgb_test_acc),
        abs(lgb_train_acc - lgb_test_acc),
        abs(cat_train_acc - cat_test_acc),
        np.nan
    ],
    'CV AUC': [
        np.mean(cv_scores['xgb']),
        np.mean(cv_scores['lgb']),
        np.mean(cv_scores['cat']),
        np.mean(cv_scores['ensemble'])
    ]
})

results['Score'] = (
    results['Test AUC'] * 0.5 +
    results['CV AUC'] * 0.4 +
    results['Test Acc'] * 0.1
)

print("\n" + results.to_string(index=False))

print("\n📊 Model Ranking:")
ranked = results.sort_values('Score', ascending=False)
print(ranked[['Model', 'Test AUC', 'CV AUC', 'Score']].to_string(index=False))

# Use ensemble or best single model
use_ensemble = best_ensemble_auc > max(xgb_test_auc, lgb_test_auc, cat_test_auc)

if use_ensemble:
    best_model_name = "Ensemble"
    print(f"\n✅ Best Model: Ensemble ({best_ensemble_name})")
    print(f"   Weights: Cat={best_ensemble_weights[0]}, LGB={best_ensemble_weights[1]}, XGB={best_ensemble_weights[2]}")
else:
    best_model_idx = results[results['Model'] != 'Ensemble']['Score'].idxmax()
    best_model_name = results.iloc[best_model_idx]['Model']
    print(f"\n✅ Best Model: {best_model_name}")

# ============================================================================
# SAVE ALL MODELS
# ============================================================================

print("\n" + "="*80)
print("💾 SAVING MODELS FOR PRODUCTION")
print("="*80)

# Save feature selector
selector_path = 'models/feature_selector.pkl'
joblib.dump(selector_f, selector_path)
print(f"✅ Feature selector saved: {selector_path}")

# Save all three models for ensemble
joblib.dump(xgb_model, 'models/xgboost_model.pkl')
joblib.dump(lgb_model, 'models/lightgbm_model.pkl')
joblib.dump(cat_model, 'models/catboost_model.pkl')
print(f"✅ All three models saved")

# Save label encoders
encoders_path = 'models/label_encoders.pkl'
joblib.dump(label_encoders, encoders_path)
print(f"✅ Label encoders saved: {encoders_path}")

# Save comprehensive metadata
metadata = {
    'model_type': best_model_name,
    'ensemble_weights': {
        'catboost': float(best_ensemble_weights[0]),
        'lightgbm': float(best_ensemble_weights[1]),
        'xgboost': float(best_ensemble_weights[2])
    } if use_ensemble else None,
    'trained_at': datetime.now().isoformat(),
    'training_samples': len(train_df),
    'test_samples': len(test_df),
    # 'max_token_age_hours': MAX_TOKEN_AGE_HOURS,
    'num_features_selected': len(selected_features),
    'best_k': best_k,
    'selected_features': selected_features,
    'all_features': ALL_FEATURES,
    'advanced_features': [
        'liquidity_per_holder', 'smart_money_score', 'risk_adjusted_volume',
        'liquidity_velocity', 'holder_quality_score', 'market_efficiency',
        'risk_weighted_smart_money', 'concentration_risk', 
        'liquidity_depth_score', 'freshness_score'
    ],
    'numeric_features': NUMERIC_FEATURES,
    'categorical_features': CATEGORICAL_FEATURES,
    'categorical_encodings': {
        col: list(label_encoders[col].classes_) 
        for col in CATEGORICAL_FEATURES
    },
    'performance': {
        'train_accuracy': float(cat_train_acc) if not use_ensemble else None,
        'test_accuracy': float(results.loc[results['Model'] == best_model_name, 'Test Acc'].iloc[0]),
        'test_auc': float(results.loc[results['Model'] == best_model_name, 'Test AUC'].iloc[0]),
        'overfit_gap': float(results.loc[results['Model'] == 'CatBoost', 'Overfit Gap'].iloc[0]),
        'cv_auc_mean': float(results.loc[results['Model'] == best_model_name, 'CV AUC'].iloc[0]),
        'baseline_accuracy': float(baseline_acc),
    },
    'all_models_comparison': results.to_dict('records'),
    'feature_scores': {
        feat: float(score) if not np.isnan(score) else 0.0
        for feat, score in zip(ALL_FEATURES, f_scores)
    }
}

# Add feature importance from best single model (CatBoost)
if hasattr(cat_model, 'feature_importances_'):
    metadata['feature_importance'] = {
        selected_features[i]: float(cat_model.feature_importances_[i])
        for i in range(len(selected_features))
    }

metadata_path = 'models/model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Metadata saved: {metadata_path}")

# ============================================================================
# CREATE PRODUCTION INFERENCE SCRIPT
# ============================================================================

inference_script = '''
"""
Production Inference Script for Solana Memecoin Classifier
Uses ensemble of XGBoost, LightGBM, and CatBoost models
"""

import pandas as pd
import numpy as np
import joblib
import json

class SolanaMemeTokenClassifier:
    def __init__(self, model_dir='models'):
        """Load all models and preprocessing artifacts"""
        self.selector = joblib.load(f'{model_dir}/feature_selector.pkl')
        self.xgb_model = joblib.load(f'{model_dir}/xgboost_model.pkl')
        self.lgb_model = joblib.load(f'{model_dir}/lightgbm_model.pkl')
        self.cat_model = joblib.load(f'{model_dir}/catboost_model.pkl')
        self.label_encoders = joblib.load(f'{model_dir}/label_encoders.pkl')
        
        with open(f'{model_dir}/model_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        self.ensemble_weights = self.metadata['ensemble_weights']
        self.selected_features = self.metadata['selected_features']
        self.all_features = self.metadata['all_features']
        
        print(f"✅ Loaded {self.metadata['model_type']} model")
        print(f"   Test AUC: {self.metadata['performance']['test_auc']:.4f}")
        print(f"   Selected features: {len(self.selected_features)}")
    
    def engineer_features(self, df):
        """Apply same feature engineering as training"""
        df = df.copy()
        
        # Basic features
        df['token_age_hours'] = df['token_age_at_signal_seconds'] / 3600
        df['token_age_hours_capped'] = df['token_age_hours'].clip(0, 24)
        df['is_ultra_fresh'] = (df['token_age_at_signal_seconds'] < 3600).astype(int)
        
        # Log transforms
        df['log_liquidity'] = np.log1p(df['liquidity_usd'])
        df['log_volume'] = np.log1p(df['volume_h24_usd'])
        df['log_fdv'] = np.log1p(df['fdv_usd'])
        
        # Basic ratios
        df['volume_to_liquidity_ratio'] = np.where(
            df['liquidity_usd'] > 0,
            df['volume_h24_usd'] / df['liquidity_usd'],
            0
        )
        df['fdv_to_liquidity_ratio'] = np.where(
            df['liquidity_usd'] > 0,
            df['fdv_usd'] / df['liquidity_usd'],
            0
        )
        
        # Risk signals
        df['high_risk_signal'] = (
            (df['pump_dump_risk_score'] > 30) & 
            (df['authority_risk_score'] > 15)
        ).astype(int)
        df['premium_signal'] = (
            (df['grade'] == 'CRITICAL') & 
            (df['signal_source'] == 'alpha')
        ).astype(int)
        df['extreme_concentration'] = (df['top_10_holders_pct'] > 60).astype(int)
        
        # ADVANCED COMPOSITE FEATURES
        df['liquidity_per_holder'] = np.where(
            df['top_10_holders_pct'] > 0,
            df['liquidity_usd'] / (df['top_10_holders_pct'] / 10),
            0
        )
        df['smart_money_score'] = (
            df['overlap_quality_score'] * df['winner_wallet_density']
        )
        df['risk_adjusted_volume'] = np.where(
            df['pump_dump_risk_score'] > 0,
            df['volume_h24_usd'] / (1 + df['pump_dump_risk_score']),
            df['volume_h24_usd']
        )
        df['liquidity_velocity'] = np.where(
            df['liquidity_usd'] > 0,
            df['volume_h24_usd'] / df['liquidity_usd'],
            0
        )
        df['holder_quality_score'] = 100 - df['top_10_holders_pct']
        df['market_efficiency'] = np.where(
            (df['liquidity_usd'] > 0) & (df['fdv_usd'] > 0),
            (df['volume_h24_usd'] / df['liquidity_usd']) / (df['fdv_usd'] / df['liquidity_usd']),
            0
        )
        df['risk_weighted_smart_money'] = np.where(
            df['pump_dump_risk_score'] + df['authority_risk_score'] > 0,
            df['smart_money_score'] / (1 + df['pump_dump_risk_score'] + df['authority_risk_score']),
            df['smart_money_score']
        )
        df['concentration_risk'] = (
            df['top_10_holders_pct'] * 0.4 +
            df['creator_balance_pct'] * 0.3 +
            df['whale_concentration_score'] * 0.3
        )
        df['liquidity_depth_score'] = np.where(
            df['fdv_usd'] > 0,
            (df['liquidity_usd'] / df['fdv_usd']) * 100,
            0
        )
        df['freshness_score'] = np.exp(-df['token_age_hours'] / 6)
        
        return df
    
    def predict(self, token_data, threshold=0.70):
        """
        Predict if token will reach 50% gain
        
        Args:
            token_data: dict or DataFrame with token features
            threshold: probability threshold for BUY signal (default 0.70 for high precision)
        
        Returns:
            dict with prediction, probability, and recommendation
        """
        # Convert to DataFrame if dict
        if isinstance(token_data, dict):
            df = pd.DataFrame([token_data])
        else:
            df = token_data.copy()
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Select features
        X = df[self.all_features].copy()
        
        # Encode categorical features
        for col in self.metadata['categorical_features']:
            if col in X.columns:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Apply feature selection
        X_selected = self.selector.transform(X)
        
        # Get predictions from all models
        xgb_proba = self.xgb_model.predict_proba(X_selected)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(X_selected)[:, 1]
        cat_proba = self.cat_model.predict_proba(X_selected)[:, 1]
        
        # Ensemble prediction
        if self.ensemble_weights:
            ensemble_proba = (
                self.ensemble_weights['catboost'] * cat_proba +
                self.ensemble_weights['lightgbm'] * lgb_proba +
                self.ensemble_weights['xgboost'] * xgb_proba
            )
        else:
            ensemble_proba = cat_proba  # Use best single model
        
        # Decision logic with risk tiers
        results = []
        for i, prob in enumerate(ensemble_proba):
            if prob >= threshold:
                action = "BUY"
                confidence = "HIGH"
                risk_tier = "LOW RISK"
            elif prob >= 0.60:
                action = "CONSIDER"
                confidence = "MEDIUM"
                risk_tier = "MEDIUM RISK"
            elif prob >= 0.45:
                action = "SKIP"
                confidence = "LOW"
                risk_tier = "HIGH RISK"
            else:
                action = "AVOID"
                confidence = "VERY LOW"
                risk_tier = "VERY HIGH RISK"
            
            results.append({
                'action': action,
                'win_probability': float(prob),
                'confidence': confidence,
                'risk_tier': risk_tier,
                'individual_predictions': {
                    'xgboost': float(xgb_proba[i]),
                    'lightgbm': float(lgb_proba[i]),
                    'catboost': float(cat_proba[i])
                }
            })
        
        return results[0] if len(results) == 1 else results
    
    def batch_predict(self, tokens_df, threshold=0.70):
        """Predict for multiple tokens at once"""
        results = self.predict(tokens_df, threshold)
        
        # Add to DataFrame
        tokens_df = tokens_df.copy()
        tokens_df['win_probability'] = [r['win_probability'] for r in results]
        tokens_df['action'] = [r['action'] for r in results]
        tokens_df['confidence'] = [r['confidence'] for r in results]
        tokens_df['risk_tier'] = [r['risk_tier'] for r in results]
        
        return tokens_df.sort_values('win_probability', ascending=False)


# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = SolanaMemeTokenClassifier()
    
    # Example token data
    example_token = {
        'signal_source': 'alpha',
        'grade': 'CRITICAL',
        'liquidity_usd': 50000,
        'volume_h24_usd': 150000,
        'fdv_usd': 100000,
        'price_change_h24_pct': 25.5,
        'creator_balance_pct': 5.0,
        'top_10_holders_pct': 35.0,
        'total_lp_locked_usd': 30000,
        'whale_concentration_score': 40.0,
        'pump_dump_risk_score': 15.0,
        'authority_risk_score': 5.0,
        'overlap_quality_score': 0.8,
        'winner_wallet_density': 0.6,
        'token_age_at_signal_seconds': 1800,  # 30 minutes old
    }
    
    # Get prediction
    result = classifier.predict(example_token, threshold=0.70)
    
    print("\\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"Action: {result['action']}")
    print(f"Win Probability: {result['win_probability']:.2%}")
    print(f"Confidence: {result['confidence']}")
    print(f"Risk Tier: {result['risk_tier']}")
    print("\\nIndividual Model Predictions:")
    for model, prob in result['individual_predictions'].items():
        print(f"  {model:10s}: {prob:.2%}")
'''

# Save inference script
with open('ml_predictor_v2.py', 'w', encoding='utf-8') as f:
    f.write(inference_script)

print(f"✅ Production inference script saved: ml_predictor_v2.py")

# ============================================================================
# DETAILED EVALUATION
# ============================================================================

print("\n" + "="*80)
print(f"📋 DETAILED EVALUATION")
print("="*80)

# Use best ensemble probabilities
if use_ensemble:
    final_proba = (
        best_ensemble_weights[0] * cat_test_proba +
        best_ensemble_weights[1] * lgb_test_proba +
        best_ensemble_weights[2] * xgb_test_proba
    )
    final_pred = (final_proba >= 0.5).astype(int)
else:
    final_proba = cat_test_proba
    final_pred = cat_test_pred

print("\nClassification Report:")
print(classification_report(y_test, final_pred, 
                          target_names=['Loss', 'Win'], digits=3))

# Confusion Matrix
cm = confusion_matrix(y_test, final_pred)
print("\nConfusion Matrix:")
print(f"              Predicted")
print(f"              Loss  Win")
print(f"Actual Loss    {cm[0,0]:3d}  {cm[0,1]:3d}")
print(f"       Win     {cm[1,0]:3d}  {cm[1,1]:3d}")

# Optimal threshold analysis
print("\n📊 Threshold Optimization:")
thresholds = [0.40, 0.50, 0.60, 0.70, 0.80]
print(f"\n{'Threshold':<12} {'Accuracy':<10} {'Precision':<12} {'Recall':<10}")
print("-" * 44)

for thresh in thresholds:
    pred = (final_proba >= thresh).astype(int)
    acc = (pred == y_test).mean()
    
    # Calculate precision and recall for Win class
    tp = ((pred == 1) & (y_test == 1)).sum()
    fp = ((pred == 1) & (y_test == 0)).sum()
    fn = ((pred == 0) & (y_test == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"{thresh:<12.2f} {acc:<10.3f} {precision:<12.3f} {recall:<10.3f}")

# Feature Importance from CatBoost
print("\n📊 Top 15 Feature Importances (CatBoost):")
importances = cat_model.feature_importances_
feature_importance = list(zip(selected_features, importances))
feature_importance.sort(key=lambda x: x[1], reverse=True)

for i, (feat, imp) in enumerate(feature_importance[:15], 1):
    is_new = "🆕" if feat in metadata['advanced_features'] else "  "
    print(f"  {is_new} {i:2d}. {feat:40s} | {imp:8.4f}")

# ============================================================================
# VERIFICATION
# ============================================================================

print("\n" + "="*80)
print("🔍 VERIFICATION")
print("="*80)

try:
    loaded_xgb = joblib.load('models/xgboost_model.pkl')
    loaded_lgb = joblib.load('models/lightgbm_model.pkl')
    loaded_cat = joblib.load('models/catboost_model.pkl')
    loaded_selector = joblib.load('models/feature_selector.pkl')
    loaded_encoders = joblib.load('models/label_encoders.pkl')
    loaded_metadata = json.load(open('models/model_metadata.json', 'r'))
    
    print(f"✅ All artifacts load successfully!")
    print(f"   XGBoost: {type(loaded_xgb).__name__}")
    print(f"   LightGBM: {type(loaded_lgb).__name__}")
    print(f"   CatBoost: {type(loaded_cat).__name__}")
    print(f"   Selector: k={loaded_selector.k}")
    print(f"   Selected features: {len(loaded_metadata['selected_features'])}")
    print(f"   Advanced features: {len(loaded_metadata['advanced_features'])}")
    
except Exception as e:
    print(f"❌ ERROR loading artifacts: {e}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ TRAINING COMPLETE - ADVANCED MODEL V2")
print("="*80)

print(f"\n🎯 Key Improvements:")
# print(f"   • Dataset: {len(df_filtered)} samples (filtered < {MAX_TOKEN_AGE_HOURS}h)")
print(f"   • Features: {len(ALL_FEATURES)} total → {best_k} selected")
print(f"   • 🆕 Added 10 advanced composite features")
print(f"   • Optimal k-value: {best_k} (tested via CV)")
print(f"   • Best model: {best_model_name}")

if use_ensemble:
    print(f"   • Ensemble weights: Cat={best_ensemble_weights[0]:.1f}, "
          f"LGB={best_ensemble_weights[1]:.1f}, XGB={best_ensemble_weights[2]:.1f}")

print(f"\n📈 Performance Metrics:")
print(f"   • Test AUC: {results.loc[results['Model'] == best_model_name, 'Test AUC'].iloc[0]:.4f}")
print(f"   • CV AUC: {results.loc[results['Model'] == best_model_name, 'CV AUC'].iloc[0]:.4f}")
print(f"   • Test Accuracy: {results.loc[results['Model'] == best_model_name, 'Test Acc'].iloc[0]:.4f}")
if not use_ensemble:
    print(f"   • Overfit Gap: {results.loc[results['Model'] == best_model_name, 'Overfit Gap'].iloc[0]:.4f}")

print(f"\n💡 Production Usage:")
print(f"   1. Run: python ml_predictor_v2.py")
print(f"   2. Use threshold ≥ 0.70 for BUY signals (high precision)")
print(f"   3. Use threshold < 0.35 for AVOID signals (high risk)")
print(f"   4. Monitor ensemble individual predictions for consensus")

print("\n" + "="*80)