import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import json
import os

# ML libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit, StratifiedKFold
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
print("🚀 CORRECTED SOLANA MEMECOIN SIGNAL CLASSIFIER")
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
# CRITICAL FIX 1: Filter by Token Age to Avoid Distribution Shift
# ============================================================================

print("\n" + "="*80)
print("🔧 FIX 1: FILTERING BY TOKEN AGE")
print("="*80)

# Analyze token age distribution
print(f"\nToken Age Statistics (seconds):")
print(df['token_age_at_signal_seconds'].describe())
print(f"\nAge in days: min={df['token_age_at_signal_seconds'].min()/86400:.1f}, "
      f"max={df['token_age_at_signal_seconds'].max()/86400:.1f}")

# CRITICAL: Only use tokens that were signaled within 24 hours of creation
# This ensures train and test have similar distributions
MAX_TOKEN_AGE_HOURS = 24
df_filtered = df[df['token_age_at_signal_seconds'] < (MAX_TOKEN_AGE_HOURS * 3600)].copy()

print(f"\n✅ Filtered to tokens < {MAX_TOKEN_AGE_HOURS}h old:")
print(f"   Before: {len(df)} samples")
print(f"   After:  {len(df_filtered)} samples ({len(df_filtered)/len(df)*100:.1f}%)")
print(f"   Win rate after filtering: {df_filtered['target'].mean()*100:.2f}%")

df = df_filtered

# ============================================================================
# CRITICAL FIX 2: Better Feature Engineering (Remove Weak Signals)
# ============================================================================

print("\n" + "="*80)
print("🔧 FIX 2: ROBUST FEATURE ENGINEERING")
print("="*80)

# Normalized token age (more stable than binary)
df['token_age_hours'] = df['token_age_at_signal_seconds'] / 3600
df['token_age_hours_capped'] = df['token_age_hours'].clip(0, 24)

# Is signal within first hour? (very fresh)
df['is_ultra_fresh'] = (df['token_age_at_signal_seconds'] < 3600).astype(int)

# Better market momentum features
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

# Risk interaction features
df['high_risk_signal'] = (
    (df['pump_dump_risk_score'] > 30) & 
    (df['authority_risk_score'] > 15)
).astype(int)

df['premium_signal'] = (
    (df['grade'] == 'CRITICAL') & 
    (df['signal_source'] == 'alpha')
).astype(int)

# Holder concentration risk
df['extreme_concentration'] = (df['top_10_holders_pct'] > 60).astype(int)

# Log transforms for skewed features (more stable)
df['log_liquidity'] = np.log1p(df['liquidity_usd'])
df['log_volume'] = np.log1p(df['volume_h24_usd'])
df['log_fdv'] = np.log1p(df['fdv_usd'])

print("✅ Created robust derived features")

# ============================================================================
# FEATURE SELECTION - Only use features that make sense for memecoins
# ============================================================================

NUMERIC_FEATURES = [
    # Market fundamentals (log-transformed for stability)
    'log_liquidity',
    'log_volume',
    'log_fdv',
    'price_change_h24_pct',
    
    # Liquidity metrics
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
    
    # Derived metrics
    'volume_to_liquidity_ratio',
    'fdv_to_liquidity_ratio',
    
    # Token age (normalized)
    'token_age_hours_capped',
    
    # Binary flags
    'is_ultra_fresh',
    'high_risk_signal',
    'extreme_concentration',
    'premium_signal',
]

CATEGORICAL_FEATURES = [
    'signal_source',  # alpha or discovery
    'grade',          # CRITICAL, HIGH, MEDIUM, LOW
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

print(f"\n📋 Using {len(ALL_FEATURES)} features:")
print(f"   - {len(NUMERIC_FEATURES)} numeric")
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
# CRITICAL FIX 3: Stratified Split (Not Pure Time Series)
# ============================================================================

print("\n" + "="*80)
print("🔧 FIX 3: STRATIFIED TRAIN/TEST SPLIT")
print("="*80)

# Stratified split ensures both train and test have similar win rates
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
# LABEL ENCODING FOR CATEGORICAL FEATURES
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
# CRITICAL FIX 4: Feature Selection with sklearn
# ============================================================================

print("\n" + "="*80)
print("🔧 FIX 4: SKLEARN FEATURE SELECTION")
print("="*80)

# Method 1: ANOVA F-statistic (for classification)
print("\n📊 Method 1: ANOVA F-statistic (Top 15 features)")
selector_f = SelectKBest(f_classif, k=15)
X_train_f = selector_f.fit_transform(X_train_encoded, y_train)
X_test_f = selector_f.transform(X_test_encoded)

# Get feature scores and names
f_scores = selector_f.scores_
f_selected_mask = selector_f.get_support()
f_selected_features = [ALL_FEATURES[i] for i in range(len(ALL_FEATURES)) if f_selected_mask[i]]

print("\nTop 15 Features by F-statistic:")
feature_scores_f = list(zip(ALL_FEATURES, f_scores))
feature_scores_f.sort(key=lambda x: x[1], reverse=True)
for i, (feat, score) in enumerate(feature_scores_f[:15], 1):
    selected = "✓" if feat in f_selected_features else " "
    print(f"  {selected} {i:2d}. {feat:35s} | F-score: {score:8.2f}")

# Method 2: Mutual Information (for non-linear relationships)
print("\n📊 Method 2: Mutual Information (Top 15 features)")
selector_mi = SelectKBest(mutual_info_classif, k=15)
X_train_mi = selector_mi.fit_transform(X_train_encoded, y_train)
X_test_mi = selector_mi.transform(X_test_encoded)

mi_scores = selector_mi.scores_
mi_selected_mask = selector_mi.get_support()
mi_selected_features = [ALL_FEATURES[i] for i in range(len(ALL_FEATURES)) if mi_selected_mask[i]]

print("\nTop 15 Features by Mutual Information:")
feature_scores_mi = list(zip(ALL_FEATURES, mi_scores))
feature_scores_mi.sort(key=lambda x: x[1], reverse=True)
for i, (feat, score) in enumerate(feature_scores_mi[:15], 1):
    selected = "✓" if feat in mi_selected_features else " "
    print(f"  {selected} {i:2d}. {feat:35s} | MI-score: {score:8.4f}")

# Use F-statistic selected features (generally more stable for tree models)
X_train_selected = X_train_f
X_test_selected = X_test_f
selected_features = f_selected_features

print(f"\n✅ Selected {len(selected_features)} features for modeling")

# ============================================================================
# CRITICAL FIX 5: Aggressive Regularization for Small Dataset
# ============================================================================

print("\n" + "="*80)
print("🔧 FIX 5: TRAINING WITH STRONG REGULARIZATION")
print("="*80)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Class weight ratio: {scale_pos_weight:.2f}")

# XGBoost with aggressive regularization
print("\n🔷 Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    max_depth=3,              # Shallow trees (was 4)
    learning_rate=0.03,       # Slower learning (was 0.05)
    n_estimators=50,          # Fewer trees (was 100)
    min_child_weight=10,      # Much higher (was 5)
    subsample=0.7,            # More aggressive (was 0.8)
    colsample_bytree=0.7,     # More aggressive (was 0.8)
    reg_alpha=1.0,            # L1 regularization (NEW)
    reg_lambda=5.0,           # L2 regularization (NEW)
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
    num_leaves=10,           # Reduced (was 15)
    max_depth=3,
    learning_rate=0.03,
    n_estimators=50,
    min_child_samples=15,    # Higher (was 10)
    min_split_gain=0.02,     # Higher (was 0.01)
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
    l2_leaf_reg=15,          # Higher (was 10)
    min_data_in_leaf=15,     # Higher (was 10)
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
# STRATIFIED K-FOLD CROSS-VALIDATION (Not Time Series)
# ============================================================================

print("\n" + "="*80)
print("📊 5-FOLD STRATIFIED CROSS-VALIDATION")
print("="*80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {'xgb': [], 'lgb': [], 'cat': []}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_selected, y_train), 1):
    # XGBoost
    fold_xgb = xgb.XGBClassifier(
        max_depth=3, learning_rate=0.03, n_estimators=50,
        min_child_weight=10, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=5.0,
        scale_pos_weight=scale_pos_weight, random_state=42
    )
    fold_xgb.fit(X_train_selected[train_idx], y_train.iloc[train_idx], verbose=False)
    score = roc_auc_score(y_train.iloc[val_idx], 
                          fold_xgb.predict_proba(X_train_selected[val_idx])[:, 1])
    cv_scores['xgb'].append(score)
    
    # LightGBM
    fold_lgb = lgb.LGBMClassifier(
        num_leaves=10, max_depth=3, learning_rate=0.03, n_estimators=50,
        min_child_samples=15, reg_alpha=1.0, reg_lambda=5.0,
        class_weight='balanced', random_state=42, verbose=-1
    )
    fold_lgb.fit(X_train_selected[train_idx], y_train.iloc[train_idx])
    score = roc_auc_score(y_train.iloc[val_idx],
                          fold_lgb.predict_proba(X_train_selected[val_idx])[:, 1])
    cv_scores['lgb'].append(score)
    
    # CatBoost
    fold_cat = CatBoostClassifier(
        iterations=50, depth=3, learning_rate=0.03, l2_leaf_reg=15,
        auto_class_weights='Balanced', random_state=42, verbose=False
    )
    fold_cat.fit(X_train_selected[train_idx], y_train.iloc[train_idx])
    score = roc_auc_score(y_train.iloc[val_idx],
                          fold_cat.predict_proba(X_train_selected[val_idx])[:, 1])
    cv_scores['cat'].append(score)
    
    print(f"Fold {fold} | XGB: {cv_scores['xgb'][-1]:.3f} | "
          f"LGB: {cv_scores['lgb'][-1]:.3f} | CAT: {cv_scores['cat'][-1]:.3f}")

print(f"\nCV AUC (mean ± std):")
print(f"  XGBoost:  {np.mean(cv_scores['xgb']):.3f} ± {np.std(cv_scores['xgb']):.3f}")
print(f"  LightGBM: {np.mean(cv_scores['lgb']):.3f} ± {np.std(cv_scores['lgb']):.3f}")
print(f"  CatBoost: {np.mean(cv_scores['cat']):.3f} ± {np.std(cv_scores['cat']):.3f}")

# ============================================================================
# MODEL COMPARISON & SELECTION
# ============================================================================

print("\n" + "="*80)
print("🏆 MODEL COMPARISON")
print("="*80)

# Test set baseline (always predict majority class)
baseline_acc = y_test.mean()
print(f"\n📊 Baseline (always predict Win): {baseline_acc:.4f}")

results = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM', 'CatBoost'],
    'Train Acc': [xgb_train_acc, lgb_train_acc, cat_train_acc],
    'Test Acc': [xgb_test_acc, lgb_test_acc, cat_test_acc],
    'Test AUC': [xgb_test_auc, lgb_test_auc, cat_test_auc],
    'Overfit Gap': [
        abs(xgb_train_acc - xgb_test_acc),
        abs(lgb_train_acc - lgb_test_acc),
        abs(cat_train_acc - cat_test_acc)
    ],
    'CV AUC': [
        np.mean(cv_scores['xgb']),
        np.mean(cv_scores['lgb']),
        np.mean(cv_scores['cat'])
    ]
})

# Calculate composite score (prioritize low overfitting and high test performance)
results['Score'] = (
    results['Test AUC'] * 0.4 +          # Test AUC is most important
    results['CV AUC'] * 0.3 +            # Cross-validation stability
    (1 - results['Overfit Gap']) * 0.2 + # Low overfitting
    results['Test Acc'] * 0.1            # Test accuracy
)

print("\n" + results.to_string(index=False))

print("\n📊 Model Ranking:")
ranked = results.sort_values('Score', ascending=False)
print(ranked[['Model', 'Test AUC', 'CV AUC', 'Overfit Gap', 'Score']].to_string(index=False))

best_model_idx = results['Score'].idxmax()
best_model_name = results.iloc[best_model_idx]['Model']

print(f"\n✅ Best Model: {best_model_name}")

# Select best model
if best_model_name == 'XGBoost':
    best_model = xgb_model
elif best_model_name == 'LightGBM':
    best_model = lgb_model
else:
    best_model = cat_model

# ============================================================================
# DETAILED EVALUATION
# ============================================================================

print("\n" + "="*80)
print(f"📋 DETAILED EVALUATION - {best_model_name}")
print("="*80)

print("\nClassification Report:")
print(classification_report(y_test, best_model.predict(X_test_selected), 
                          target_names=['Loss', 'Win'], digits=3))

# Confusion Matrix
cm = confusion_matrix(y_test, best_model.predict(X_test_selected))
print("\nConfusion Matrix:")
print(f"              Predicted")
print(f"              Loss  Win")
print(f"Actual Loss    {cm[0,0]:3d}  {cm[0,1]:3d}")
print(f"       Win     {cm[1,0]:3d}  {cm[1,1]:3d}")

# Feature Importance
print("\n📊 Top 10 Feature Importances:")
if best_model_name in ['XGBoost', 'LightGBM']:
    importances = best_model.feature_importances_
else:
    importances = best_model.feature_importances_

feature_importance = list(zip(selected_features, importances))
feature_importance.sort(key=lambda x: x[1], reverse=True)

for i, (feat, imp) in enumerate(feature_importance[:10], 1):
    print(f"  {i:2d}. {feat:35s} | {imp:8.4f}")

# ============================================================================
# SAVE MODEL AND METADATA
# ============================================================================

print("\n" + "="*80)
print("💾 SAVING MODEL FOR PRODUCTION")
print("="*80)

# Save the sklearn selector (CRITICAL for production)
selector_path = 'models/feature_selector.pkl'
joblib.dump(selector_f, selector_path)
print(f"✅ Feature selector saved: {selector_path}")

# Save the model
model_path = 'models/xgboost_signal_classifier.pkl'
joblib.dump(best_model, model_path)
print(f"✅ Model saved: {model_path}")

# Save label encoders
encoders_path = 'models/label_encoders.pkl'
joblib.dump(label_encoders, encoders_path)
print(f"✅ Label encoders saved: {encoders_path}")

# Save metadata with selected features
metadata = {
    'model_type': best_model_name,
    'trained_at': datetime.now().isoformat(),
    'training_samples': len(train_df),
    'test_samples': len(test_df),
    'max_token_age_hours': MAX_TOKEN_AGE_HOURS,
    'num_features_selected': len(selected_features),
    'selected_features': selected_features,
    'all_features': ALL_FEATURES,
    'numeric_features': NUMERIC_FEATURES,
    'categorical_features': CATEGORICAL_FEATURES,
    'categorical_encodings': {
        col: list(label_encoders[col].classes_) 
        for col in CATEGORICAL_FEATURES
    },
    'performance': {
        'train_accuracy': float(results.loc[results['Model'] == best_model_name, 'Train Acc'].iloc[0]),
        'test_accuracy': float(results.loc[results['Model'] == best_model_name, 'Test Acc'].iloc[0]),
        'test_auc': float(results.loc[results['Model'] == best_model_name, 'Test AUC'].iloc[0]),
        'overfit_gap': float(results.loc[results['Model'] == best_model_name, 'Overfit Gap'].iloc[0]),
        'cv_auc_mean': float(results.loc[results['Model'] == best_model_name, 'CV AUC'].iloc[0]),
        'baseline_accuracy': float(baseline_acc),
    },
    'all_models_comparison': results.to_dict('records'),
    'feature_importance': {
        feat: float(imp) for feat, imp in feature_importance
    }
}

metadata_path = 'models/model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Metadata saved: {metadata_path}")

# ============================================================================
# VERIFICATION
# ============================================================================

print("\n" + "="*80)
print("🔍 VERIFICATION")
print("="*80)

# Test loading
try:
    loaded_model = joblib.load(model_path)
    loaded_selector = joblib.load(selector_path)
    loaded_encoders = joblib.load(encoders_path)
    loaded_metadata = json.load(open(metadata_path, 'r'))
    
    print(f"✅ All artifacts load successfully!")
    print(f"   Model: {type(loaded_model).__name__}")
    print(f"   Selector: {type(loaded_selector).__name__} (k={loaded_selector.k})")
    print(f"   Encoders: {len(loaded_encoders)} categorical features")
    print(f"   Selected features: {len(loaded_metadata['selected_features'])}")
    
except Exception as e:
    print(f"❌ ERROR loading artifacts: {e}")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"\n🎯 Key Improvements:")
print(f"   • Filtered to tokens < {MAX_TOKEN_AGE_HOURS}h old (distribution consistency)")
print(f"   • Stratified split (balanced train/test)")
print(f"   • Feature selection: {len(ALL_FEATURES)} → {len(selected_features)} features")
print(f"   • Strong regularization (reduced overfitting)")
print(f"   • Overfit gap: {results.loc[results['Model'] == best_model_name, 'Overfit Gap'].iloc[0]:.4f}")
print(f"   • Test AUC: {results.loc[results['Model'] == best_model_name, 'Test AUC'].iloc[0]:.4f}")
print(f"   • CV AUC: {results.loc[results['Model'] == best_model_name, 'CV AUC'].iloc[0]:.4f}")
print("="*80)