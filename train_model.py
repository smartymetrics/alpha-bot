import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import json
import os

# ML libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Load data
df = pd.read_csv('data/token_datasets.csv')

print(f"📊 Dataset: {df.shape[0]} samples")
print(f"🎯 Win rate: {(df['label_status'] == 'win').mean() * 100:.2f}%")

# ============================================================================
# FEATURE ENGINEERING - MUST MATCH ml_predictor.py
# ============================================================================

# Encode target
df['target'] = (df['label_status'] == 'win').astype(int)

# Better temporal features (NEW - added to predictor)
df['is_asia_prime'] = df['time_of_day_utc'].between(6, 14).astype(int)
df['is_us_prime'] = df['time_of_day_utc'].between(13, 21).astype(int)
df['is_eu_prime'] = df['time_of_day_utc'].between(8, 16).astype(int)
df['is_dead_hours'] = df['time_of_day_utc'].between(0, 6).astype(int)

# Derived market features (already in predictor)
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

df['is_new_token'] = (df['token_age_at_signal_seconds'] < 43200).astype(int)

# Risk interaction features (NEW - added to predictor)
df['high_risk_combo'] = (
    (df['pump_dump_risk_score'] > 30) & 
    (df['authority_risk_score'] > 15)
).astype(int)

df['critical_signal_quality'] = (
    (df['grade'] == 'CRITICAL') & 
    (df['signal_source'] == 'alpha')
).astype(int)

# ============================================================================
# FEATURE DEFINITIONS - MATCHES ml_predictor.py
# ============================================================================

NUMERIC_FEATURES = [
    # Market features (at signal time)
    'fdv_usd', 
    'liquidity_usd', 
    'volume_h24_usd', 
    'price_change_h24_pct',
    
    # Security features
    'creator_balance_pct', 
    'top_10_holders_pct', 
    'total_lp_locked_usd',
    
    # Smart money features
    'overlap_quality_score', 
    'winner_wallet_density',
    
    # Risk features
    'whale_concentration_score', 
    'pump_dump_risk_score',
    'authority_risk_score',
    
    # Derived features
    'volume_to_liquidity_ratio',
    'fdv_to_liquidity_ratio',
    
    # Temporal features
    'time_of_day_utc', 
    'day_of_week_utc',
    'is_asia_prime', 
    'is_us_prime', 
    'is_eu_prime', 
    'is_dead_hours',
    
    # Token features
    'is_new_token',
    
    # Interaction features
    'high_risk_combo', 
    'critical_signal_quality'
]

CATEGORICAL_FEATURES = [
    'signal_source',        # alpha or discovery
    'grade',                # CRITICAL, HIGH, MEDIUM, LOW
    'hour_category',        # dead_hours, asia_hours, eu_hours, us_hours
    'rugcheck_risk_level'   # from probation_meta
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

print(f"\n✅ Using {len(ALL_FEATURES)} features:")
print(f"   - {len(NUMERIC_FEATURES)} numeric")
print(f"   - {len(CATEGORICAL_FEATURES)} categorical")

# Handle missing values
for col in NUMERIC_FEATURES:
    if col in df.columns:
        df[col] = df[col].fillna(0)
    else:
        print(f"⚠️  Missing column in data: {col}, creating with default value 0")
        df[col] = 0

for col in CATEGORICAL_FEATURES:
    if col in df.columns:
        df[col] = df[col].fillna('UNKNOWN')
    else:
        print(f"⚠️  Missing column in data: {col}, creating with default value 'UNKNOWN'")
        df[col] = 'UNKNOWN'

# ============================================================================
# TIME-SERIES SPLIT
# ============================================================================

df_sorted = df.sort_values('checked_at_timestamp').reset_index(drop=True)

split_idx = int(len(df_sorted) * 0.8)
train_df = df_sorted.iloc[:split_idx].copy()
test_df = df_sorted.iloc[split_idx:].copy()

print(f"\n✂️ Train: {len(train_df)} | Test: {len(test_df)}")
print(f"Train win rate: {train_df['target'].mean()*100:.2f}%")
print(f"Test win rate: {test_df['target'].mean()*100:.2f}%")

X_train = train_df[ALL_FEATURES].copy()
y_train = train_df['target'].copy()
X_test = test_df[ALL_FEATURES].copy()
y_test = test_df['target'].copy()

# ============================================================================
# LABEL ENCODING FOR CATEGORICAL FEATURES
# ============================================================================

X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

label_encoders = {}
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    X_train_encoded[col] = le.fit_transform(X_train_encoded[col].astype(str))
    X_test_encoded[col] = le.transform(X_test_encoded[col].astype(str))
    label_encoders[col] = le
    print(f"  Encoded {col}: {list(le.classes_)}")

# ============================================================================
# MODEL 1: XGBoost
# ============================================================================

print("\n" + "="*60)
print("🔷 MODEL 1: XGBoost")
print("="*60)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Class weight ratio: {scale_pos_weight:.2f}")

xgb_model = xgb.XGBClassifier(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=100,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='auc'
)

xgb_model.fit(X_train_encoded, y_train, verbose=False)

xgb_train_pred = xgb_model.predict(X_train_encoded)
xgb_test_pred = xgb_model.predict(X_test_encoded)
xgb_test_proba = xgb_model.predict_proba(X_test_encoded)[:, 1]

xgb_train_acc = (xgb_train_pred == y_train).mean()
xgb_test_acc = (xgb_test_pred == y_test).mean()
xgb_test_auc = roc_auc_score(y_test, xgb_test_proba)

print(f"Training Accuracy: {xgb_train_acc:.4f}")
print(f"Test Accuracy: {xgb_test_acc:.4f}")
print(f"Test AUC: {xgb_test_auc:.4f}")
print(f"Overfit Gap: {abs(xgb_train_acc - xgb_test_acc):.4f}")

# ============================================================================
# MODEL 2: LightGBM
# ============================================================================

print("\n" + "="*60)
print("🔶 MODEL 2: LightGBM")
print("="*60)

lgb_model = lgb.LGBMClassifier(
    num_leaves=15,
    max_depth=4,
    learning_rate=0.05,
    n_estimators=100,
    min_child_samples=10,
    min_split_gain=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)

lgb_model.fit(X_train_encoded, y_train)

lgb_train_pred = lgb_model.predict(X_train_encoded)
lgb_test_pred = lgb_model.predict(X_test_encoded)
lgb_test_proba = lgb_model.predict_proba(X_test_encoded)[:, 1]

lgb_train_acc = (lgb_train_pred == y_train).mean()
lgb_test_acc = (lgb_test_pred == y_test).mean()
lgb_test_auc = roc_auc_score(y_test, lgb_test_proba)

print(f"Training Accuracy: {lgb_train_acc:.4f}")
print(f"Test Accuracy: {lgb_test_acc:.4f}")
print(f"Test AUC: {lgb_test_auc:.4f}")
print(f"Overfit Gap: {abs(lgb_train_acc - lgb_test_acc):.4f}")

# ============================================================================
# MODEL 3: CatBoost
# ============================================================================

print("\n" + "="*60)
print("🔸 MODEL 3: CatBoost")
print("="*60)

# CatBoost needs original categorical values, not encoded
X_train_cat = X_train.copy()
X_test_cat = X_test.copy()

cat_feature_indices = [X_train_cat.columns.get_loc(col) for col in CATEGORICAL_FEATURES]

cat_model = CatBoostClassifier(
    iterations=100,
    depth=4,
    learning_rate=0.05,
    l2_leaf_reg=10,
    min_data_in_leaf=10,
    auto_class_weights='Balanced',
    cat_features=cat_feature_indices,
    random_state=42,
    verbose=False
)

cat_model.fit(X_train_cat, y_train)

cat_train_pred = cat_model.predict(X_train_cat)
cat_test_pred = cat_model.predict(X_test_cat)
cat_test_proba = cat_model.predict_proba(X_test_cat)[:, 1]

cat_train_acc = (cat_train_pred == y_train).mean()
cat_test_acc = (cat_test_pred == y_test).mean()
cat_test_auc = roc_auc_score(y_test, cat_test_proba)

print(f"Training Accuracy: {cat_train_acc:.4f}")
print(f"Test Accuracy: {cat_test_acc:.4f}")
print(f"Test AUC: {cat_test_auc:.4f}")
print(f"Overfit Gap: {abs(cat_train_acc - cat_test_acc):.4f}")

# ============================================================================
# CROSS-VALIDATION (5 folds)
# ============================================================================

print("\n" + "="*60)
print("📊 5-FOLD TIME-SERIES CROSS-VALIDATION")
print("="*60)

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = {'xgb': [], 'lgb': [], 'cat': []}

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
    # XGBoost
    fold_xgb = xgb.XGBClassifier(
        max_depth=4, learning_rate=0.05, n_estimators=100,
        scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='auc'
    )
    fold_xgb.fit(X_train_encoded.iloc[train_idx], y_train.iloc[train_idx], verbose=False)
    score = roc_auc_score(y_train.iloc[val_idx], 
                          fold_xgb.predict_proba(X_train_encoded.iloc[val_idx])[:, 1])
    cv_scores['xgb'].append(score)
    
    # LightGBM
    fold_lgb = lgb.LGBMClassifier(
        num_leaves=15, max_depth=4, learning_rate=0.05, n_estimators=100,
        class_weight='balanced', random_state=42, verbose=-1
    )
    fold_lgb.fit(X_train_encoded.iloc[train_idx], y_train.iloc[train_idx])
    score = roc_auc_score(y_train.iloc[val_idx],
                          fold_lgb.predict_proba(X_train_encoded.iloc[val_idx])[:, 1])
    cv_scores['lgb'].append(score)
    
    # CatBoost
    fold_cat = CatBoostClassifier(
        iterations=100, depth=4, learning_rate=0.05,
        auto_class_weights='Balanced', cat_features=cat_feature_indices,
        random_state=42, verbose=False
    )
    fold_cat.fit(X_train_cat.iloc[train_idx], y_train.iloc[train_idx])
    score = roc_auc_score(y_train.iloc[val_idx],
                          fold_cat.predict_proba(X_train_cat.iloc[val_idx])[:, 1])
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

print("\n" + "="*60)
print("🏆 MODEL COMPARISON")
print("="*60)

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

# Calculate composite score (prioritize test accuracy and low overfitting)
results['Score'] = (
    results['Test Acc'] * 0.5 +
    (1 - results['Overfit Gap']) * 0.3 +
    results['CV AUC'] * 0.2
)

print(results.to_string(index=False))

print("\n📊 Model Ranking:")
ranked = results.sort_values('Score', ascending=False)
print(ranked[['Model', 'Test Acc', 'Overfit Gap', 'Score']].to_string(index=False))

best_model_idx = results['Score'].idxmax()
best_model_name = results.iloc[best_model_idx]['Model']

print(f"\n✅ Best Model: {best_model_name}")

# Select best model
if best_model_name == 'XGBoost':
    best_model = xgb_model
    X_test_final = X_test_encoded
elif best_model_name == 'LightGBM':
    best_model = lgb_model
    X_test_final = X_test_encoded
else:
    best_model = cat_model
    X_test_final = X_test_cat

# Classification report
print(f"\n📋 Detailed Classification Report ({best_model_name}):")
print(classification_report(y_test, best_model.predict(X_test_final), 
                          target_names=['Loss', 'Win'], digits=3))

# ============================================================================
# SAVE MODEL - COMPATIBLE WITH ml_predictor.py
# ============================================================================

print("\n" + "="*60)
print("💾 SAVING MODEL FOR PRODUCTION")
print("="*60)

# IMPORTANT: Save with the exact filename that ml_predictor.py expects
model_path = 'models/xgboost_signal_classifier.pkl'
joblib.dump(best_model, model_path)
print(f"✅ Model saved: {model_path}")

# Save label encoders
encoders_path = 'models/label_encoders.pkl'
joblib.dump(label_encoders, encoders_path)
print(f"✅ Label encoders saved: {encoders_path}")

# Save feature names (exact order matters!)
features_path = 'models/feature_names.json'
with open(features_path, 'w') as f:
    json.dump(ALL_FEATURES, f, indent=2)
print(f"✅ Feature names saved: {features_path}")

# Save detailed metadata
metadata = {
    'model_type': best_model_name,
    'trained_at': datetime.now().isoformat(),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'num_features': len(ALL_FEATURES),
    'features': {
        'numeric': NUMERIC_FEATURES,
        'categorical': CATEGORICAL_FEATURES,
        'all': ALL_FEATURES
    },
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
    },
    'all_models_comparison': results.to_dict('records')
}

metadata_path = 'models/model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Metadata saved: {metadata_path}")

# Verification
print("\n" + "="*60)
print("🔍 VERIFICATION")
print("="*60)

print(f"Model file: {os.path.basename(model_path)}")
print(f"Features count: {len(ALL_FEATURES)}")
print(f"Categorical encoders: {list(label_encoders.keys())}")

# Test loading (simulate what ml_predictor.py does)
try:
    loaded_model = joblib.load(model_path)
    loaded_encoders = joblib.load(encoders_path)
    loaded_features = json.load(open(features_path, 'r'))
    print(f"\n✅ Model loads successfully!")
    print(f"   Loaded {len(loaded_features)} features")
    print(f"   Loaded {len(loaded_encoders)} encoders")
except Exception as e:
    print(f"\n❌ ERROR loading model: {e}")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE - Model ready for ml_predictor.py!")
print("="*60)