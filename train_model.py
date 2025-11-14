import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import json

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score
)
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

import os

# Create output directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("✅ Created output directories")

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

#  load dataset
df = pd.read_csv('data/token_datasets.csv')

print(f"📊 Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n🎯 Target distribution:")
print(df['label_status'].value_counts())
print(f"\nWin rate: {(df['label_status'] == 'win').mean() * 100:.2f}%")


# ----------------------------------------------------------------------------
# MARKET FEATURES (Used in ML)
# ----------------------------------------------------------------------------
MARKET_FEATURES = [
    'fdv_usd',
    'liquidity_usd',
    'volume_h24_usd',
    'price_change_h24_pct',
]

# ----------------------------------------------------------------------------
# SECURITY FEATURES (Used in ML)
# ----------------------------------------------------------------------------
SECURITY_FEATURES = [
    'has_mint_authority',
    'has_freeze_authority',
    'creator_balance_pct',
    'top_10_holders_pct',
    'is_lp_locked_95_plus',
    'total_lp_locked_usd',
]

# ----------------------------------------------------------------------------
# ADDITIONAL SECURITY FEATURES (Not in ML model)
# ----------------------------------------------------------------------------
ADDITIONAL_SECURITY_FEATURES = [
    'total_insider_networks',
    'largest_insider_network_size',
    'total_insider_token_amount',
    'rugcheck_risk_level',
]

# ----------------------------------------------------------------------------
# TIME FEATURES (Used in ML)
# ----------------------------------------------------------------------------
TIME_FEATURES = [
    'time_of_day_utc',
    'day_of_week_utc',
    'is_weekend_utc',
    'is_public_holiday_any',
]

# ----------------------------------------------------------------------------
# SIGNAL FEATURES 
# ----------------------------------------------------------------------------
SIGNAL_FEATURES = [
    'signal_source',  # discovery or alpha
    'grade'          # CRITICAL, HIGH, MEDIUM, LOW
]

# ----------------------------------------------------------------------------
# TOKEN AGE FEATURES 
# ----------------------------------------------------------------------------
ADDITIONAL_TOKEN_AGE_FEATURES = [
    'token_age_at_signal_seconds'
]

# ----------------------------------------------------------------------------
# DERIVED SMART MONEY FEATURES 
# ----------------------------------------------------------------------------
DERIVED_SMART_MONEY_FEATURES = [
    'overlap_quality_score',
    'winner_wallet_density'
]

# ----------------------------------------------------------------------------
# DERIVED CONCENTRATION FEATURES
# ----------------------------------------------------------------------------
DERIVED_CONCENTRATION_FEATURES = [
    'whale_concentration_score',
]

# ----------------------------------------------------------------------------
# DERIVED RISK FEATURES
# ----------------------------------------------------------------------------
DERIVED_RISK_FEATURES = [
    'pump_dump_risk_score'
]

# ----------------------------------------------------------------------------
# ADDITIONAL DERIVED RISK FEATURES
# ----------------------------------------------------------------------------
ADDITIONAL_DERIVED_RISK_FEATURES = [
    'authority_risk_score',
    'creator_dumped',
]

# ----------------------------------------------------------------------------
# DERIVED TEMPORAL FEATURES (Used in ML)
# ----------------------------------------------------------------------------
DERIVED_TEMPORAL_FEATURES = [
    'hour_category',
    'is_last_day_of_week'
]

# ----------------------------------------------------------------------------
# ML TRAINING FEATURES
# ----------------------------------------------------------------------------
ML_TRAINING_FEATURES = (
    MARKET_FEATURES + 
    SECURITY_FEATURES + 
    TIME_FEATURES + 
    SIGNAL_FEATURES + 
    DERIVED_SMART_MONEY_FEATURES +
    DERIVED_CONCENTRATION_FEATURES +
    DERIVED_RISK_FEATURES +
    DERIVED_TEMPORAL_FEATURES
)

# ----------------------------------------------------------------------------
# ALL FEATURES (Everything extracted from snapshot)
# ----------------------------------------------------------------------------
FEATURE_COLS = (
    MARKET_FEATURES +
    SECURITY_FEATURES +
    ADDITIONAL_SECURITY_FEATURES +
    TIME_FEATURES +
    SIGNAL_FEATURES +
    ADDITIONAL_TOKEN_AGE_FEATURES +
    DERIVED_SMART_MONEY_FEATURES +
    DERIVED_CONCENTRATION_FEATURES +
    DERIVED_RISK_FEATURES +
    ADDITIONAL_DERIVED_RISK_FEATURES +
    DERIVED_TEMPORAL_FEATURES 
)

print(f"✅ Using {len(FEATURE_COLS)} features from dataset:")
print("\nFeatures:")
for f in FEATURE_COLS:
    print(f"  - {f}")

# Data Preprocessing
df_clean = df.copy()

# Handle missing values
print("🔍 Checking for missing values...")
df_clean[FEATURE_COLS].isnull().sum()

df_clean.dtypes

# Encode categorical features
print("\n🏷️  Encoding categorical features...")

label_encoders = {}
df_encoded = df_clean.copy()

for col in FEATURE_COLS:
    if df_encoded[col].dtype == 'object':
        print(f"  Encoding: {col}")
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

print(f"✅ Encoded {len(label_encoders)} categorical features")

# Encode target
print("\n🎯 Encoding target variable...")
le_target = LabelEncoder()
df_encoded['target'] = le_target.fit_transform(df_clean['label_status'])
label_encoders['target'] = le_target

print(f"  0 = {le_target.classes_[0]}")
print(f"  1 = {le_target.classes_[1]}")

# Create derived features
print("\n🔬 Creating derived features...")

df_encoded['volume_to_liquidity_ratio'] = np.where(
    df_encoded['liquidity_usd'] > 0,
    df_encoded['volume_h24_usd'] / df_encoded['liquidity_usd'],
    0
)

df_encoded['fdv_to_liquidity_ratio'] = np.where(
    df_encoded['liquidity_usd'] > 0,
    df_encoded['fdv_usd'] / df_encoded['liquidity_usd'],
    0
)

df_encoded['is_new_token'] = (df_encoded['token_age_at_signal_seconds'] < 43200).astype(int)  # <12 hours

# Add derived features to feature list
FEATURE_COLS.extend([
    'volume_to_liquidity_ratio',
    'fdv_to_liquidity_ratio', 
    'is_new_token'
])

print(f"✅ Total features: {len(FEATURE_COLS)}")

# Time-Series Train-Test Split
# Sort by timestamp
print("📅 Sorting by timestamp...")
df_encoded = df_encoded.sort_values('checked_at_timestamp')
print(f"  Date range: {df_encoded['checked_at_timestamp'].min()} to {df_encoded['checked_at_timestamp'].max()}")

# Time-series split: 80% train, 20% test
# Test set is the MOST RECENT data
split_idx = int(len(df_encoded) * 0.8)

train_df = df_encoded.iloc[:split_idx]
test_df = df_encoded.iloc[split_idx:]

print(f"\n✂️  Train-Test Split:")
print(f"  Training samples: {len(train_df)} ({len(train_df)/len(df_encoded)*100:.1f}%)")
print(f"  Test samples: {len(test_df)} ({len(test_df)/len(df_encoded)*100:.1f}%)")

# Check distribution
print(f"\n📊 Distribution:")
print(f"  Training win rate: {train_df['target'].mean()*100:.2f}%")
print(f"  Test win rate: {test_df['target'].mean()*100:.2f}%")

# Prepare X and y
X_train = train_df[FEATURE_COLS].copy()
y_train = train_df['target'].copy()

X_test = test_df[FEATURE_COLS].copy()
y_test = test_df['target'].copy()

print(f"  X_train: {X_train.shape}")
print(f"  X_test: {X_test.shape}")

# Feature Correlation with target
print("🔍 Top 15 Features Correlated with Target:")
print("-" * 60)

correlations = train_df[FEATURE_COLS + ['target']].corr()['target'].drop('target').sort_values(ascending=False)

for i, (feature, corr) in enumerate(correlations.head(15).items(), 1):
    print(f"{i:2}. {feature:40} | r = {corr:6.3f}")

# Plot
plt.figure(figsize=(12, 8))
correlations.head(20).plot(kind='barh')
plt.xlabel('Correlation with Target')
plt.title('Top 20 Features Correlated with Win/Loss')
plt.tight_layout()
plt.savefig('outputs/feature_correlations.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Feature correlation analysis complete")

print("🚀 Training XGBoost Model...")
print("-" * 60)

# Calculate scale_pos_weight for class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Class imbalance ratio: {scale_pos_weight:.2f}")

# Initialize XGBoost
model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    objective='binary:logistic',
    eval_metric='auc',
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,  
    random_state=42,
    early_stopping_rounds=20,
    n_jobs=-1
)

# Train with validation set
eval_set = [(X_train, y_train), (X_test, y_test)]

model.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=True
)

print("\n✅ Model training complete!")

# Cross-Validation

print("🔄 Running 5-Fold Time-Series Cross-Validation...")

# Use TimeSeriesSplit for proper time-series CV
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]
    
    # Train
    fold_model = xgb.XGBClassifier(
        max_depth=6, learning_rate=0.1, n_estimators=200,
        scale_pos_weight=scale_pos_weight, random_state=42
    )
    fold_model.fit(X_fold_train, y_fold_train, verbose=False)
    
    # Evaluate
    score = roc_auc_score(y_fold_val, fold_model.predict_proba(X_fold_val)[:, 1])
    cv_scores.append(score)
    print(f"  Fold {fold}: AUC = {score:.4f}")

print(f"\n📊 Cross-Validation Results:")
print(f"  Mean AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Model Evaluation

print("="*60)
print("📊 MODEL EVALUATION")
print("="*60)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

y_train_proba = model.predict_proba(X_train)[:, 1]
y_test_proba = model.predict_proba(X_test)[:, 1]

# Accuracy
train_acc = (y_train_pred == y_train).mean()
test_acc = (y_test_pred == y_test).mean()

print(f"\n📈 Accuracy:")
print(f"  Training: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"  Test:     {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"  Gap:      {abs(train_acc - test_acc):.4f}")

# AUC
train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)

print(f"\n📈 ROC AUC:")
print(f"  Training: {train_auc:.4f}")
print(f"  Test:     {test_auc:.4f}")

# Confusion Matrix
print("\n" + "="*60)
print("📋 CONFUSION MATRIX")
print("="*60)

cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

print("\nTraining Set:")
print(cm_train)
print(f"  TN: {cm_train[0][0]}, FP: {cm_train[0][1]}")
print(f"  FN: {cm_train[1][0]}, TP: {cm_train[1][1]}")

print("\nTest Set:")
print(cm_test)
print(f"  TN: {cm_test[0][0]}, FP: {cm_test[0][1]}")
print(f"  FN: {cm_test[1][0]}, TP: {cm_test[1][1]}")

# Classification Report
print("\n" + "="*60)
print("📊 CLASSIFICATION REPORT")
print("="*60)

print("\nTest Set:")
print(classification_report(
    y_test, y_test_pred,
    target_names=['Loss', 'Win'],
    digits=4
))

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
# Prediction Analysis

print("="*60)
print("🔍 PREDICTION ANALYSIS")
print("="*60)

# Analyze predictions by confidence
test_df_pred = test_df.copy()
test_df_pred['predicted_proba'] = y_test_proba
test_df_pred['predicted_class'] = y_test_pred
test_df_pred['correct'] = (y_test_pred == y_test).astype(int)

# Group by confidence bins
bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
test_df_pred['confidence_bin'] = pd.cut(test_df_pred['predicted_proba'], bins=bins)

print("\n📊 Performance by Confidence Level:")
print("-" * 60)
conf_analysis = test_df_pred.groupby('confidence_bin').agg({
    'correct': ['count', 'sum', 'mean'],
    'target': 'mean'
})
print(conf_analysis)

# High confidence predictions
high_conf = test_df_pred[test_df_pred['predicted_proba'] >= 0.7]
print(f"\n🎯 High Confidence Predictions (>=0.7):")
print(f"  Count: {len(high_conf)}")
print(f"  Accuracy: {high_conf['correct'].mean():.4f}")
print(f"  Win rate: {high_conf['target'].mean():.4f}")

# Misclassifications
misclassified = test_df_pred[test_df_pred['correct'] == 0]
print(f"\n❌ Misclassified Samples: {len(misclassified)}")

if len(misclassified) > 0:
    print("\nMisclassification Analysis:")
    print(f"  False Positives (predicted win, actual loss): {((misclassified['predicted_class'] == 1) & (misclassified['target'] == 0)).sum()}")
    print(f"  False Negatives (predicted loss, actual win): {((misclassified['predicted_class'] == 0) & (misclassified['target'] == 1)).sum()}")

# Save Model for Production

print("="*60)
print("💾 SAVING MODEL FOR PRODUCTION")
print("="*60)

import os
os.makedirs('models', exist_ok=True)

# Save model
model_path = 'models/xgboost_signal_classifier.pkl'
joblib.dump(model, model_path)
print(f"✅ Model saved: {model_path}")

# Save label encoders
encoders_path = 'models/label_encoders.pkl'
joblib.dump(label_encoders, encoders_path)
print(f"✅ Label encoders saved: {encoders_path}")

# Save feature names
features_path = 'models/feature_names.json'
with open(features_path, 'w') as f:
    json.dump(FEATURE_COLS, f, indent=2)
print(f"✅ Feature names saved: {features_path}")

# Save metadata
metadata = {
    'model_type': 'XGBClassifier',
    'trained_at': datetime.now().isoformat(),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'features': FEATURE_COLS,
    'performance': {
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'train_auc': float(train_auc),
        'test_auc': float(test_auc),
        'cv_auc_mean': float(np.mean(cv_scores)),
        'cv_auc_std': float(np.std(cv_scores))
    },
    'hyperparameters': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'scale_pos_weight': float(scale_pos_weight)
    }
}

metadata_path = 'models/model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Metadata saved: {metadata_path}")

print("\n" + "="*60)
print("✅ MODEL READY FOR PRODUCTION DEPLOYMENT!")
print("="*60)
