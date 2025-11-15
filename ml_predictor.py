#!/usr/bin/env python3
"""
ml_predictor.py (V4 - Compatible with train_model.py)

Loads trained XGBoost model and predicts win probability.
Uses feature selector and matches train_model.py feature engineering.
"""

import joblib
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import aiohttp
import asyncio
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from functools import wraps
import logging

# --- Configuration ---
MODEL_DIR = './models'
MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_signal_classifier.pkl')
ENCODERS_PATH = os.path.join(MODEL_DIR, 'label_encoders.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'model_metadata.json')
SELECTOR_PATH = os.path.join(MODEL_DIR, 'feature_selector.pkl')

# --- Logging ---
log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=logging.INFO)

# --- Load Model Artifacts ---
try:
    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
    feature_selector = joblib.load(SELECTOR_PATH)
    metadata = json.load(open(METADATA_PATH, 'r'))
    
    all_features = metadata['all_features']
    selected_features = metadata['selected_features']
    numeric_features = metadata['numeric_features']
    categorical_features = metadata['categorical_features']
    
    log.info(f"✅ [ML_Predictor] Loaded model: {metadata['model_type']}")
    log.info(f"   All features: {len(all_features)}")
    log.info(f"   Selected features: {len(selected_features)}")
    log.info(f"   Categorical encoders: {list(label_encoders.keys())}")
    log.info(f"   Test AUC: {metadata['performance']['test_auc']:.4f}")
    
except Exception as e:
    log.error(f"❌ [ML_Predictor] CRITICAL: Could not load model: {e}")
    model = None
    label_encoders = {}
    feature_selector = None
    metadata = {}
    all_features = []
    selected_features = []
    numeric_features = []
    categorical_features = []

def _safe_divide(a, b, default=0.0):
    """Safe division with None handling."""
    if a is None or b is None:
        return default
    try:
        a_f = float(a)
        b_f = float(b)
        return a_f / b_f if b_f != 0 else default
    except (ValueError, TypeError):
        return default

def _safe_float(val, default=0.0):
    """Safely convert to float with None handling."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

async def _prepare_features(
    data_dict: dict, 
    signal_source: str,
    session: aiohttp.ClientSession
) -> pd.DataFrame | None:
    """
    Converts raw token data into model features.
    MUST match train_model.py's feature engineering exactly!
    """
    if not model or not all_features:
        log.error("❌ Model not loaded")
        return None

    try:
        features = {}
        
        # === 1. RAW DATA EXTRACTION ===
        raw_dex = data_dict.get('raw_dexscreener', data_dict.get('raw', {}))
        if not isinstance(raw_dex, dict):
            raw_dex = {}
        
        raw_rug = data_dict.get('raw_rugcheck', data_dict.get('raw', {}))
        if not isinstance(raw_rug, dict):
            raw_rug = {}
        
        # === 2. TIMESTAMP & TOKEN AGE ===
        ts_str = data_dict.get('checked_at', data_dict.get('ts', datetime.now(timezone.utc).isoformat()))
        dt = pd.to_datetime(ts_str).tz_convert(timezone.utc)
        checked_at_timestamp = int(dt.timestamp())
        
        # Token age calculation
        token_age_seconds = None
        block_time = data_dict.get('block_time')
        pair_created_at = raw_dex.get('pairCreatedAt')
        
        if block_time:
            token_age_seconds = max(0, checked_at_timestamp - int(block_time))
        elif pair_created_at:
            token_age_seconds = max(0, checked_at_timestamp - int(pair_created_at / 1000))
        
        if token_age_seconds is None:
            token_age_seconds = 100000  # Default for unknown
        
        features['token_age_at_signal_seconds'] = token_age_seconds
        
        # === 3. MARKET FUNDAMENTALS ===
        features['fdv_usd'] = _safe_float(data_dict.get('fdv_usd', raw_dex.get('fdv')))
        features['liquidity_usd'] = _safe_float(data_dict.get('liquidity_usd', data_dict.get('total_lp_usd')))
        features['volume_h24_usd'] = _safe_float(
            data_dict.get('volume_h24_usd', raw_dex.get('volume', {}).get('h24'))
        )
        features['price_change_h24_pct'] = _safe_float(
            data_dict.get('price_change_h24_pct', raw_dex.get('priceChange', {}).get('h24'))
        )
        
        # === 4. SECURITY/RISK FEATURES ===
        features['creator_balance_pct'] = _safe_float(data_dict.get('creator_balance_pct'))
        features['top_10_holders_pct'] = _safe_float(data_dict.get('top_10_holders_pct'))
        
        # LP locked USD
        markets = (raw_rug.get('data') or raw_rug.get('raw') or {}).get('markets', [])
        features['total_lp_locked_usd'] = sum(m.get('lp', {}).get('lpLockedUSD', 0) for m in markets)
        
        # === 5. SMART MONEY FEATURES ===
        overlap_count = _safe_float(data_dict.get('overlap_count'))
        weighted_concentration = _safe_float(data_dict.get('weighted_concentration'))
        total_winner_wallets = _safe_float(data_dict.get('total_winner_wallets'), 1.0)
        holder_count = _safe_float(data_dict.get('holder_count', data_dict.get('total_holders')), 1.0)
        
        features['overlap_quality_score'] = _safe_divide(
            overlap_count * weighted_concentration, total_winner_wallets
        )
        features['winner_wallet_density'] = _safe_divide(overlap_count, holder_count)
        
        # === 6. WHALE CONCENTRATION SCORE ===
        top1_pct = _safe_float(data_dict.get('top1_holder_pct'))
        top10_pct = features['top_10_holders_pct']
        features['whale_concentration_score'] = (
            (top1_pct * 3) + (top10_pct - top1_pct) if top10_pct >= top1_pct else top1_pct * 3
        )
        
        # === 7. AUTHORITY RISK SCORE ===
        has_mint = int(data_dict.get('has_mint_authority', data_dict.get('mint_authority') is not None))
        has_freeze = int(data_dict.get('has_freeze_authority', data_dict.get('freeze_authority') is not None))
        transfer_fee = _safe_float(data_dict.get('transfer_fee_pct'))
        
        features['authority_risk_score'] = (
            (50 if has_mint else 0) +
            (50 if has_freeze else 0) +
            (transfer_fee * 100 if transfer_fee else 0)
        )
        
        # === 8. PUMP DUMP RISK SCORE ===
        lp_locked_pct = _safe_float(data_dict.get('lp_locked_pct', data_dict.get('overall_lp_locked_pct')))
        
        insider_networks = (raw_rug.get('data') or raw_rug.get('raw') or {}).get('insiderNetworks', [])
        largest_insider_network_size = max((n.get('size', 0) for n in insider_networks), default=0)
        
        creator_dumped = int(
            (features['creator_balance_pct'] == 0) and 
            (token_age_seconds < 86400)
        )
        
        pump_dump_score = 0
        if largest_insider_network_size > 100: pump_dump_score += 30
        elif largest_insider_network_size > 50: pump_dump_score += 20
        elif largest_insider_network_size > 20: pump_dump_score += 10
        
        if top1_pct > 30: pump_dump_score += 25
        elif top1_pct > 20: pump_dump_score += 15
        elif top1_pct > 10: pump_dump_score += 8
        
        if lp_locked_pct < 50: pump_dump_score += 20
        elif lp_locked_pct < 80: pump_dump_score += 12
        elif lp_locked_pct < 95: pump_dump_score += 5
        
        pump_dump_score += min(features['authority_risk_score'] / 100 * 15, 15)
        
        if creator_dumped: pump_dump_score += 10
        elif features['creator_balance_pct'] > 10: pump_dump_score += 7
        elif features['creator_balance_pct'] > 5: pump_dump_score += 4
        
        features['pump_dump_risk_score'] = pump_dump_score
        
        # === 9. DERIVED FEATURES (from train_model.py) ===
        
        # Token age features
        features['token_age_hours'] = token_age_seconds / 3600
        features['token_age_hours_capped'] = min(features['token_age_hours'], 24)
        features['is_ultra_fresh'] = int(token_age_seconds < 3600)
        
        # Market ratios
        features['volume_to_liquidity_ratio'] = _safe_divide(
            features['volume_h24_usd'], features['liquidity_usd']
        )
        features['fdv_to_liquidity_ratio'] = _safe_divide(
            features['fdv_usd'], features['liquidity_usd']
        )
        
        # Risk interactions
        features['high_risk_signal'] = int(
            (pump_dump_score > 30) and (features['authority_risk_score'] > 15)
        )
        
        features['premium_signal'] = int(
            (data_dict.get('grade') == 'CRITICAL') and (signal_source == 'alpha')
        )
        
        features['extreme_concentration'] = int(features['top_10_holders_pct'] > 60)
        
        # Log transforms (for stability)
        features['log_liquidity'] = np.log1p(features['liquidity_usd'])
        features['log_volume'] = np.log1p(features['volume_h24_usd'])
        features['log_fdv'] = np.log1p(features['fdv_usd'])
        
        # === 10. CATEGORICAL FEATURES ===
        features['signal_source'] = signal_source
        features['grade'] = data_dict.get('grade', 'NONE')
        
        # === 11. CREATE DATAFRAME WITH ALL FEATURES ===
        df = pd.DataFrame([features])
        
        # Add any missing numeric features
        for col in numeric_features:
            if col not in df.columns:
                df[col] = 0.0
        
        # Add any missing categorical features
        for col in categorical_features:
            if col not in df.columns:
                df[col] = 'UNKNOWN'
        
        # === 12. ENCODE CATEGORICAL FEATURES ===
        df_encoded = df.copy()
        for col in categorical_features:
            if col in label_encoders:
                le = label_encoders[col]
                val = str(df[col].iloc[0])
                if val not in le.classes_:
                    val = le.classes_[0]
                df_encoded[col] = le.transform([val])[0]
        
        # === 13. ENSURE CORRECT FEATURE ORDER ===
        df_ordered = df_encoded[all_features]
        
        # === 14. APPLY FEATURE SELECTOR ===
        df_selected = feature_selector.transform(df_ordered)
        
        log.debug(f"[ML_Predictor] Features prepared: {df_ordered.shape} -> {df_selected.shape}")
        
        return df_selected
        
    except Exception as e:
        log.error(f"❌ Failed to prepare features: {e}", exc_info=True)
        return None

async def predict_token_win_probability(
    data_dict: dict,
    signal_source: str,
    session: aiohttp.ClientSession
) -> float | None:
    """
    Main prediction function.
    
    Args:
        data_dict: Token data from monitor
        signal_source: 'discovery' or 'alpha'
        session: aiohttp session (kept for compatibility, not used)
        
    Returns:
        Win probability [0.0-1.0] or None on failure
    """
    if not model or not feature_selector:
        log.error("❌ Model or feature selector not loaded")
        return None
    
    features_array = await _prepare_features(data_dict, signal_source, session)
    
    if features_array is None:
        return None
    
    try:
        probabilities = model.predict_proba(features_array)
        win_probability = float(probabilities[0][1])
        
        log.debug(f"[ML_Predictor] P(win)={win_probability:.3f} for {signal_source} signal")
        return win_probability
        
    except Exception as e:
        log.error(f"❌ Prediction failed: {e}", exc_info=True)
        return None

# === OPTIONAL: Batch Prediction ===
async def predict_batch(
    data_list: List[dict],
    signal_sources: List[str],
    session: aiohttp.ClientSession
) -> List[Optional[float]]:
    """
    Predict win probabilities for multiple tokens.
    
    Args:
        data_list: List of token data dicts
        signal_sources: List of signal sources (same length as data_list)
        session: aiohttp session
        
    Returns:
        List of win probabilities (or None for failures)
    """
    if len(data_list) != len(signal_sources):
        log.error("❌ data_list and signal_sources must have same length")
        return [None] * len(data_list)
    
    results = []
    for data_dict, signal_source in zip(data_list, signal_sources):
        prob = await predict_token_win_probability(data_dict, signal_source, session)
        results.append(prob)
    
    return results

# === UTILITY: Get Model Info ===
def get_model_info() -> dict:
    """Returns model metadata."""
    if not metadata:
        return {}
    return {
        'model_type': metadata.get('model_type'),
        'trained_at': metadata.get('trained_at'),
        'num_features': len(all_features),
        'selected_features': len(selected_features),
        'test_auc': metadata.get('performance', {}).get('test_auc'),
        'test_accuracy': metadata.get('performance', {}).get('test_accuracy'),
        'max_token_age_hours': metadata.get('max_token_age_hours'),
    }

# === TEST FUNCTION ===
async def test_predictor():
    """Test predictor with dummy data."""
    test_data = {
        'grade': 'CRITICAL',
        'fdv_usd': 1000000,
        'liquidity_usd': 50000,
        'volume_h24_usd': 200000,
        'price_change_h24_pct': 150,
        'creator_balance_pct': 2.5,
        'top_10_holders_pct': 35,
        'top1_holder_pct': 12,
        'block_time': int(datetime.now(timezone.utc).timestamp()) - 1800,  # 30 min old
        'overlap_count': 5,
        'weighted_concentration': 0.3,
        'total_winner_wallets': 100,
        'holder_count': 200,
    }
    
    prob = await predict_token_win_probability(test_data, 'alpha')
    
    if prob is not None:
        log.info(f"✅ Test prediction: {prob:.3f}")
        log.info(f"   Model info: {get_model_info()}")
        return True
    else:
        log.error("❌ Test prediction failed")
        return False

if __name__ == "__main__":
    # Test the predictor
    asyncio.run(test_predictor())