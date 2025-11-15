#!/usr/bin/env python3
"""
ml_predictor.py (V3 - Feature Compatible)

Loads trained XGBoost model and predicts win probability.
NOW INCLUDES NEW FEATURES: is_asia_prime, is_us_prime, etc.
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
FEATURES_PATH = os.path.join(MODEL_DIR, 'feature_names.json')

# --- Logging ---
log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=logging.INFO)

# --- Holiday API Logic (from collector.py) ---

HOLIDAY_COUNTRY_CODES = "US,GB,DE,JP,SG,KR,CN,CA,AU".split(',')

def async_ttl_cache(ttl_seconds: int):
    """Decorator for async in-memory TTL caching."""
    cache: Dict[Any, Tuple[Any, float]] = {}
    lock = asyncio.Lock()

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.monotonic()

            async with lock:
                if key in cache:
                    result, expiry = cache[key]
                    if now < expiry:
                        return result
                    del cache[key]

            new_result = await func(*args, **kwargs)
            async with lock:
                cache[key] = (new_result, now + ttl_seconds)
            return new_result
        return wrapper
    return decorator

class BaseAPIClient:
    def __init__(self, session: aiohttp.ClientSession, max_retries: int, name: str):
        self.session = session
        self.max_retries = max_retries
        self.name = name

    async def async_get(self, url: str, timeout: int) -> Optional[Dict]:
        for attempt in range(self.max_retries):
            try:
                if not self.session or self.session.closed:
                    log.error(f"[{self.name}] HTTP session is closed.")
                    return None
                    
                async with self.session.get(url, timeout=timeout) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    if resp.status >= 500 or resp.status == 429:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        return None
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                log.warning(f"[{self.name}] Request failed: {e}")
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        return None

class HolidayClient(BaseAPIClient):
    BASE_URL = "https://date.nager.at/api/v3"

    def __init__(self, session: aiohttp.ClientSession, max_retries: int = 3):
        super().__init__(session, max_retries, "HolidayAPI")
        self.timeout = 5

    @async_ttl_cache(ttl_seconds=3600 * 24)
    async def _fetch_holidays_for_year(self, year: int, country_code: str) -> Set[str]:
        url = f"{self.BASE_URL}/PublicHolidays/{year}/{country_code}"
        data = await self.async_get(url, self.timeout)
        if data and isinstance(data, list):
            return {item['date'] for item in data if 'date' in item}
        return set()

    async def is_holiday(self, dt: datetime, country_codes: List[str]) -> bool:
        date_str = dt.strftime('%Y-%m-%d')
        year = dt.year
        tasks = [self._fetch_holidays_for_year(year, code) for code in country_codes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, set) and date_str in res:
                return True
        return False

# --- Load Model Artifacts ---
try:
    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
    feature_list = json.load(open(FEATURES_PATH, 'r'))
    log.info(f"✅ [ML_Predictor] Loaded model with {len(feature_list)} features")
    log.info(f"   Categorical encoders: {list(label_encoders.keys())}")
except Exception as e:
    log.error(f"❌ [ML_Predictor] CRITICAL: Could not load model: {e}")
    model = None
    label_encoders = {}
    feature_list = []

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
    MUST match training script's feature engineering exactly!
    """
    if not model or not feature_list:
        log.error("❌ Model not loaded")
        return None

    try:
        features = {}
        
        # === 1. TIMESTAMP ===
        ts_str = data_dict.get('checked_at', data_dict.get('ts', datetime.now(timezone.utc).isoformat()))
        dt = pd.to_datetime(ts_str).tz_convert(timezone.utc)
        checked_at_timestamp = int(dt.timestamp())
        
        # === 2. TIME FEATURES ===
        hour = dt.hour
        features['time_of_day_utc'] = hour
        features['day_of_week_utc'] = dt.dayofweek
        features['is_weekend_utc'] = int(dt.dayofweek >= 5)
        
        # NEW: Regional prime time indicators
        features['is_asia_prime'] = int(6 <= hour < 14)
        features['is_us_prime'] = int(13 <= hour < 21)
        features['is_eu_prime'] = int(8 <= hour < 16)
        features['is_dead_hours'] = int(0 <= hour < 6)
        
        # Hour category
        if 0 <= hour < 7:
            features['hour_category'] = 'dead_hours'
        elif 7 <= hour < 15:
            features['hour_category'] = 'asia_hours'
        elif 15 <= hour < 19:
            features['hour_category'] = 'eu_hours'
        else:
            features['hour_category'] = 'us_hours'
        
        features['is_last_day_of_week'] = int(dt.dayofweek == 4)
        
        # === 3. HOLIDAY FEATURE ===
        try:
            holiday_client = HolidayClient(session)
            features['is_public_holiday_any'] = await holiday_client.is_holiday(dt, HOLIDAY_COUNTRY_CODES)
        except Exception as e:
            log.warning(f"Holiday check failed: {e}")
            features['is_public_holiday_any'] = False
        
        # === 4. SIGNAL FEATURES ===
        features['signal_source'] = signal_source
        features['grade'] = data_dict.get('grade', 'NONE')
        
        # === 5. MARKET FEATURES ===
        raw_dex = data_dict.get('raw_dexscreener', data_dict.get('raw', {}))
        if not isinstance(raw_dex, dict):
            raw_dex = {}
        
        features['fdv_usd'] = _safe_float(data_dict.get('fdv_usd', raw_dex.get('fdv')))
        features['liquidity_usd'] = _safe_float(data_dict.get('liquidity_usd', data_dict.get('total_lp_usd')))
        features['volume_h24_usd'] = _safe_float(
            data_dict.get('volume_h24_usd', raw_dex.get('volume', {}).get('h24'))
        )
        features['price_change_h24_pct'] = _safe_float(
            data_dict.get('price_change_h24_pct', raw_dex.get('priceChange', {}).get('h24'))
        )
        
        # === 6. SECURITY FEATURES ===
        raw_rug = data_dict.get('raw_rugcheck', data_dict.get('raw', {}))
        if not isinstance(raw_rug, dict):
            raw_rug = {}
        
        features['has_mint_authority'] = int(data_dict.get('has_mint_authority', data_dict.get('mint_authority') is not None))
        features['has_freeze_authority'] = int(data_dict.get('has_freeze_authority', data_dict.get('freeze_authority') is not None))
        features['creator_balance_pct'] = _safe_float(data_dict.get('creator_balance_pct'))
        features['top_10_holders_pct'] = _safe_float(data_dict.get('top_10_holders_pct'))
        
        lp_locked_pct = _safe_float(data_dict.get('lp_locked_pct', data_dict.get('overall_lp_locked_pct')))
        features['is_lp_locked_95_plus'] = int(lp_locked_pct >= 95.0)
        
        markets = (raw_rug.get('data') or raw_rug.get('raw') or {}).get('markets', [])
        features['total_lp_locked_usd'] = sum(m.get('lp', {}).get('lpLockedUSD', 0) for m in markets)
        
        features['rugcheck_risk_level'] = (data_dict.get('probation_meta') or {}).get('level', 'unknown')
        
        # === 7. TOKEN AGE ===
        token_age_seconds = None
        block_time = data_dict.get('block_time')
        pair_created_at = raw_dex.get('pairCreatedAt')
        
        if block_time:
            token_age_seconds = max(0, checked_at_timestamp - int(block_time))
        elif pair_created_at:
            token_age_seconds = max(0, checked_at_timestamp - int(pair_created_at / 1000))
        
        features['token_age_at_signal_seconds'] = token_age_seconds if token_age_seconds is not None else 100000
        features['is_new_token'] = int((token_age_seconds or 100000) < 43200)
        
        # === 8. SMART MONEY FEATURES ===
        overlap_count = _safe_float(data_dict.get('overlap_count'))
        weighted_concentration = _safe_float(data_dict.get('weighted_concentration'))
        total_winner_wallets = _safe_float(data_dict.get('total_winner_wallets'), 1.0)
        holder_count = _safe_float(data_dict.get('holder_count', data_dict.get('total_holders')), 1.0)
        
        features['overlap_quality_score'] = _safe_divide(
            overlap_count * weighted_concentration, total_winner_wallets
        )
        features['winner_wallet_density'] = _safe_divide(overlap_count, holder_count)
        
        # === 9. DERIVED MARKET FEATURES ===
        features['volume_to_liquidity_ratio'] = _safe_divide(
            features['volume_h24_usd'], features['liquidity_usd']
        )
        features['fdv_to_liquidity_ratio'] = _safe_divide(
            features['fdv_usd'], features['liquidity_usd']
        )
        
        # === 10. RISK FEATURES ===
        top1_pct = _safe_float(data_dict.get('top1_holder_pct'))
        top10_pct = features['top_10_holders_pct']
        features['whale_concentration_score'] = (
            (top1_pct * 3) + (top10_pct - top1_pct) if top10_pct >= top1_pct else top1_pct * 3
        )
        
        transfer_fee = _safe_float(data_dict.get('transfer_fee_pct'))
        features['authority_risk_score'] = (
            (50 if features['has_mint_authority'] else 0) +
            (50 if features['has_freeze_authority'] else 0) +
            (transfer_fee * 100 if transfer_fee else 0)
        )
        
        features['creator_dumped'] = int(
            (features['creator_balance_pct'] == 0) and 
            (features['token_age_at_signal_seconds'] < 86400)
        )
        
        # Pump dump risk score
        insider_networks = (raw_rug.get('data') or raw_rug.get('raw') or {}).get('insiderNetworks', [])
        largest_insider_network_size = max((n.get('size', 0) for n in insider_networks), default=0)
        
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
        
        if features['creator_dumped']: pump_dump_score += 10
        elif features['creator_balance_pct'] > 10: pump_dump_score += 7
        elif features['creator_balance_pct'] > 5: pump_dump_score += 4
        
        features['pump_dump_risk_score'] = pump_dump_score
        
        # === 11. INTERACTION FEATURES (NEW) ===
        features['high_risk_combo'] = int(
            (pump_dump_score > 30) and (features['authority_risk_score'] > 15)
        )
        features['critical_signal_quality'] = int(
            (features['grade'] == 'CRITICAL') and (signal_source == 'alpha')
        )
        
        # === 12. ENCODE CATEGORICAL FEATURES ===
        for col in label_encoders:
            if col in features:
                le = label_encoders[col]
                val = str(features[col])
                if val not in le.classes_:
                    val = le.classes_[0]
                features[col] = le.transform([val])[0]
        
        # === 13. CREATE DATAFRAME ===
        df = pd.DataFrame([features])
        
        # Add missing features with defaults
        for col in feature_list:
            if col not in df.columns:
                if col in label_encoders:
                    df[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]
                elif any(k in col for k in ['pct', 'usd', 'score', 'ratio']):
                    df[col] = 0.0
                elif any(k in col for k in ['is_', 'has_']):
                    df[col] = 0
                else:
                    df[col] = 0
        
        # Return in exact feature order
        return df[feature_list]
        
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
        session: aiohttp session
        
    Returns:
        Win probability [0.0-1.0] or None on failure
    """
    if not model:
        return None
    
    features_df = await _prepare_features(data_dict, signal_source, session)
    
    if features_df is None:
        return None
    
    try:
        probabilities = model.predict_proba(features_df)
        win_probability = float(probabilities[0][1])
        
        log.debug(f"[ML_Predictor] P(win)={win_probability:.3f} for {signal_source} signal")
        return win_probability
        
    except Exception as e:
        log.error(f"❌ Prediction failed: {e}", exc_info=True)
        return None