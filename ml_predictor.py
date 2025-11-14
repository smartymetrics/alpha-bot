#!/usr/bin/env python3
"""
ml_predictor.py

Loads the trained XGBoost model and provides a simple interface
to predict the "win" probability for a given token.

(V2) Now includes feature-enrichment logic from collector.py,
such as the HolidayClient, to fully build the feature set.
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
# Use a simple logger if this file is run standalone
log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=logging.INFO)

# --- Logic transplanted from collector.py ---

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
                        log.debug(f"[Cache HIT] for {func.__name__}")
                        return result
                    else:
                        log.debug(f"[Cache EXPIRED] for {func.__name__}")
                        del cache[key]

            log.debug(f"[Cache MISS] for {func.__name__}")
            new_result = await func(*args, **kwargs)

            async with lock:
                cache[key] = (new_result, now + ttl_seconds)
            
            return new_result
        return wrapper
    return decorator

class BaseAPIClient:
    """Base class for API clients with retries and backoff."""
    def __init__(self, session: aiohttp.ClientSession, max_retries: int, name: str):
        self.session = session
        self.max_retries = max_retries
        self.name = name

    async def async_get(self, url: str, timeout: int) -> Optional[Dict]:
        """Performs an async GET request with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                # Use the session passed to the client
                if not self.session or self.session.closed:
                    log.error(f"[{self.name}] HTTP session is closed or None.")
                    return None
                    
                async with self.session.get(url, timeout=timeout) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    log.warning(f"[{self.name}] API Error: Status {resp.status} for {url}")
                    if resp.status >= 500 or resp.status == 429:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        return None
            except asyncio.TimeoutError:
                log.warning(f"[{self.name}] Timeout for {url} (attempt {attempt+1})")
            except aiohttp.ClientError as e:
                log.error(f"[{self.name}] ClientError for {url}: {e}")
            
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
        """Fetches all holidays for a year/country and returns a set of 'YYYY-MM-DD' strings."""
        url = f"{self.BASE_URL}/PublicHolidays/{year}/{country_code}"
        data = await self.async_get(url, self.timeout)
        if data and isinstance(data, list):
            return {item['date'] for item in data if 'date' in item}
        return set()

    async def is_holiday(self, dt: datetime, country_codes: List[str]) -> bool:
        """Checks if the given date is a holiday in any of the specified countries."""
        date_str = dt.strftime('%Y-%m-%d')
        year = dt.year
        tasks = [self._fetch_holidays_for_year(year, code) for code in country_codes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, set) and date_str in res:
                return True
        return False

# --- End of transplanted logic ---


# --- Load Model and Artifacts ---
try:
    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
    feature_list = json.load(open(FEATURES_PATH, 'r'))
    print("✅ [ML_Predictor] Model, encoders, and feature list loaded.")
except Exception as e:
    log.error(f"❌ [ML_Predictor] CRITICAL ERROR: Could not load model artifacts: {e}")
    model = None
    label_encoders = {}
    feature_list = []

def _safe_divide(a, b, default=0.0):
    """Helper function for safe division."""
    if a is None or b is None:
        return default
    try:
        a_f = float(a)
        b_f = float(b)
        if b_f == 0:
            return default
        return a_f / b_f
    except (ValueError, TypeError):
        return default

async def _prepare_features(
    data_dict: dict, 
    signal_source: str,
    session: aiohttp.ClientSession
) -> pd.DataFrame | None:
    """
    Converts a raw token data dictionary from the monitors into a
    single-row DataFrame that matches the model's required features.
    
    This function transplants logic from 'extract_datasets.py' and
    'collector.py' (for holidays).
    """
    if not model or not feature_list:
        log.error("❌ [ML_Predictor] Model not loaded. Cannot prepare features.")
        return None

    try:
        features = {}
        
        # --- 1. Get Timestamp ---
        ts_str = data_dict.get('checked_at', data_dict.get('ts', datetime.now(timezone.utc).isoformat()))
        dt = pd.to_datetime(ts_str).tz_convert(timezone.utc)
        checked_at_timestamp = int(dt.timestamp())
        
        # --- 2. Time Features (from extract_datasets.py) ---
        features['checked_at_timestamp'] = checked_at_timestamp
        features['time_of_day_utc'] = dt.hour
        features['day_of_week_utc'] = dt.dayofweek
        features['is_weekend_utc'] = (dt.dayofweek >= 5)
        
        # --- 3. Holiday Feature (from collector.py) ---
        try:
            holiday_client = HolidayClient(session)
            features['is_public_holiday_any'] = await holiday_client.is_holiday(dt, HOLIDAY_COUNTRY_CODES)
        except Exception as e:
            log.warning(f"[ML_Predictor] Holiday check failed: {e}. Defaulting to False.")
            features['is_public_holiday_any'] = False
        
        # --- 4. Signal Features (from extract_datasets.py) ---
        features['signal_source'] = signal_source
        features['grade'] = data_dict.get('grade', 'NONE')

        # --- 5. Market Features (from extract_datasets.py) ---
        raw_dex = data_dict.get('raw_dexscreener', data_dict.get('raw', {})) # Check both keys
        if not isinstance(raw_dex, dict):
            raw_dex = {}
            
        features['price_usd'] = float(data_dict.get('price_usd', raw_dex.get('priceUsd', 0.0) or 0.0))
        features['fdv_usd'] = float(data_dict.get('fdv_usd', raw_dex.get('fdv', 0.0) or 0.0))
        features['liquidity_usd'] = float(data_dict.get('liquidity_usd', data_dict.get('total_lp_usd', 0.0) or 0.0))
        features['volume_h24_usd'] = float(data_dict.get('volume_h24_usd', raw_dex.get('volume', {}).get('h24', 0.0) or 0.0))
        features['price_change_h24_pct'] = float(data_dict.get('price_change_h24_pct', raw_dex.get('priceChange', {}).get('h24', 0.0) or 0.0))
        
        pair_created_at_ms = raw_dex.get('pairCreatedAt')
        pair_created_at_timestamp = _safe_divide(pair_created_at_ms, 1000) if pair_created_at_ms else None

        # --- 6. Security Features (from extract_datasets.py) ---
        raw_rug = data_dict.get('raw_rugcheck', data_dict.get('raw', {})) # Check both keys
        if not isinstance(raw_rug, dict):
            raw_rug = {}

        features['has_mint_authority'] = data_dict.get('has_mint_authority', data_dict.get('mint_authority') is not None)
        features['has_freeze_authority'] = data_dict.get('has_freeze_authority', data_dict.get('freeze_authority') is not None)
        features['creator_balance_pct'] = float(data_dict.get('creator_balance_pct', 0.0) or 0.0)
        features['top_10_holders_pct'] = float(data_dict.get('top_10_holders_pct', 0.0) or 0.0)
        
        lp_locked_pct = float(data_dict.get('lp_locked_pct', data_dict.get('overall_lp_locked_pct', 0.0) or 0.0))
        features['is_lp_locked_95_plus'] = (lp_locked_pct >= 95.0)
        
        markets_from_raw = (raw_rug.get('data') or raw_rug.get('raw') or {}).get('markets', [])
        total_lp_locked_usd_from_raw = 0.0
        if markets_from_raw:
            total_lp_locked_usd_from_raw = sum(m.get('lp', {}).get('lpLockedUSD', 0) for m in markets_from_raw)
        features['total_lp_locked_usd'] = total_lp_locked_usd_from_raw

        features['rugcheck_risk_level'] = (data_dict.get('probation_meta') or {}).get('level', 'unknown')
        
        # --- 7. Token Age (from extract_datasets.py) ---
        token_age_at_signal_seconds = None
        block_time = data_dict.get('block_time')
        
        if block_time:
            token_age_at_signal_seconds = max(0, checked_at_timestamp - int(block_time))
        elif pair_created_at_timestamp:
            token_age_at_signal_seconds = max(0, checked_at_timestamp - int(pair_created_at_timestamp))
        
        features['token_age_at_signal_seconds'] = token_age_at_signal_seconds if token_age_at_signal_seconds is not None else 100_000

        # --- 8. DERIVED FEATURES (from extract_datasets.py) ---
        
        # (*** BUG FIX ***) Apply robust None-handling to these float conversions
        overlap_count = float(data_dict.get('overlap_count') or 0.0)
        weighted_concentration = float(data_dict.get('weighted_concentration') or 0.0)
        total_winner_wallets = float(data_dict.get('total_winner_wallets') or 1.0)
        
        # This is the line that crashed
        holder_count_val = data_dict.get('holder_count', data_dict.get('total_holders'))
        holder_count = float(holder_count_val or 1.0)
        
        features['overlap_quality_score'] = _safe_divide(overlap_count * weighted_concentration, total_winner_wallets, 0)
        features['winner_wallet_density'] = _safe_divide(overlap_count, holder_count, 0)

        total_liquidity = features['liquidity_usd']
        features['volume_to_liquidity_ratio'] = _safe_divide(features['volume_h24_usd'], total_liquidity, 0)
        features['fdv_to_liquidity_ratio'] = _safe_divide(features['fdv_usd'], total_liquidity, 0)

        top1_pct = float(data_dict.get('top1_holder_pct', 0.0) or 0.0)
        top10_pct = features['top_10_holders_pct']
        features['whale_concentration_score'] = (top1_pct * 3) + (top10_pct - top1_pct) if top10_pct >= top1_pct else top1_pct * 3

        ts_age = features['token_age_at_signal_seconds']
        features['is_new_token'] = int(ts_age < 43200)

        transfer_fee = float(data_dict.get('transfer_fee_pct', 0.0) or 0.0)
        features['authority_risk_score'] = (
            (50 if features['has_mint_authority'] else 0) + 
            (50 if features['has_freeze_authority'] else 0) + 
            (transfer_fee * 100 if transfer_fee else 0)
        )
        
        features['creator_dumped'] = (features['creator_balance_pct'] == 0) and (ts_age < 86400)

        insider_networks = (raw_rug.get('data') or raw_rug.get('raw') or {}).get('insiderNetworks') or []
        largest_insider_network_size = 0
        if insider_networks:
            try:
                largest_insider_network_size = max(n.get('size', 0) for n in insider_networks)
            except ValueError:
                pass 

        pump_dump_components = []
        if largest_insider_network_size > 100: pump_dump_components.append(30)
        elif largest_insider_network_size > 50: pump_dump_components.append(20)
        elif largest_insider_network_size > 20: pump_dump_components.append(10)
        
        if top1_pct > 30: pump_dump_components.append(25)
        elif top1_pct > 20: pump_dump_components.append(15)
        elif top1_pct > 10: pump_dump_components.append(8)

        if lp_locked_pct < 50: pump_dump_components.append(20)
        elif lp_locked_pct < 80: pump_dump_components.append(12)
        elif lp_locked_pct < 95: pump_dump_components.append(5)

        pump_dump_components.append(min(features['authority_risk_score'] / 100 * 15, 15))
        
        if features.get('creator_dumped'): pump_dump_components.append(10)
        elif features['creator_balance_pct'] > 10: pump_dump_components.append(7)
        elif features['creator_balance_pct'] > 5: pump_dump_components.append(4)
        
        features['pump_dump_risk_score'] = sum(pump_dump_components)
        
        hour = features['time_of_day_utc']
        if 0 <= hour < 7: features['hour_category'] = 'dead_hours'
        elif 7 <= hour < 15: features['hour_category'] = 'asia_hours'
        elif 15 <= hour < 19: features['hour_category'] = 'eu_hours'
        else: features['hour_category'] = 'us_hours'
        
        features['is_last_day_of_week'] = (features['day_of_week_utc'] == 4)
        
        # --- 9. Encoding Categorical Features ---
        for col in label_encoders:
            if col in features and col in label_encoders:
                le = label_encoders[col]
                val = str(features[col])
                if val not in le.classes_:
                    val = le.classes_[0] 
                features[col] = le.transform([val])[0]

        # --- 10. Final DataFrame Preparation ---
        df = pd.DataFrame([features])
        
        for col in feature_list:
            if col not in df.columns:
                if col in label_encoders:
                    df[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]
                elif any(c in col for c in ['pct', 'usd', 'score', 'ratio']):
                     df[col] = 0.0
                elif any(c in col for c in ['is_', 'has_']):
                    df[col] = False
                else:
                    df[col] = 0
        
        return df[feature_list]

    except Exception as e:
        log.error(f"❌ [ML_Predictor] Failed to prepare features: {e}", exc_info=True)
        return None

async def predict_token_win_probability(
    data_dict: dict, 
    signal_source: str,
    session: aiohttp.ClientSession
) -> float | None:
    """
    Main prediction function.
    
    Args:
        data_dict: A combined dictionary of token data from the monitor.
        signal_source: 'discovery' (for token_monitor) or 'alpha' (for winner_monitor).
        session: The aiohttp.ClientSession from the calling monitor.
        
    Returns:
        A float (0.0 to 1.0) for the 'win' probability, or None on failure.
    """
    if not model:
        return None
        
    features_df = await _prepare_features(data_dict, signal_source, session)
    
    if features_df is None:
        return None
        
    try:
        probabilities = model.predict_proba(features_df)
        win_probability = probabilities[0][1]
        return float(win_probability)
        
    except Exception as e:
        log.error(f"❌ [ML_Predictor] Failed to predict: {e}", exc_info=True)
        return None