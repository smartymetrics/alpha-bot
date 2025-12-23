#!/usr/bin/env python3
"""
extract_datasets.py

Extracts training data from your Supabase datasets (created by collector.py)
and prepares it in the EXACT format needed for model training.
"""

import os
import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import argparse
from supabase import create_client
from dotenv import load_dotenv
import logging
from typing import Optional, List, Dict

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract ML features from collector.py snapshots."""
    
    # Updated Mapping to include creator_address and all requested metrics
    FEATURE_MAPPING = {
        # Identifiers
        'mint': 'mint',
        'creator_address': 'creator_address',
        
        # Market features
        'price_usd': 'price_usd',
        'fdv_usd': 'fdv_usd',
        'liquidity_usd': 'liquidity_usd',
        'volume_h24_usd': 'volume_h24_usd',
        'price_change_h24_pct': 'price_change_h24_pct',
        
        # Ratios
        'volume_to_liquidity_ratio': 'volume_to_liquidity_ratio',
        'fdv_to_liquidity_ratio': 'fdv_to_liquidity_ratio',
        'liquidity_to_volume_ratio': 'liquidity_to_volume_ratio',
        
        # Security features
        'creator_balance_pct': 'creator_balance_pct',
        'top_10_holders_pct': 'top_10_holders_pct',
        'total_lp_locked_usd': 'total_lp_locked_usd',
        'has_mint_authority': 'has_mint_authority',
        'has_freeze_authority': 'has_freeze_authority',
        'is_lp_locked_95_plus': 'is_lp_locked_95_plus',
        
        # Insider / Supply Features
        'token_supply': 'token_supply',
        'total_insider_networks': 'total_insider_networks',
        'largest_insider_network_size': 'largest_insider_network_size',
        'total_insider_token_amount': 'total_insider_token_amount',
        
        # Risk encoding
        'rugcheck_risk_level': 'rugcheck_risk_level',
        'pump_dump_risk_score': 'pump_dump_risk_score',
        
        # Time features
        'time_of_day_utc': 'time_of_day_utc',
        'day_of_week_utc': 'day_of_week_utc',
        'is_weekend_utc': 'is_weekend_utc',
        'is_public_holiday_any': 'is_public_holiday_any',
        
        # Signal features
        'signal_source': 'signal_source',
        'grade': 'grade',
        
        # Token age
        'token_age_at_signal_seconds': 'token_age_at_signal_seconds',
        
        # Timestamp for sorting
        'checked_at_timestamp': 'checked_at_timestamp',
        
        # Internal Use (Needed for deduplication logic)
        'checked_at_utc': 'checked_at_utc',
        'token_age_hours_at_signal': 'token_age_hours_at_signal',
    }
    
    def __init__(self):
        self.supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        self.bucket = os.getenv("SUPABASE_BUCKET", "monitor-data")
        
    async def download_dataset_file(self, remote_path: str) -> Optional[Dict]:
        """Download a single dataset JSON from Supabase."""
        try:
            response = await asyncio.to_thread(
                self.supabase.storage.from_(self.bucket).download,
                remote_path
            )
            return json.loads(response)
        except Exception as e:
            logger.debug(f"Failed to download {remote_path}: {e}")
            return None
    
    async def list_all_dataset_folders(self, pipeline: str) -> List[str]:
        try:
            folders = await asyncio.to_thread(
                self.supabase.storage.from_(self.bucket).list,
                f"datasets/{pipeline}"
            )
            date_folders = []
            for folder in folders:
                name = folder.get('name', '')
                if name == 'expired_no_label': continue
                if len(name) == 10 and name[4] == '-' and name[7] == '-':
                    try:
                        datetime.strptime(name, '%Y-%m-%d')
                        date_folders.append(name)
                    except ValueError: continue
            return sorted(date_folders)
        except Exception as e:
            logger.error(f"Failed to list folders for {pipeline}: {e}")
            return []
    
    async def list_dataset_files(self, pipeline: str, date_str: str) -> List[str]:
        folder = f"datasets/{pipeline}/{date_str}"
        try:
            files = await asyncio.to_thread(
                self.supabase.storage.from_(self.bucket).list,
                folder
            )
            return [f['name'] for f in files if f['name'].endswith('.json')]
        except Exception as e:
            return []
    
    def extract_features_from_snapshot(self, snapshot: dict) -> dict:
        """
        Robust extraction logic that specifically targets nested JSON structures
        to fix zero-value issues.
        """
        features_dict = {}
        
        # --- 1. Helper Functions ---
        def safe_divide(a, b, default=0.0):
            try:
                if not b or float(b) == 0: return default
                return float(a) / float(b)
            except (ValueError, TypeError):
                return default

        def get_nested(data, path, default=None):
            """Safely retrieve nested keys: get_nested(d, 'a.b.c')"""
            keys = path.split('.')
            curr = data
            for k in keys:
                if isinstance(curr, dict) and k in curr:
                    curr = curr[k]
                else:
                    return default
            return curr

        # --- 2. Data Source Normalization ---
        inputs = snapshot.get('inputs', {})
        features_block = snapshot.get('features', {})
        
        # Signal Data (Reliable fallback for market data)
        signal_result = get_nested(inputs, 'signal_data.result', {})
        
        # === RUGCHECK DATA STRATEGY ===
        rc_raw_wrapper = inputs.get('rugcheck_raw', {})
        
        # Primary Source: The deep 'raw' object
        rc_inner = rc_raw_wrapper.get('raw')
        
        # Fallback Source: signal_data -> security -> rugcheck_raw -> raw
        if not rc_inner:
            rc_inner = get_nested(signal_result, 'security.rugcheck_raw.raw', {})
        
        # Ensure rc_inner is a dict
        if not isinstance(rc_inner, dict):
            rc_inner = {}
            
        # Summary Source (for simple flags like lp_locked_pct)
        rc_summary = get_nested(signal_result, 'security.rugcheck', {})

        # === DEXSCREENER DATA STRATEGY ===
        dex_raw = inputs.get('dexscreener_raw', {})
        dex_inner = {}
        
        # 1. Check if dexscreener_raw has pair data directly
        if dex_raw.get('pair_exists') or (dex_raw.get('pairs') and len(dex_raw['pairs']) > 0):
            pairs = dex_raw.get('pairs', [])
            if pairs: dex_inner = pairs[0]
        # 2. Fallback to signal_data
        else:
            dex_signal = get_nested(signal_result, 'dexscreener', {})
            if dex_signal.get('ok'):
                raw_signal = dex_signal.get('raw', {})
                # Check if raw_signal is the pair directly or contains pairs
                if 'pairs' in raw_signal:
                    pairs = raw_signal.get('pairs', [])
                    if pairs: dex_inner = pairs[0]
                elif 'pairAddress' in raw_signal:
                    dex_inner = raw_signal
                elif 'baseToken' in raw_signal:
                    dex_inner = raw_signal

        # --- 3. Feature Extraction ---

        # Identifiers
        features_dict['mint'] = features_block.get('mint', signal_result.get('mint'))
        
        # Creator Address: From rc_inner (raw data)
        features_dict['creator_address'] = rc_inner.get('creator', 'UNKNOWN')
        
        features_dict['signal_source'] = features_block.get('signal_source', 'unknown')
        features_dict['grade'] = features_block.get('grade', signal_result.get('grade'))
        
        # Timestamps
        features_dict['checked_at_utc'] = features_block.get('checked_at_utc', signal_result.get('checked_at'))
        features_dict['checked_at_timestamp'] = features_block.get('checked_at_timestamp')
        
        features_dict['time_of_day_utc'] = features_block.get('time_of_day_utc')
        features_dict['day_of_week_utc'] = features_block.get('day_of_week_utc')
        features_dict['is_weekend_utc'] = features_block.get('is_weekend_utc')
        
        holiday_check = inputs.get('holiday_check', {})
        features_dict['is_public_holiday_any'] = features_block.get('is_public_holiday_any', holiday_check.get('is_holiday'))

        # --- MARKET DATA ---
        
        # Price
        price = features_block.get('price_usd')
        if price is None or price == 0:
            price = dex_inner.get('priceUsd')
        if price is None or price == 0:
            price = get_nested(signal_result, 'dexscreener.price_usd')
        if price is None or price == 0:
            price = rc_inner.get('price')
        features_dict['price_usd'] = float(price) if price else 0.0

        # FDV
        fdv = features_block.get('fdv_usd')
        if fdv is None or fdv == 0:
            fdv = dex_inner.get('fdv')
        if fdv is None or fdv == 0:
            fdv = dex_inner.get('marketCap')
        if fdv is None or fdv == 0:
            fdv = get_nested(signal_result, 'dexscreener.raw.fdv')
        if fdv is None or fdv == 0:
            fdv = get_nested(signal_result, 'dexscreener.raw.marketCap')
        features_dict['fdv_usd'] = float(fdv) if fdv else 0.0
        
        # Liquidity
        liq = features_block.get('liquidity_usd')
        if liq is None or liq == 0:
            liq = get_nested(dex_inner, 'liquidity.usd')
        if liq is None or liq == 0:
            liq = rc_summary.get('total_liquidity_usd')
        if liq is None or liq == 0:
            liq = rc_raw_wrapper.get('total_lp_usd')
        features_dict['liquidity_usd'] = float(liq) if liq else 0.0

        # Volume H24
        vol = features_block.get('volume_h24_usd')
        if vol is None or vol == 0:
            vol = get_nested(dex_inner, 'volume.h24')
        if vol is None or vol == 0:
            vol = get_nested(signal_result, 'dexscreener.raw.volume.h24')
        features_dict['volume_h24_usd'] = float(vol) if vol else 0.0
        
        # Price Change H24
        chg = features_block.get('price_change_h24_pct')
        if chg is None or chg == 0:
            chg = get_nested(dex_inner, 'priceChange.h24')
        if chg is None or chg == 0:
            chg = get_nested(signal_result, 'dexscreener.raw.priceChange.h24')
        features_dict['price_change_h24_pct'] = float(chg) if chg else 0.0

        # Pair Age & Token Age
        pair_created_at_ms = dex_inner.get('pairCreatedAt')
        if not pair_created_at_ms:
            pair_created_at_ms = get_nested(signal_result, 'dexscreener.raw.pairCreatedAt')

        pair_created_ts = safe_divide(pair_created_at_ms, 1000) if pair_created_at_ms else None
        
        if features_dict['checked_at_timestamp'] and pair_created_ts:
            features_dict['token_age_at_signal_seconds'] = features_dict['checked_at_timestamp'] - pair_created_ts
        else:
            features_dict['token_age_at_signal_seconds'] = 0

        # --- SECURITY / RUGCHECK DATA ---
        
        # Token Supply and Decimals
        token_supply = get_nested(rc_inner, 'token.supply', 0)
        token_decimals = get_nested(rc_inner, 'token.decimals', 0)
        
        # Ensure numeric types (handle None gracefully)
        token_supply = float(token_supply) if token_supply is not None else 0.0
        token_decimals = int(token_decimals) if token_decimals is not None else 0
        
        features_dict['token_supply'] = token_supply

        # Authorities
        has_mint = features_block.get('has_mint_authority')
        if has_mint is None:
            raw_auth = get_nested(rc_inner, 'token.mintAuthority')
            has_mint = raw_auth is not None
        features_dict['has_mint_authority'] = int(has_mint)
        
        has_freeze = features_block.get('has_freeze_authority')
        if has_freeze is None:
            raw_auth = get_nested(rc_inner, 'token.freezeAuthority')
            has_freeze = raw_auth is not None
        features_dict['has_freeze_authority'] = int(has_freeze)
        
        # Creator Balance - FIXED: Normalize by decimals
        creator_bal = rc_inner.get('creatorBalance', 0)
        creator_pct = features_block.get('creator_balance_pct')
        
        # Coerce creator_bal to float safely
        try:
            creator_bal = float(creator_bal) if creator_bal is not None else 0.0
        except (ValueError, TypeError):
            creator_bal = 0.0
        
        # Calculate manually if missing or zero
        # IMPORTANT: Normalize creator_bal by decimals before dividing by supply
        if (creator_pct is None or creator_pct == 0) and token_supply > 0:
            creator_bal_normalized = 0.0
            if creator_bal > 0 and token_decimals > 0:
                try:
                    # Normalize the raw creator balance by decimals
                    creator_bal_normalized = creator_bal / (10 ** token_decimals)
                except (ValueError, TypeError, ZeroDivisionError):
                    creator_bal_normalized = 0.0
            
            if creator_bal_normalized > 0:
                creator_pct = (creator_bal_normalized / token_supply) * 100
            else:
                creator_pct = 0.0
        
        features_dict['creator_balance_pct'] = float(creator_pct) if creator_pct else 0.0

        # Top 10 Holders
        top_holders = rc_inner.get('topHolders') or []
        top_10_pct = features_block.get('top_10_holders_pct')
        if top_10_pct is None or top_10_pct == 0:
            top_10_pct = sum(h.get('pct', 0) for h in top_holders[:10])
        features_dict['top_10_holders_pct'] = float(top_10_pct)
        
        top_1_pct = sum(h.get('pct', 0) for h in top_holders[:1])

        # LP Locked
        lp_locked_usd = features_block.get('total_lp_locked_usd')
        if lp_locked_usd is None or lp_locked_usd == 0:
            lp_locked_usd = rc_raw_wrapper.get('total_lp_usd')
        if lp_locked_usd is None or lp_locked_usd == 0:
            markets = rc_inner.get('markets', [])
            lp_locked_usd = sum(m.get('lp', {}).get('lpLockedUSD', 0) for m in markets)
        features_dict['total_lp_locked_usd'] = float(lp_locked_usd) if lp_locked_usd else 0.0

        lp_locked_pct = rc_summary.get('lp_locked_pct')
        if lp_locked_pct is None:
            lp_locked_pct = rc_raw_wrapper.get('overall_lp_locked_pct', 0)
        
        # Safely convert to float
        try:
            lp_locked_pct = float(lp_locked_pct) if lp_locked_pct is not None else 0.0
        except (ValueError, TypeError):
            lp_locked_pct = 0.0
        
        features_dict['is_lp_locked_95_plus'] = int(lp_locked_pct >= 95)

        features_dict['rugcheck_risk_level'] = features_block.get('rugcheck_risk_level', 'unknown')

        # --- INSIDERS (Fix for zero values) ---
        insider_networks = rc_inner.get('insiderNetworks')
        
        # Handle case where insiderNetworks is None (JSON null) or missing
        if insider_networks is None:
            insider_networks = []
        
        features_dict['total_insider_networks'] = len(insider_networks)
        
        if insider_networks:
            features_dict['largest_insider_network_size'] = max((n.get('size', 0) for n in insider_networks), default=0)
            features_dict['total_insider_token_amount'] = sum((n.get('tokenAmount', 0) for n in insider_networks), start=0)
        else:
            features_dict['largest_insider_network_size'] = 0
            features_dict['total_insider_token_amount'] = 0

        # --- DERIVED RATIOS ---
        f_fdv = features_dict['fdv_usd']
        f_liq = features_dict['liquidity_usd']
        f_vol = features_dict['volume_h24_usd']
        
        features_dict['volume_to_liquidity_ratio'] = safe_divide(f_vol, f_liq)
        features_dict['fdv_to_liquidity_ratio'] = safe_divide(f_fdv, f_liq)
        features_dict['liquidity_to_volume_ratio'] = safe_divide(f_liq, f_vol)

        # --- RISK SCORE ---
        pump_dump_components = []
        
        # Insider risk - ensure i_size is numeric
        i_size = features_dict.get('largest_insider_network_size', 0) or 0
        if isinstance(i_size, (int, float)):
            if i_size > 100: pump_dump_components.append(30)
            elif i_size > 50: pump_dump_components.append(20)
            elif i_size > 20: pump_dump_components.append(10)
        
        # Whale risk - ensure top_1_pct is numeric
        top_1_pct = sum(h.get('pct', 0) for h in top_holders[:1]) if top_holders else 0
        if isinstance(top_1_pct, (int, float)):
            if top_1_pct > 30: pump_dump_components.append(25)
            elif top_1_pct > 20: pump_dump_components.append(15)
            elif top_1_pct > 10: pump_dump_components.append(8)
        
        # LP Risk
        lp_pct_val = float(lp_locked_pct) if lp_locked_pct is not None else 0
        if lp_pct_val < 50: pump_dump_components.append(20)
        elif lp_pct_val < 80: pump_dump_components.append(12)
        elif lp_pct_val < 95: pump_dump_components.append(5)
        
        # Authority Risk
        transfer_fee = rc_summary.get('transfer_fee_pct', 0)
        auth_score = (50 if has_mint else 0) + (50 if has_freeze else 0) + (transfer_fee * 100)
        pump_dump_components.append(min(auth_score / 100 * 15, 15))
        
        # Creator Risk / Token Age Check
        finalization = snapshot.get('finalization', {})
        raw_age = finalization.get('token_age_hours_at_signal')
        
        # Force cast string ages to float
        try:
            token_age_hours = float(raw_age) if raw_age is not None else None
        except (ValueError, TypeError):
            token_age_hours = None
            
        features_dict['token_age_hours_at_signal'] = token_age_hours
        
        creator_dumped = (creator_pct == 0) and (token_age_hours is not None and token_age_hours < 24)
        if creator_dumped: pump_dump_components.append(10)
        elif isinstance(creator_pct, (int, float)) and creator_pct > 10: pump_dump_components.append(7)
        elif isinstance(creator_pct, (int, float)) and creator_pct > 5: pump_dump_components.append(4)
        
        features_dict['pump_dump_risk_score'] = sum(pump_dump_components)

        # --- 4. Clean Final Dictionary ---
        clean_row = {}
        for ml_key, map_key in self.FEATURE_MAPPING.items():
            val = features_dict.get(ml_key)
            
            # Type Enforcement
            if ml_key.endswith('_ratio') or ml_key.endswith('_usd') or ml_key.endswith('_pct'):
                clean_row[ml_key] = float(val) if val is not None else 0.0
            elif ml_key.startswith('is_') or ml_key.startswith('has_'):
                clean_row[ml_key] = int(val) if val else 0
            elif ml_key in ['token_supply', 'largest_insider_network_size', 'total_insider_networks', 'total_insider_token_amount']:
                clean_row[ml_key] = float(val) if val is not None else 0.0
            elif val is None:
                clean_row[ml_key] = 'UNKNOWN' if isinstance(clean_row.get(ml_key, ''), str) else 0
            else:
                clean_row[ml_key] = val
                
        # --- 5. Add Labels ---
        label = snapshot.get('label', {})
        if label:
            clean_row['label_status'] = label.get('status', 'unknown')
            clean_row['label_ath_roi'] = label.get('ath_roi', 0)
            clean_row['label_final_roi'] = label.get('final_roi', 0)
            clean_row['label_hit_50_percent'] = int(label.get('hit_50_percent', False))
        else:
            clean_row['label_status'] = 'expired'
            clean_row['label_ath_roi'] = 0
            clean_row['label_final_roi'] = 0
            clean_row['label_hit_50_percent'] = 0

        return clean_row
    
    async def extract_all_data(self, pipeline: str) -> List[Dict]:
        logger.info(f"Discovering all date folders for {pipeline}...")
        date_folders = await self.list_all_dataset_folders(pipeline)
        if not date_folders: return []
        
        all_features = []
        for date_str in date_folders:
            files = await self.list_dataset_files(pipeline, date_str)
            if files:
                logger.info(f"Processing {pipeline}/{date_str}: {len(files)} files")
                for filename in files:
                    remote_path = f"datasets/{pipeline}/{date_str}/{filename}"
                    snapshot = await self.download_dataset_file(remote_path)
                    if snapshot:
                        try:
                            features = self.extract_features_from_snapshot(snapshot)
                            all_features.append(features)
                        except Exception as e:
                            logger.error(f"Error in {filename}: {e}")
        return all_features
    
    async def extract_date_range(self, pipeline: str, start_date: str, end_date: str) -> List[Dict]:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        all_features = []
        current_date = start_dt
        
        while current_date <= end_dt:
            date_str = current_date.strftime('%Y-%m-%d')
            files = await self.list_dataset_files(pipeline, date_str)
            if files:
                logger.info(f"Processing {pipeline}/{date_str}: {len(files)} files")
                for filename in files:
                    remote_path = f"datasets/{pipeline}/{date_str}/{filename}"
                    snapshot = await self.download_dataset_file(remote_path)
                    if snapshot:
                        try:
                            features = self.extract_features_from_snapshot(snapshot)
                            all_features.append(features)
                        except Exception as e:
                            logger.error(f"Error in {filename}: {e}")
            current_date += timedelta(days=1)
        return all_features
    
    async def create_training_dataset(self, start_date: Optional[str], end_date: Optional[str], output_file: str):
        """
        Create training dataset with optional 7-day incremental extraction.
        
        Strategy:
          - If output_file exists: INCREMENTAL MODE (extract last 7 days, merge with existing)
          - If output_file doesn't exist: FULL EXTRACTION MODE (extract all data)
        
        This reduces extraction time from ~300 min to ~6 min (50x speedup).
        """
        
        # ===== DETERMINE EXTRACTION MODE =====
        if os.path.exists(output_file):
            # INCREMENTAL MODE: Use 7-day lookback window
            logger.info(f"📦 INCREMENTAL MODE: Loading existing {output_file}")
            
            # Get CSV creation date
            csv_stat = os.stat(output_file)
            csv_creation_datetime = datetime.fromtimestamp(csv_stat.st_mtime)
            csv_age_days = (datetime.now() - csv_creation_datetime).days
            logger.info(f"   CSV created: {csv_creation_datetime.strftime('%Y-%m-%d %H:%M:%S')} ({csv_age_days} days ago)")
            
            # Load existing dataset
            df_existing = pd.read_csv(output_file)
            existing_rows = len(df_existing)
            logger.info(f"   Loaded: {existing_rows} existing records")
            
            # Define 7-day lookback boundary
            boundary_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            logger.info(f"   Extraction boundary: {boundary_date} (7 days ago)")
            
            # Extract only the last 7 days
            logger.info(f"Creating training dataset: {boundary_date} to today (INCREMENTAL)")
            discovery_features = await self.extract_date_range('discovery', boundary_date, datetime.now().strftime('%Y-%m-%d'))
            alpha_features = await self.extract_date_range('alpha', boundary_date, datetime.now().strftime('%Y-%m-%d'))
            
            # Combine new extracted data
            new_features = discovery_features + alpha_features
            logger.info(f"   ✅ Extracted {len(new_features)} new samples (last 7 days)")
            
            if not new_features:
                logger.warning("   ⚠️  No new data extracted for last 7 days, using existing dataset")
                df = df_existing.copy()
            else:
                # Filter existing data to keep only pre-boundary records
                df_existing['checked_at_utc'] = pd.to_datetime(df_existing['checked_at_utc'], errors='coerce')
                df_pre_boundary = df_existing[df_existing['checked_at_utc'] < boundary_date].copy()
                logger.info(f"   Kept from old: {len(df_pre_boundary)} records (before {boundary_date})")
                
                # Create DataFrame from new features
                df_new = pd.DataFrame(new_features)
                logger.info(f"   Appending: {len(df_new)} new records ({boundary_date} to today)")
                
                # Merge: old pre-boundary + new 7-day data
                df = pd.concat([df_pre_boundary, df_new], ignore_index=True)
                logger.info(f"   ✅ Merged: {existing_rows} old + {len(df_new)} new = {len(df)} total")
        
        else:
            # FULL EXTRACTION MODE: Extract all available data (first run)
            if start_date and end_date:
                logger.info(f"🆕 FULL EXTRACTION MODE: Creating training dataset: {start_date} to {end_date}")
                discovery_features = await self.extract_date_range('discovery', start_date, end_date)
                alpha_features = await self.extract_date_range('alpha', start_date, end_date)
            else:
                logger.info(f"🆕 FULL EXTRACTION MODE: Creating training dataset with ALL AVAILABLE DATA")
                discovery_features = await self.extract_all_data('discovery')
                alpha_features = await self.extract_all_data('alpha')
            
            all_features = discovery_features + alpha_features
            logger.info(f"   ✅ Extracted: {len(all_features)} total samples")
            
            if not all_features:
                logger.error("No data extracted!")
                return None
            
            df = pd.DataFrame(all_features)
        
        logger.info(f"\n✅ Combined: {len(df)} total samples")
        
        if df.empty:
            logger.error("No data available!")
            return None
        
        # ===== DEDUPLICATION LOGIC =====
        if 'checked_at_utc' in df.columns and 'mint' in df.columns:
            logger.info(f"\n📊 Deduplication by tracking period...")
            
            def filter_by_tracking_period(group):
                rows_to_keep = []
                tracking_end_time = pd.NaT
                for index, row in group.iterrows():
                    current_time = row['checked_at_utc']
                    if pd.isna(current_time): continue
                    
                    if pd.isna(tracking_end_time) or current_time >= tracking_end_time:
                        rows_to_keep.append(index)
                        
                        # Defensive type conversion here
                        raw_age = row.get('token_age_hours_at_signal')
                        try:
                            token_age = float(raw_age) if pd.notna(raw_age) else None
                        except (ValueError, TypeError):
                            token_age = None
                            
                        is_new_token = (token_age is not None and token_age < 12)
                        tracking_duration_hours = 24 if is_new_token else 168
                        tracking_end_time = current_time + pd.Timedelta(hours=tracking_duration_hours)
                return group.loc[rows_to_keep]
            df['checked_at_utc'] = pd.to_datetime(df['checked_at_utc'], errors='coerce')
            df = df.sort_values(by=['mint', 'signal_source', 'checked_at_utc'])
            df = df.groupby(['mint', 'signal_source']).apply(filter_by_tracking_period).reset_index(drop=True)

        
        # ===== REMOVE UNLABELED DATA =====
        if 'label_status' in df.columns:
            unlabeled_count = len(df[~df['label_status'].isin(['win', 'loss'])])
            df = df[df['label_status'].isin(['win', 'loss'])]
            if unlabeled_count > 0:
                logger.info(f"   Removed {unlabeled_count} unlabeled records")
        
        logger.info(f"   Final rows after dedup: {len(df)}")
        
        # ===== VALIDATION & EXPORT =====
        logger.info(f"\n✔️ VALIDATION CHECKS:")
        logger.info(f"   Total records: {len(df)}")
        logger.info(f"   Columns: {len(df.columns)}")
        logger.info(f"   Null values: {df.isnull().sum().sum()}")
        
        if 'signal_source' in df.columns:
            signal_dist = df['signal_source'].value_counts()
            logger.info(f"\n   Signal distribution:")
            for signal, count in signal_dist.items():
                pct = (count / len(df)) * 100
                logger.info(f"     - {signal}: {count} ({pct:.1f}%)")
        
        # Save
        df.to_csv(output_file, index=False)
        file_size = os.path.getsize(output_file) / (1024*1024)
        logger.info(f"\n💾 Exported: {output_file}")
        logger.info(f"   File size: {file_size:.2f} MB")
        logger.info(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Print sample statistics
        logger.info(f"\n📊 Dataset Statistics:")
        logger.info(f"Total rows: {len(df)}")
        if 'mint' in df.columns:
            logger.info(f"Unique tokens: {df['mint'].nunique()}")
        if 'label_status' in df.columns:
            logger.info(f"Win/Loss distribution:\n{df['label_status'].value_counts()}")
        
        # Check for zero values in problematic columns
        problem_cols = ['fdv_usd', 'volume_h24_usd', 'creator_balance_pct', 'token_supply']
        for col in problem_cols:
            if col in df.columns:
                zero_count = (df[col] == 0).sum()
                if zero_count > 0:
                    logger.warning(f"⚠️  {col}: {zero_count} zero values ({zero_count/len(df)*100:.1f}%)")
        
        return df


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='data/token_datasets.csv')
    parser.add_argument('--start-date', type=str, default=None)
    parser.add_argument('--end-date', type=str, default=None)
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    extractor = FeatureExtractor()
    await extractor.create_training_dataset(args.start_date, args.end_date, args.output)

if __name__ == "__main__":
    asyncio.run(main())