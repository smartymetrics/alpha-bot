#!/usr/bin/env python3
"""
winner_monitor.py
Tracks high-frequency winner wallets and monitors their latest token purchases.
- Ranks wallets by appearance frequency in Dune winner cache (past 7 days)
- Checks top 50 wallets every 15 minutes for new token purchases (Moralis API)
- Enforces 6-hour cooldown per wallet
- Analyzes discovered tokens for overlap with winner wallets
- Applies RugCheck security validation for tokens with LOW+ grades
- Tokens that fail RugCheck, fail security requirements, or are in probation do NOT get overlap checks.
- Failed tokens stay in memory for 24h hourly rechecks (NOT uploaded to Supabase).
- ONLY tokens passing ALL RugCheck requirements AND having NON-ZERO overlap get uploaded to Supabase.

*** MODIFIED: ***
- Now fetches and embeds Dexscreener data for all tokens that pass security and overlap checks.
- This provides the market data (price, etc.) for collector.py, removing the need
  for the collector to make a separate, failing live API call.
- *** INTEGRATED ML PREDICTOR V2 ***
- Runs ml_predictor_v2.py on all tokens that pass security/overlap checks.
- Adds 'ml_prediction' (probability, confidence, risk_tier) to the final
  results file for data confirmation, NOT as a filter.
- *** REMOVED SHYFT / ADDED HELIUS RPC ***
- Replaced all Shyft API calls with direct Helius RPC calls (getTokenSupply, etc.)
- Added fallback to public Solana RPC.
"""

import asyncio
import aiohttp
import os
import joblib
import time
import json
import threading
import pandas as pd
import math
import random
from dotenv import load_dotenv
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

# --- Local Imports ---
# Assuming these are correct from your environment
from shared.moralis_client import MoralisClient
from birdeye_client import BirdeyeClient
from supabase_utils import (
    download_dune_cache_file, 
    upload_alpha_overlap_results
)
from token_monitor import (
    SolanaAlphaClient,
    HolderAggregator,
    DuneWinnersCache,
    calculate_overlap_grade,
    evaluate_probation_from_rugcheck
)
from ml_predictor import SolanaTokenPredictor
from smart_money_scorer import SmartMoneyScorer, apply_smart_money_boost

load_dotenv()

# --- Environment Config ---
MORALIS_API_KEYS = [
    k.strip() for k in os.getenv("MORALIS_API_KEY", "").split(",") if k.strip()
]
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY")
WINNER_POLL_INTERVAL_SECONDS = int(os.getenv("WINNER_POLL_INTERVAL_SECONDS", "300"))
WINNER_WALLET_COOLDOWN_SECONDS = int(os.getenv("WINNER_WALLET_COOLDOWN_SECONDS", "21600"))
WINNER_TOP_N_WALLETS = int(os.getenv("WINNER_TOP_N_WALLETS", "70"))
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "monitor-data")

# --- Probation Config ---
PROBATION_TOP_N = int(os.getenv("PROBATION_TOP_N", "3"))
PROBATION_THRESHOLD_PCT = float(os.getenv("PROBATION_THRESHOLD_PCT", "40"))
ML_PREDICTION_THRESHOLD = float(os.getenv("ML_PREDICTION_THRESHOLD", "0.50"))
ML_ACTION_THRESHOLD = float(os.getenv("ML_ACTION_THRESHOLD", "0.70"))

# --- Monitoring Window Config ---
TOKEN_MONITORING_WINDOW_HOURS = int(os.getenv("TOKEN_MONITORING_WINDOW_HOURS", "6"))

# --- Smart Money Config ---
SMART_MONEY_ENABLED = os.getenv("SMART_MONEY_ENABLED", "true").lower() == "true"
SMART_MONEY_TOP_N_LABEL = int(os.getenv("SMART_MONEY_TOP_N_LABEL", "500"))
SMART_MONEY_CLUSTER_WINDOW_SECONDS = int(os.getenv("SMART_MONEY_CLUSTER_WINDOW_SECONDS", "1800"))
BIRDEYE_API_KEYS = [k.strip() for k in os.getenv("BIRDEYE_API_KEYS", "").split(",") if k.strip()]

# --- MINIMUM SECURITY REQUIREMENTS ---
MIN_HOLDER_COUNT = 50
MIN_LIQUIDITY_USD = 10000.0
MIN_LP_LOCKED_PCT = 80.0 
MAX_TRANSFER_FEE_PCT = 5
MAX_CREATOR_BALANCE_PCT = 20.0 

# -----------------------------------------------
# Re-usable Helpers
# -----------------------------------------------

def _sanitize_dict(d: Dict[Any, Any]) -> Dict[str, Any]:
    """Recursively sanitize a dictionary so all keys are strings."""
    clean: Dict[str, Any] = {}
    for k, v in d.items():
        key = "null" if k is None else str(k)
        if isinstance(v, dict):
            v = _sanitize_dict(v)
        elif isinstance(v, list):
            v = [_sanitize_dict(x) if isinstance(x, dict) else x for x in v]
        clean[key] = v
    return clean

def _sanitize_maybe(obj: Any) -> Any:
    """Sanitize objects that may contain dicts/lists."""
    if isinstance(obj, dict):
        return _sanitize_dict(obj)
    if isinstance(obj, list):
        return [_sanitize_maybe(x) for x in obj]
    return obj

def safe_load_overlap(overlap_store_or_obj: Any) -> Dict[str, List[Dict]]:
    """Load and normalize overlap store contents into a mapping: mint -> list[entries]."""
    obj = (
        overlap_store_or_obj.load() 
        if hasattr(overlap_store_or_obj, "load") 
        else overlap_store_or_obj
    )
    
    if not obj:
        return {}
    
    if isinstance(obj, pd.DataFrame):
        try:
            rows = obj.to_dict(orient="records")
        except Exception:
            rows = []
        mapping = {}
        for r in rows:
            mint = r.get("mint") or r.get("token") or r.get("token_mint") or "_unknown"
            mapping.setdefault(mint, []).append(r)
        return mapping

    if isinstance(obj, dict):
        is_good = True
        if not obj:
            return {}
        for v in obj.values():
            if not isinstance(v, list):
                is_good = False
                break
        if is_good:
            return obj

        list_lengths = [len(v) for v in obj.values() if isinstance(v, (list, tuple))]
        if list_lengths:
            try:
                length = max(list_lengths)
                keys = list(obj.keys())
                rows = []
                for i in range(length):
                    row = {}
                    for k in keys:
                        try:
                            row[k] = obj[k][i]
                        except Exception:
                            row[k] = None
                    rows.append(row)
                
                mapping = {}
                for r in rows:
                    mint = r.get("mint") or r.get("token") or r.get("token_mint") or "_unknown"
                    mapping.setdefault(mint, []).append(r)
                return mapping
            except Exception:
                pass

    if isinstance(obj, list):
        mapping = {}
        for item in obj:
            if isinstance(item, dict):
                mint = item.get("mint") or item.get("token") or item.get("token_mint") or "_unknown"
                mapping.setdefault(mint, []).append(item)
        return mapping

    print(f"⚠️ safe_load_overlap: Unknown data type {type(obj)}, returning empty dict.")
    return {}


async def retry_with_backoff(func, *args, retries: int = 5, base_delay: float = 0.5, **kwargs):
    """Retry an async function with exponential backoff and jitter."""
    for attempt in range(1, retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            wait = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.3)
            if hasattr(func, '__name__'):
                print(f"[Retry] {func.__name__} failed (attempt {attempt}/{retries}): {e}. Retrying in {wait:.2f}s")
            else:
                print(f"[Retry] A function failed (attempt {attempt}/{retries}): {e}. Retrying in {wait:.2f}s")
            await asyncio.sleep(wait)
    raise RuntimeError(f"Function failed after {retries} retries")


def prune_old_overlap_entries(data: dict, expiry_hours: int = 24) -> dict:
    """
    Robust pruning of overlap entries older than expiry_hours
    or with timestamps far in the future (handles clock skew).
    """
    if not data:
        return {}
        
    now = datetime.now(timezone.utc)
    cutoff_past = now - timedelta(hours=expiry_hours)
    cutoff_future = now + timedelta(days=1)
    
    pruned = {}

    for mint, entries in (data or {}).items():
        if not isinstance(entries, list):
            entries = [entries]

        new_entries = []
        for entry in entries:
            ts_val = None
            if isinstance(entry, dict):
                ts_val = (
                    entry.get("ts") or 
                    entry.get("timestamp") or 
                    entry.get("checked_at") or
                    entry.get("result", {}).get("checked_at")
                )
            
            if ts_val is None:
                new_entries.append(entry)
                continue

            try:
                if isinstance(ts_val, (int, float)):
                    ts = datetime.fromtimestamp(int(ts_val), tz=timezone.utc)
                elif isinstance(ts_val, datetime):
                    ts = ts_val
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                elif isinstance(ts_val, str):
                    s = ts_val.replace("Z", "+00:00")
                    if '.' in s and '+' in s:
                         s = s.split('.')[0] + s[s.find('+'):]
                    ts = datetime.fromisoformat(s)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                else:
                    new_entries.append(entry)
                    continue
                
                if ts and (ts > cutoff_past and ts < cutoff_future):
                    new_entries.append(entry)
                
            except Exception:
                new_entries.append(entry)

        if new_entries:
            pruned[mint] = new_entries

    return pruned


# -----------------------------------------------
# Rate Limiter Class
# -----------------------------------------------

class AsyncRateLimiter:
    """Enhanced token bucket rate limiter with minimum spacing between requests."""
    def __init__(self, rate: float, per: float = 1.0, min_interval: float = 0.0, debug: bool = False):
        """
        Args:
            rate: Number of requests allowed per time period
            per: Time period in seconds (default 1.0)
            min_interval: Minimum seconds between requests (enforces spacing)
            debug: Enable debug logging
        """
        self.rate = rate
        self.per = per
        self.min_interval = min_interval or (per / rate)
        self.tokens = rate
        self.last_refill = time.monotonic()
        self.last_acquire = 0.0
        self._lock = asyncio.Lock()
        self.debug = debug
        self.name = "RateLimiter"

    async def _refill_tokens(self):
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        
        if elapsed > 0:
            tokens_to_add = (elapsed / self.per) * self.rate
            self.tokens = min(self.rate, self.tokens + tokens_to_add)
            self.last_refill = now

    async def acquire(self):
        """Acquire a token, sleeping if necessary to respect both rate and spacing."""
        async with self._lock:
            now = time.monotonic()
            time_since_last = now - self.last_acquire
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                if self.debug:
                    print(f"[{self.name}] Enforcing min spacing: sleeping {sleep_time:.3f}s")
                await asyncio.sleep(sleep_time)
                now = time.monotonic()
            
            await self._refill_tokens()
            
            while self.tokens < 1:
                time_needed = (1 - self.tokens) * (self.per / self.rate)
                if self.debug:
                    print(f"[{self.name}] Rate limited. Sleeping for {time_needed:.4f}s")
                await asyncio.sleep(time_needed + 0.01)
                await self._refill_tokens()
            
            self.tokens -= 1
            self.last_acquire = time.monotonic()
            
            if self.debug:
                print(f"[{self.name}] Token acquired. Remaining: {self.tokens:.2f}")

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


# -----------------------------------------------
# RugCheck Client with Retry Logic
# -----------------------------------------------

class RugCheckClient:
    """Dedicated RugCheck API client with rate limiting and retry logic."""
    BASE_URL = "https://api.rugcheck.xyz/v1"
    
    def __init__(
        self, 
        session: aiohttp.ClientSession,
        rate_limiter: AsyncRateLimiter,
        max_retries: int = 3,
        debug: bool = False
    ):
        self.session = session
        self.rate_limiter = rate_limiter
        self.max_retries = max_retries
        self.debug = debug
        
    async def get_token_report(self, mint: str) -> Dict[str, Any]:
        """Fetch token report with automatic retries for transient errors."""
        url = f"{self.BASE_URL}/tokens/{mint}/report"
        
        for attempt in range(self.max_retries):
            try:
                async with self.rate_limiter:
                    if self.debug and attempt > 0:
                        print(f"[RugCheck] Retry {attempt + 1}/{self.max_retries} for {mint}")
                    
                    async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return {"ok": True, "data": data}
                        
                        elif resp.status == 429:
                            retry_after = int(resp.headers.get('Retry-After', 60))
                            if self.debug:
                                print(f"[RugCheck] Rate limited for {mint}. Retry after {retry_after}s")
                            
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(retry_after)
                                continue
                            
                            return {
                                "ok": False,
                                "error": "rate_limited",
                                "error_text": f"Rate limited, retry after {retry_after}s",
                                "status_code": 429
                            }
                        
                        elif resp.status in (502, 503):
                            text = await resp.text()
                            if self.debug:
                                print(f"[RugCheck] {resp.status} error for {mint}: {text[:200]}")
                            
                            if attempt < self.max_retries - 1:
                                backoff = min(2 ** attempt, 30)
                                if self.debug:
                                    print(f"[RugCheck] Backing off {backoff}s before retry...")
                                await asyncio.sleep(backoff)
                                continue
                            
                            return {
                                "ok": False,
                                "error": f"server_error_{resp.status}",
                                "error_text": text[:500],
                                "status_code": resp.status
                            }
                        
                        elif resp.status == 404:
                            return {
                                "ok": False,
                                "error": "token_not_found",
                                "error_text": "Token not found in RugCheck database",
                                "status_code": 404
                            }
                        
                        else:
                            text = await resp.text()
                            if self.debug:
                                print(f"[RugCheck] Unexpected status {resp.status} for {mint}: {text[:200]}")
                            
                            return {
                                "ok": False,
                                "error": f"http_error_{resp.status}",
                                "error_text": text[:500],
                                "status_code": resp.status
                            }
                            
            except asyncio.TimeoutError:
                if self.debug:
                    print(f"[RugCheck] Timeout for {mint} (attempt {attempt + 1})")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                
                return {
                    "ok": False,
                    "error": "timeout",
                    "error_text": "Request timed out after 30s"
                }
            
            except Exception as e:
                if self.debug:
                    print(f"[RugCheck] Exception for {mint}: {e}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                
                return {
                    "ok": False,
                    "error": "exception",
                    "error_text": str(e)
                }
        
        return {
            "ok": False,
            "error": "max_retries_exceeded",
            "error_text": f"Failed after {self.max_retries} attempts"
        }


# -----------------------------------------------
# Component 1: WinnerWalletRanker
# -----------------------------------------------

class WinnerWalletRanker:
    """Download Dune cache files from Supabase and rank wallets by appearance frequency."""
    def __init__(
        self, 
        dune_cache_dir: str = "./data/dune_cache",
        supabase_bucket: str = "monitor-data",
        debug: bool = False
    ):
        self.dune_cache_dir = dune_cache_dir
        self.supabase_bucket = supabase_bucket
        self.debug = debug
        os.makedirs(self.dune_cache_dir, exist_ok=True)
        self._ranked_wallets: Dict[str, int] = {}
        if self.debug:
            print("[WalletRanker] Initialized.")

    async def download_dune_caches_for_last_7_days(self) -> List[str]:
        """Download last 7 days of cache files from Supabase."""
        if self.debug:
            print("[WalletRanker] Downloading last 7 days of Dune cache...")
            
        days = [
            (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y%m%d") 
            for i in range(7)
        ]
        
        paths = []
        for day in days:
            fname = f"dune_cache_{day}.pkl"
            local_path = os.path.join(self.dune_cache_dir, fname)
            
            if os.path.exists(local_path):
                if self.debug:
                    print(f"[WalletRanker] Found local cache: {fname}")
                paths.append(local_path)
                continue
            
            try:
                if self.debug:
                    print(f"[WalletRanker] Downloading {fname} from Supabase...")
                success = download_dune_cache_file(
                    local_path, fname, self.supabase_bucket
                )
                if success:
                    paths.append(local_path)
                else:
                    if self.debug:
                        print(f"[WalletRanker] ⚠️ {fname} not found in Supabase.")
            except Exception as e:
                if self.debug:
                    print(f"[WalletRanker] ❌ Failed to download {fname}: {e}")
            
            await asyncio.sleep(0.1)
            
        if self.debug:
            print(f"[WalletRanker] Total cache files loaded: {len(paths)}")
        return paths

    def extract_and_rank_wallets(self, cache_files: List[str]) -> Dict[str, int]:
        """Parse cache files and rank wallets by frequency across unique tokens."""
        wallet_freq = defaultdict(int)
        processed_tokens: Set[str] = set()
        
        for fpath in cache_files:
            try:
                obj = joblib.load(fpath)
                token_to_holders = obj.get("token_to_top_holders", {})
                
                for token, holders in token_to_holders.items():
                    if token in processed_tokens:
                        continue
                    processed_tokens.add(token)
                    
                    unique_holders_for_token = set(
                        h for h in holders if isinstance(h, str)
                    )
                    
                    for w in unique_holders_for_token:
                        wallet_freq[w] += 1
                        
            except Exception as e:
                if self.debug:
                    print(f"[WalletRanker] ❌ Failed to load {fpath}: {e}")

        sorted_wallets = sorted(
            wallet_freq.items(), key=lambda item: item[1], reverse=True
        )
        
        self._ranked_wallets = dict(sorted_wallets)
        
        if self.debug:
            print(f"[WalletRanker] Ranked {len(self._ranked_wallets)} unique wallets from {len(processed_tokens)} tokens.")
            if self._ranked_wallets:
                top_wallet = list(self._ranked_wallets.items())[0]
                print(f"[WalletRanker] Top wallet: {top_wallet[0]} (Freq: {top_wallet[1]})")

        return self._ranked_wallets

    def get_top_n_wallets(self, n: int = 50) -> List[Tuple[str, int]]:
        """Return top N wallets as list of (wallet_address, frequency) tuples."""
        return list(self._ranked_wallets.items())[:n]


# -----------------------------------------------
# Component 2: WalletCheckScheduler
# -----------------------------------------------

class WalletCheckScheduler:
    """Manage wallet check timing and 6-hour cooldowns with persistent state."""
    def __init__(
        self,
        state_file: str = "./data/winner_wallet_state.pkl",
        cooldown_seconds: int = 21600,
        max_wallet_history: int = 5000,
        debug: bool = False
    ):
        self.state_file = state_file
        self.cooldown_seconds = cooldown_seconds
        self.max_wallet_history = max_wallet_history
        self.debug = debug
        self._lock = threading.Lock()
        self.state = self.load_state()
        self.total_checks_this_session = 0
        if self.debug:
            print(f"[WalletScheduler] Initialized with {len(self.state)} wallets from state.")
            print(f"[WalletScheduler] Cooldown period: {cooldown_seconds/3600:.1f} hours")

    def load_state(self) -> Dict[str, Any]:
        """Load state from disk using joblib.load()."""
        with self._lock:
            if not os.path.exists(self.state_file):
                return {}
            try:
                return joblib.load(self.state_file)
            except Exception as e:
                if self.debug:
                    print(f"[WalletScheduler] ❌ Failed to load state: {e}")
                return {}

    def save_state(self, state: Dict[str, Any]):
        """Save state to disk with memory management."""
        with self._lock:
            s = state
            if len(s) > self.max_wallet_history:
                if self.debug:
                    print(f"[WalletScheduler] Pruning state from {len(s)} to {self.max_wallet_history} wallets...")
                sorted_items = sorted(
                    s.items(), 
                    key=lambda item: item[1].get('last_checked_ts', 0), 
                    reverse=True
                )
                s = dict(sorted_items[:self.max_wallet_history])
            
            try:
                joblib.dump(_sanitize_maybe(s), self.state_file)
                self.state = s
            except Exception as e:
                if self.debug:
                    print(f"[WalletScheduler] ❌ Failed to save state: {e}")

    def can_check_wallet(self, wallet: str) -> bool:
        """Return True if wallet hasn't been checked in last 6 hours."""
        wallet_data = self.state.get(wallet)
        if not wallet_data:
            return True
            
        last_checked_ts = wallet_data.get('last_checked_ts', 0)
        now = int(datetime.now(timezone.utc).timestamp())
        
        can_check = (now - last_checked_ts) >= self.cooldown_seconds
        
        if not can_check and self.debug:
            remaining = (last_checked_ts + self.cooldown_seconds) - now
            if remaining > 0:
                print(f"[WalletScheduler] ⏱️ Wallet {wallet[:6]}... on cooldown ({remaining/60:.0f}m left)")
            
        return can_check

    def mark_wallet_checked(
        self, wallet: str, tokens_found: int = 0, key_used: str = None
    ):
        """Record wallet check with current timestamp."""
        now_ts = int(datetime.now(timezone.utc).timestamp())
        
        current_data = self.state.get(wallet, {})
        
        current_data.update({
            "last_checked_ts": now_ts,
            "total_checks": current_data.get("total_checks", 0) + 1,
            "last_tokens_found": tokens_found,
            "last_key_used": key_used
        })
        
        self.state[wallet] = current_data
        self.total_checks_this_session += 1
        
        threading.Thread(target=self.save_state, args=(self.state.copy(),)).start()

    def get_next_batch(
        self, 
        ranked_wallets: List[Tuple[str, int]], 
        limit: int = 50
    ) -> List[str]:
        """
        Filter ranked wallets through cooldown check.
        Returns up to `limit` wallets that are eligible (not on cooldown).
        """
        batch: List[str] = []
        scanned = 0
        on_cooldown = 0
        
        for wallet, freq in ranked_wallets:
            scanned += 1
            if len(batch) >= limit:
                break
            
            if self.can_check_wallet(wallet):
                batch.append(wallet)
            else:
                on_cooldown += 1
                
        if self.debug:
            print(f"[WalletScheduler] Scanned {scanned} wallets: {len(batch)} eligible, {on_cooldown} on cooldown")
            if batch:
                total_wallet_pool = len(ranked_wallets)
                coverage_pct = (scanned / total_wallet_pool * 100) if total_wallet_pool > 0 else 0
                print(f"[WalletScheduler] Coverage: scanned {scanned}/{total_wallet_pool} ({coverage_pct:.1f}%) of wallet pool")
            
        return batch


# -----------------------------------------------
# Component 3: AlphaOverlapStore
# -----------------------------------------------

class AlphaOverlapStore:
    """Manage storage of alpha token results with 24-hour expiry and Supabase sync."""
    def __init__(
        self,
        filepath: str = "./data/overlap_results_alpha.pkl",
        supabase_bucket: str = "monitor-data",
        debug: bool = False
    ):
        self.filepath = filepath
        self.supabase_bucket = supabase_bucket
        self.debug = debug
        self._last_upload = 0
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if self.debug:
            print(f"[AlphaOverlapStore] Initialized. Using file: {self.filepath}")

    def load(self) -> Dict[str, Any]:
        """Load existing results from local PKL file."""
        with self._lock:
            return safe_load_overlap(self._load_internal())

    def _load_internal(self) -> Dict[str, Any]:
        """Internal load method for safe_load_overlap compatibility."""
        if os.path.exists(self.filepath):
            try:
                return joblib.load(self.filepath)
            except Exception as e:
                if self.debug:
                    print(f"[AlphaOverlapStore] ❌ Load failed: {e}")
        return {}

    def save(self, obj: Dict[str, Any], expiry_hours: int = 24):
        """Save overlap results with pruning and throttled Supabase upload."""
        now = time.time()
        with self._lock:
            try:
                normalized = safe_load_overlap(obj)
                
                # Prune by time first (handles Requirement 1)
                pruned = prune_old_overlap_entries(
                    normalized, expiry_hours=expiry_hours
                )
                
                # --- Requirement 2: Reinforce Quality Gate ---
                # This filter runs every time the .pkl file is saved, ensuring
                # only entries that meet quality standards are persisted.
                filtered_for_quality = {}
                has_uploadable_data = False
                for mint, entries in pruned.items():
                    valid_entries = []
                    for entry in entries:
                        # 1. Check for explicit "passed" security status
                        is_security_passed = entry.get("security") == "passed"
                        
                        # 2. Check for a grade above "NONE"
                        grade = entry.get("result", {}).get("grade", "NONE")
                        is_graded = grade not in ("NONE", "UNKNOWN")
                        
                        # Only keep entries that pass BOTH checks
                        if is_security_passed and is_graded:
                            valid_entries.append(entry)
                            has_uploadable_data = True # Set flag for upload check
                        elif self.debug:
                            security = entry.get("security", "N/A")
                            ts = entry.get("ts", "N/A")
                            print(f"[AlphaOverlapStore] 🗑️ Discarding old/failed entry from PKL for {mint} (TS: {ts}, Sec: {security}, Grade: {grade})")
                            
                    if valid_entries:
                        filtered_for_quality[mint] = valid_entries
                        
                joblib.dump(_sanitize_maybe(filtered_for_quality), self.filepath)
                # --- END Requirement 2 ---

                # IMPORTANT: Only upload if save is not throttled
                if now - self._last_upload < 120:
                    if self.debug:
                        print("[AlphaOverlapStore] 💾 Save throttled (recent upload). Local save OK.")
                    return

                # Use the flag set during filtering
                if not has_uploadable_data:
                     if self.debug:
                         print("[AlphaOverlapStore] ⚠️ Upload skipped (no new PASSED/GRADED tokens since last upload)")
                     return

                if self.debug:
                    print("[AlphaOverlapStore] 🚀 Uploading to Supabase...")
                success = upload_alpha_overlap_results(
                    self.filepath, self.supabase_bucket, debug=self.debug
                )

                if success:
                    self._last_upload = now
                    if self.debug:
                        print(f"[AlphaOverlapStore] ✅ Saved and uploaded at {time.ctime(now)}")
                else:
                    if self.debug:
                        print("[AlphaOverlapStore] ⚠️ Upload failed or skipped by utility (check utility logs)")

            except Exception as e:
                if self.debug:
                    print(f"[AlphaOverlapStore] ❌ Save failed: {e}")

    def get_all_tracked_tokens(self) -> List[str]:
        """Return list of all token mints currently in store."""
        data = self.load()
        return list(data.keys())


# -----------------------------------------------
# Component 4: AlphaTokenAnalyzer
# -----------------------------------------------

class AlphaTokenAnalyzer:
    """Perform overlap and security analysis on discovered tokens."""
    def __init__(
        self,
        http_session: aiohttp.ClientSession,
        holder_agg: HolderAggregator,
        dune_cache: DuneWinnersCache,
        helius_limiter: AsyncRateLimiter,
        rugcheck_client: RugCheckClient,
        dex_limiter: AsyncRateLimiter,
        ml_classifier: SolanaTokenPredictor,
        smart_money_scorer: Optional["SmartMoneyScorer"] = None,
        debug: bool = False
    ):
        self.http_session = http_session
        self.holder_agg = holder_agg
        self.dune_cache = dune_cache
        self.helius_limiter = helius_limiter
        self.rugcheck_client = rugcheck_client
        self.dex_limiter = dex_limiter
        self.ml_classifier = ml_classifier
        self.smart_money_scorer = smart_money_scorer   # None = SM disabled
        self.debug = debug
        
        if self.debug:
            sm_status = "ENABLED" if smart_money_scorer else "DISABLED"
            print(f"[TokenAnalyzer] Initialized. Smart Money: {sm_status}")

    async def _get_supply_and_decimals_rpc(self, mint: str) -> tuple:
        """
        Fallback method to get supply and decimals from Helius RPC (with public RPC backup).
        Returns (supply, decimals) or (0, 0) if failed.
        """
        # Endpoints
        helius_url = f"https://mainnet.helius-rpc.com/?api-key={os.getenv('HELIUS_API_KEY')}"
        fallback_url = "https://api.mainnet-beta.solana.com"
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTokenSupply",
            "params": [mint]
        }

        # Attempt 1: Helius
        try:
            async with self.helius_limiter:
                async with self.http_session.post(helius_url, json=payload, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        val = data.get("result", {}).get("value", {})
                        if val:
                            supply = int(val.get("amount", "0"))
                            decimals = int(val.get("decimals", 0))
                            if self.debug:
                                print(f"[TokenAnalyzer] ✅ Got supply from Helius: {supply}, decimals: {decimals}")
                            return (supply, decimals)
        except Exception as e:
            if self.debug:
                print(f"[TokenAnalyzer] ⚠️ Helius supply fetch failed for {mint}: {e}")
        
        # Attempt 2: Public RPC (Fallback)
        try:
            # We don't use the Helius limiter for public RPC, but we should be gentle.
            # Assuming low volume of fallbacks.
            async with self.http_session.post(fallback_url, json=payload, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    val = data.get("result", {}).get("value", {})
                    if val:
                        supply = int(val.get("amount", "0"))
                        decimals = int(val.get("decimals", 0))
                        if self.debug:
                            print(f"[TokenAnalyzer] ✅ Got supply from Public RPC: {supply}, decimals: {decimals}")
                        return (supply, decimals)
        except Exception as e:
            if self.debug:
                print(f"[TokenAnalyzer] ⚠️ Public RPC supply fetch failed for {mint}: {e}")

        return (0, 0)

    async def _get_creator_balance_rpc(self, mint: str, decimals: int) -> tuple:
        """
        Fallback method to get creator balance via RPC.
        Logic:
         1. Get Mint Info to find 'mintAuthority' (often the creator/deployer).
         2. Get Token Accounts for that authority to find their balance of this mint.
        Returns (creator_balance_normalized, success_flag).
        """
        helius_url = f"https://mainnet.helius-rpc.com/?api-key={os.getenv('HELIUS_API_KEY')}"
        fallback_url = "https://api.mainnet-beta.solana.com"
        
        # Step 1: Get Mint Authority
        mint_authority = None
        
        info_payload = {
            "jsonrpc": "2.0",
            "id": 1, 
            "method": "getAccountInfo",
            "params": [mint, {"encoding": "jsonParsed"}]
        }
        
        # Try Helius for Mint Info
        try:
            async with self.helius_limiter:
                async with self.http_session.post(helius_url, json=info_payload, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Navigate: result -> value -> data -> parsed -> info -> mintAuthority
                        info = data.get("result", {}).get("value", {}).get("data", {}).get("parsed", {}).get("info", {})
                        mint_authority = info.get("mintAuthority")
        except Exception:
            pass # Fallback logic below
            
        # Fallback to Public RPC if Helius failed
        if not mint_authority:
            try:
                async with self.http_session.post(fallback_url, json=info_payload, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        info = data.get("result", {}).get("value", {}).get("data", {}).get("parsed", {}).get("info", {})
                        mint_authority = info.get("mintAuthority")
            except Exception:
                pass
        
        if not mint_authority:
            if self.debug:
                print(f"[TokenAnalyzer] ⚠️ Could not determine mint authority (creator) for {mint}")
            return (0.0, False)
            
        # Step 2: Get Balance for Mint Authority
        balance_payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "getTokenAccountsByOwner",
            "params": [
                mint_authority,
                {"mint": mint},
                {"encoding": "jsonParsed"}
            ]
        }
        
        raw_balance = 0.0
        success = False
        
        # Try Helius for Balance
        try:
            async with self.helius_limiter:
                async with self.http_session.post(helius_url, json=balance_payload, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        accounts = data.get("result", {}).get("value", [])
                        for acc in accounts:
                            # Add up balance from all accounts (usually just one)
                            amt_info = acc.get("account", {}).get("data", {}).get("parsed", {}).get("info", {}).get("tokenAmount", {})
                            raw_balance += float(amt_info.get("amount", 0))
                        success = True
        except Exception:
             success = False # Fallback below

        # Fallback to Public RPC
        if not success:
            try:
                 async with self.http_session.post(fallback_url, json=balance_payload, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        accounts = data.get("result", {}).get("value", [])
                        for acc in accounts:
                            amt_info = acc.get("account", {}).get("data", {}).get("parsed", {}).get("info", {}).get("tokenAmount", {})
                            raw_balance += float(amt_info.get("amount", 0))
                        success = True
            except Exception:
                pass

        if success:
            # Normalize
            if decimals > 0:
                normalized = raw_balance / (10 ** decimals)
            else:
                normalized = raw_balance
                
            if self.debug:
                print(f"[TokenAnalyzer] ✅ RPC Creator Balance ({mint_authority}): {normalized} (raw: {raw_balance})")
            return (normalized, True)
            
        return (0.0, False)

    async def _run_rugcheck_check(self, mint: str) -> Dict[str, Any]:
        """
        Check token security using RugCheck API with comprehensive risk analysis.
        Uses RugCheckClient for rate limiting and retry logic.
        """
        # Delegate to the RugCheckClient which handles rate limiting and retries
        result = await self.rugcheck_client.get_token_report(mint)
        
        # Handle None or invalid responses
        if result is None:
            if self.debug:
                print(f"[TokenAnalyzer] ⚠️ RugCheck returned None for {mint} (rate limit or service issue)")
            return {"ok": False, "error": "rugcheck_none_response", "error_text": "RugCheck API returned None"}
        
        if not result.get("ok"):
            if self.debug:
                print(f"[TokenAnalyzer] ⚠️ RugCheck failed for {mint}: {result.get('error')}")
            return result
        
        data = result.get("data")
        
        # Handle None data
        if data is None:
            if self.debug:
                print(f"[TokenAnalyzer] ⚠️ RugCheck data is None for {mint}")
            return {"ok": False, "error": "rugcheck_null_data", "error_text": "RugCheck returned null data"}
        
        # Extract key security metrics
        probation_result = evaluate_probation_from_rugcheck(data)

        # --- FIX: Handle data.get() returning None if key exists but value is null ---
        top_holders = data.get("topHolders") or []
        rugged = data.get("rugged", False)
        
        # Authorities and Creator Balance
        freeze_authority = data.get("freezeAuthority")
        mint_authority = data.get("mintAuthority")
        has_authorities = bool(freeze_authority or mint_authority)

        # --- FIXED: Handle inconsistent creatorBalance type with None default ---
        creator_balance_raw = data.get("creatorBalance")
        creator_balance_pct = None  # Changed from 0.0 to None to distinguish "missing" from "zero"
        
        if isinstance(creator_balance_raw, dict):
            try:
                creator_balance_pct = float(creator_balance_raw.get('pct', 0.0) or 0.0)
            except (ValueError, TypeError):
                creator_balance_pct = 0.0
        elif creator_balance_raw is not None:
            try:
                # If it's a raw number, normalize by decimals and supply
                creator_balance_raw_num = float(creator_balance_raw)
                
                # Extract token metadata for normalization
                token_data = data.get("token", {})
                decimals = int(token_data.get("decimals", 0)) if token_data.get("decimals") is not None else 0
                supply = float(token_data.get("supply", 0)) if token_data.get("supply") is not None else 0
                
                if decimals > 0 and supply > 0:
                    # Normalize: raw_amount / (10^decimals) / supply * 100
                    creator_balance_normalized = creator_balance_raw_num / (10 ** decimals)
                    creator_balance_pct = (creator_balance_normalized / supply) * 100
                else:
                    # Fallback: just use raw number as-is (already percentage)
                    creator_balance_pct = creator_balance_raw_num
            except (ValueError, TypeError, ZeroDivisionError):
                creator_balance_pct = 0.0
        
        # NOTE: If creator_balance_raw was None, creator_balance_pct remains None
        creator_balance = creator_balance_pct

        transfer_fee_pct = data.get("transferFee", {}).get("pct", 0)
        
        # Holder metrics
        total_holders = data.get("totalHolders", 0)
        top1_holder_pct = top_holders[0].get("pct", 0) if top_holders else 0.0
        top_10_holders_pct = sum(h.get("pct", 0) for h in top_holders[:10])

        # --- Liquidity Aggregation ---
        # --- FIX: Handle data.get() returning None if key exists but value is null ---
        markets = data.get("markets") or []
        total_lp_locked_usd = 0.0
        total_lp_usd = 0.0
        lp_lock_details = []

        for market in markets:
            # --- FIX: Add check to ensure market item is not None ---
            if not market:
                continue
                
            lp_data = market.get("lp", {})
            if lp_data:
                lp_locked_usd = 0.0
                lp_unlocked_usd = 0.0
                
                try:
                    lp_locked_usd = float(lp_data.get("lpLockedUSD", 0.0) or 0.0)
                except (ValueError, TypeError):
                    lp_locked_usd = 0.0
                
                try:
                    lp_unlocked_usd = float(lp_data.get("lpUnlocked", 0.0) or 0.0)
                except (ValueError, TypeError):
                    lp_unlocked_usd = 0.0
                
                lp_total_usd_market = lp_locked_usd + lp_unlocked_usd
                
                total_lp_locked_usd += lp_locked_usd
                total_lp_usd += lp_total_usd_market
                
                lp_lock_details.append({
                    "market_type": market.get("marketType"),
                    "locked_usd": lp_locked_usd,
                    "total_usd": lp_total_usd_market,
                    "locked_pct": lp_data.get("lpLockedPct", 0)
                })
        
        overall_lp_locked_pct = (total_lp_locked_usd / total_lp_usd * 100) if total_lp_usd > 0 else 0.0
        lp_lock_sufficient = overall_lp_locked_pct >= 95.0

        return {
            "ok": True,
            "rugged": rugged,
            "total_holders": total_holders,
            "top1_holder_pct": top1_holder_pct,
            "top_10_holders_pct": top_10_holders_pct,
            "creator_balance": creator_balance,
            "freeze_authority": freeze_authority,
            "mint_authority": mint_authority,
            "has_authorities": has_authorities,
            "transfer_fee_pct": transfer_fee_pct,
            "total_lp_usd": total_lp_usd,
            "total_lp_locked_usd": total_lp_locked_usd,
            "overall_lp_locked_pct": overall_lp_locked_pct,
            "lp_lock_sufficient": lp_lock_sufficient,
            "probation": probation_result["probation"],
            "probation_meta": probation_result,
            "lp_lock_details": lp_lock_details,
            "raw": data
        }

    async def _run_dexscreener_check(self, mint: str) -> Dict[str, Any]:
        """
        Fetches token data from DexScreener.
        Designed to be used with `retry_with_backoff`, so it raises exceptions on failure
        (HTTP errors, no pairs found, invalid price) to trigger retries.
        """
        url = f"https://api.dexscreener.com/latest/dex/tokens/{mint}"
        session = self.http_session
        async with self.dex_limiter: # Use the new Dexscreener limiter
            # Let retry_with_backoff handle exceptions from this block
            async with session.get(url, timeout=30) as resp:
                if resp.status == 429:
                    # Specific error for rate limiting to make logs clearer
                    raise aiohttp.ClientResponseError(
                        resp.request_info, resp.history, status=resp.status,
                        message=f"Rate limited by DexScreener for {mint}"
                    )
                # Raise an exception for any other 4xx/5xx status
                resp.raise_for_status()
                data = await resp.json()

        pairs = data.get("pairs") or []
        if not pairs:
            # A successful response but no data is a retryable condition
            raise ValueError(f"DexScreener: No pairs found for mint {mint}")

        # Use the first pair as the canonical source
        p0 = pairs[0]
        
        # Validate priceUsd: must exist and be a positive float
        price_str = None # Initialize
        try:
            price_str = p0.get("priceUsd")
            if price_str is None:
                raise ValueError("priceUsd is null")
            price_usd = float(price_str)
            if price_usd <= 0:
                raise ValueError(f"price is zero or negative: {price_usd}")
        except (ValueError, TypeError, KeyError) as e:
            # Re-raise a more informative error to trigger retry
            raise ValueError(f"DexScreener: Could not parse valid price for {mint}. Raw value: '{price_str}'. Details: {e}") from e

        # Extract liquidity, converting to float safely
        try:
            liquidity = float(p0.get("liquidity", {}).get("usd", 0.0) or 0.0)
        except (ValueError, TypeError):
            liquidity = 0.0

        return {
            "ok": True,
            "pair_exists": True,
            "price_usd": price_usd,
            "raw": p0
        }

    async def analyze_token(
        self,
        mint: str,
        wallet_buy_timestamps: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive token analysis.
        FIXED: Handle zero supply edge case in security requirements check
        FIXED: Only fallback to RPC when creator_balance is None (missing), not 0 (valid zero)
        FIXED: Properly calculate creator balance percentage from raw amounts

        wallet_buy_timestamps: {wallet_address: ISO_timestamp} — when the discovering
            wallet first bought this token. Used for early buyer insider detection.
        """
        if self.debug:
            print(f"[TokenAnalyzer] 🔬 Analyzing token: {mint}")

        # 1. RUN RUGCHECK FIRST
        security_report = {
            "rugcheck_passed": None,
            "probation": False,
            "probation_reason": None,
            "rugcheck": {},
            "rugcheck_raw": None
        }
        
        try:
            rugcheck = await self._run_rugcheck_check(mint)
            security_report["rugcheck_raw"] = rugcheck
            
            if rugcheck and rugcheck.get("ok"):
                security_report["rugcheck_passed"] = True
                security_report["probation"] = rugcheck.get("probation", False)
                security_report["probation_reason"] = (
                    rugcheck.get("probation_meta", {}).get("explanation")
                    if security_report["probation"] else None
                )
                security_report["rugcheck"] = {
                    "rugged": rugcheck.get("rugged"),
                    "top1_holder_pct": rugcheck.get("top1_holder_pct"),
                    "top_10_holders_pct": rugcheck.get("top_10_holders_pct"),
                    "holder_count": rugcheck.get("total_holders"),
                    "has_authorities": rugcheck.get("has_authorities"),
                    "creator_balance": rugcheck.get("creator_balance"),
                    "freeze_authority": rugcheck.get("freeze_authority"),
                    "mint_authority": rugcheck.get("mint_authority"),
                    "transfer_fee_pct": rugcheck.get("transfer_fee_pct"),
                    "lp_locked_pct": rugcheck.get("overall_lp_locked_pct"),
                    "total_liquidity_usd": rugcheck.get("total_lp_usd")
                }
            else:
                security_report["rugcheck_passed"] = False
                if self.debug:
                    print(f"[TokenAnalyzer] ❌ RugCheck API call failed for {mint}: {rugcheck.get('error')}")
                    
        except Exception as e:
            security_report["rugcheck_passed"] = False
            if self.debug:
                print(f"[TokenAnalyzer] ❌ RugCheck exception for {mint}: {e}")

        # --- PRE-OVERLAP SECURITY GATE ---
        
        # 2. CHECK RUGCHECK API SUCCESS
        if security_report["rugcheck_passed"] is False:
            if self.debug:
                print(f"[TokenAnalyzer] ⚠️ RugCheck API failed for {mint} - SKIPPING overlap check")
            return {
                "mint": mint,
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "overlap_count": 0,
                "overlap_percentage": 0.0,
                "concentration": 0.0,
                "weighted_concentration": 0.0,
                "total_winner_wallets": 0,
                "grade": "NONE",
                "overlap_wallets_sample": [],
                "security": security_report,
                "needs_monitoring": True,
                "skip_reason": "rugcheck_api_failed"
            }
        
        # 3. CHECK RUGGED STATUS (only if rugcheck passed)
        if security_report["rugcheck_passed"] and security_report["rugcheck"].get("rugged"):
            if self.debug:
                print(f"[TokenAnalyzer] ⚠️ Token {mint} marked as RUGGED - SKIPPING overlap check")
            return {
                "mint": mint,
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "overlap_count": 0,
                "overlap_percentage": 0.0,
                "concentration": 0.0,
                "weighted_concentration": 0.0,
                "total_winner_wallets": 0,
                "grade": "NONE",
                "overlap_wallets_sample": [],
                "security": security_report,
                "needs_monitoring": True,
                "skip_reason": "rugged"
            }
        
        # 4. CHECK PROBATION STATUS (only if rugcheck passed)
        if security_report["rugcheck_passed"] and security_report["probation"]:
            if self.debug:
                print(f"[TokenAnalyzer] ⚠️ Token {mint} in PROBATION - SKIPPING overlap check")
                print(f"[TokenAnalyzer] Reason: {security_report['probation_reason']}")
            return {
                "mint": mint,
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "overlap_count": 0,
                "overlap_percentage": 0.0,
                "concentration": 0.0,
                "weighted_concentration": 0.0,
                "total_winner_wallets": 0,
                "grade": "NONE",
                "overlap_wallets_sample": [],
                "security": security_report,
                "needs_monitoring": True,
                "skip_reason": "probation"
            }
        
        # 5. CHECK MINIMUM SECURITY REQUIREMENTS
        # *** FIXED: Handle zero supply edge case and null rugcheck_details ***
        rugcheck_details = security_report.get("rugcheck") or {}
        security_failures = []
        
        # Check for active authorities (Mint/Freeze) - only if rugcheck_passed
        if security_report.get("rugcheck_passed") and rugcheck_details.get("has_authorities"):
            security_failures.append("has_active_authorities")
        
        # *** FIX: Get supply and decimals from RugCheck, fallback to RPC if needed ***
        raw_data = security_report.get("rugcheck_raw") or {}
        raw_data = raw_data.get("raw") or {}
        token_data = raw_data.get("token") or {}
        
        supply = token_data.get("supply", 0)
        decimals = token_data.get("decimals", 0)
        data_source = "rugcheck"
        
        # If RugCheck didn't provide valid supply/decimals, try RPC
        if not supply or not decimals:
            if self.debug:
                print(f"[TokenAnalyzer] ⚠️ RugCheck missing supply/decimals for {mint}, trying RPC...")
            supply, decimals = await self._get_supply_and_decimals_rpc(mint)
            data_source = "rpc"
            if supply and decimals and self.debug:
                print(f"[TokenAnalyzer] ✅ Using supply from {data_source}: {supply}, decimals: {decimals}")
        
        # Calculate total supply with zero check
        total_supply = 0
        if supply and decimals:
            try:
                # Use float for supply to handle large numbers
                total_supply = float(supply) / (10 ** int(decimals))
                if self.debug:
                    print(f"[TokenAnalyzer] 📊 Calculated total_supply from {data_source}: {total_supply}")
            except (ZeroDivisionError, ValueError, TypeError, OverflowError):
                total_supply = 0
                if self.debug:
                    print(f"[TokenAnalyzer] ❌ Error calculating total_supply for {mint}")
        
        # --- FIXED: Check for creator balance (handle raw amounts, not percentages) ---
        creator_balance_raw = rugcheck_details.get("creator_balance")  # This is RAW amount or None
        creator_balance_pct = None  # The calculated percentage

        # Only use RPC fallback if RugCheck didn't provide ANY data (None)
        if creator_balance_raw is None:
            if self.debug:
                print(f"[TokenAnalyzer] ⚠️ RugCheck missing creator balance for {mint}, trying RPC...")
            
            creator_balance_normalized, rpc_success = await self._get_creator_balance_rpc(mint, decimals)
            
            if rpc_success and total_supply > 0:
                creator_balance_pct = (creator_balance_normalized / total_supply) * 100
                if self.debug:
                    print(f"[TokenAnalyzer] ✅ Using creator balance from RPC: {creator_balance_normalized:.2f} tokens ({creator_balance_pct:.2f}%)")
            else:
                creator_balance_pct = 0.0  # Default to 0 if RPC also fails
                if self.debug:
                    print(f"[TokenAnalyzer] ⚠️ RPC fallback failed, defaulting to 0%")
        else:
            # RugCheck provided a RAW value (could be 0 or positive integer)
            # We need to calculate the percentage ourselves
            try:
                creator_raw_amount = float(creator_balance_raw)
                
                if decimals > 0 and total_supply > 0:
                    # Normalize: raw_amount / (10^decimals) to get actual token count
                    creator_balance_normalized = creator_raw_amount / (10 ** int(decimals))
                    # Calculate percentage
                    creator_balance_pct = (creator_balance_normalized / total_supply) * 100
                    
                    if self.debug:
                        if creator_balance_pct == 0.0:
                            print(f"[TokenAnalyzer] ✅ RugCheck: creator holds 0 tokens (0.00%)")
                        else:
                            print(f"[TokenAnalyzer] 💰 RugCheck: creator holds {creator_balance_normalized:.2f} tokens ({creator_balance_pct:.2f}%)")
                else:
                    # Cannot calculate percentage without supply/decimals
                    creator_balance_pct = 0.0
                    if self.debug:
                        print(f"[TokenAnalyzer] ⚠️ Cannot calculate creator %: total_supply={total_supply}, decimals={decimals}")
            except (ValueError, TypeError, ZeroDivisionError) as e:
                creator_balance_pct = 0.0
                if self.debug:
                    print(f"[TokenAnalyzer] ⚠️ Error calculating creator balance %: {e}")

        # Validate creator balance percentage
        if creator_balance_pct is None:
            creator_balance_pct = 0.0  # Safety fallback

        if total_supply > 0 and creator_balance_pct > MAX_CREATOR_BALANCE_PCT:
            security_failures.append(f"creator_balance_pct:{creator_balance_pct:.2f}_req_{MAX_CREATOR_BALANCE_PCT}%")
        elif total_supply == 0 and creator_balance_raw is not None and float(creator_balance_raw) > 0:
            # If total_supply is 0 but creator has balance, this is suspicious
            security_failures.append(f"invalid_supply:creator_has_balance_but_supply_is_zero_source_{data_source}")
            if self.debug:
                print(f"[TokenAnalyzer] ⚠️ Suspicious: creator has raw balance {creator_balance_raw} but total_supply=0")
        
        # Check transfer fee (must be low)
        transfer_fee_val = 0.0
        try:
            transfer_fee_val = float(rugcheck_details.get("transfer_fee_pct", 0.0) or 0.0)
        except (ValueError, TypeError):
            transfer_fee_val = 0.0
            
        if transfer_fee_val > MAX_TRANSFER_FEE_PCT:
            security_failures.append(f"transfer_fee:{transfer_fee_val}%_req_<{MAX_TRANSFER_FEE_PCT}%")
        
        # Check holder count
        holder_count_val = 0
        try:
            holder_count_val = int(rugcheck_details.get("holder_count", 0) or 0)
        except (ValueError, TypeError):
            holder_count_val = 0
            
        if holder_count_val < MIN_HOLDER_COUNT:
            security_failures.append(f"holder_count:{holder_count_val}_req_{MIN_HOLDER_COUNT}")
            
        # Check LP lock percentage
        lp_locked_val = 0.0
        try:
            lp_locked_val = float(rugcheck_details.get("lp_locked_pct", 0.0) or 0.0)
        except (ValueError, TypeError):
            lp_locked_val = 0.0
            
        if lp_locked_val < MIN_LP_LOCKED_PCT:
            security_failures.append(f"lp_locked:{lp_locked_val}%_req_{MIN_LP_LOCKED_PCT}%")
            
        # Check total liquidity USD
        total_liq_val = 0.0
        try:
            total_liq_val = float(rugcheck_details.get("total_liquidity_usd", 0.0) or 0.0)
        except (ValueError, TypeError):
            total_liq_val = 0.0
            
        if total_liq_val < MIN_LIQUIDITY_USD:
            security_failures.append(f"liquidity_usd:{total_liq_val:.2f}_req_{MIN_LIQUIDITY_USD:.0f}")
        
        # If any security requirement fails, SKIP overlap check
        if security_failures:
            if self.debug:
                print(f"[TokenAnalyzer] ⚠️ Token {mint} failed security requirements - SKIPPING overlap check")
                print(f"[TokenAnalyzer] Failures: {', '.join(security_failures)}")
            return {
                "mint": mint,
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "overlap_count": 0,
                "overlap_percentage": 0.0,
                "concentration": 0.0,
                "weighted_concentration": 0.0,
                "total_winner_wallets": 0,
                "grade": "NONE",
                "overlap_wallets_sample": [],
                "security": security_report,
                "needs_monitoring": True,
                "skip_reason": "security_requirements_failed",
                "security_failures": security_failures
            }
            
        # --- END PRE-OVERLAP SECURITY GATE ---
        
        # 6. ONLY IF ALL RUGCHECK REQUIREMENTS PASS, PROCEED WITH OVERLAP CHECK
        if self.debug:
            print(f"[TokenAnalyzer] ✅ RugCheck passed ALL requirements for {mint} - proceeding with overlap check")
        
        # Fetch token holders (expensive step)
        try:
            async with self.helius_limiter:
                holders_list = await self.holder_agg.get_token_holders(
                    mint, limit=1000, max_pages=2, decimals=None
                )
        except Exception as e:
            if self.debug:
                print(f"[TokenAnalyzer] ❌ Holder fetch failed for {mint}: {e}")
            return {
                "mint": mint,
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "overlap_count": 0,
                "overlap_percentage": 0.0,
                "concentration": 0.0,
                "weighted_concentration": 0.0,
                "total_winner_wallets": 0,
                "grade": "NONE",
                "overlap_wallets_sample": [],
                "security": security_report,
                "error": f"Holder fetch failed: {e}",
                "needs_monitoring": True,
                "skip_reason": "holder_fetch_failed"
            }

        # HYBRID SAMPLING RULE
        n = len(holders_list)
        if n <= 200:
            top_holders = [h.get("wallet") for h in holders_list if h.get("wallet")]
        else:
            top10_pct = max(1, int(n * 0.10))
            top_holders = [
                h.get("wallet") 
                for h in holders_list[:min(500, top10_pct)] 
                if h.get("wallet")
            ]
        top_set = set(top_holders)
        top_count = len(top_set)
        
        if top_count == 0:
            if self.debug:
                print(f"[TokenAnalyzer] ⚠️ No holders found for {mint}")
            return {
                "mint": mint,
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "overlap_count": 0,
                "overlap_percentage": 0.0,
                "concentration": 0.0,
                "weighted_concentration": 0.0,
                "total_winner_wallets": 0,
                "grade": "NONE",
                "overlap_wallets_sample": [],
                "security": security_report,
                "error": "No holders found",
                "needs_monitoring": True,
                "skip_reason": "zero_holders"
            }

        try:
            token_to_top_holders, wallet_freq, _ = self.dune_cache.load_last_7_days()
            winners_union: Set[str] = set()
            for holders in token_to_top_holders.values():
                winners_union.update(holders)
            total_winner_wallets = len(winners_union)
            total_winner_weights = sum(wallet_freq.values()) if wallet_freq else 0
        except Exception as e:
            if self.debug:
                print(f"[TokenAnalyzer] ❌ Failed to load Dune winner cache: {e}")
            return {
                "mint": mint,
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "overlap_count": 0,
                "overlap_percentage": 0.0,
                "concentration": 0.0,
                "weighted_concentration": 0.0,
                "total_winner_wallets": 0,
                "grade": "NONE",
                "overlap_wallets_sample": [],
                "security": security_report,
                "error": f"Dune cache load failed: {e}",
                "needs_monitoring": True,
                "skip_reason": "dune_cache_failed"
            }

        # 7. OVERLAP CALCULATION
        overlap = top_set.intersection(winners_union)
        overlap_count = len(overlap)
        
        concentration = (
            (overlap_count / total_winner_wallets * 100.0) 
            if total_winner_wallets else 0.0
        )
        
        overlap_pct = (overlap_count / top_count * 100.0) if top_count > 0 else 0.0
        
        overlap_weight = sum(wallet_freq.get(w, 0) for w in overlap)
        weighted_concentration = (
            (overlap_weight / total_winner_weights * 100.0) 
            if total_winner_weights else 0.0
        )

        # 8. GRADE CALCULATION
        grade = calculate_overlap_grade(
            overlap_count=overlap_count,
            overlap_percentage=overlap_pct, 
            concentration=concentration,
            weighted_concentration=weighted_concentration,
            total_new_holders=top_count,
            total_winner_wallets=total_winner_wallets
        )
        
        # 9. FINAL CHECK: ONLY UPLOAD IF OVERLAP IS FOUND
        final_needs_monitoring = False
        final_skip_reason = None
        
        if overlap_count == 0:
            # If it passed all security but has no overlap, monitor it in case winners buy later.
            final_needs_monitoring = True
            final_skip_reason = "zero_overlap_after_security_pass"
            if self.debug:
                print(f"[TokenAnalyzer] ⚠️ {mint} passed security but ZERO overlap - setting to monitoring.")
        elif grade in ("NONE", "UNKNOWN"):
            # If overlap exists but doesn't meet grade thresholds, also monitor
            final_needs_monitoring = True
            final_skip_reason = "overlap_below_grade_threshold"
            if self.debug:
                print(f"[TokenAnalyzer] ⚠️ {mint} has overlap ({overlap_count}) but grade is {grade} - setting to monitoring.")
        
        # Only fetch DexScreener data for tokens with valid grades (LOW+)
        dex_data = None
        if not final_needs_monitoring and grade not in ("NONE", "UNKNOWN"):
            if self.debug:
                print(f"[TokenAnalyzer] 💎 {mint} PASSED (Grade: {grade}), fetching DexScreener data...")
            try:
                dex_data_result = await retry_with_backoff(
                    self._run_dexscreener_check, mint, retries=8, base_delay=1.5
                )
                dex_data = dex_data_result
            except Exception as e:
                if self.debug:
                    print(f"[TokenAnalyzer] ⚠️ {mint} Dexscreener fetch failed after retries: {e}")
                dex_data = {"ok": False, "error": str(e)}
        elif self.debug and not final_needs_monitoring:
            print(f"[TokenAnalyzer] ⚠️ {mint} skipping DexScreener (Grade: {grade})")
        
        result = {
            "mint": mint,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "overlap_count": overlap_count,
            "overlap_percentage": round(overlap_pct, 2),
            "concentration": round(concentration, 2),
            "weighted_concentration": round(weighted_concentration, 2),
            "total_winner_wallets": total_winner_wallets,
            "grade": grade,
            "overlap_wallets_sample": list(overlap)[:20],
            "security": security_report,
            "needs_monitoring": final_needs_monitoring,
            "skip_reason": final_skip_reason,
            "dexscreener": dex_data
        }
        
        # --- ML PREDICTION ---
        ml_prediction_result = None
        ml_passed = False

        is_cleared_for_upload = (
            not final_needs_monitoring and 
            grade not in ("NONE", "UNKNOWN")
        )
        
        if is_cleared_for_upload:
            try:
                if self.debug:
                    print(f"[TokenAnalyzer] 🔮 Calling ML predictor for mint: {mint}")

                ml_prediction_result = self.ml_classifier.predict(
                    mint, 
                    threshold=ML_PREDICTION_THRESHOLD,
                    action_threshold=ML_ACTION_THRESHOLD,
                    signal_type='alpha'
                )
                
                if not ml_prediction_result or ml_prediction_result.get("error"):
                    raise ValueError(f"ML prediction returned an error: {ml_prediction_result.get('error', 'Unknown')}")
                
            except Exception as e:
                if self.debug:
                    print(f"[TokenAnalyzer] ⚠️ ML prediction failed for {mint}: {e}")
                ml_prediction_result = {
                    'action': 'ERROR',
                    'win_probability': 0.0,
                    'confidence': 'NONE',
                    'risk_tier': 'UNKNOWN',
                    'error': str(e)
                }
            
            # Calculate ML_PASSED based on probability vs threshold
            ml_probability = ml_prediction_result.get("win_probability")
            if ml_probability is not None:
                try:
                    ml_passed = float(ml_probability) >= ML_PREDICTION_THRESHOLD
                except (ValueError, TypeError):
                    ml_passed = False
            
            result["ml_prediction"] = {
                "probability": ml_prediction_result.get("win_probability"),
                "confidence": ml_prediction_result.get("confidence"),
                "risk_tier": ml_prediction_result.get("risk_tier"),
                "action": ml_prediction_result.get("action"),
                "error": ml_prediction_result.get("error"),
                "key_metrics": ml_prediction_result.get("key_metrics"),
                "warnings": ml_prediction_result.get("warnings")
            }
        
        # Set ML_PASSED in result for all tokens (cleared or not)
        result["ML_PASSED"] = ml_passed
        
        if self.debug and not final_needs_monitoring:
            ml_prob_str = "N/A (Skipped)"
            if "ml_prediction" in result:
                prob_val = result.get('ml_prediction', {}).get('probability')
                if prob_val is not None:
                    try:
                        ml_prob_str = f"{float(prob_val):.1%}"
                    except (ValueError, TypeError):
                        ml_prob_str = "N/A (Invalid)"
                else:
                    ml_prob_str = "N/A (None)"
            
            cleared_status = "CLEARED for upload" if is_cleared_for_upload else "NOT CLEARED (No Grade)"

            print(f"[TokenAnalyzer] ✅ {mint} -> Grade: {grade}, Overlap: {overlap_count}, ML Prob: {ml_prob_str}, Status: {cleared_status}")

        # --- SMART MONEY SCORING (Post-Prediction Overlay) ---
        # Only runs for tokens that are cleared for upload (passed all gates).
        # The scorer enriches the result dict but never blocks the upload path.
        if is_cleared_for_upload and self.smart_money_scorer is not None and overlap_count > 0:
            try:
                if self.debug:
                    print(f"[TokenAnalyzer] 🧠 Running Smart Money scorer for {mint}...")

                # Pass the effective overlap wallets and their Dune frequencies
                sm_score = await self.smart_money_scorer.score_token(
                    mint=mint,
                    overlap_wallets=list(overlap),
                    wallet_freq=wallet_freq,
                    ml_passed=ml_passed,
                    pair_created_at_ms=int(
                        (result.get("dexscreener") or {})
                        .get("raw", {})
                        .get("pairCreatedAt") or 0
                    ) or None,
                    wallet_buy_timestamps=wallet_buy_timestamps,
                    rugcheck_raw=(
                        (result.get("security") or {})
                        .get("rugcheck_raw", {})
                        .get("raw")
                    ),
                )

                # Apply post-prediction grade boost
                final_grade, alert_label, is_super_alpha = apply_smart_money_boost(
                    current_grade=grade,
                    ml_prediction=result.get("ml_prediction"),
                    smart_money_score=sm_score,
                )

                # Embed enrichment into result (non-destructive to existing keys)
                result["smart_money"] = {
                    "enabled": True,
                    "boost_tier": sm_score.boost_tier,
                    "boost_reason": sm_score.boost_reason,
                    "alert_label": alert_label,
                    "is_super_alpha": is_super_alpha,
                    "final_grade": final_grade,
                    "effective_overlap_count": sm_score.effective_overlap_count,
                    "smart_money_weighted_score": sm_score.smart_money_weighted_score,
                    "positive_entity_wallets": sm_score.positive_entity_wallets,
                    "negative_entity_wallets": sm_score.negative_entity_wallets,
                    "insider_wallets": sm_score.insider_wallets,
                    "has_cluster": sm_score.has_smart_money_cluster,
                    "cluster_wallets": sm_score.cluster_wallets,
                    "wallet_profiles": sm_score.wallet_profiles,
                }

                # Promote grade if Smart Money boosts it
                if final_grade != grade:
                    if self.debug:
                        print(
                            f"[TokenAnalyzer] 🚀 {mint} grade boosted: "
                            f"{grade} → {final_grade} [{alert_label}]"
                        )
                    result["grade"] = final_grade
                    result["grade_boosted_from"] = grade

                if is_super_alpha:
                    result["SUPER_ALPHA"] = True
                    if self.debug:
                        print(f"[TokenAnalyzer] 🔥 {mint} promoted to SUPER-ALPHA!")

                # FILTERED: Smart Money detected only bots/CEX — suppress upload
                if sm_score.boost_tier == "FILTERED":
                    if self.debug:
                        print(
                            f"[TokenAnalyzer] 🚫 {mint} FILTERED by Smart Money "
                            f"(all overlap wallets are negative entities). "
                            f"Moving to monitoring."
                        )
                    result["needs_monitoring"] = True
                    result["skip_reason"] = "smart_money_filtered_negative_entities"

            except Exception as sm_err:
                # Smart Money failure must NEVER block the main pipeline
                if self.debug:
                    print(f"[TokenAnalyzer] ⚠️ Smart Money scorer error for {mint}: {sm_err}")
                result["smart_money"] = {
                    "enabled": True,
                    "error": str(sm_err),
                    "boost_tier": "NONE",
                    "alert_label": "STANDARD 📊",
                    "is_super_alpha": False,
                }
        else:
            # Smart Money disabled or token not cleared
            result["smart_money"] = {
                "enabled": False,
                "boost_tier": "NONE",
                "alert_label": "STANDARD 📊",
                "is_super_alpha": False,
            }
        # --- END SMART MONEY SCORING ---

        return result

    async def batch_analyze_tokens(
        self,
        mints: List[str],
        wallet_buy_context: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple tokens with semaphore-controlled concurrency.

        wallet_buy_context: { mint: { wallet_address: ISO_timestamp } }
            Tells score_token when the discovering wallet first bought each token.
            Used for early buyer / insider detection at zero extra API cost.
        """
        if not mints:
            return []

        wallet_buy_context = wallet_buy_context or {}
        tasks = [
            asyncio.create_task(
                self.analyze_token(
                    m,
                    wallet_buy_timestamps=wallet_buy_context.get(m),
                )
            )
            for m in mints
        ]
        results: List[Dict[str, Any]] = []
        
        api_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, res in enumerate(api_results):
            if isinstance(res, Exception):
                # This catches the format string error if it still occurs
                err_str = str(res)
                if "unsupported format string" in err_str:
                     print(f"[TokenAnalyzer] ❌ CRITICAL BATCH ERROR for {mints[i]}: {err_str}")
                     import traceback
                     traceback.print_exc()
                else:
                    if self.debug:
                        print(f"[TokenAnalyzer] ❌ Batch analysis error for {mints[i]}: {res}")
                results.append({"mint": mints[i], "error": str(res)})
            elif res and not res.get("error"):
                results.append(res)
            elif res and res.get("error"):
                if self.debug:
                    print(f"[TokenAnalyzer] ⚠️ Analysis failed for {mints[i]}: {res.get('error')}")
                results.append(res)
                    
        return results


# -----------------------------------------------
# Conviction Summary Builder
# -----------------------------------------------

def build_conviction_summary(overlap_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a professional trader-facing conviction summary from a fully
    analysed overlap result. Designed to be stored alongside the result
    and surfaced directly in Telegram alerts or dashboards.

    Follows the "Gold Standard" approach used by GMGN, AlphaScan, and
    top Solana monitors — aggregates over individuals, not single-wallet stats.

    Key philosophy:
      - Cluster averages beat top-wallet stats (one lucky wallet is noise)
      - Sentiment label > raw numbers (traders act on conviction framing)
      - Recency matters (a cold wallet with $1M profit is not a signal)
      - Consistency beats magnitude (84% WR across 5 wallets > 1 wallet 10x)
    """
    sm          = overlap_result.get("smart_money", {})
    ml          = overlap_result.get("ml_prediction") or {}
    dex_raw     = (overlap_result.get("dexscreener") or {}).get("raw") or {}
    sec         = overlap_result.get("security") or {}
    rugcheck    = sec.get("rugcheck") or {} if isinstance(sec, dict) else {}
    profiles    = sm.get("wallet_profiles") or {}

    # =========================================================================
    # SECTION 1: Wallet cluster analysis
    # =========================================================================

    tier_counts  = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "NONE": 0}
    pnl_counts   = {"ELITE": 0, "STRONG": 0, "ACTIVE": 0, "NOISE": 0, "NEGATIVE": 0}
    all_wallets  = []

    # Accumulators for cluster-level PnL aggregates
    profits_with_data: List[float] = []    # realized profit USD for wallets with PnL data
    win_rates_with_data: List[float] = []  # win rate % for wallets with PnL data
    trade_counts_with_data: List[int] = [] # trade counts for wallets with PnL data
    insider_count  = 0
    sniper_count   = 0    # bought within SM_SNIPER_WINDOW_MINUTES
    early_buy_count = 0   # bought within SM_INSIDER_WINDOW_MINUTES

    for addr, p in profiles.items():
        freq_tier    = p.get("alpha_winner_tier", "NONE")
        pnl_tier     = p.get("pnl_tier") or "NOISE"
        insider_type = p.get("insider_type")

        tier_counts[freq_tier] = tier_counts.get(freq_tier, 0) + 1
        pnl_counts[pnl_tier]   = pnl_counts.get(pnl_tier, 0) + 1

        if p.get("is_insider"):
            insider_count += 1
            if insider_type == "SNIPER":
                sniper_count += 1
            elif insider_type in ("EARLY_BUYER", "RUGCHECK"):
                early_buy_count += 1

        # Only include wallets with real PnL data in aggregates
        profit = p.get("total_realized_profit_usd")
        wr     = p.get("win_rate_pct")
        tc     = p.get("total_count_of_trades")

        if profit is not None:
            try:
                profits_with_data.append(float(profit))
            except (ValueError, TypeError):
                pass
        if wr is not None:
            try:
                win_rates_with_data.append(float(wr))
            except (ValueError, TypeError):
                pass
        if tc is not None:
            try:
                trade_counts_with_data.append(int(tc))
            except (ValueError, TypeError):
                pass

        all_wallets.append({
            "address":             addr,
            "short":               addr[:6] + "..." + addr[-4:],
            "frequency":           p.get("dune_frequency", 0),
            "alpha_winner_tier":   freq_tier,
            "pnl_tier":            pnl_tier,
            "realized_profit_usd": profit,
            "realized_profit_pct": p.get("total_realized_profit_pct"),
            "trade_count":         tc,
            "win_rate_pct":        wr,
            "is_insider":          p.get("is_insider", False),
            "insider_type":        insider_type,         # "SNIPER" / "EARLY_BUYER" / "RUGCHECK"
            "insider_rank":        p.get("insider_rank"), # minutes after launch
            "label_from_cache":    p.get("label_from_cache", False),
        })

    all_wallets.sort(key=lambda x: x["frequency"], reverse=True)
    top5 = all_wallets[:5]

    # ── Cluster PnL aggregates (the "Gold Standard" approach) ─────────────────
    n_pnl = len(profits_with_data)   # wallets with actual PnL data

    cluster_combined_profit   = round(sum(profits_with_data), 2) if profits_with_data else None
    cluster_avg_win_rate      = round(sum(win_rates_with_data) / len(win_rates_with_data), 1) \
                                if win_rates_with_data else None
    cluster_avg_trade_count   = round(sum(trade_counts_with_data) / len(trade_counts_with_data)) \
                                if trade_counts_with_data else None
    cluster_total_trades      = sum(trade_counts_with_data) if trade_counts_with_data else None

    # Best single wallet (highest realized profit among those with data)
    best_wallet = None
    if profits_with_data:
        best = max(all_wallets, key=lambda w: w.get("realized_profit_usd") or 0)
        if best.get("realized_profit_usd") is not None:
            best_wallet = {
                "short":       best["short"],
                "pnl_tier":    best["pnl_tier"],
                "profit_usd":  best["realized_profit_usd"],
                "win_rate":    best["win_rate_pct"],
                "trade_count": best["trade_count"],
                "is_insider":  best["is_insider"],
            }

    # ── Dune frequency conviction score ───────────────────────────────────────
    # Weighted quality of the overlap: HIGH=3, MEDIUM=2, LOW=1, NOISE=0
    tier_weights   = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}
    raw_score      = sum(tier_weights[t] * c for t, c in tier_counts.items())
    max_possible   = tier_weights["HIGH"] * (overlap_result.get("overlap_count", 1) or 1)
    wallet_conviction_pct = round((raw_score / max_possible * 100) if max_possible else 0, 1)

    # ── Support tier labels (GMGN-style: Elite / Strong / Active) ─────────────
    elite_count  = pnl_counts["ELITE"]
    strong_count = pnl_counts["STRONG"]
    active_count = pnl_counts["ACTIVE"]
    qualified_count = elite_count + strong_count + active_count  # excludes NOISE/NEGATIVE

    # =========================================================================
    # SECTION 2: Smart money signals
    # =========================================================================

    boost_tier     = sm.get("boost_tier", "NONE")
    alert_label    = sm.get("alert_label", "STANDARD 📊")
    is_super_alpha = sm.get("is_super_alpha", False)
    has_cluster    = sm.get("has_cluster", False)
    cluster_wallets = sm.get("cluster_wallets") or []
    pos_entities   = sm.get("positive_entity_wallets") or []
    neg_entities   = sm.get("negative_entity_wallets") or []
    insiders       = sm.get("insider_wallets") or []
    weighted_score = sm.get("smart_money_weighted_score", 0)
    eff_overlap    = sm.get("effective_overlap_count", overlap_result.get("overlap_count", 0))

    # =========================================================================
    # SECTION 3: ML signals
    # =========================================================================

    ml_prob       = ml.get("probability")
    ml_confidence = ml.get("confidence", "NONE")
    ml_risk       = ml.get("risk_tier", "UNKNOWN")
    ml_action     = ml.get("action", "UNKNOWN")
    ml_passed     = overlap_result.get("ML_PASSED", False)

    try:
        ml_prob_f   = float(ml_prob) if ml_prob is not None else 0.0
        ml_prob_pct = f"{ml_prob_f:.1%}"
    except (ValueError, TypeError):
        ml_prob_f   = 0.0
        ml_prob_pct = "N/A"

    # =========================================================================
    # SECTION 4: Market data
    # =========================================================================

    price_usd    = (overlap_result.get("dexscreener") or {}).get("price_usd")
    volume       = dex_raw.get("volume") or {}
    price_chg    = dex_raw.get("priceChange") or {}
    liquidity    = dex_raw.get("liquidity") or {}
    txns         = dex_raw.get("txns") or {}
    mcap         = dex_raw.get("marketCap")
    fdv          = dex_raw.get("fdv")
    dex_url      = dex_raw.get("url", "")
    pair_created = dex_raw.get("pairCreatedAt")
    base_token   = dex_raw.get("baseToken") or {}
    token_name   = base_token.get("name", "Unknown")
    token_symbol = base_token.get("symbol", "???")
    dex_id       = dex_raw.get("dexId", "unknown")

    pair_age_mins = None
    if pair_created:
        try:
            pair_age_mins = round((time.time() * 1000 - int(pair_created)) / 60000, 1)
        except Exception:
            pass

    buys_5m  = (txns.get("m5") or {}).get("buys", 0)
    sells_5m = (txns.get("m5") or {}).get("sells", 0)
    buys_1h  = (txns.get("h1") or {}).get("buys", 0)
    sells_1h = (txns.get("h1") or {}).get("sells", 0)
    buy_pressure_5m = round(buys_5m / (buys_5m + sells_5m) * 100) if (buys_5m + sells_5m) else None
    buy_pressure_1h = round(buys_1h / (buys_1h + sells_1h) * 100) if (buys_1h + sells_1h) else None

    # =========================================================================
    # SECTION 5: Security highlights
    # =========================================================================

    lp_locked = rugcheck.get("lp_locked_pct")
    holders   = rugcheck.get("holder_count")
    liq_usd   = rugcheck.get("total_liquidity_usd")

    # =========================================================================
    # SECTION 6: Alpha Score (0–100) + Trader Sentiment
    # =========================================================================
    # Combines five independent signals into one actionable number.
    # Weighted so that wallet quality and PnL consistency dominate —
    # a single lucky wallet cannot carry the score.
    #
    # Component weights:
    #   Overlap grade tier      25 pts  — how many high-frequency winners hold this
    #   Cluster avg win rate    25 pts  — consistency of the group, not just one wallet
    #   ML probability          20 pts  — model confidence
    #   SM boost tier           20 pts  — entity / cluster signal
    #   Insider detection       10 pts  — early buyers = informed money

    grade = overlap_result.get("grade", "NONE")

    # Component 1: overlap grade (0-25)
    grade_pts = {"VERY_HIGH": 25, "HIGH": 20, "MEDIUM": 14, "LOW": 8, "UNKNOWN": 3, "NONE": 0}.get(grade, 0)

    # Component 2: cluster avg win rate (0-25)
    # Scale: 70%+ WR → full 25pts; 50% → 12pts; <40% → 0pts
    if cluster_avg_win_rate is not None:
        wr_pts = round(max(0.0, min(25.0, (cluster_avg_win_rate - 40) / 30 * 25)))
    else:
        wr_pts = 0   # no PnL data available yet

    # Component 3: ML probability (0-20)
    ml_pts = round(max(0.0, min(20.0, ml_prob_f * 20)))

    # Component 4: SM boost tier (0-20)
    sm_pts = {"SUPER_ALPHA": 20, "STRONG": 14, "STANDARD": 8, "NONE": 4, "FILTERED": 0}.get(boost_tier, 4)

    # Component 5: insider detection (0-10)
    # Snipers (≤5 min) = 5 pts each, early buyers (≤30 min) = 2 pts each, max 10
    insider_pts = min(10, sniper_count * 5 + early_buy_count * 2)

    alpha_score = grade_pts + wr_pts + ml_pts + sm_pts + insider_pts   # 0-100

    # ── Trader sentiment label ────────────────────────────────────────────────
    # Psychology-first framing — tells the trader how to feel about the signal
    if alpha_score >= 80:
        trader_sentiment     = "EXTREME BULLISH 🔥"
        sentiment_confidence = "VERY HIGH"
    elif alpha_score >= 65:
        trader_sentiment     = "STRONG BULLISH ⭐"
        sentiment_confidence = "HIGH"
    elif alpha_score >= 50:
        trader_sentiment     = "BULLISH ✅"
        sentiment_confidence = "MEDIUM"
    elif alpha_score >= 35:
        trader_sentiment     = "CAUTIOUS ⚠️"
        sentiment_confidence = "LOW"
    else:
        trader_sentiment     = "BEARISH ❌"
        sentiment_confidence = "VERY LOW"

    # ── Support tier label (GMGN-style) ──────────────────────────────────────
    # e.g. "2 Elite (Top 0.1%) | 3 Strong (Top 1%)"
    tier_parts = []
    if elite_count:
        tier_parts.append(f"{elite_count} Elite 🏅 (Top 0.1%)")
    if strong_count:
        tier_parts.append(f"{strong_count} Strong 💪 (Top 1%)")
    if active_count:
        tier_parts.append(f"{active_count} Active ✅")
    support_tier_label = " | ".join(tier_parts) if tier_parts else "No Qualified Wallets"

    # ── Overall signal rating (backward-compat with previous format) ──────────
    if alpha_score >= 80:   overall_rating = "🔥 ELITE"
    elif alpha_score >= 65: overall_rating = "⭐ STRONG"
    elif alpha_score >= 50: overall_rating = "✅ SOLID"
    elif alpha_score >= 35: overall_rating = "⚠️ SPECULATIVE"
    else:                   overall_rating = "❌ WEAK"

    # =========================================================================
    # SECTION 7: Assemble — flat, JSON-safe dict
    # =========================================================================
    return {
        # ── Identity ──────────────────────────────────────────────────────────
        "token_name":           token_name,
        "token_symbol":         token_symbol,
        "mint":                 overlap_result.get("mint"),
        "dex":                  dex_id,
        "dex_url":              dex_url,

        # ── Headline signal ────────────────────────────────────────────────────
        "alpha_score":          alpha_score,            # 0-100 composite
        "trader_sentiment":     trader_sentiment,       # "EXTREME BULLISH 🔥"
        "sentiment_confidence": sentiment_confidence,   # VERY HIGH / HIGH / MEDIUM / LOW / VERY LOW
        "overall_rating":       overall_rating,         # 🔥 ELITE / ⭐ STRONG / ...
        "alert_label":          alert_label,            # SUPER-ALPHA 🔥 / ALPHA ⭐ / ...
        "grade":                grade,
        "is_super_alpha":       is_super_alpha,

        # ── Cluster PnL aggregates (the signal, not one wallet) ───────────────
        "cluster_wallets_with_pnl":    n_pnl,
        "cluster_combined_profit_usd": cluster_combined_profit,   # sum of all realized profits
        "cluster_avg_win_rate_pct":    cluster_avg_win_rate,      # avg win rate across wallets
        "cluster_avg_trade_count":     cluster_avg_trade_count,   # avg # trades per wallet
        "cluster_total_trades":        cluster_total_trades,       # total trades across cluster
        "cluster_qualified_wallets":   qualified_count,            # ELITE + STRONG + ACTIVE

        # ── Support tier breakdown (GMGN-style) ───────────────────────────────
        "support_tier_label":   support_tier_label,     # "2 Elite | 3 Strong"
        "pnl_tier_breakdown": {
            "ELITE":    elite_count,
            "STRONG":   strong_count,
            "ACTIVE":   active_count,
            "NOISE":    pnl_counts["NOISE"],
            "NEGATIVE": pnl_counts["NEGATIVE"],
        },

        # ── Insider / sniper detection ─────────────────────────────────────────
        "insider_count":        insider_count,          # total insiders (all types)
        "sniper_count":         sniper_count,           # bought within SM_SNIPER_WINDOW_MINUTES
        "early_buyer_count":    early_buy_count,        # bought within SM_INSIDER_WINDOW_MINUTES
        "sniper_detected":      sniper_count >= 1,
        "early_buyer_detected": early_buy_count >= 1,

        # ── Best single wallet (AlphaScan-style: proof of competence) ─────────
        "best_wallet":          best_wallet,            # None if no PnL data in top-N gate

        # ── Full wallet breakdown (top-5 by frequency) ────────────────────────
        "top_wallets":          top5,
        "overlap_count":        overlap_result.get("overlap_count", 0),
        "effective_overlap":    eff_overlap,
        "excluded_wallets":     len(neg_entities),
        "wallet_conviction_pct": wallet_conviction_pct,
        "dune_tier_breakdown": {
            "HIGH_freq":   tier_counts["HIGH"],
            "MEDIUM_freq": tier_counts["MEDIUM"],
            "LOW_freq":    tier_counts["LOW"],
            "NOISE":       tier_counts["NONE"],
        },

        # ── SM scoring internals ───────────────────────────────────────────────
        "sm_boost_tier":        boost_tier,
        "sm_weighted_score":    round(weighted_score, 2),
        "sm_positive_wallets":  len(pos_entities),
        "sm_negative_wallets":  len(neg_entities),
        "sm_has_cluster":       has_cluster,
        "sm_cluster_size":      len(cluster_wallets),

        # ── Alpha score components (transparent breakdown) ─────────────────────
        "alpha_score_breakdown": {
            "overlap_grade_pts":  grade_pts,   # /25
            "win_rate_pts":       wr_pts,       # /25
            "ml_prob_pts":        ml_pts,       # /20
            "sm_boost_pts":       sm_pts,       # /20
            "insider_pts":        insider_pts,  # /10
        },

        # ── ML ─────────────────────────────────────────────────────────────────
        "ml_probability":       ml_prob,
        "ml_probability_pct":   ml_prob_pct,
        "ml_confidence":        ml_confidence,
        "ml_risk_tier":         ml_risk,
        "ml_action":            ml_action,
        "ml_passed":            ml_passed,

        # ── Market ─────────────────────────────────────────────────────────────
        "price_usd":            price_usd,
        "market_cap_usd":       mcap,
        "fdv_usd":              fdv,
        "pair_age_mins":        pair_age_mins,
        "liquidity_usd":        liquidity.get("usd"),
        "volume_5m_usd":        volume.get("m5"),
        "volume_1h_usd":        volume.get("h1"),
        "volume_6h_usd":        volume.get("h6"),
        "volume_24h_usd":       volume.get("h24"),
        "price_change_5m_pct":  price_chg.get("m5"),
        "price_change_1h_pct":  price_chg.get("h1"),
        "price_change_6h_pct":  price_chg.get("h6"),
        "price_change_24h_pct": price_chg.get("h24"),
        "buys_5m":              buys_5m,
        "sells_5m":             sells_5m,
        "buys_1h":              buys_1h,
        "sells_1h":             sells_1h,
        "buy_pressure_5m_pct":  buy_pressure_5m,
        "buy_pressure_1h_pct":  buy_pressure_1h,

        # ── Security ───────────────────────────────────────────────────────────
        "lp_locked_pct":        lp_locked,
        "holder_count":         holders,
        "security_liquidity_usd": liq_usd,

        # ── Meta ────────────────────────────────────────────────────────────────
        "checked_at":           overlap_result.get("checked_at"),
        "generated_at":         datetime.now(timezone.utc).isoformat(),
    }


# -----------------------------------------------
# Component 5: WinnerMonitor (Orchestrator)
# -----------------------------------------------

class WinnerMonitor:
    """Coordinate all components with three concurrent async loops."""
    def __init__(
        self,
        http_session: aiohttp.ClientSession,
        moralis_client: MoralisClient,
        wallet_ranker: WinnerWalletRanker,
        wallet_scheduler: WalletCheckScheduler,
        token_analyzer: AlphaTokenAnalyzer,
        overlap_store: AlphaOverlapStore,
        *,
        poll_interval_seconds: int = 900,
        top_n_wallets: int = 50,
        monitoring_window_hours: int = 6,
        debug: bool = False
    ):
        self.http_session = http_session
        self.moralis_client = moralis_client
        self.wallet_ranker = wallet_ranker
        self.wallet_scheduler = wallet_scheduler
        self.token_analyzer = token_analyzer
        self.overlap_store = overlap_store
        
        self.poll_interval_seconds = poll_interval_seconds
        self.top_n_wallets = top_n_wallets
        self.monitoring_window_hours = monitoring_window_hours
        self.debug = debug
        
        # Track ALL tokens under monitoring (including NONE grade, probation, security fail)
        self.pending_tokens: Dict[str, Dict[str, Any]] = {}
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        
        if self.debug:
            print("🚀 [WinnerMonitor] Orchestrator initialized.")
            print(f"  - Poll Interval: {self.poll_interval_seconds}s")
            print(f"  - Wallet Cooldown: {self.wallet_scheduler.cooldown_seconds}s")
            print(f"  - Top N Wallets: {self.top_n_wallets}")
            print(f"  - Monitoring Window: {self.monitoring_window_hours}h")

    async def startup(self):
        """One-time initialization on monitor start."""
        if self.debug:
            print("🚀 [WinnerMonitor] Performing startup...")
            
        cache_files = await self.wallet_ranker.download_dune_caches_for_last_7_days()
        self.wallet_ranker.extract_and_rank_wallets(cache_files)
        self.wallet_scheduler.load_state()

        # Download Smart Money label cache from Supabase and merge with local
        # This recovers all previously discovered entity labels after a restart
        if hasattr(self.moralis_client, "download_label_cache_from_supabase"):
            if self.debug:
                print("🚀 [WinnerMonitor] Syncing Smart Money label cache from Supabase...")
            self.moralis_client.download_label_cache_from_supabase()
        
        if self.debug:
            print("🚀 [WinnerMonitor] Startup complete. Starting loops...")

    async def _security_gate_and_save(
        self,
        overlap_result: Dict[str, Any],
        current_store_state: Dict[str, List[Dict]],
        check_type: str = "new_discovery"
    ) -> bool:
        """
        Properly handles tokens that need monitoring vs. upload.
        - Tokens with needs_monitoring=True: Enter 24h monitoring loop (NOT uploaded to Supabase)
        - Tokens passing ALL requirements AND overlap_count > 0: Uploaded to Supabase immediately
        """
        mint = overlap_result.get("mint")
        if not mint:
            return False
        
        needs_monitoring = overlap_result.get("needs_monitoring", False)
        skip_reason = overlap_result.get("skip_reason")
        security_failures = overlap_result.get("security_failures", [])
        grade = overlap_result.get("grade", "NONE")
        overlap_count = overlap_result.get("overlap_count", 0)
        
        # 1. CRITICAL: If needs_monitoring flag is set, do NOT upload. Start monitoring.
        if needs_monitoring:
            reasons = [skip_reason] if skip_reason else []
            reasons.extend(security_failures)
            
            if self.debug:
                print(f"[SecurityGate] 🔒 Token {mint} needs monitoring - NOT uploading to Supabase")
                print(f"[SecurityGate] Reasons: {', '.join(reasons)}")
            
            # Start/update monitoring loop (stays in memory only)
            await self._start_or_update_monitoring(mint, overlap_result, reasons, grade)
            return False  # Return False to prevent Supabase upload

        # 2. Token passed ALL checks (RugCheck, Probation, Security, AND has overlap > 0) - upload
        # THIS IS THE ONLY PATH TO PERSISTENT STORAGE
        if overlap_count > 0 and grade not in ("NONE", "UNKNOWN"):
            
            # Build conviction summary for trader-facing display
            conviction = build_conviction_summary(overlap_result)

            # Build entry for storage
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "result": overlap_result,
                "check_type": check_type,
                "security": "passed",
                "ML_PASSED": overlap_result.get("ML_PASSED", False),
                "conviction_summary": conviction,   # ← trader-facing summary
            }
            
            current_store_state.setdefault(mint, []).append(entry)
            
            # Remove from monitoring if it was there (e.g. if a recheck brought it here)
            if mint in self.pending_tokens:
                if self.debug:
                    print(f"[SecurityGate] ✅ Token {mint} passed and removed from monitoring.")
                self.pending_tokens.pop(mint, None)
                task = self._monitoring_tasks.pop(mint, None)
                if task and not task.done():
                    task.cancel()
            
            if self.debug:
                print(f"[SecurityGate] ✅ Token {mint} PASSED ALL CHECKS - uploading to Supabase [Grade: {grade}]")
            
            return True  # Upload to Supabase

        # 3. If here, needs_monitoring was False, but overlap_count was 0 or grade was NONE.
        if self.debug:
            print(f"[SecurityGate] ⚠️ Token {mint} passed security/probation but has ZERO overlap or {grade} grade. Not uploading.")
            
        return False # Do not upload zero-overlap/NONE-grade tokens

    async def poll_wallets_loop(self):
        """Main 15-minute loop: check top wallets for new tokens."""
        while True:
            start_time = time.monotonic() 
            
            if self.debug:
                print(f"[PollLoop] -----------------------------------------------")
                print(f"[PollLoop] Starting new poll cycle")
            
            try:
                # Ranker should be run periodically for fresh data (now in refresh_dune_cache_loop)
                all_ranked = self.wallet_ranker.get_top_n_wallets(n=10000)
                
                if self.debug:
                    print(f"[PollLoop] Total ranked wallets available: {len(all_ranked)}")
                
                eligible = self.wallet_scheduler.get_next_batch(all_ranked, self.top_n_wallets)
                
                if not eligible:
                    if self.debug:
                        print(f"[PollLoop] ⚠️ No eligible wallets found (all on cooldown). Sleeping...")
                    
                    work_duration = time.monotonic() - start_time
                    sleep_time = max(0, self.poll_interval_seconds - work_duration)
                    if self.debug:
                        print(f"[PollLoop] Cycle work took {work_duration:.2f}s. Sleeping for {sleep_time:.2f}s.")
                    await asyncio.sleep(sleep_time)
                    continue
                
                if self.debug:
                    print(f"[PollLoop] Processing {len(eligible)} eligible wallets this cycle")
                
                # Load state once before batch processing
                current_store_state = self.overlap_store.load()
                store_changed = False
                
                for wallet in eligible:
                    key_used = None
                    try:
                        txs = await self.moralis_client.fetch_wallet_transfers(wallet)
                        key_used = self.moralis_client._last_key_per_wallet.get(wallet)
                        
                        mints = await self.moralis_client.extract_unique_tokens(txs)
                        # Extract buy timestamps for insider detection — {mint: ISO_ts}
                        # Tells us when THIS wallet first bought each token
                        mint_buy_timestamps = await self.moralis_client.extract_tokens_with_timestamps(txs)
                        
                        if not mints:
                            self.wallet_scheduler.mark_wallet_checked(wallet, 0, key_used)
                            continue
                        
                        # Only check new tokens or tokens failing monitoring.
                        # Do NOT check tokens already in the main store or pending monitoring.
                        new_mints_to_check = [
                            m for m in mints 
                            if m not in current_store_state 
                            and m not in self.pending_tokens
                        ]
                        
                        if not new_mints_to_check:
                            if self.debug:
                                print(f"[PollLoop] 🔎 Wallet {wallet[:6]}... found {len(mints)} tokens (all known).")
                            self.wallet_scheduler.mark_wallet_checked(wallet, len(mints), key_used)
                            continue
                        
                        if self.debug:
                            print(f"[PollLoop] 💎 Wallet {wallet[:6]}... found {len(new_mints_to_check)} NEW tokens.")

                        # Build a per-token wallet buy context:
                        # { mint: { wallet_address: ISO_timestamp } }
                        # so score_token knows when THIS wallet first bought each token
                        token_wallet_buy_context: Dict[str, Dict[str, str]] = {}
                        for m in new_mints_to_check:
                            ts = mint_buy_timestamps.get(m)
                            if ts:
                                token_wallet_buy_context[m] = {wallet: ts}

                        results = await self.token_analyzer.batch_analyze_tokens(
                            new_mints_to_check,
                            wallet_buy_context=token_wallet_buy_context,
                        )
                        
                        self.wallet_scheduler.mark_wallet_checked(wallet, len(mints), key_used)
                        
                        for res in results:
                            if "error" in res:
                                continue
                            
                            # The security gate handles whether it goes to monitoring or the store.
                            uploaded_to_be_stored = await self._security_gate_and_save(
                                res, current_store_state, "new_discovery"
                            )
                            if uploaded_to_be_stored:
                                store_changed = True
                            
                    except Exception as e:
                        if self.debug:
                            print(f"[PollLoop] ❌ Error processing wallet {wallet}: {e}")
                            import traceback
                            traceback.print_exc()
                        self.wallet_scheduler.mark_wallet_checked(wallet, 0, key_used)

                if store_changed:
                    # Save and upload if any token was cleared for upload in this batch
                    self.overlap_store.save(current_store_state)
                
                if self.debug:
                    print(f"[PollLoop] -----------------------------------------------")
                    print(f"[PollLoop] ✅ Cycle complete: Processed {len(eligible)} wallets")
                    print(f"[PollLoop] Total checks this session: {self.wallet_scheduler.total_checks_this_session}")

            except Exception as e:
                if self.debug:
                    print(f"[PollLoop] ❌ CRITICAL Error in main poll loop: {e}")
                    import traceback
                    traceback.print_exc()

            work_duration = time.monotonic() - start_time
            sleep_time = max(0, self.poll_interval_seconds - work_duration)
            
            if self.debug:
                print(f"[PollLoop] Cycle work took {work_duration:.2f}s.")
                if sleep_time > 0:
                    print(f"[PollLoop] Next cycle in {sleep_time:.2f}s...")
            
            await asyncio.sleep(sleep_time)

    async def refresh_dune_cache_loop(self):
        """Background task: Download fresh Dune cache once per day at 3:30 AM UTC."""
        while True:
            try:
                now = datetime.now(timezone.utc)
                next_run = now.replace(hour=3, minute=30, second=0, microsecond=0)
                if now >= next_run:
                    next_run += timedelta(days=1)
                
                sleep_seconds = (next_run - now).total_seconds()
                if self.debug:
                    print(f"[RefreshLoop] 😴 Next Dune cache refresh in {sleep_seconds/3600:.2f} hours.")
                await asyncio.sleep(sleep_seconds)
                
                if self.debug:
                    print("[RefreshLoop] 🌞 Waking up to refresh Dune cache...")

                cache_files = await self.wallet_ranker.download_dune_caches_for_last_7_days()
                self.wallet_ranker.extract_and_rank_wallets(cache_files)
                
                if self.debug:
                    print("[RefreshLoop] ✅ Dune cache refreshed and wallets re-ranked.")

            except Exception as e:
                if self.debug:
                    print(f"[RefreshLoop] ❌ Error in Dune refresh loop: {e}")
                await asyncio.sleep(3600)

    async def recheck_tokens_loop(self):
        """Background task: Recheck all tracked tokens every 30 minutes."""
        while True:
            await asyncio.sleep(1800)
            
            try:
                if self.debug:
                    print("[RecheckLoop] -----------------------------------------------")
                    print("[RecheckLoop] 🔄 Starting hourly recheck...")
                
                # Load current store for recheck eligibility
                current_store = self.overlap_store.load()
                
                # We only recheck tokens that are currently in the store and have a grade (passed the initial gate)
                tokens_to_recheck = []
                for mint, entries in current_store.items():
                    # Do not recheck tokens already in the pending_tokens monitoring loop
                    if mint in self.pending_tokens:
                        continue
                    
                    if not entries or not isinstance(entries, list):
                        continue
                        
                    try:
                        latest_entry = entries[-1]
                        latest_grade = latest_entry.get("result", {}).get("grade", "NONE")
                        
                        # Recheck tokens that have a grade higher than NONE AND passed security
                        if latest_grade not in ("NONE", "UNKNOWN") and latest_entry.get("security") == "passed":
                            tokens_to_recheck.append(mint)
                            
                    except Exception:
                        # Fallback: recheck if entry is malformed but still exists
                        tokens_to_recheck.append(mint)

                if not tokens_to_recheck:
                    if self.debug:
                        print("[RecheckLoop] No graded (non-NONE/UNKNOWN) tokens to recheck.")
                    continue
                
                if self.debug:
                    print(f"[RecheckLoop] Rechecking {len(tokens_to_recheck)} (non-NONE/UNKNOWN) tokens...")
                    
                results = await self.token_analyzer.batch_analyze_tokens(tokens_to_recheck)
                
                # Re-load store as it may have changed during analysis (via monitoring passes)
                current_store = self.overlap_store.load()
                changed = False
                
                for res in results:
                    mint = res.get("mint")
                    if not mint or "error" in res:
                        continue
                        
                    history = current_store.get(mint, [])
                    if not history:
                        # If a token was removed between load and now, skip
                        continue 
                    
                    latest_entry = history[-1]
                    latest_grade = latest_entry.get("result", {}).get("grade", "NONE")
                    new_grade = res.get("grade", "NONE")
                    
                    # If the recheck changed the grade OR the security/monitoring status changed
                    security_check_passed = not res.get("needs_monitoring", False)
                    overlap_check_passed = res.get("overlap_count", 0) > 0
                    
                    # If the security status is now failing (e.g. market cap drop, new authority)
                    if not security_check_passed:
                        if self.debug:
                            print(f"[RecheckLoop] ⚠️ {mint} FAILED SECURITY on recheck (Grade: {latest_grade} -> NONE). Moving to monitoring.")
                        
                        # Use the security gate to put it into monitoring (since needs_monitoring=True)
                        await self._security_gate_and_save(
                            res, current_store, "hourly_recheck_fail"
                        )
                        # We don't mark as changed yet, as this is purely an in-memory change.
                        
                    # If the grade or overlap changed
                    elif latest_grade != new_grade or (latest_grade not in ("NONE", "UNKNOWN") and overlap_check_passed):
                        if self.debug:
                            print(f"[RecheckLoop] 📊 Status change {mint}: {latest_grade} -> {new_grade}")
                        
                        # Use the security gate, which will append a new entry with "security": "passed"
                        gated = await self._security_gate_and_save(
                            res, current_store, "hourly_recheck"
                        )
                        if gated:
                            changed = True

                if changed:
                    self.overlap_store.save(current_store)

            except Exception as e:
                if self.debug:
                    print(f"[RecheckLoop] ❌ Error in recheck loop: {e}")
                    import traceback
                    traceback.print_exc()

    async def _start_or_update_monitoring(
        self, mint: str, overlap_result: Dict[str, Any], reasons: List[str], grade: str
    ):
        """Start or update the monitoring state for a token."""
        now_iso = datetime.now(timezone.utc).isoformat()
        now_ts = int(datetime.now(timezone.utc).timestamp())
        
        # Determine the correct first_seen_ts: keep the original if updating
        first_seen_ts = self.pending_tokens.get(mint, {}).get("first_seen_ts", now_ts)
        
        self.pending_tokens[mint] = {
            "first_seen_ts": first_seen_ts,
            "last_checked": now_iso,
            "attempts": self.pending_tokens.get(mint, {}).get("attempts", 0) + 1,
            "reasons": list(dict.fromkeys(self.pending_tokens.get(mint, {}).get("reasons", []) + reasons)),
            "overlap_result": overlap_result,
            "grade": grade
        }

        if mint in self._monitoring_tasks and not self._monitoring_tasks[mint].done():
            if self.debug:
                print(f"[Monitoring] Updated monitoring for {mint} [Grade: {grade}]; reasons={self.pending_tokens[mint]['reasons']}")
            return

        task = asyncio.create_task(
            self._monitoring_recheck_loop(mint)
        )
        self._monitoring_tasks[mint] = task
        if self.debug:
            print(f"[Monitoring] 🔍 Started 6h monitoring for {mint} [Grade: {grade}]; reasons={self.pending_tokens[mint]['reasons']}")

    async def _monitoring_recheck_loop(
        self, mint: str
    ):
        """Recheck token every 30 mins for 6 hours (applies to ALL grades including NONE) that failed the gate."""
        entry = self.pending_tokens.get(mint)
        if not entry:
            return
            
        first_seen_ts = entry["first_seen_ts"]
        deadline = first_seen_ts + self.monitoring_window_hours * 3600
        interval = 1800  # 30 mins

        while True:
            now_ts = int(datetime.now(timezone.utc).timestamp())
            if now_ts >= deadline:
                if self.debug:
                    final_grade = entry.get("grade", "NONE")
                    final_reasons = ", ".join(entry.get("reasons", ["no_reason"]))
                    print(f"[Monitoring] ⏰ {mint} completed 6h monitoring [Final Grade: {final_grade}, Reasons: {final_reasons}] -> NOT uploading to Supabase.")
                
                # Remove from monitoring memory
                self.pending_tokens.pop(mint, None)
                self._monitoring_tasks.pop(mint, None)
                break
            
            await asyncio.sleep(interval)
            
            # Check if token was removed during sleep (e.g. if it passed and was uploaded)
            if mint not in self.pending_tokens:
                if self.debug:
                    print(f"[Monitoring] {mint} no longer in pending list. Exiting monitoring loop.")
                self._monitoring_tasks.pop(mint, None)
                break
            
            try:
                if self.debug:
                    current_grade = self.pending_tokens[mint].get("grade", "NONE")
                    print(f"[Monitoring] 🔄 Hourly recheck for {mint} [Current Grade: {current_grade}]...")
                    
                fresh_result = await self.token_analyzer.analyze_token(mint)
                
                # The crucial difference: we use the security gate.
                # If it passes the gate (meaning security passes AND overlap > 0), it gets uploaded and removed from monitoring.
                
                store = self.overlap_store.load()
                uploaded = await self._security_gate_and_save(
                    fresh_result, store, "monitoring_recheck"
                )
                
                if uploaded:
                    self.overlap_store.save(store)
                    # _security_gate_and_save already removes it from pending_tokens and monitoring_tasks
                    if self.debug:
                        print(f"[Monitoring] 🎉 {mint} PASSED ALL REQUIREMENTS (recheck) - Uploaded and removed from monitoring!")
                    break 
                else:
                    # Still failing to pass all requirements for upload (security/probation fail OR zero overlap)
                    # Update monitoring state if it's still in pending_tokens
                    if mint in self.pending_tokens:
                        if self.debug:
                            print(f"[Monitoring] 🔍 {mint} still failing: {fresh_result.get('skip_reason', 'unknown')}")
                        
                        entry = self.pending_tokens[mint]
                        entry["last_checked"] = datetime.now(timezone.utc).isoformat()
                        entry["attempts"] += 1
                        
                        # Update reasons from the fresh result
                        new_reasons = []
                        if fresh_result.get("skip_reason"):
                            new_reasons.append(fresh_result.get("skip_reason"))
                        if fresh_result.get("security_failures"):
                            new_reasons.extend(fresh_result.get("security_failures"))
                        
                        if new_reasons:
                            entry["reasons"] = list(dict.fromkeys(new_reasons))
                        entry["grade"] = fresh_result.get("grade", "NONE")
                    
            except Exception as e:
                if self.debug:
                    print(f"[Monitoring] ❌ Error during recheck for {mint}: {e}")
                    import traceback
                    traceback.print_exc()
                entry = self.pending_tokens.get(mint, {})
                entry["last_checked"] = datetime.now(timezone.utc).isoformat()
                entry["attempts"] = entry.get("attempts", 0) + 1

    async def run(self):
        """Orchestrate all loops as concurrent tasks."""
        await asyncio.gather(
            self.poll_wallets_loop(),
            self.refresh_dune_cache_loop(),
            self.recheck_tokens_loop()
        )


# -----------------------------------------------
# Main Entry Point
# -----------------------------------------------

async def main():
    """Initialize and run winner monitor with dependency injection."""
    if not MORALIS_API_KEYS:
        raise RuntimeError("No Moralis API keys found in MORALIS_API_KEYS")
        
    async with aiohttp.ClientSession() as http_session:
        
        debug_mode = True

        sol_client = SolanaAlphaClient()
        holder_agg = HolderAggregator(sol_client, debug=debug_mode)
        dune_cache = DuneWinnersCache(
            cache_dir="./data/dune_cache",
            debug=debug_mode,
            supabase_bucket=SUPABASE_BUCKET
        )
        
        moralis_client = MoralisClient(
            http_session,
            MORALIS_API_KEYS,
            debug=debug_mode,
            supabase_bucket=SUPABASE_BUCKET,
        )

        # Attach BirdeyeClient — shares moralis_client's PnL cache
        if SMART_MONEY_ENABLED and BIRDEYE_API_KEYS:
            birdeye_client = BirdeyeClient(
                http_session=http_session,
                api_keys=BIRDEYE_API_KEYS,
                pnl_cache=moralis_client._label_cache,
                pnl_cache_lock=moralis_client._label_cache_lock,
                save_cache_fn=moralis_client._save_label_cache,
                get_cached_fn=moralis_client.get_cached_label,
                classify_tier_fn=moralis_client._classify_pnl_tier,
                no_entity_sentinel=moralis_client._NO_ENTITY_SENTINEL_INSTANCE,
                positive_tiers=moralis_client.POSITIVE_PNL_TIERS_INSTANCE,
                negative_tiers=moralis_client.NEGATIVE_PNL_TIERS_INSTANCE,
                debug=debug_mode,
            )
            moralis_client.set_birdeye_client(birdeye_client)
            print(f"[Main] ✅ BirdeyeClient attached ({len(BIRDEYE_API_KEYS)} key(s))")
        elif SMART_MONEY_ENABLED:
            print("[Main] ⚠️  BIRDEYE_API_KEYS not set — PnL layer will return NOISE. "
                  "Set BIRDEYE_API_KEYS in .env to enable.")
        wallet_ranker = WinnerWalletRanker(
            supabase_bucket=SUPABASE_BUCKET, debug=debug_mode
        )
        wallet_scheduler = WalletCheckScheduler(
            cooldown_seconds=WINNER_WALLET_COOLDOWN_SECONDS, debug=debug_mode
        )
        
        helius_limiter = AsyncRateLimiter(
            rate=40, 
            per=1.0, 
            min_interval=0.025,
            debug=debug_mode
        )
        helius_limiter.name = "HeliusLimiter"
        
        rugcheck_limiter = AsyncRateLimiter(
            rate=2,
            per=1.0,
            min_interval=0.5,
            debug=debug_mode
        )
        rugcheck_limiter.name = "RugCheckLimiter"
        
        dex_limiter = AsyncRateLimiter(
            rate=2,
            per=1.0,
            min_interval=0.5,
            debug=debug_mode
        )
        dex_limiter.name = "DexLimiter"
        
        rugcheck_client = RugCheckClient(
            session=http_session,
            rate_limiter=rugcheck_limiter,
            max_retries=3,
            debug=debug_mode
        )

        # --- NEW: Initialize ML Classifier ---
        try:
            ml_classifier = SolanaTokenPredictor(model_dir='models')
        except Exception as e:
            print(f"❌ CRITICAL: Failed to load ML models: {e}")
            print("--- 💀 Winner Monitor Halted ---")
            return
        # --- END NEW ---

        token_analyzer = AlphaTokenAnalyzer(
            http_session=http_session,
            holder_agg=holder_agg,
            dune_cache=dune_cache,
            helius_limiter=helius_limiter,
            rugcheck_client=rugcheck_client,
            dex_limiter=dex_limiter,
            ml_classifier=ml_classifier,
            smart_money_scorer=(
                SmartMoneyScorer(
                    moralis_client=moralis_client,
                    wallet_ranker=wallet_ranker,
                    top_n_label_threshold=SMART_MONEY_TOP_N_LABEL,
                    cluster_window_seconds=SMART_MONEY_CLUSTER_WINDOW_SECONDS,
                    debug=debug_mode,
                )
                if SMART_MONEY_ENABLED
                else None
            ),
            debug=debug_mode
        )
        
        overlap_store = AlphaOverlapStore(
            supabase_bucket=SUPABASE_BUCKET, debug=debug_mode
        )
        
        monitor = WinnerMonitor(
            http_session=http_session,
            moralis_client=moralis_client,
            wallet_ranker=wallet_ranker,
            wallet_scheduler=wallet_scheduler,
            token_analyzer=token_analyzer,
            overlap_store=overlap_store,
            poll_interval_seconds=WINNER_POLL_INTERVAL_SECONDS,
            top_n_wallets=WINNER_TOP_N_WALLETS,
            monitoring_window_hours=TOKEN_MONITORING_WINDOW_HOURS,
            debug=debug_mode
        )
        
        await monitor.startup()
        await monitor.run()

if __name__ == "__main__":
    print("--- 🚀 Starting Winner Monitor ---")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Winner Monitor interrupted, exiting gracefully")
    except RuntimeError as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print("--- 💀 Winner Monitor Halted ---")
    except Exception as e:
        print(f"\n❌ UNHANDLED EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        print("--- 💀 Winner Monitor Halted ---")