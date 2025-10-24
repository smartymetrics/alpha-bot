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
from dotenv import load_dotenv
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

# --- Local Imports ---
# Assuming these are correct from your environment
from shared.moralis_client import MoralisClient
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

load_dotenv()

# --- Environment Config ---
MORALIS_API_KEYS = [
    k.strip() for k in os.getenv("MORALIS_API_KEY", "").split(",") if k.strip()
]
WINNER_POLL_INTERVAL_SECONDS = int(os.getenv("WINNER_POLL_INTERVAL_SECONDS", "1800"))
WINNER_WALLET_COOLDOWN_SECONDS = int(os.getenv("WINNER_WALLET_COOLDOWN_SECONDS", "21600"))
WINNER_TOP_N_WALLETS = int(os.getenv("WINNER_TOP_N_WALLETS", "70"))
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "monitor-data")

# --- Probation Config ---
PROBATION_TOP_N = int(os.getenv("PROBATION_TOP_N", "3"))
PROBATION_THRESHOLD_PCT = float(os.getenv("PROBATION_THRESHOLD_PCT", "40"))

# --- Monitoring Window Config ---
TOKEN_MONITORING_WINDOW_HOURS = int(os.getenv("TOKEN_MONITORING_WINDOW_HOURS", "24"))

# --- MINIMUM SECURITY REQUIREMENTS (Reinforced to match user request) ---
MIN_HOLDER_COUNT = 500
MIN_LIQUIDITY_USD = 30000.0
MIN_LP_LOCKED_PCT = 95.0
MAX_TRANSFER_FEE_PCT = 5
MAX_CREATOR_BALANCE = 0.0 # Must be exactly 0

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


def prune_old_overlap_entries(data: dict, expiry_hours: int = 24) -> dict:
    """
    Robust pruning of overlap entries older than expiry_hours
    or with timestamps far in the future (handles clock skew).
    """
    if not data:
        return {}
        
    # --- MODIFIED: Requirement 1 ---
    # Define both a past cutoff and a future cutoff
    now = datetime.now(timezone.utc)
    cutoff_past = now - timedelta(hours=expiry_hours)
    # Prune entries with timestamps more than 1 day in the future
    cutoff_future = now + timedelta(days=1)
    # --- End Modification ---
    
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
                # Keep entries without a recognizable timestamp
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
                    # Keep unrecognizable timestamp formats
                    new_entries.append(entry)
                    continue
                
                # --- MODIFIED: Requirement 1 ---
                # Keep entry only if it's within the valid time window
                # (i.e., not expired AND not in the future)
                if ts and (ts > cutoff_past and ts < cutoff_future):
                    new_entries.append(entry)
                # --- End Modification ---
                
            except Exception:
                # Keep entries with parsing errors
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
        debug: bool = False
    ):
        self.http_session = http_session
        self.holder_agg = holder_agg
        self.dune_cache = dune_cache
        self.helius_limiter = helius_limiter
        self.rugcheck_client = rugcheck_client
        self.debug = debug
        
        if self.debug:
            print("[TokenAnalyzer] Initialized.")

    async def _run_rugcheck_check(self, mint: str) -> Dict[str, Any]:
        """Check token security using RugCheck API."""
        result = await self.rugcheck_client.get_token_report(mint)
        
        if not result.get("ok"):
            if self.debug:
                print(f"[TokenAnalyzer] ⚠️ RugCheck failed for {mint}: {result.get('error')}")
            return result
        
        data = result.get("data", {})
        
        probation_result = evaluate_probation_from_rugcheck(
            data, top_n=PROBATION_TOP_N, threshold_pct=PROBATION_THRESHOLD_PCT
        )
        
        top_n_pct = probation_result.get("top_n_pct", 0)
        if top_n_pct > 150.0:
            if self.debug:
                print(f"[RugCheck] ⚠️ {mint} - SUSPICIOUS DATA: Top {probation_result.get('top_n')} holders = {top_n_pct:.2f}%")
        
        elif self.debug and probation_result.get("probation"):
            print(f"[RugCheck] {mint} - Probation triggered!")
            print(f"[RugCheck] {mint} - {probation_result.get('explanation')}")
        
        top_holders = data.get("topHolders", [])
        rugged = data.get("rugged", False)
        
        freeze_authority = data.get("freezeAuthority")
        mint_authority = data.get("mintAuthority")
        has_authorities = bool(freeze_authority or mint_authority)
        creator_balance = data.get("creatorBalance", 0)
        transfer_fee_pct = data.get("transferFee", {}).get("pct", 0)
        total_holders = data.get("totalHolders", 0)
        
        top1_holder_pct = top_holders[0].get("pct", 0) if top_holders else 0.0

        markets = data.get("markets", [])
        total_lp_locked_usd = 0.0
        total_lp_usd = 0.0

        for market in markets:
            lp_data = market.get("lp", {})
            if lp_data:
                # Ensure the values are treated as numbers
                lp_locked_usd = float(lp_data.get("lpLockedUSD") or 0.0)
                lp_unlocked_usd = float(lp_data.get("lpUnlocked") or 0.0)
                lp_total_usd_market = lp_locked_usd + lp_unlocked_usd
                
                # Check for locked status for this market
                market_lp_locked_pct = (lp_locked_usd / lp_total_usd_market * 100) if lp_total_usd_market > 0 else 0.0
                
                # Only count markets with sufficient lock towards totals? No, let's keep all
                total_lp_locked_usd += lp_locked_usd
                total_lp_usd += lp_total_usd_market
        
        overall_lp_locked_pct = (total_lp_locked_usd / total_lp_usd * 100) if total_lp_usd > 0 else 0.0

        return {
            "ok": True,
            "rugged": rugged,
            "total_holders": total_holders,
            "top1_holder_pct": top1_holder_pct,
            "creator_balance": creator_balance,
            "freeze_authority": freeze_authority,
            "mint_authority": mint_authority,
            "has_authorities": has_authorities,
            "transfer_fee_pct": transfer_fee_pct,
            "total_lp_usd": total_lp_usd,
            "overall_lp_locked_pct": overall_lp_locked_pct,
            "lp_lock_sufficient": overall_lp_locked_pct >= MIN_LP_LOCKED_PCT,
            "probation": probation_result["probation"],
            "probation_meta": probation_result,
            "raw": data
        }

    async def analyze_token(self, mint: str) -> Dict[str, Any]:
        """
        Comprehensive token analysis.
        If RugCheck fails OR probation=True OR security requirements fail, returns immediately without checking overlap.
        Failed tokens enter 24h monitoring loop and are NOT uploaded to Supabase.
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
        
        # 2. CHECK RUGCHECK API SUCCESS - RETURN IMMEDIATELY IF FAILED
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
                "needs_monitoring": True, # Keep in memory for recheck
                "skip_reason": "rugcheck_api_failed"
            }
        
        # 3. CHECK RUGGED STATUS
        if security_report["rugcheck"].get("rugged"):
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
                "needs_monitoring": True, # Keep in memory for recheck
                "skip_reason": "rugged"
            }
        
        # 4. CHECK PROBATION STATUS
        if security_report["probation"]:
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
                "needs_monitoring": True, # Keep in memory for recheck
                "skip_reason": "probation"
            }
        
        # 5. CHECK MINIMUM SECURITY REQUIREMENTS (User-Requested Filtering)
        rugcheck_details = security_report.get("rugcheck", {})
        security_failures = []
        
        # Check for active authorities (Mint/Freeze)
        if rugcheck_details.get("has_authorities"):
            security_failures.append("has_active_authorities")
        
        # Check for creator balance (must be zero)
        if rugcheck_details.get("creator_balance", 0) > MAX_CREATOR_BALANCE:
            security_failures.append(f"creator_balance:{rugcheck_details.get('creator_balance')}_req_{MAX_CREATOR_BALANCE}")
        
        # Check transfer fee (must be low)
        if rugcheck_details.get("transfer_fee_pct", 0) > MAX_TRANSFER_FEE_PCT:
            security_failures.append(f"transfer_fee:{rugcheck_details.get('transfer_fee_pct')}%_req_<{MAX_TRANSFER_FEE_PCT}%")
        
        # Check holder count
        if rugcheck_details.get("holder_count", 0) < MIN_HOLDER_COUNT:
            security_failures.append(f"holder_count:{rugcheck_details.get('holder_count')}_req_{MIN_HOLDER_COUNT}")
             
        # Check LP lock percentage
        if rugcheck_details.get("lp_locked_pct", 0) < MIN_LP_LOCKED_PCT:
            security_failures.append(f"lp_locked:{rugcheck_details.get('lp_locked_pct')}%_req_{MIN_LP_LOCKED_PCT}%")
             
        # Check total liquidity USD
        if rugcheck_details.get("total_liquidity_usd", 0) < MIN_LIQUIDITY_USD:
            security_failures.append(f"liquidity_usd:{rugcheck_details.get('total_liquidity_usd'):.2f}_req_{MIN_LIQUIDITY_USD:.0f}")
        
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
                "needs_monitoring": True, # Keep in memory for recheck
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
                "needs_monitoring": True, # Should retry holder fetch
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
                "needs_monitoring": True, # Keep monitoring in case of delayed data
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
                "needs_monitoring": True, # Should retry later
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
        
        # 9. FINAL CHECK: ONLY UPLOAD IF OVERLAP IS FOUND (User requirement: only clean/quality tokens)
        final_needs_monitoring = False
        final_skip_reason = None
        
        if overlap_count == 0:
            # If it passed all security but has no overlap, monitor it in case winners buy later.
            final_needs_monitoring = True
            final_skip_reason = "zero_overlap_after_security_pass"
            if self.debug:
                 print(f"[TokenAnalyzer] ⚠️ {mint} passed security but ZERO overlap - setting to monitoring.")
            
        
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
            "needs_monitoring": final_needs_monitoring, # Only False if overlap > 0
            "skip_reason": final_skip_reason
        }
        
        if self.debug and not final_needs_monitoring:
            print(f"[TokenAnalyzer] ✅ {mint} -> Grade: {grade}, Overlap: {overlap_count}, CLEARED for upload")
            
        return result

    async def batch_analyze_tokens(self, mints: List[str]) -> List[Dict[str, Any]]:
        """Process multiple tokens with semaphore-controlled concurrency."""
        if not mints:
            return []
            
        tasks = [asyncio.create_task(self.analyze_token(m)) for m in mints]
        results: List[Dict[str, Any]] = []
        
        api_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, res in enumerate(api_results):
            if isinstance(res, Exception):
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
        monitoring_window_hours: int = 24,
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
            
            # Build entry for storage
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "result": overlap_result,
                "check_type": check_type,
                "security": "passed"  # This is the "passed" status checked by Requirement 2
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

                        results = await self.token_analyzer.batch_analyze_tokens(new_mints_to_check)
                        
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
        """Background task: Recheck all tracked tokens every 1 hour."""
        while True:
            await asyncio.sleep(3600)
            
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
            print(f"[Monitoring] 🔍 Started 24h monitoring for {mint} [Grade: {grade}]; reasons={self.pending_tokens[mint]['reasons']}")

    async def _monitoring_recheck_loop(
        self, mint: str
    ):
        """Recheck token every hour for 24 hours (applies to ALL grades including NONE) that failed the gate."""
        entry = self.pending_tokens.get(mint)
        if not entry:
            return
            
        first_seen_ts = entry["first_seen_ts"]
        deadline = first_seen_ts + self.monitoring_window_hours * 3600
        interval = 3600  # 1 hour

        while True:
            now_ts = int(datetime.now(timezone.utc).timestamp())
            if now_ts >= deadline:
                if self.debug:
                    final_grade = entry.get("grade", "NONE")
                    final_reasons = ", ".join(entry.get("reasons", ["no_reason"]))
                    print(f"[Monitoring] ⏰ {mint} completed 24h monitoring [Final Grade: {final_grade}, Reasons: {final_reasons}] -> NOT uploading to Supabase.")
                
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
        
        moralis_client = MoralisClient(http_session, MORALIS_API_KEYS, debug=debug_mode)
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
        
        rugcheck_client = RugCheckClient(
            session=http_session,
            rate_limiter=rugcheck_limiter,
            max_retries=3,
            debug=debug_mode
        )

        token_analyzer = AlphaTokenAnalyzer(
            http_session=http_session,
            holder_agg=holder_agg,
            dune_cache=dune_cache,
            helius_limiter=helius_limiter,
            rugcheck_client=rugcheck_client,
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