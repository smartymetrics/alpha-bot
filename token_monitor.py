#!/usr/bin/env python3
"""
Enhanced token monitor with persistent scheduling and hierarchical overlap scoring.
Now using CoinGecko Pro API for new token discovery.

--- REFACTOR NOTES ---
- This script is now structured like winner_monitor.py to reduce Helius API calls.
- Helius (HolderAggregator) is now ONLY called AFTER a token passes
  all RugCheck and minimum security requirements.
- The core logic is consolidated into `run_token_analysis_step`.
- Removed functions: _security_gate_and_route, _schedule_first_check_only, _schedule_repeat_check_only.
- Renamed `check_holders_overlap` to `_fetch_and_calculate_overlap` for clarity.

*** MODIFIED (RPC FALLBACK) ***
- Implemented `_fetch_rpc_security_report` to handle RugCheck 400/429 errors.
- Uses Helius RPC to manually check Authorities, Supply, and Creator Balance.
- Uses DexScreener to check Liquidity during fallback.
"""
import asyncio
import aiohttp
import joblib
import os
import time
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import pandas as pd
import traceback
import math
from supabase_utils import upload_overlap_results
from supabase_utils import upload_file
import threading
from dotenv import load_dotenv

import random

from typing import List, Set

import sqlite3
import threading

from ml_predictor import SolanaTokenPredictor

load_dotenv()

PROBATION_TOP_N = int(os.getenv("PROBATION_TOP_N", "3"))
PROBATION_THRESHOLD_PCT = float(os.getenv("PROBATION_THRESHOLD_PCT", "40"))
ML_PREDICTION_THRESHOLD = float(os.getenv("ML_PREDICTION_THRESHOLD", "0.50"))
ML_ACTION_THRESHOLD = float(os.getenv("ML_ACTION_THRESHOLD", "0.70"))
MAX_CREATOR_PCT = 20

COINGECKO_PRO_API_KEY = os.environ.get("GECKO_API")
DUNE_API_KEY = os.environ.get("DUNE_API_KEY")
DUNE_QUERY_ID = int(os.environ.get("DUNE_QUERY_ID"))

SHYFT_API_KEY=os.getenv("SHYFT_API_KEY")

def _load_keys_list(env_name: str) -> List[str]:
    """Load comma-separated API keys from environment variable"""
    value = os.getenv(env_name, "")
    if not value:
        return []
    return [k.strip() for k in value.split(",") if k.strip()]

try:
    from dune_client.client import DuneClient
except Exception:
    DuneClient = None

# Load Helius keys
HELIUS_KEYS = _load_keys_list("HELIUS_API_KEY")
if not HELIUS_KEYS:
    raise RuntimeError("No Helius API keys found in HELIUS_KEYS or HELIUS_API_KEY")

_helius_idx = 0
_bad_helius_keys: Set[str] = set()

def _next_helius_key() -> str:
    """Round-robin through valid Helius keys"""
    global _helius_idx
    valid = [k for k in HELIUS_KEYS if k not in _bad_helius_keys]
    if not valid:
        _bad_helius_keys.clear()  # Reset if all keys exhausted
        valid = HELIUS_KEYS
    key = valid[_helius_idx % len(valid)]
    _helius_idx += 1
    return key

async def retry_with_backoff(func, *args, retries: int = 5, base_delay: float = 0.5, **kwargs):
    """
    Retry an async function with exponential backoff and jitter.
    Now handles 429 rate limiting with Retry-After header support.
    """
    for attempt in range(1, retries + 1):
        try:
            return await func(*args, **kwargs)
            
        except aiohttp.ClientResponseError as e:
            # Special handling for 429 rate limiting
            if e.status == 429:
                retry_after = None
                
                # Try to parse Retry-After header
                if e.headers and 'Retry-After' in e.headers:
                    try:
                        retry_after = int(e.headers['Retry-After'])
                    except (ValueError, TypeError):
                        pass
                
                # Determine wait time
                if retry_after:
                    wait = retry_after
                    print(f"[Retry] {func.__name__} rate limited (attempt {attempt}/{retries}). "
                          f"Retry-After header: {retry_after}s")
                else:
                    # Use exponential backoff without jitter for rate limits
                    wait = base_delay * (2 ** (attempt - 1))
                    print(f"[Retry] {func.__name__} rate limited (attempt {attempt}/{retries}). "
                          f"Using exponential backoff: {wait:.2f}s")
                
                # If we haven't exhausted retries, wait and continue
                if attempt < retries:
                    await asyncio.sleep(wait)
                    continue
                else:
                    # Max retries exceeded for rate limiting
                    raise RuntimeError(
                        f"{func.__name__} failed after {retries} retries due to rate limiting (HTTP 429)"
                    )
            else:
                # For non-429 ClientResponseErrors, re-raise immediately
                raise
                
        except Exception as e:
            # For other exceptions, use exponential backoff with jitter
            wait = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.3)
            print(f"[Retry] {func.__name__} failed (attempt {attempt}/{retries}): {e}. "
                  f"Retrying in {wait:.2f}s")
            
            if attempt < retries:
                await asyncio.sleep(wait)
            else:
                # Max retries exceeded for general errors
                raise RuntimeError(f"{func.__name__} failed after {retries} retries")

BUCKET_NAME = os.getenv("SUPABASE_BUCKET", "monitor-data")

# -----------------------
# Sanitizer utilities
# -----------------------
def _sanitize_dict(d: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Recursively sanitize a dictionary so all keys are strings (replace None with "null"),
    and nested dicts/lists are processed. This prevents joblib/pickle errors like
    "Cannot serialize non-str key None".
    """
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
    """Sanitize objects that may contain dicts/lists/TradingStart."""
    if isinstance(obj, dict):
        return _sanitize_dict(obj)
    if isinstance(obj, list):
        return [_sanitize_maybe(x) for x in obj]
    if isinstance(obj, TradingStart):
        d = asdict(obj)
        d["extra"] = _sanitize_dict(d.get("extra") or {})
        return d
    return obj

# -----------------------
# Solana RPC client
# -----------------------
class SolanaAlphaClient:
    def __init__(self):
        self.headers = {"Content-Type": "application/json"}

    def _get_url(self):
        """Get URL with next available key"""
        key = _next_helius_key()
        return f"https://mainnet.helius-rpc.com/?api-key={key}"

    async def make_rpc_call(self, method: str, params: List[Any]) -> Dict[str, Any]:
        payload = {"jsonrpc": "2.0", "id": "1", "method": method, "params": params}
        
        max_attempts = min(len(HELIUS_KEYS) * 2, 10)
        
        for attempt in range(max_attempts):
            url = self._get_url()
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(url, json=payload, headers=self.headers, timeout=40) as resp:
                        if resp.status == 429:
                            print(f"[Helius] Rate limited, rotating key (attempt {attempt+1})")
                            await asyncio.sleep(1)
                            continue
                        if resp.status == 401:
                            current_key = url.split("api-key=")[-1]
                            _bad_helius_keys.add(current_key)
                            print(f"[Helius] Key unauthorized, blacklisting and rotating")
                            continue
                        resp.raise_for_status()
                        return await resp.json()
                except asyncio.TimeoutError:
                    print(f"[Helius] Timeout on attempt {attempt+1}, rotating key")
                    await asyncio.sleep(0.5)
                    continue
                except Exception as e:
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(0.5)
                        continue
                    return {"error": str(e)}
        
        return {"error": "All Helius keys exhausted or rate limited"}

    async def test_connection(self) -> bool:
        r = await self.make_rpc_call("getHealth", [])
        return r.get("result") == "ok"

# -----------------------
# Domain objects
# -----------------------
@dataclass
class TradingStart:
    mint: Optional[str] = None
    block_time: Optional[int] = None
    program_id: Optional[str] = None
    detected_via: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None
    fdv_usd: Optional[float] = None
    volume_usd: Optional[float] = None
    source_dex: Optional[str] = None
    price_change_percentage: Optional[float] = None

# -----------------------
# Token discovery (CoinGecko + Rugcheck + FULLY ASYNC Dune)
# -----------------------
class TokenDiscovery:
    def __init__(
        self,
        client: Optional[Any] = None,
        *,
        coingecko_pro_api_key: Optional[str] = None,
        dune_api_key: Optional[str] = None,
        dune_query_id: Optional[int] = None,
        dune_cache_file: str = "./data/dune_recent.pkl",
        timestamp_cache_file: str = "./data/last_timestamp.pkl",
        debug: bool = False,
    ):
        self.client = client
        self.debug = bool(debug)
        
        # --- CoinGecko / GeckoTerminal Config ---
        # Switch from Pro to Public GeckoTerminal API
        self.coingecko_pro_api_key = coingecko_pro_api_key or os.environ.get("GECKO_API")
        self.geckoterminal_url = "https://api.geckoterminal.com/api/v2/networks/solana/new_pools"
        
        # --- BirdEye Config ---
        self.birdeye_keys = _load_keys_list("BIRDEYE_API_KEY")
        self.birdeye_url = "https://public-api.birdeye.so/defi/v2/tokens/new_listing"
        self._birdeye_idx = 0
        self._bad_birdeye_keys: Set[str] = set()
        self.last_birdeye_call = 0
        
        # Throttling for BirdEye: 30,000 CUs per month per key.
        # User analysis suggests up to 100 CUs per call for new listings.
        # If N keys, max requests = (N * 30000) / 100 = N * 300 requests/month.
        # Interval (seconds) = (30 days * 24h * 3600s) / (N * 300).
        # Which simplifies to 8,640 / N_KEYS.
        num_keys = len(self.birdeye_keys) or 1
        self.birdeye_interval = (30 * 24 * 3600 * 100) / (num_keys * 30000)
        
        # --- DexScreener Config ---
        self.dexscreener_profiles_url = "https://api.dexscreener.com/token-profiles/latest/v1"
        self.dexscreener_boosts_url = "https://api.dexscreener.com/token-boosts/latest/v1"
        self.last_dexscreener_boost_call = 0
        self.dexscreener_boost_interval = 300 # 5 minutes
        
        # --- RugCheck Config ---
        self.rugcheck_new_url = "https://api.rugcheck.xyz/v1/stats/new_tokens"
        
        # --- State / Cache ---
        self.last_processed_timestamp = self._load_last_timestamp(timestamp_cache_file)
        self.timestamp_cache_file = timestamp_cache_file
        
        # --- Dune Config ---
        self.dune_api_key = dune_api_key or os.environ.get("DUNE_API_KEY")
        self.dune_query_id = dune_query_id
        if DuneClient and self.dune_api_key:
            try:
                self.dune_client = DuneClient(self.dune_api_key)
            except Exception:
                self.dune_client = None
        else:
            self.dune_client = None
        self.dune_cache_file = dune_cache_file
        
        if self.debug:
            print(f"TokenDiscovery initialized with Multi-Source: GT Public, BirdEye ({num_keys} keys), DexScreener, RugCheck, and ASYNC Dune")

    def _load_last_timestamp(self, cache_file: str) -> Optional[int]:
        if os.path.exists(cache_file):
            try:
                return joblib.load(cache_file)
            except Exception:
                pass
        return None

    def _save_last_timestamp(self):
        try:
            joblib.dump(self.last_processed_timestamp, self.timestamp_cache_file)
        except Exception as e:
            if self.debug:
                print(f"Failed to save last timestamp: {e}")

    # ---------------- Dune helpers ----------------
    def _rows_from_dune_payload(self, payload: Any) -> List[Dict[str, Any]]:
        if payload is None:
            return []
        if hasattr(payload, "result"):
            try:
                r = getattr(payload, "result")
                if isinstance(r, dict) and "rows" in r and isinstance(r["rows"], list):
                    return r["rows"]
                if hasattr(r, "rows"):
                    return list(getattr(r, "rows") or [])
            except Exception:
                pass
        if isinstance(payload, dict):
            if "result" in payload and isinstance(payload["result"], dict) and "rows" in payload["result"]:
                return payload["result"]["rows"]
            if "rows" in payload and isinstance(payload["rows"], list):
                return payload["rows"]
            if "data" in payload and isinstance(payload["data"], list):
                return payload["data"]
        if isinstance(payload, list):
            return payload
        if hasattr(payload, "rows"):
            r = getattr(payload, "rows")
            if isinstance(r, list):
                return r
        return []

    def fetch_dune_latest_rows(self) -> List[Dict[str, Any]]:
        """Synchronous wrapper for Dune results."""
        if not self.dune_client or not self.dune_query_id:
            raise RuntimeError("Dune client or query_id not configured")
        if self.debug:
            print(f"[Dune] fetching latest result for query {self.dune_query_id}")
        payload = self.dune_client.get_latest_result(self.dune_query_id)
        rows = self._rows_from_dune_payload(payload)
        if self.debug:
            print(f"[Dune] extracted {len(rows)} rows")
        return rows

    async def fetch_dune_force_refresh(self) -> List[Dict[str, Any]]:
            """
            Fully async Dune refresh using the new API endpoints (v1).
            1. POST /execute -> Get execution_id
            2. GET /results -> Poll until QUERY_STATE_COMPLETED
            """
            if not self.dune_api_key or not self.dune_query_id:
                raise RuntimeError("Dune credentials or query_id not configured")

            base_url = "https://api.dune.com/api/v1"
            headers = {"X-DUNE-API-KEY": self.dune_api_key}
            
            # We use a localized session for this distinct operation
            async with aiohttp.ClientSession() as session:
                # === STEP 1: Execute Query ===
                exec_url = f"{base_url}/query/{self.dune_query_id}/execute"
                if self.debug:
                    print(f"[Dune API] 🚀 POST executing query {self.dune_query_id}...")

                try:
                    async with session.post(exec_url, headers=headers) as resp:
                        if resp.status == 429:
                            raise RuntimeError("Dune API Rate Limited (POST execute)")
                        if resp.status != 200:
                            text = await resp.text()
                            raise RuntimeError(f"Dune Execute failed {resp.status}: {text}")
                        
                        data = await resp.json()
                        execution_id = data.get("execution_id")
                        state = data.get("state")
                        
                        if not execution_id:
                            raise RuntimeError(f"No execution_id returned: {data}")
                        
                        if self.debug:
                            print(f"[Dune API] Execution ID: {execution_id} (State: {state})")

                except Exception as e:
                    if self.debug:
                        print(f"[Dune API] ❌ Execute failed: {e}")
                    return []

                # === STEP 2: Poll for Results ===
                results_url = f"{base_url}/execution/{execution_id}/results"
                attempt = 0
                
                while True:
                    attempt += 1
                    # Wait before polling (backoff slightly)
                    await asyncio.sleep(min(3, 1 + (attempt * 0.2)))
                    
                    try:
                        async with session.get(results_url, headers=headers) as resp:
                            if resp.status == 429:
                                if self.debug:
                                    print("[Dune API] Rate limit polling results, waiting 5s...")
                                await asyncio.sleep(5)
                                continue
                                
                            if resp.status != 200:
                                # 202 is sometimes used for pending, but Dune v1 usually returns 200 with state
                                text = await resp.text()
                                raise RuntimeError(f"Dune Results failed {resp.status}: {text}")
                                
                            data = await resp.json()
                            state = data.get("state")
                            
                            if state == "QUERY_STATE_COMPLETED":
                                rows = data.get("result", {}).get("rows", [])
                                if self.debug:
                                    print(f"[Dune API] ✅ Query completed! Returned {len(rows)} rows.")
                                return rows
                                
                            elif state in ["QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED", "QUERY_STATE_EXPIRED"]:
                                raise RuntimeError(f"Query failed with state: {state}")
                                
                            else:
                                # QUERY_STATE_PENDING, QUERY_STATE_EXECUTING
                                if self.debug and attempt % 5 == 0:
                                    print(f"[Dune API] Status: {state} (poll #{attempt})...")
                                    
                    except Exception as e:
                        if self.debug:
                            print(f"[Dune API] ⚠️ Polling error: {e}")
                        # If it's a transient network error, we might want to continue, 
                        # but for now we break to avoid infinite loops on hard errors
                        break
            
            return []

    async def get_tokens_launched_yesterday_cached(self, cache_max_age_days: int = 7) -> List[TradingStart]:
        """Async retrieval of Dune tokens for 'yesterday' with local caching."""
        cache_path = self.dune_cache_file
        
        # Helper to convert raw Dune rows to TradingStart objects
        def rows_to_trading_starts(rows: List[Dict[str, Any]], target_yesterday: datetime.date) -> List[TradingStart]:
            if not rows: return []
            df = pd.DataFrame(rows)
            
            # Identify columns
            date_col = next((c for c in ("first_buy_date", "first_buy_date_utc", "block_date", "first_trade_date") if c in df.columns), None)
            mint_col = next((c for c in ("mint_address", "mint", "token_bought_mint_address") if c in df.columns), None)
            
            if not date_col or not mint_col: return []
            
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            filtered = df[df[date_col].dt.date == target_yesterday]
            out = []
            for _, row in filtered.iterrows():
                try:
                    dt = pd.to_datetime(row[date_col])
                    if pd.isna(dt): continue
                    if dt.tzinfo is None: dt = dt.tz_localize("UTC")
                    ts = int(dt.tz_convert("UTC").timestamp())
                except Exception:
                    continue
                out.append(TradingStart(
                    mint=row[mint_col], 
                    block_time=ts, 
                    program_id="dune", 
                    detected_via="dune", 
                    extra={date_col: str(row[date_col])}
                ))
            return out

        current_yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1))
        need_fetch = True

        # Check existing cache
        if os.path.exists(cache_path):
            try:
                cache_obj = joblib.load(cache_path)
                cached_rows = cache_obj.get("rows", [])
                fetched_at = None
                try:
                    fetched_at = datetime.fromisoformat(cache_obj["fetched_at"])
                    if fetched_at.tzinfo is None: fetched_at = fetched_at.replace(tzinfo=timezone.utc)
                except Exception: fetched_at = None

                if cached_rows and fetched_at:
                    df = pd.DataFrame(cached_rows)
                    if "first_buy_date" in df.columns:
                        try:
                            first_buy = pd.to_datetime(df["first_buy_date"].iloc[0]).date()
                            if first_buy == current_yesterday:
                                age_days = (datetime.now(timezone.utc) - fetched_at).days
                                if age_days <= cache_max_age_days:
                                    starts = rows_to_trading_starts(cached_rows, current_yesterday)
                                    if starts:
                                        if self.debug:
                                            print(f"[Dune/cache] using valid cache for {current_yesterday}")
                                        need_fetch = False
                                        return starts
                        except Exception: pass
            except Exception: pass

        if need_fetch:
            try:
                rows = await self.fetch_dune_force_refresh()
                if rows:
                    joblib.dump({
                        "rows": rows,
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                        "target_yesterday": current_yesterday.isoformat()
                    }, cache_path)
                    return rows_to_trading_starts(rows, current_yesterday)
            except Exception as e:
                if self.debug: print(f"[Dune ASYNC] ❌ fetch failure: {e}")
        
        return []

    # ---------------- GeckoTerminal Logic (Verified) ----------------
    async def _fetch_geckoterminal_new_pools(self, limit: int = 500, timeout: int = 30) -> List[Dict[str, Any]]:
        """Fetch raw pool data from GeckoTerminal Public API."""
        params = {
            "include": "base_token,quote_token",
            "include_gt_community_data": "true",
            "page": "1" # Public tier pagination is limited
        }
        headers = {"accept": "application/json"}
        
        # If we have a Pro key, we could still use it via headers if supported by the public endpoint
        if self.coingecko_pro_api_key:
            headers["x-cg-pro-api-key"] = self.coingecko_pro_api_key

        if self.debug:
            print("[GeckoTerminal] Fetching latest pools...")

        async with aiohttp.ClientSession() as sess:
            async with sess.get(self.geckoterminal_url, headers=headers, params=params, timeout=timeout) as resp:
                if resp.status != 200:
                    if self.debug:
                        print(f"[GeckoTerminal] Failed: Status {resp.status}")
                    return []
                data = await resp.json()
                return data.get("data", [])

    @staticmethod
    def _parse_iso_timestamp(val: Any) -> Optional[int]:
        if not val: return None
        try:
            # Handle formats like 2026-01-15T13:28:28Z or 2026-01-15T13:50:45
            ts_str = str(val).replace("Z", "+00:00")
            if "T" in ts_str and len(ts_str) == 19: # 2026-01-15T13:50:45
                ts_str += "+00:00"
            dt = datetime.fromisoformat(ts_str)
            if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except Exception: return None

    def _parse_geckoterminal_pool(self, pool: Dict[str, Any]) -> TradingStart:
        attr = pool["attributes"]
        base_token = pool["relationships"]["base_token"]["data"]
        mint = base_token["id"].replace("eth_", "").replace("solana_", "")
        block_time = self._parse_iso_timestamp(attr["pool_created_at"])
        
        # Parse volume - it's nested in verified GT response
        vol_h24 = 0.0
        vol_data = attr.get("volume_usd")
        if isinstance(vol_data, dict):
            vol_h24 = float(vol_data.get("h24") or 0.0)
        elif isinstance(vol_data, (int, float, str)):
            vol_h24 = float(vol_data)

        return TradingStart(
            mint=mint,
            block_time=block_time,
            program_id="geckoterminal",
            detected_via="geckoterminal",
            extra={
                "name": attr["name"].split(" / ")[0],
                "fdv_usd": float(attr.get("fdv_usd") or 0.0),
                "market_cap_usd": float(attr.get("market_cap_usd") or attr.get("fdv_usd") or 0.0),
                "volume_usd": vol_h24,
                "source_dex": pool["relationships"]["dex"]["data"]["id"],
                "price_change_percentage": float(attr.get("price_change_percentage", {}).get("h24") or 0.0),
            },
            fdv_usd=float(attr.get("fdv_usd") or 0.0),
                volume_usd=vol_h24,
            source_dex=pool["relationships"]["dex"]["data"]["id"],
            price_change_percentage=float(attr.get("price_change_percentage", {}).get("h24") or 0.0),
        )

    # ---------------- BirdEye Logic (Verified) ----------------
    def _next_birdeye_key(self) -> Optional[str]:
        """Round-robin through valid BirdEye keys"""
        valid = [k for k in self.birdeye_keys if k not in self._bad_birdeye_keys]
        if not valid:
            if self.birdeye_keys:
                self._bad_birdeye_keys.clear() # reset
                valid = self.birdeye_keys
            else:
                return None
        key = valid[self._birdeye_idx % len(valid)]
        self._birdeye_idx += 1
        return key

    async def _fetch_birdeye_new_tokens(self) -> List[TradingStart]:
        """Fetch the latest tokens from BirdEye API with rotation and throttling."""
        # Throttling check
        now = time.time()
        if now - self.last_birdeye_call < self.birdeye_interval:
            if self.debug:
                print(f"[BirdEye] Skipping (throttling: {now - self.last_birdeye_call:.1f}s < {self.birdeye_interval:.1f}s)")
            return []
            
        key = self._next_birdeye_key()
        if not key: return []
        
        headers = {
            "accept": "application/json",
            "x-chain": "solana",
            "X-API-KEY": key
        }
        params = {"limit": "20", "meme_platform_enabled": "true"}
        
        if self.debug:
            print(f"[BirdEye] Fetching new listings (key {key[:4]}...)")

        try:
            self.last_birdeye_call = now
            async with aiohttp.ClientSession() as sess:
                async with sess.get(self.birdeye_url, headers=headers, params=params, timeout=20) as resp:
                    if resp.status == 401 or resp.status == 403:
                        self._bad_birdeye_keys.add(key)
                        return []
                    if resp.status != 200:
                        return []
                    data = await resp.json()
                    items = data.get("data", {}).get("items", [])
                    
                    out = []
                    for item in items:
                        mint = item.get("address")
                        if not mint: continue
                        
                        block_time = self._parse_iso_timestamp(item.get("liquidityAddedAt"))
                        out.append(TradingStart(
                            mint=mint,
                            block_time=block_time or int(now),
                            program_id="birdeye",
                            detected_via="birdeye",
                            extra={
                                "symbol": item.get("symbol"),
                                "name": item.get("name"),
                                "source_dex": item.get("source"),
                                "liquidity": item.get("liquidity")
                            },
                            fdv_usd=0.0, # BirdEye listed tokens often lack FDV in this endpoint
                            volume_usd=0.0,
                            source_dex=item.get("source") or "unknown",
                            price_change_percentage=0.0
                        ))
                    return out
        except Exception as e:
            if self.debug: print(f"[BirdEye] Error: {e}")
            return []

    # ---------------- DexScreener Logic ----------------
    async def _fetch_dexscreener_new_tokens(self) -> List[TradingStart]:
        """Fetch latest tokens from DexScreener Profiles."""
        if self.debug:
            print("[DexScreener] Fetching latest profiles...")

        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(self.dexscreener_profiles_url, timeout=20) as resp:
                    if resp.status != 200: return []
                    data = await resp.json()
                    
                    # Profiles are returned as a list
                    if not isinstance(data, list): return []
                    
                    out = []
                    now_ts = int(time.time())
                    for item in data:
                        if item.get("chainId") != "solana": continue
                        mint = item.get("tokenAddress")
                        if not mint: continue
                        
                        out.append(TradingStart(
                            mint=mint,
                            block_time=now_ts, # DexScreener profiles are very fresh
                            program_id="dexscreener_profile",
                            detected_via="dexscreener",
                            extra={
                                "name": item.get("tokenName"), # NEW
                                "symbol": item.get("tokenSymbol"), # NEW
                                "description": item.get("description"),
                                "url": item.get("url"),
                                "icon": item.get("icon") # NEW
                            },
                            fdv_usd=0.0,
                            volume_usd=0.0,
                            source_dex="dexscreener",
                            price_change_percentage=0.0
                        ))
                    return out
        except Exception as e:
            if self.debug: print(f"[DexScreener Profile] Error: {e}")
            return []

    async def _fetch_dexscreener_boosted_tokens(self) -> List[TradingStart]:
        """Fetch latest boosted tokens from DexScreener."""
        now = time.time()
        if now - self.last_dexscreener_boost_call < self.dexscreener_boost_interval:
            if self.debug:
                print(f"[DexScreener Boosts] Skipping (throttling: {now - self.last_dexscreener_boost_call:.1f}s < {self.dexscreener_boost_interval:.1f}s)")
            return []

        if self.debug:
            print("[DexScreener Boosts] Fetching latest boosts...")

        try:
            self.last_dexscreener_boost_call = now
            async with aiohttp.ClientSession() as sess:
                async with sess.get(self.dexscreener_boosts_url, timeout=20) as resp:
                    if resp.status != 200: return []
                    data = await resp.json()
                    
                    if not isinstance(data, list): return []
                    
                    out = []
                    now_ts = int(now)
                    for item in data:
                        if item.get("chainId") != "solana": continue
                        mint = item.get("tokenAddress")
                        if not mint: continue
                        
                        out.append(TradingStart(
                            mint=mint,
                            block_time=now_ts,
                            program_id="dexscreener_boost",
                            detected_via="dexscreener_boost",
                            extra={
                                "url": item.get("url"),
                                "description": item.get("description"),
                                "icon": item.get("icon"),
                                "header": item.get("header"),
                                "links": item.get("links"),
                                "totalAmount": item.get("totalAmount"),
                                "amount": item.get("amount")
                            },
                            fdv_usd=0.0,
                            volume_usd=0.0,
                            source_dex="dexscreener",
                            price_change_percentage=0.0
                        ))
                    return out
        except Exception as e:
            if self.debug: print(f"[DexScreener Boosts] Error: {e}")
            return []

    # ---------------- RugCheck Logic (Primary Source) ----------------
    async def _fetch_rugcheck_new_tokens(self, timeout: int = 50) -> List[TradingStart]:
        """Fetch the latest tokens from RugCheck API."""
        if self.debug:
            print("[RugCheck Discovery] Fetching new tokens...")

        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(self.rugcheck_new_url, timeout=timeout) as resp:
                    if resp.status != 200:
                        if self.debug:
                            print(f"[RugCheck Discovery] Failed: Status {resp.status}")
                        return []
                    data = await resp.json()
        except Exception as e:
            if self.debug:
                print(f"[RugCheck Discovery] Error: {e}")
            return []

        if not isinstance(data, list):
            return []

        out = []
        now_ts = int(datetime.now(timezone.utc).timestamp())

        for item in data:
            mint = item.get("mint")
            if not mint:
                continue
            
            # RugCheck detections are fresh; use current time for block_time
            # to ensure the scheduler picks them up immediately.
            out.append(TradingStart(
                mint=mint,
                block_time=now_ts,
                program_id="rugcheck_new",
                detected_via="rugcheck",
                extra={
                    "symbol": item.get("symbol"),
                    "name": item.get("name"),
                    "supply": item.get("supply"),
                    "uri": item.get("uri")
                },
                # Default these to 0 as RugCheck 'new_tokens' often lack market data
                fdv_usd=0.0,
                volume_usd=0.0,
                source_dex="unknown",
                price_change_percentage=0.0,
            ))

        if self.debug and out:
            print(f"[RugCheck Discovery] Found {len(out)} tokens")
        return out

    # ---------------- Aggregated Discovery ----------------
    async def get_tokens_created_today(self, limit: int = 500) -> List[TradingStart]:
        """
        Fetches tokens from GeckoTerminal, BirdEye, DexScreener, and RugCheck in parallel.
        Merges results, prioritizing GeckoTerminal data if duplicates exist.
        """
        # 1. Define tasks for parallel execution
        gt_task = retry_with_backoff(
            self._fetch_geckoterminal_new_pools,
            limit=limit,
            retries=3,
            base_delay=2.0
        )
        
        be_task = self._fetch_birdeye_new_tokens()
        dx_task = self._fetch_dexscreener_new_tokens()
        dx_boost_task = self._fetch_dexscreener_boosted_tokens()
        
        rc_task = retry_with_backoff(
            self._fetch_rugcheck_new_tokens,
            retries=3,
            base_delay=2.0
        )

        # 2. Run concurrently
        results = await asyncio.gather(gt_task, be_task, dx_task, dx_boost_task, rc_task, return_exceptions=True)
        
        # 3. Handle Results
        gt_raw_pools = results[0] if not isinstance(results[0], Exception) else []
        be_starts = results[1] if not isinstance(results[1], Exception) else []
        dx_starts = results[2] if not isinstance(results[2], Exception) else []
        dx_boosted_starts = results[3] if not isinstance(results[3], Exception) else []
        rc_starts = results[4] if not isinstance(results[4], Exception) else []
        
        # Log failures if any
        if self.debug:
            for i, name in enumerate(["GeckoTerminal", "BirdEye", "DexScreener", "DexScreenerBoost", "RugCheck"]):
                if isinstance(results[i], Exception):
                    print(f"[TokenDiscovery] {name} failed: {results[i]}")

        # 4. Process GeckoTerminal Data
        gt_starts = []
        now = int(datetime.now(timezone.utc).timestamp())
        cutoff = now - 24 * 3600  # last 24 hours

        for pool in gt_raw_pools:
            block_time = self._parse_iso_timestamp(pool["attributes"]["pool_created_at"])
            if not block_time: continue
            if block_time < cutoff: continue

            ts = self._parse_geckoterminal_pool(pool)
            gt_starts.append(ts)

            # Update timestamp tracker
            if self.last_processed_timestamp is None or block_time > self.last_processed_timestamp:
                self.last_processed_timestamp = block_time

        if gt_starts:
            self._save_last_timestamp()

        # 5. Merge and Deduplicate (Priority order: RC < BE < DX < GT)
        unique_tokens = {}
        
        # Lower priority sources first
        for ts in rc_starts:
            unique_tokens[ts.mint] = ts
        
        for ts in be_starts:
            unique_tokens[ts.mint] = ts
            
        for ts in dx_starts:
            unique_tokens[ts.mint] = ts

        for ts in dx_boosted_starts:
            unique_tokens[ts.mint] = ts

        for ts in gt_starts:
            unique_tokens[ts.mint] = ts

        final_list = list(unique_tokens.values())

        if self.debug:
            print(f"[TokenDiscovery] Combined Total: {len(final_list)} unique tokens "
                  f"(GT: {len(gt_starts)}, BE: {len(be_starts)}, DX: {len(dx_starts)}, DX_BOOST: {len(dx_boosted_starts)}, RC: {len(rc_starts)})")

        return final_list

# -----------------------
# Dune 7-day rolling cache + builder (updated for async)
# -----------------------
class DuneWinnersCache:
    """
    Rolling, per-day cache of Dune "winner" token holders with Supabase sync.
    Files are stored locally under: ./data/dune_cache/dune_cache_YYYYMMDD.pkl
    And synced to Supabase storage bucket in dune_cache/ folder.
    """
    def __init__(self, cache_dir: str = "./data/dune_cache", debug: bool = False, supabase_bucket: str = "monitor-data"):
        self.cache_dir = cache_dir
        self.debug = debug
        self.supabase_bucket = supabase_bucket
        os.makedirs(self.cache_dir, exist_ok=True)

    def _today_key(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d")

    def _path_for(self, yyyymmdd: str) -> str:
        return os.path.join(self.cache_dir, f"dune_cache_{yyyymmdd}.pkl")

    def save_today(self, token_to_top_holders: Dict[str, List[str]]):
        """Save today's token->holders snapshot to a per-day file and upload to Supabase."""
        y = self._today_key()
        obj = {
            "token_to_top_holders": _sanitize_maybe(token_to_top_holders),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "day": y,
        }
        
        local_path = self._path_for(y)
        try:
            joblib.dump(obj, local_path)
            if self.debug:
                tot_wallets = len({w for v in token_to_top_holders.values() for w in v})
                print(f"[DuneCache] saved {len(token_to_top_holders)} tokens, ~{tot_wallets} unique wallets for {y}")
            
            # Upload to Supabase dune_cache folder
            try:
                from supabase_utils import upload_dune_cache_file
                upload_dune_cache_file(local_path, self.supabase_bucket)
                if self.debug:
                    print(f"[DuneCache] uploaded {y} to Supabase dune_cache folder")
            except Exception as e:
                if self.debug:
                    print(f"[DuneCache] Supabase upload failed for {y}: {e}")
                    
        except Exception as e:
            if self.debug:
                print(f"[DuneCache] save_today failed: {e}")

    def _download_from_supabase(self, yyyymmdd: str) -> bool:
        """Download a cache file from Supabase dune_cache folder if it doesn't exist locally."""
        local_path = self._path_for(yyyymmdd)
        if os.path.exists(local_path):
            return True
            
        try:
            from supabase_utils import download_dune_cache_file
            file_name = f"dune_cache_{yyyymmdd}.pkl"
            success = download_dune_cache_file(local_path, file_name, self.supabase_bucket)
            if success and self.debug:
                print(f"[DuneCache] downloaded {yyyymmdd} from Supabase dune_cache folder")
            return success
        except Exception as e:
            if self.debug:
                print(f"[DuneCache] download from Supabase failed for {yyyymmdd}: {e}")
            return False

    def load_last_7_days(self) -> Tuple[Dict[str, List[str]], Dict[str, int], List[str]]:
        """Load and merge per-day files for the past 7 UTC dates. Downloads from Supabase if needed."""
        days = [(datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y%m%d") for i in range(7)]
        token_to_top_holders: Dict[str, List[str]] = {}
        wallet_freq: Dict[str, int] = defaultdict(int)
        loaded_days: List[str] = []

        # Try to download missing files from Supabase first
        for d in days:
            self._download_from_supabase(d)

        # merge from oldest -> newest so newest day's holders can overwrite if needed
        for d in reversed(days):
            path = self._path_for(d)
            if not os.path.exists(path):
                continue
            try:
                obj = joblib.load(path)
                per_day = obj.get("token_to_top_holders", {})
                # per_day is mapping token -> list[wallets]
                for token, holders in per_day.items():
                    token_to_top_holders[token] = holders
                    for w in holders:
                        wallet_freq[w] += 1
                loaded_days.append(d)
            except Exception as e:
                if self.debug:
                    print(f"[DuneCache] failed to load {path}: {e}")

        # cleanup older files (both local and Supabase)
        self._cleanup_old_files(days)

        if self.debug:
            uniq_wallets = len(wallet_freq)
            print(f"[DuneCache] Loaded {len(loaded_days)} days: {len(token_to_top_holders)} tokens, {uniq_wallets} unique wallets")

        return token_to_top_holders, dict(wallet_freq), loaded_days

    def _cleanup_old_files(self, keep_days: List[str]):
        """Clean up old cache files locally and optionally from Supabase."""
        # Local cleanup
        try:
            for fname in os.listdir(self.cache_dir):
                if not fname.startswith("dune_cache_") or not fname.endswith(".pkl"):
                    continue
                ymd = fname[len("dune_cache_"):-4]
                if ymd not in keep_days:
                    try:
                        os.remove(os.path.join(self.cache_dir, fname))
                        if self.debug:
                            print(f"[DuneCache] removed old local cache file {fname}")
                    except Exception as e:
                        if self.debug:
                            print(f"[DuneCache] local cleanup failed for {fname}: {e}")
        except Exception as e:
            if self.debug:
                print(f"[DuneCache] local cleanup scan failed: {e}")

        # Note: Supabase cleanup could be implemented here if you have a list_files function
        # For now, old Supabase files will remain but won't be downloaded

    def sync_to_supabase(self):
        """Manually sync all local cache files to Supabase dune_cache folder."""
        try:
            for fname in os.listdir(self.cache_dir):
                if not fname.startswith("dune_cache_") or not fname.endswith(".pkl"):
                    continue
                    
                local_path = os.path.join(self.cache_dir, fname)
                
                try:
                    from supabase_utils import upload_dune_cache_file
                    upload_dune_cache_file(local_path, self.supabase_bucket)
                    if self.debug:
                        print(f"[DuneCache] synced {fname} to Supabase dune_cache folder")
                except Exception as e:
                    if self.debug:
                        print(f"[DuneCache] sync failed for {fname}: {e}")
                        
        except Exception as e:
            if self.debug:
                print(f"[DuneCache] sync_to_supabase failed: {e}")

class DuneWinnersBuilder:
    """
    Builds today's per-day winners cache by fetching holders for tokens reported by Dune.
    Uses bounded concurrency and the hybrid sampling rule.
    """
    def __init__(self, cache: DuneWinnersCache, debug: bool = False, max_concurrency: int = 8):
        self.cache = cache
        self.debug = debug
        self.sema = None  # Removed concurrency semaphore for sequential processing

    async def _fetch_top_sampled_holders(self, holder_agg: 'HolderAggregator', mint: str) -> List[str]:
        """Fetch holders for `mint` respecting concurrency semaphore and the hybrid sampling rule.
        Returns a list of wallet addresses (strings).
        """
        # Sequential processing; concurrency removed
        # async with self.sema:
        try:
            if self.debug:
                print(f"[DuneBuilder] Fetching holders for {mint}")
                
            holders = await holder_agg.get_token_holders(mint, limit=1000, max_pages=2, decimals=None)
                
            if self.debug:
                print(f"[DuneBuilder] Raw holders fetched for {mint}: {len(holders)}")
                
            if not holders:
                if self.debug:
                    print(f"[DuneBuilder] No holders returned for {mint}")
                    return []
                
            n = len(holders)
            if n <= 200:
                selected = holders
            else:
                top10 = max(1, int(n * 0.10))
                selected = holders[: min(500, top10)]
                
            wallets = [h.get("wallet") for h in selected if h.get("wallet")]
                
            if self.debug:
                print(f"[DuneBuilder] {mint}: {len(holders)} total -> {len(selected)} selected -> {len(wallets)} valid wallets")
                
            return wallets
                
        except Exception as e:
            if self.debug:
                print(f"[DuneBuilder] ERROR fetching holders for {mint}: {e}")
                import traceback
                traceback.print_exc()
            return []

    async def build_today_from_dune(self, token_discovery: TokenDiscovery, holder_agg: 'HolderAggregator') -> Dict[str, List[str]]:
        """
        🚀 NOW ASYNC: Fetch tokens from Dune (yesterday), fetch sampled holders concurrently, save today's per-day cache,
        and return the token->holders mapping for today.
        """
        if self.debug:
            print(f"[DuneBuilder ASYNC] 🚀 Starting build_today_from_dune")
            
        # 🚀 ASYNC: Get tokens from yesterday's Dune data
        starts = await token_discovery.get_tokens_launched_yesterday_cached()
        if self.debug:
            print(f"[DuneBuilder ASYNC] 🚀 Dune returned {len(starts)} tokens for yesterday")
            if starts:
                print(f"[DuneBuilder ASYNC] Sample tokens: {[(s.mint, s.block_time) for s in starts[:3]]}")
                # Validate token mint addresses
                valid_tokens = [s for s in starts if s.mint and len(s.mint) >= 32]
                invalid_tokens = len(starts) - len(valid_tokens)
                if invalid_tokens > 0:
                    print(f"[DuneBuilder ASYNC] Warning: {invalid_tokens} tokens have invalid mint addresses")
            else:
                print(f"[DuneBuilder ASYNC] No tokens returned by Dune - checking why...")
        
        if not starts:
            if self.debug:
                print("[DuneBuilder ASYNC] No tokens from Dune, returning empty mapping")
            # Still save an empty mapping for today to mark that we tried
            self.cache.save_today({})
            return {}
        
        # Filter out tokens with invalid mint addresses
        valid_starts = [s for s in starts if s.mint and len(s.mint) >= 32]
        if len(valid_starts) != len(starts):
            if self.debug:
                print(f"[DuneBuilder ASYNC] Filtered {len(starts) - len(valid_starts)} tokens with invalid mint addresses")
        starts = valid_starts
        
        if self.debug:
            print(f"[DuneBuilder ASYNC] 🚀 Processing {len(starts)} tokens to fetch holders")
        
        # Process tokens to get holders
        token_to_top_holders: Dict[str, List[str]] = {}
        successful_fetches = 0
        failed_fetches = 0
        
        # Process tokens sequentially with better error handling
        for i, s in enumerate(starts):
            if not s.mint:
                if self.debug:
                    print(f"[DuneBuilder ASYNC] Token {i}: No mint address, skipping")
                continue
            
            try:
                if self.debug:
                    print(f"[DuneBuilder ASYNC] Token {i+1}/{len(starts)}: Processing {s.mint}")
                
                holders = await self._fetch_top_sampled_holders(holder_agg, s.mint)
                token_to_top_holders[s.mint] = holders
                
                if holders:
                    successful_fetches += 1
                    if self.debug:
                        print(f"[DuneBuilder ASYNC] Token {i} ({s.mint}): Got {len(holders)} holders")
                else:
                    failed_fetches += 1
                    if self.debug:
                        print(f"[DuneBuilder ASYNC] Token {i} ({s.mint}): No holders returned")
                        
                # Add a delay to avoid overwhelming the RPC, with backoff for failures
                if holders:
                    await asyncio.sleep(0.3)  # Shorter delay for successful requests
                else:
                    await asyncio.sleep(1.0)  # Longer delay after failures
                
            except Exception as e:
                failed_fetches += 1
                if self.debug:
                    print(f"[DuneBuilder ASYNC] Token {i} ({s.mint}): Exception - {e}")
                # Longer delay after exceptions
                await asyncio.sleep(2.0)

        # Calculate final statistics
        total_tokens_with_holders = len([holders for holders in token_to_top_holders.values() if holders])
        total_unique_wallets = len({w for holders in token_to_top_holders.values() for w in holders})
        
        if self.debug:
            print(f"[DuneBuilder ASYNC] ✅ Processing complete: {successful_fetches} successful, {failed_fetches} failed")
            print(f"[DuneBuilder ASYNC] ✅ Final result: {len(token_to_top_holders)} tokens processed")
            print(f"[DuneBuilder ASYNC] - {total_tokens_with_holders} tokens have holders")
            print(f"[DuneBuilder ASYNC] - {total_unique_wallets} unique wallet addresses")
            if total_tokens_with_holders > 0:
                print(f"[DuneBuilder ASYNC] - Average holders per token: {total_unique_wallets / total_tokens_with_holders:.1f}")
        
        # Save to today's per-day cache file
        try:
            self.cache.save_today(token_to_top_holders)
            if self.debug:
                print(f"[DuneBuilder ASYNC] ✅ Successfully saved today's cache")
        except Exception as e:
            if self.debug:
                print(f"[DuneBuilder ASYNC] ❌ Failed to save today's cache: {e}")
        
        return token_to_top_holders

# -----------------------
# Holder aggregation
# -----------------------
class HolderAggregator:
    def __init__(self, client: SolanaAlphaClient, debug: bool = False):
        self.client = client
        self.debug = debug

    async def get_token_holders(self, token_mint: str, *, sleep_between: float = 0.15, limit: int = 1000, max_pages: Optional[int] = None, decimals: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch token holders with improved error handling and validation.
        """
        if self.debug:
            print(f"[HolderAgg] Fetching holders for token: {token_mint}")
            
        # Validate token mint format
        if not token_mint or len(token_mint) < 32:
            if self.debug:
                print(f"[HolderAgg] Invalid token mint format: {token_mint}")
            return []
            
        page = 1
        owner_balances = defaultdict(int)
        owner_token_account_counts = defaultdict(int)
        total_accounts_processed = 0
        
        while True:
            payload_params = {"mint": token_mint, "page": page, "limit": limit, "displayOptions": {}}
            
            try:
                data = await retry_with_backoff(
                    self.client.make_rpc_call, 
                    "getTokenAccounts", 
                    payload_params,
                    retries=3,
                    base_delay=1.0
                )
            except Exception as e:
                if self.debug:
                    print(f"[HolderAgg] RPC call failed for {token_mint} page {page}: {e}")
                break
                
            if "error" in data:
                if self.debug:
                    print(f"[HolderAgg] RPC error for {token_mint}: {data['error']}")
                break
                
            token_accounts = data.get("result", {}).get("token_accounts", [])
            
            if not token_accounts:
                if self.debug and page == 1:
                    print(f"[HolderAgg] No token accounts found for {token_mint}")
                    # Check if token exists by trying a different approach
                    try:
                        supply_data = await self.client.make_rpc_call("getTokenSupply", [token_mint])
                        if "error" in supply_data:
                            print(f"[HolderAgg] Token supply check failed: {supply_data['error']}")
                        else:
                            print(f"[HolderAgg] Token exists but has no holders")
                    except Exception as e:
                        print(f"[HolderAgg] Token supply check error: {e}")
                break
                
            if self.debug and page == 1:
                print(f"[HolderAgg] Found {len(token_accounts)} token accounts on page {page}")
                
            for ta in token_accounts:
                owner = ta.get("owner") or ta.get("address")
                amt_raw = ta.get("amount", 0)
                
                # Handle nested account structure
                if "account" in ta and isinstance(ta["account"], dict):
                    acct = ta["account"]
                    owner = owner or acct.get("owner")
                    amt_raw = acct.get("amount", 0)
                    
                # Parse amount with better error handling
                if isinstance(amt_raw, dict):
                    amt_raw = int(float(amt_raw.get("amount") or amt_raw.get("uiAmount", 0)))
                else:
                    try:
                        amt_raw = int(amt_raw)
                    except Exception:
                        try:
                            amt_raw = int(float(amt_raw)) if amt_raw else 0
                        except Exception:
                            amt_raw = 0
                            
                if owner:
                    owner_balances[owner] += amt_raw
                    owner_token_account_counts[owner] += 1
                    total_accounts_processed += 1
                    
            page += 1
            if max_pages and page > max_pages:
                if self.debug:
                    print(f"[HolderAgg] Reached max pages ({max_pages}) for {token_mint}")
                break
                
            await asyncio.sleep(sleep_between)
            
        if self.debug:
            print(f"[HolderAgg] Processed {total_accounts_processed} token accounts, {len(owner_balances)} unique holders")
            
        holders = []
        for owner, raw in owner_balances.items():
            human_balance = raw / (10 ** decimals) if decimals else None
            holders.append({"wallet": owner, "balance_raw": raw, "balance": human_balance, "balance_formatted": (f"{human_balance:,.{decimals}f}" if human_balance is not None and decimals is not None else str(raw)), "num_token_accounts": owner_token_account_counts[owner]})
            
        holders.sort(key=lambda x: x["balance_raw"], reverse=True)
        
        if self.debug:
            print(f"[HolderAgg] Final result: {len(holders)} holders for {token_mint}")
            if holders:
                print(f"[HolderAgg] Top holder balance: {holders[0]['balance_raw']}")
                
        return holders
    
def _normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {("null" if k is None else str(k)): _normalize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_normalize(v) for v in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    elif isinstance(obj, TradingStart):
        d = asdict(obj)
        d["extra"] = _normalize(d.get("extra") or {})
        return _normalize(d)
    else:
        return obj
    
# -----------------------
# JobLibTokenUpdater
# -----------------------
# --- BEGIN REPLACEMENT ---
# This is the new SQLite-backed version of JobLibTokenUpdater.

class JobLibTokenUpdater:
    def __init__(self, data_dir: str = "./data/token_data", expiry_hours: int = 6, debug: bool = False):
        self.data_dir = os.path.abspath(data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Use SQLite database file instead of a pickle file
        self.db_file = os.path.join(self.data_dir, "tokens.db")
        self.expiry_hours = expiry_hours
        self.debug = debug
        # Use a lock for thread-safe database writes from asyncio.to_thread
        self.lock = threading.Lock()
        self._init_db()
        if self.debug:
            print(f"JobLibTokenUpdater: Initialized with SQLite db at {self.db_file}")

    def _init_db(self):
        """Initializes the SQLite database and table."""
        with self.lock:
            try:
                with sqlite3.connect(self.db_file) as conn:
                    cursor = conn.cursor()
                    # Create table:
                    # - mint: Primary key for automatic existence checks
                    # - block_time: Indexed for fast cleanup and sorting
                    # - data: JSON blob of the full TradingStart object
                    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tokens (
                        mint TEXT PRIMARY KEY,
                        block_time INTEGER,
                        data TEXT
                    )
                    """)
                    # Create index for fast sorting and cleanup
                    cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_block_time
                    ON tokens (block_time)
                    """)
                    conn.commit()
            except sqlite3.Error as e:
                if self.debug:
                    print(f"JobLibTokenUpdater: Database init error: {e}")

    async def save_trading_starts_async(self, trading_starts: List[TradingStart], skip_existing: bool = True) -> Dict[str, int]:
        """
        Asynchronously saves a list of TradingStart objects to the SQLite database.
        This operation is run in a separate thread to avoid blocking asyncio.
        """
        # Run the blocking DB operations in a separate thread
        return await asyncio.to_thread(
            self._save_tokens_sync, trading_starts, skip_existing
        )

    def _save_tokens_sync(self, trading_starts: List[TradingStart], skip_existing: bool) -> Dict[str, int]:
        """Synchronous (blocking) implementation for saving tokens."""
        saved = 0
        skipped = 0
        errors = 0
        
        # Prepare data for insertion
        data_to_insert = []
        for s in trading_starts:
            if not s.mint:
                errors += 1
                continue
            try:
                # Normalize and serialize the full object to JSON
                s_dict = asdict(s)
                # Use the existing _normalize helper to handle NaN/Inf
                s_json = json.dumps(_normalize(s_dict)) 
                data_to_insert.append((s.mint, s.block_time or 0, s_json))
            except Exception as e:
                if self.debug:
                    print(f"JobLibTokenUpdater: Serialization error for {s.mint}: {e}")
                errors += 1
        
        if not data_to_insert:
            return {"saved": 0, "skipped": 0, "errors": errors}

        # Use a lock to ensure thread-safe database access
        with self.lock:
            try:
                # Use a context manager for the connection
                with sqlite3.connect(self.db_file) as conn:
                    cursor = conn.cursor()
                    
                    if skip_existing:
                        # INSERT OR IGNORE = skip if mint (PRIMARY KEY) already exists
                        # This is the SQL equivalent of the previous logic
                        cursor.executemany(
                            "INSERT OR IGNORE INTO tokens (mint, block_time, data) VALUES (?, ?, ?)",
                            data_to_insert
                        )
                        saved = cursor.rowcount
                        skipped = len(data_to_insert) - saved
                    else:
                        # INSERT OR REPLACE = overwrite if mint (PRIMARY KEY) already exists
                        cursor.executemany(
                            "INSERT OR REPLACE INTO tokens (mint, block_time, data) VALUES (?, ?, ?)",
                            data_to_insert
                        )
                        saved = cursor.rowcount # This counts all successful inserts/replacements
                    
                    conn.commit()
                    
            except sqlite3.Error as e:
                if self.debug:
                    print(f"JobLibTokenUpdater: DB save error: {e}")
                errors += len(data_to_insert) # Assume all failed on batch error
                saved = 0
                skipped = 0
        
        if self.debug:
            # We can't easily get the total_now without another query, so we report the action
            print(f"JobLibTokenUpdater: saved={saved} skipped={skipped} errors={errors}")

        return {"saved": saved, "skipped": skipped, "errors": errors}

    async def cleanup_old_tokens_async(self) -> int:
        """
        Asynchronously cleans up old tokens from the database based on expiry_hours.
        This operation is run in a separate thread.
        """
        return await asyncio.to_thread(self._cleanup_sync)

    def _cleanup_sync(self) -> int:
        """Synchronous (blocking) implementation for cleaning up old tokens."""
        now = datetime.now(timezone.utc)
        cutoff = int((now - timedelta(hours=self.expiry_hours)).timestamp())
        deleted = 0
        
        with self.lock:
            try:
                with sqlite3.connect(self.db_file) as conn:
                    cursor = conn.cursor()
                    # Delete rows where block_time is older than the cutoff
                    cursor.execute("DELETE FROM tokens WHERE block_time < ?", (cutoff,))
                    deleted = cursor.rowcount
                    conn.commit()
            except sqlite3.Error as e:
                if self.debug:
                    print(f"JobLibTokenUpdater: DB cleanup error: {e}")
        
        if deleted > 0 and self.debug:
             print(f"JobLibTokenUpdater: cleaned {deleted} tokens older than {self.expiry_hours} hours")
        
        return deleted

    async def get_tracked_tokens_async(self, limit: Optional[int] = None) -> List[TradingStart]:
        """
        Asynchronously retrieves tracked tokens from the database.
        This operation is run in a separate thread.
        """
        return await asyncio.to_thread(self._get_tokens_sync, limit)

    def _get_tokens_sync(self, limit: Optional[int]) -> List[TradingStart]:
        """Synchronous (blocking) implementation for fetching tokens."""
        rows = []
        with self.lock:
            try:
                with sqlite3.connect(self.db_file) as conn:
                    # Use row_factory to get dict-like rows (though we only need one column)
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    # Select the JSON data, already sorted by block_time by the DB
                    query = "SELECT data FROM tokens ORDER BY block_time DESC"
                    params = []
                    
                    if limit:
                        query += " LIMIT ?"
                        params.append(limit)
                        
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
            except sqlite3.Error as e:
                if self.debug:
                    print(f"JobLibTokenUpdater: DB read error: {e}")
                return [] # Return empty on error
        
        # --- DESERIALIZATION (Copied from original class) ---
        norm: List[TradingStart] = []
        for row in rows:
            try:
                data_json = row["data"]
                t = json.loads(data_json) # t is a dict
                
                # Original logic to convert dict back to TradingStart
                if isinstance(t, TradingStart):
                     norm.append(t) # Should not happen from JSON, but good safety
                elif isinstance(t, dict):
                    try:
                        norm.append(TradingStart(**t))
                    except Exception:
                        # Handle dicts that might have extra/missing keys
                        allowed = {"mint","block_time","program_id","detected_via","extra","fdv_usd","volume_usd","source_dex","price_change_percentage"}
                        clean = {k:v for k,v in t.items() if k in allowed}
                        norm.append(TradingStart(**clean))
                        
            except Exception as e:
                if self.debug:
                    print(f"JobLibTokenUpdater: Deserialization error: {e}")

        # No need to sort, SQL's `ORDER BY` already did it.
        return norm

# --- END REPLACEMENT ---


class DuneHolderCache:
    def __init__(self, cache_file: str = "./data/dune_holders.pkl", cache_max_days: int = 7, debug: bool = False):
        self.cache_file = cache_file
        self.cache_max_days = cache_max_days
        self.debug = debug
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

    def _load_cache(self) -> Dict[str, Any]:
        if os.path.exists(self.cache_file):
            try:
                return joblib.load(self.cache_file)
            except Exception as e:
                if self.debug:
                    print("DuneHolderCache: load failed", e)
        return {}

    def _save_cache(self, obj: Dict[str, Any]):
        try:
            joblib.dump(_sanitize_maybe(obj), self.cache_file)
        except Exception as e:
            if self.debug:
                print("DuneHolderCache: save failed", e)

    async def build_cache(self, token_discovery: TokenDiscovery, holder_agg: HolderAggregator, top_n_per_token: int = 500) -> Dict[str, Set[str]]:
        # 🚀 NOW ASYNC
        starts = await token_discovery.get_tokens_launched_yesterday_cached()
        if self.debug:
            print(f"DuneHolderCache ASYNC: found {len(starts)} dune tokens")
        mapping: Dict[str, Set[str]] = {}
        for s in starts:
            if not s.mint:
                continue
            try:
                holders = await holder_agg.get_token_holders(s.mint, limit=1000, max_pages=2, decimals=None)
                top_wallets = {h["wallet"] for h in holders[:top_n_per_token]}
                mapping[s.mint] = top_wallets
                if self.debug:
                    print(f"DuneHolderCache ASYNC: token {s.mint} -> {len(top_wallets)} top holders")
            except Exception as e:
                if self.debug:
                    print(f"DuneHolderCache ASYNC: error fetching holders for {s.mint}: {e}")
                mapping[s.mint] = set()
        cache_obj = {"mapping": mapping, "fetched_at": datetime.now(timezone.utc).isoformat()}
        self._save_cache(cache_obj)
        return mapping

    def load_mapping(self) -> Tuple[Dict[str, Set[str]], Optional[datetime]]:
        obj = self._load_cache()
        if not obj:
            return {}, None
        mapping = obj.get("mapping", {})
        fetched_at = None
        try:
            fetched_at = datetime.fromisoformat(obj.get("fetched_at"))
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        except Exception:
            fetched_at = None
        return mapping, fetched_at

# -----------------------
# Overlap & scheduling stores
# -----------------------


def prune_old_overlap_entries(data: dict, expiry_hours: int = 6) -> dict:
    """
    Robust pruning of overlap entries.
    - Accepts several input shapes (mapping mint->list, DataFrame-like dicts, lists).
    - Treats entries without a parsable timestamp as recent (keeps them).
    - Supports ts as ISO string, numeric epoch seconds, or datetime.
    """
    if not data:
        return {}
    cutoff = datetime.now(timezone.utc) - timedelta(hours=expiry_hours)
    pruned = {}

    for mint, entries in (data or {}).items():
        # Normalize entries into a list of dicts
        if isinstance(entries, dict):
            # Possibly DataFrame-orient='list' mapping: convert to list of rows
            lists = entries
            try:
                # determine max length among lists
                lengths = [len(v) for v in lists.values() if isinstance(v, (list, tuple))]
                length = max(lengths) if lengths else 0
                rows = []
                for i in range(length):
                    row = {}
                    for k, v in lists.items():
                        try:
                            row[k] = v[i]
                        except Exception:
                            row[k] = None
                    rows.append(row)
                entries = rows
            except Exception:
                # Fallback: wrap the dict as single entry
                entries = [entries]
        elif not isinstance(entries, list):
            # Unknown shape -> wrap
            entries = [entries]

        new_entries = []
        for entry in entries:
            ts_val = None
            if isinstance(entry, dict):
                # prefer 'ts' but accept several common names
                ts_val = entry.get("ts") or entry.get("timestamp") or entry.get("saved_at") or entry.get("fetched_at") or entry.get("created_at")
            else:
                ts_val = entry

            # If no timestamp available, keep the entry (safer)
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
                    # try ISO parse; support both 'Z' and offset forms
                    s = ts_val
                    if s.endswith("Z"):
                        s = s.replace("Z", "+00:00")
                    ts = datetime.fromisoformat(s)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                else:
                    # Unknown type -> keep entry
                    new_entries.append(entry)
                    continue

                # Keep entry if it's newer than cutoff
                if ts and ts > cutoff:
                    new_entries.append(entry)
            except Exception:
                # Parsing failed -> keep entry (fail-open)
                new_entries.append(entry)

        if new_entries:
            pruned[mint] = new_entries

    return pruned

class OverlapStore:
    def __init__(self, filepath: str = "./data/overlap_results.pkl", debug: bool = False):
        self.filepath = filepath
        self.debug = debug
        self._last_upload = 0       # instance-specific cooldown tracker
        self._lock = threading.Lock()  # ensure thread safety
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def load(self) -> Dict[str, Any]:
        """Load overlap results from disk safely."""
        if os.path.exists(self.filepath):
            try:
                return joblib.load(self.filepath)
            except Exception as e:
                if self.debug:
                    print("OverlapStore: load failed", e)
        return {}

    def save(self, obj: Dict[str, Any], expiry_hours: int = 6):
        """
        Save overlap results to disk and periodically upload to Supabase.
        Always filters NONE grades before uploading.
        """
        now = time.time()
        with self._lock:  # ensure only one thread saves at a time
            try:
                # Normalize input into mapping mint -> list[entries]
                try:
                    normalized = (
                        safe_load_overlap(obj)
                        if not isinstance(obj, dict) or any(not isinstance(v, list) for v in obj.values())
                        else obj
                    )
                except Exception:
                    # Fallback: attempt to coerce using safe_load_overlap
                    normalized = safe_load_overlap(obj)

                # prune old entries before saving (safe, fail-open)
                pruned = prune_old_overlap_entries(normalized, expiry_hours=expiry_hours)

                # Always save the pruned, normalized object locally
                joblib.dump(_sanitize_maybe(pruned), self.filepath)

                # throttle uploads to Supabase (once every 120 secs)
                if now - self._last_upload < 120:
                    if self.debug:
                        print("OverlapStore: save throttled (recent upload). Local save completed.")
                    return

                # Use specialized uploader (will skip if only NONE grades)
                success = upload_overlap_results(self.filepath, BUCKET_NAME, debug=self.debug)

                if success:
                    self._last_upload = now
                    if self.debug:
                        print(f"OverlapStore: saved and uploaded at {time.ctime(now)}")
                else:
                    if self.debug:
                        print("OverlapStore: upload skipped (empty or NONE-only data)")

            except Exception as e:
                if self.debug:
                    print("OverlapStore: save failed", e)

class SchedulingStore:
    def __init__(self, filepath: str = "./data/scheduling_state.pkl", debug: bool = False):
        self.filepath = filepath
        self.debug = debug
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def load(self) -> Dict[str, Any]:
        if os.path.exists(self.filepath):
            try:
                return joblib.load(self.filepath)
            except Exception as e:
                if self.debug:
                    print("SchedulingStore: load failed", e)
        return {}

    def save(self, obj: Dict[str, Any]):
        try:
            joblib.dump(_sanitize_maybe(obj), self.filepath)
        except Exception as e:
            if self.debug:
                print("SchedulingStore: save failed", e)

    def update_token_state(self, token_mint: str, state_update: Dict[str, Any]):
        current_state = self.load()
        if token_mint not in current_state:
            current_state[token_mint] = {}
        current_state[token_mint].update(state_update)
        self.save(current_state)

    def get_token_state(self, token_mint: str) -> Dict[str, Any]:
        current_state = self.load()
        return current_state.get(token_mint, {})

    def cleanup_old_states(self, cutoff_timestamp: int = None):
        current_state = self.load()
        now = datetime.now(timezone.utc)
        cutoff = cutoff_timestamp or int((now - timedelta(hours=6)).timestamp())
        cleaned_state = {}
        for token_mint, state in current_state.items():
            launch_time = state.get("launch_time", 0)
            if launch_time > cutoff:
                cleaned_state[token_mint] = state
        removed_count = len(current_state) - len(cleaned_state)
        if removed_count > 0:
            self.save(cleaned_state)
            if self.debug:
                print(f"SchedulingStore: cleaned {removed_count} old scheduling states")
        return removed_count


def safe_load_overlap(overlap_store):
    """
    Load and normalize overlap store contents into a mapping: mint -> list[entries].
    Handles:
      - joblib-saved dict mapping mint->list[dict]
      - DataFrame objects
      - DataFrame-like dicts (orient='list')
      - Unexpected shapes (attempts best-effort normalization)
    """
    obj = overlap_store.load() if hasattr(overlap_store, "load") else overlap_store
    if not obj:
        return {}
    # If it's a DataFrame, convert to records then group by mint
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

    # If it's already a mapping mint -> list[dict], validate and return
    if isinstance(obj, dict):
        # Quick check: are values lists of dicts?
        is_good = True
        for v in obj.values():
            if not isinstance(v, list):
                is_good = False
                break
            if v and not isinstance(v[0], dict):
                # Could still be DataFrame-like (orient='list')
                is_good = False
                break
        if is_good:
            return obj

        # If it's a dict of column lists (DataFrame orient='list'), rebuild rows then group
        list_lengths = [len(v) for v in obj.values() if isinstance(v, (list, tuple))]
        if list_lengths:
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

    # If none of the above worked, try to coerce a sensible mapping:
    # - if it's a list: assume each item has 'mint'
    if isinstance(obj, list):
        mapping = {}
        for item in obj:
            if isinstance(item, dict):
                mint = item.get("mint") or item.get("token") or item.get("token_mint") or "_unknown"
                mapping.setdefault(mint, []).append(item)
        return mapping

    return {}


# -----------------------
# Security Analysis Helpers
# -----------------------
def evaluate_probation_from_rugcheck(report: dict, top_n: int = PROBATION_TOP_N, threshold_pct: float = PROBATION_THRESHOLD_PCT) -> dict:
    """
    Evaluates if a token should be on probation based on top holder concentration from a RugCheck report.
    This is a pure function for testability.

    Configuration is managed via environment variables:
      - PROBATION_TOP_N (default: 10)
      - PROBATION_THRESHOLD_PCT (default: 40.0)

    Returns a dict with:
      - probation (bool): True if combined top N pct > threshold_pct
      - top_n_pct (float): Sum of top N holders' pct values
      - top_n (int): How many holders were actually processed
      - threshold_pct (float): The threshold used for the check
      - explanation (str): A short human-readable reason for the result
    """
    top_holders = report.get("topHolders")
    if not isinstance(top_holders, list) or not top_holders:
        return {
            "probation": False,
            "top_n_pct": 0.0,
            "top_n": 0,
            "threshold_pct": threshold_pct,
            "explanation": "topHolders missing or empty in RugCheck report"
        }

    total_pct = 0.0
    processed_count = 0
    for holder in top_holders[:top_n]:
        pct_val = holder.get("pct")
        if pct_val is None:
            continue
        try:
            pct = float(pct_val)
            total_pct += pct
            processed_count += 1
        except (ValueError, TypeError):
            continue

    total_pct = round(total_pct, 6)
    probation = total_pct > threshold_pct
    explanation = f"Top {processed_count} holders own {total_pct:.2f}% of supply; threshold is >{threshold_pct}%. Probation={probation}."

    return {
        "probation": probation,
        "top_n_pct": total_pct,
        "top_n": processed_count,
        "threshold_pct": threshold_pct,
        "explanation": explanation
    }


# -----------------------
# Grading logic
# -----------------------
def calculate_overlap_grade(overlap_count: int, overlap_percentage: float, concentration: float, weighted_concentration: float, total_new_holders: int, total_winner_wallets: int) -> str:
    """
    Calculate overlap grade using multiple metrics including weighted concentration.
    
    Args:
        overlap_count: Number of overlapping wallet addresses
        overlap_percentage: Percentage of token holders that overlap with winners
        concentration: Distinct concentration (overlap_count / total_winner_wallets * 100)
        weighted_concentration: Weighted concentration based on wallet frequencies
        total_new_holders: Total number of holders in the new token
        total_winner_wallets: Total number of unique winner wallets across all tokens
    
    Returns:
        str: Grade level - "CRITICAL", "HIGH", "MEDIUM", "LOW", or "NONE"
    """
    
    # CRITICAL: Extremely high overlap 
    if (
        # High overlap percentage with significant count
        (overlap_percentage >= 50 and overlap_count >= 100) or
        (overlap_percentage >= 60 and overlap_count >= 50) or
        
        # High distinct concentration with significant overlap
        (concentration >= 30 and overlap_count >= 75) or
        (concentration >= 40 and overlap_count >= 50) or
        
        # High weighted concentration (indicates frequent repeat wallets)
        (weighted_concentration >= 35 and overlap_count >= 50) or
        (weighted_concentration >= 45 and overlap_count >= 30) or
        
        # Combined thresholds: moderate weighted + high distinct concentration
        (weighted_concentration >= 25 and concentration >= 25 and overlap_count >= 40) or
        
        # Very high weighted concentration even with lower overlap
        (weighted_concentration >= 60 and overlap_count >= 25)
    ):
        return "CRITICAL"
    
    # HIGH: Significant overlap
    elif (
        # Moderate-high overlap percentage
        (overlap_percentage >= 30 and overlap_count >= 50) or
        (overlap_percentage >= 40 and overlap_count >= 25) or
        
        # Moderate-high distinct concentration
        (concentration >= 20 and overlap_count >= 40) or
        (concentration >= 25 and overlap_count >= 30) or
        
        # High weighted concentration with moderate overlap
        (weighted_concentration >= 25 and overlap_count >= 35) or
        (weighted_concentration >= 30 and overlap_count >= 25) or
        
        # Combined moderate thresholds
        (weighted_concentration >= 20 and concentration >= 15 and overlap_count >= 30) or
        (weighted_concentration >= 15 and overlap_percentage >= 25 and overlap_count >= 25) or
        
        # High weighted concentration alone (frequent repeat participants)
        (weighted_concentration >= 40 and overlap_count >= 15)
    ):
        return "HIGH"
    
    # MEDIUM: Notable overlap worth monitoring
    elif (
        # Lower overlap percentage but still significant
        (overlap_percentage >= 15 and overlap_count >= 25) or
        (overlap_percentage >= 20 and overlap_count >= 15) or
        
        # Lower distinct concentration
        (concentration >= 10 and overlap_count >= 20) or
        (concentration >= 15 and overlap_count >= 15) or
        
        # Moderate weighted concentration
        (weighted_concentration >= 15 and overlap_count >= 20) or
        (weighted_concentration >= 20 and overlap_count >= 15) or
        
        # Combined lower thresholds
        (weighted_concentration >= 12 and concentration >= 8 and overlap_count >= 15) or
        (weighted_concentration >= 10 and overlap_percentage >= 12 and overlap_count >= 12) or
        
        # Weighted concentration indicating repeat behavior
        (weighted_concentration >= 25 and overlap_count >= 10)
    ):
        return "MEDIUM"
    
    # LOW: Minimal but detectable overlap
    elif (
        # Basic overlap thresholds
        (overlap_percentage >= 5 and overlap_count >= 10) or
        (overlap_count >= 5) or
        
        # Lower concentration thresholds
        (concentration >= 5 and overlap_count >= 8) or
        (concentration >= 8 and overlap_count >= 5) or
        
        # Lower weighted concentration thresholds
        (weighted_concentration >= 8 and overlap_count >= 8) or
        (weighted_concentration >= 12 and overlap_count >= 5) or
        
        # Any combination showing some pattern
        (weighted_concentration >= 6 and concentration >= 4 and overlap_count >= 5) or
        (weighted_concentration >= 4 and overlap_percentage >= 8 and overlap_count >= 4) or
        
        # Weighted concentration showing repeat participation even with low overlap
        (weighted_concentration >= 15 and overlap_count >= 3)
    ):
        return "LOW"
    
    # NONE: No significant overlap detected
    else:
        return "NONE"

# -----------------------
# Monitor (updated for async Dune)
# -----------------------
class Monitor:
    def __init__(
        self,
        sol_client: SolanaAlphaClient,
        token_discovery: TokenDiscovery,
        holder_agg: HolderAggregator,
        updater: JobLibTokenUpdater,
        dune_cache: DuneWinnersCache,
        dune_builder: DuneWinnersBuilder,
        overlap_store: OverlapStore,
        scheduling_store: SchedulingStore,
        # ------------------ 🚀 CHANGE 1: Accept http_session ------------------
        http_session: aiohttp.ClientSession,
        # --- NEW: Accept ML Classifier ---
        ml_classifier: SolanaTokenPredictor,
        *,
        coingecko_poll_interval_seconds: int = 30,
        initial_check_delay_seconds: int = 600, # 10 minutes
        repeat_interval_seconds: int = 1800, # 30 minutes
        debug: bool = False,
    ):
        self.sol_client = sol_client
        self.token_discovery = token_discovery
        self.holder_agg = holder_agg
        self.updater = updater
        self.dune_cache = dune_cache
        self.dune_builder = dune_builder
        self.overlap_store = overlap_store
        self.scheduling_store = scheduling_store
        self.coingecko_poll_interval_seconds = coingecko_poll_interval_seconds
        self.initial_check_delay_seconds = initial_check_delay_seconds
        self.repeat_interval_seconds = repeat_interval_seconds
        self.debug = debug
        self._scheduled: Set[str] = set()
        self.last_cleanup = 0
        self.last_dune_build = 0  # Track last Dune cache build time
        # Probation / risky tokens (for GoPlus + Dexscreener gating)
        self.pending_risky_tokens: Dict[str, Dict[str, Any]] = {}  # mint -> {first_seen, last_checked, attempts, reasons, overlap_result}
        self._probation_tasks: Dict[str, asyncio.Task] = {}  # mint -> asyncio.Task
        # ------------------ 🚀 CHANGE 2: Use the provided session ------------------
        self.http_session = http_session
        # --- NEW: Store ML Classifier ---
        self.ml_classifier = ml_classifier
        # concurrency guard for external API calls
        self._api_sema = asyncio.Semaphore(8)

    # ------------------ 🚀 CHANGE 3: Remove session management methods ------------------
    # async def _get_http_session(self) -> aiohttp.ClientSession:
    #     ...
    # async def _close_http_session(self):
    #     ...

    async def _cleanup_finished_tasks(self):
        """Periodically cleans up finished tasks and states from internal memory."""
        now_ts = int(datetime.now(timezone.utc).timestamp())
        cutoff_ts = now_ts - (6 * 3600)  # 4 hours ago

        # --- Cleanup _probation_tasks ---
        finished_probation_tasks = [
            mint for mint, task in self._probation_tasks.items() if task.done()
        ]
        if finished_probation_tasks:
            for mint in finished_probation_tasks:
                del self._probation_tasks[mint]
            if self.debug:
                print(f"[Cleanup] Removed {len(finished_probation_tasks)} finished probation tasks from memory.")

        # --- Cleanup _scheduled (as a safety net) ---
        scheduling_state = self.scheduling_store.load()
        expired_mints_in_mem = set()
        
        # Iterate over a copy of the set as we might modify it
        for mint in list(self._scheduled):
            state = scheduling_state.get(mint)
            if not state:
                continue

            launch_time = state.get("launch_time", 0)
            status = state.get("status")

            # Condition 1: Token is older than 24 hours
            if launch_time < cutoff_ts:
                expired_mints_in_mem.add(mint)
            # Condition 2: Token's lifecycle is explicitly finished
            elif status in ["completed", "dropped", "failed"]:
                expired_mints_in_mem.add(mint)

        if expired_mints_in_mem:
            for mint in expired_mints_in_mem:
                self._scheduled.discard(mint)
            if self.debug:
                print(f"[Cleanup] Safety net removed {len(expired_mints_in_mem)} expired/completed tokens from _scheduled set.")

# --------------------------------------------------------------------------
    # --- 🚀 MODIFIED: Security Checks with Helius RPC Fallback 🚀 ---
    # --------------------------------------------------------------------------

    async def _fetch_rpc_security_report(self, mint: str) -> Dict[str, Any]:
        """
        Fallback: Manually build a security report using Helius RPC 
        when RugCheck returns 400/404/429.
        """
        if self.debug:
            print(f"[Security] 🛡️ Initiating RPC Fallback for {mint}...")

        # 1. Get Mint Info (Authorities & Decimals)
        freeze_authority = None
        mint_authority = None
        decimals = 0
        
        try:
            info = await self.sol_client.make_rpc_call(
                "getAccountInfo", 
                [mint, {"encoding": "jsonParsed"}]
            )
            result = info.get("result")
            value = result.get("value") if result else None
            
            if value:
                parsed = value.get("data", {}).get("parsed", {}).get("info", {})
                mint_authority = parsed.get("mintAuthority")
                freeze_authority = parsed.get("freezeAuthority")
                decimals = int(parsed.get("decimals", 0))
                
                if self.debug:
                    if not mint_authority and not freeze_authority:
                        print(f"[Security] ℹ️ No authorities found for {mint} (immutable mint)")
                    else:
                        print(f"[Security] ✅ Got authorities for {mint}: mint_auth={mint_authority[:8] if mint_authority else 'None'}..., freeze_auth={freeze_authority[:8] if freeze_authority else 'None'}...")
            else:
                if self.debug:
                    print(f"[Security] ⚠️ getAccountInfo returned null for {mint} (account not found)")
        except Exception as e:
            if self.debug:
                print(f"[Security] ⚠️ RPC Authority check failed for {mint}: {e}")

        # 2. Get Supply
        supply = 0
        try:
            s_data = await self.sol_client.make_rpc_call("getTokenSupply", [mint])
            amount_str = s_data.get("result", {}).get("value", {}).get("amount", "0")
            supply = int(amount_str)
            if self.debug:
                print(f"[Security] ✅ Got supply for {mint}: {supply}")
        except Exception as e:
            if self.debug:
                print(f"[Security] ⚠️ Failed to get supply for {mint}: {e}")

        # 3. Get Creator Balance (We assume Mint Authority == Creator for new tokens)
        creator_balance_pct = 0.0
        creator_balance_raw = 0
        if mint_authority:
            try:
                # Reuse your existing helper logic here
                normalized, success = await self._get_creator_balance_rpc(mint, decimals)
                if success and supply > 0:
                    # Calculate percentage based on raw supply
                    total_supply_normalized = supply / (10 ** decimals) if decimals > 0 else supply
                    creator_balance_pct = (normalized / total_supply_normalized) * 100
                    creator_balance_raw = normalized # Store for record
                    if self.debug:
                        print(f"[Security] ✅ Creator balance for {mint}: {creator_balance_pct:.2f}%")
                elif self.debug and not success:
                    print(f"[Security] ⚠️ Failed to get creator balance for {mint} from RPC")
            except Exception as e:
                if self.debug:
                    print(f"[Security] ⚠️ Exception getting creator balance RPC for {mint}: {e}")
        else:
            # Immutable mint - try to find creator from creation transaction
            if self.debug:
                print(f"[Security] ℹ️ Immutable mint detected for {mint} - attempting to find creator from creation tx...")
            
            creator = await self._get_creator_from_creation_tx(mint)
            if creator:
                creator_balance_pct, success = await self._get_creator_balance_for_immutable_mint(
                    mint, creator, decimals, supply
                )
            else:
                if self.debug:
                    print(f"[Security] ⚠️ Could not find creator for immutable mint {mint}")

        # 4. Get Liquidity (Optional: Quick DexScreener check just for the Fallback)
        # We do this because RugCheck usually gives us this, and we need it for the gate.
        total_lp_usd = 0.0
        try:
            dex_data = await self._run_dexscreener_check(mint)
            if dex_data.get("ok"):
                total_lp_usd = dex_data.get("liquidity_usd", 0.0)
                if self.debug:
                    print(f"[Security] ✅ Got liquidity for {mint}: ${total_lp_usd:.2f}")
        except Exception as e:
            if self.debug:
                print(f"[Security] ⚠️ Failed to get liquidity for {mint}: {e}")

        return {
            "ok": True,
            "data_source": "rpc_fallback",
            "rugged": False, # We can't know for sure, but we assume False if valid on-chain
            "total_holders": 0, # RPC makes this hard to count cheaply, so we skip holder count check
            "creator_balance": creator_balance_pct, # The calculated %
            "freeze_authority": freeze_authority,
            "mint_authority": mint_authority,
            "has_authorities": bool(freeze_authority or mint_authority),
            "transfer_fee_pct": 0, # Hard to parse from RPC without complex decoding, assume 0 for fallback
            "total_lp_usd": total_lp_usd,
            "overall_lp_locked_pct": None, # Cannot get this from simple RPC
            "probation": False, # Cannot calculate holder concentration easily via RPC
            "probation_meta": {"explanation": "RPC Fallback - Concentration check skipped"},
            "raw": {}
        }

    async def _run_rugcheck_check(self, mint: str) -> Dict[str, Any]:
        """
        Check token security. 
        Primary: RugCheck API.
        Fallback: Helius RPC (if RugCheck returns 400/RateLimit).
        """
        url = f"https://api.rugcheck.xyz/v1/tokens/{mint}/report"
        session = self.http_session
        
        # --- Attempt 1: RugCheck ---
        try:
            async with self._api_sema:
                async with session.get(url, timeout=10) as resp:
                    # If 400 (Report not found/generation failed) or 429/5xx
                    if resp.status != 200:
                        if self.debug:
                            print(f"[Security] RugCheck returned status {resp.status}. Switching to RPC Fallback.")
                        # TRIGGER FALLBACK DIRECTLY
                        return await self._fetch_rpc_security_report(mint)
                    
                    data = await resp.json()

        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
            if self.debug:
                print(f"[Security] RugCheck request failed for {mint}: {e}. Switching to RPC Fallback.")
            return await self._fetch_rpc_security_report(mint)

        # --- Process RugCheck Data (If Successful) ---
        probation_result = evaluate_probation_from_rugcheck(data)
        
        # Handle None returns from data.get()
        top_holders = data.get("topHolders") or []
        markets = data.get("markets") or []
        rugged = data.get("rugged", False)
        
        # Authorities
        freeze_authority = data.get("freezeAuthority")
        mint_authority = data.get("mintAuthority")
        has_authorities = bool(freeze_authority or mint_authority)

        # --- FIXED: Creator Balance Handling (None default for missing data) ---
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
        total_lp_locked_usd = 0.0
        total_lp_usd = 0.0
        lp_lock_details = []

        for market in markets:
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
            "data_source": "rugcheck",
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

    async def _get_supply_and_decimals_rpc(self, mint: str) -> tuple:
        """
        Get supply and decimals from Helius RPC (with public RPC backup).
        Returns (supply, decimals) or (0, 0) if failed.
        """
        # Build the Helius URL using the existing key rotation system
        helius_url = self.sol_client._get_url()
        fallback_url = "https://api.mainnet-beta.solana.com"
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTokenSupply",
            "params": [mint]
        }

        # Attempt 1: Helius
        try:
            data = await self.sol_client.make_rpc_call("getTokenSupply", [mint])
            if not data.get("error"):
                val = data.get("result", {}).get("value", {})
                if val:
                    supply = int(val.get("amount", "0"))
                    decimals = int(val.get("decimals", 0))
                    if self.debug:
                        print(f"[Security] ✅ Got supply from Helius: {supply}, decimals: {decimals}")
                    return (supply, decimals)
        except Exception as e:
            if self.debug:
                print(f"[Security] ⚠️ Helius supply fetch failed for {mint}: {e}")
        
        # Attempt 2: Public RPC (Fallback)
        try:
            async with self.http_session.post(fallback_url, json=payload, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    val = data.get("result", {}).get("value", {})
                    if val:
                        supply = int(val.get("amount", "0"))
                        decimals = int(val.get("decimals", 0))
                        if self.debug:
                            print(f"[Security] ✅ Got supply from Public RPC: {supply}, decimals: {decimals}")
                        return (supply, decimals)
        except Exception as e:
            if self.debug:
                print(f"[Security] ⚠️ Public RPC supply fetch failed for {mint}: {e}")

        return (0, 0)

    async def _get_creator_from_creation_tx(self, mint: str) -> str:
        """
        Extract creator address from token creation transaction.
        When mint authority is None (immutable), we can still find the creator
        by looking at who initiated the token creation transaction.
        
        Logic:
        1. Get first signature (token creation tx) via getSignaturesForAddress
        2. Parse transaction to find who paid for it (likely the creator)
        3. Return creator address
        """
        try:
            # Get the first (oldest) transaction - this should be token creation
            sig_data = await self.sol_client.make_rpc_call(
                "getSignaturesForAddress",
                [mint, {"limit": 1}]  # Get only the first signature
            )
            
            sigs = sig_data.get("result", [])
            if not sigs:
                if self.debug:
                    print(f"[Security] ℹ️ No signatures found for {mint}")
                return None
            
            sig = sigs[0].get("signature")
            if not sig:
                return None
            
            # Parse the transaction (with version support for Solana versioned transactions)
            tx_data = await self.sol_client.make_rpc_call(
                "getTransaction",
                [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]
            )
            
            if tx_data.get("error"):
                if self.debug:
                    print(f"[Security] ⚠️ Could not fetch creation tx for {mint}: {tx_data.get('error')}")
                return None
            
            tx = tx_data.get("result", {})
            if not tx:
                # Retry with explicit version 0 support
                if self.debug:
                    print(f"[Security] ℹ️ Retrying getTransaction with explicit version support for {mint}...")
                tx_data = await self.sol_client.make_rpc_call(
                    "getTransaction",
                    [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]
                )
                if tx_data.get("error"):
                    if self.debug:
                        print(f"[Security] ⚠️ Retry also failed for {mint}: {tx_data.get('error')}")
                    return None
                tx = tx_data.get("result", {})
                if not tx:
                    return None
            
            # Get the fee payer (first signer) - usually the creator
            message = tx.get("transaction", {}).get("message", {})
            account_keys = message.get("accountKeys", [])
            
            if account_keys:
                creator = account_keys[0].get("pubkey") if isinstance(account_keys[0], dict) else str(account_keys[0])
                if self.debug:
                    print(f"[Security] ✅ Found creator from creation tx: {creator[:8] if creator else 'None'}...")
                return creator
            
            return None
            
        except Exception as e:
            if self.debug:
                print(f"[Security] ⚠️ Failed to extract creator from creation tx for {mint}: {e}")
            return None

    async def _get_creator_balance_for_immutable_mint(self, mint: str, creator: str, decimals: int, supply: int) -> tuple:
        """
        Get creator balance when mint is immutable (no mint authority).
        Uses the extracted creator address to find their token accounts.
        
        Returns (creator_balance_normalized, success_flag).
        """
        if not creator:
            return (0.0, False)
        
        try:
            data = await self.sol_client.make_rpc_call(
                "getTokenAccountsByOwner",
                [
                    creator,
                    {"mint": mint},
                    {"encoding": "jsonParsed"}
                ]
            )
            
            if data.get("error"):
                if self.debug:
                    print(f"[Security] ⚠️ Failed to get token accounts for creator {creator[:8]}... on {mint}")
                return (0.0, False)
            
            accounts = data.get("result", {}).get("value", [])
            raw_balance = 0.0
            
            for acc in accounts:
                amt_info = acc.get("account", {}).get("data", {}).get("parsed", {}).get("info", {}).get("tokenAmount", {})
                raw_balance += float(amt_info.get("amount", 0))
            
            if raw_balance > 0:
                # Normalize by decimals
                normalized = raw_balance / (10 ** decimals) if decimals > 0 else raw_balance
                creator_pct = (normalized / (supply / (10 ** decimals))) * 100 if supply > 0 and decimals > 0 else 0.0
                
                if self.debug:
                    print(f"[Security] ✅ Immutable mint creator balance: {creator_pct:.2f}% (raw: {raw_balance})")
                return (creator_pct, True)
            else:
                if self.debug:
                    print(f"[Security] ℹ️ Creator {creator[:8]}... has 0 balance in {mint}")
                return (0.0, True)
            
        except Exception as e:
            if self.debug:
                print(f"[Security] ⚠️ Exception getting immutable mint creator balance for {mint}: {e}")
            return (0.0, False)
        """
        Get creator balance via RPC.
        Logic:
        1. Get Mint Info to find 'mintAuthority' (often the creator/deployer).
        2. Get Token Accounts for that authority to find their balance of this mint.
        Returns (creator_balance_normalized, success_flag).
        """
        # Step 1: Get Mint Authority
        mint_authority = None
        
        try:
            data = await self.sol_client.make_rpc_call(
                "getAccountInfo",
                [mint, {"encoding": "jsonParsed"}]
            )
            
            if not data.get("error"):
                info = data.get("result", {}).get("value", {}).get("data", {}).get("parsed", {}).get("info", {})
                mint_authority = info.get("mintAuthority")
        except Exception:
            pass
        
        if not mint_authority:
            if self.debug:
                print(f"[Security] ⚠️ Could not determine mint authority (creator) for {mint}")
            return (0.0, False)
            
        # Step 2: Get Balance for Mint Authority
        raw_balance = 0.0
        success = False
        
        try:
            data = await self.sol_client.make_rpc_call(
                "getTokenAccountsByOwner",
                [
                    mint_authority,
                    {"mint": mint},
                    {"encoding": "jsonParsed"}
                ]
            )
            
            if not data.get("error"):
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
                print(f"[Security] ✅ RPC Creator Balance ({mint_authority}): {normalized} (raw: {raw_balance})")
            return (normalized, True)
            
        return (0.0, False)

    async def run_token_analysis_step(self, start: TradingStart, check_count: int):
        """
        Core analysis function that runs the full analysis lifecycle.
        Handles both RugCheck and Helius RPC (fallback) data sources.
        """
        mint = start.mint
        now_ts = int(datetime.now(timezone.utc).timestamp())
        
        if self.debug:
            print(f"[Analysis] Running check #{check_count} for {mint}")

        # === 1. RUN SECURITY CHECK (RUGCHECK -> RPC FALLBACK) ===
        try:
            # We call _run_rugcheck_check directly now as it handles its own fallback/retries internally
            r = await self._run_rugcheck_check(mint)
        except Exception as e:
            if self.debug:
                print(f"[Security] Fatal error checking security for {mint}: {e}")
            reasons = [f"security_check_exception:{str(e)}"]
            await self._start_or_update_probation(mint, start, {"mint": mint, "grade": "NONE"}, reasons)
            return

        # === 2. SECURITY GATE ===
        reasons = []
        if not r.get("ok"):
            reasons.append(f"security_error:{r.get('error')}")
        else:
            data_source = r.get("data_source", "unknown")
            
            # Rule 0: Probation
            if r.get("probation"):
                reasons.append(r.get("probation_meta", {}).get("explanation", "Probation"))

            # Rule 1: Rugged (Only valid for RugCheck)
            if r.get("rugged"):
                reasons.append("rugged:true")

            # Rule 2: Authorities
            if r.get("has_authorities"):
                authorities = []
                if r.get("freeze_authority"): authorities.append("freeze")
                if r.get("mint_authority"): authorities.append("mint")
                reasons.append(f"authorities:{','.join(authorities)}")
            
            # Rule 3: Creator Balance
            creator_pct = r.get("creator_balance", 0)
            if creator_pct > MAX_CREATOR_PCT:
                reasons.append(f"creator_balance_pct:{creator_pct:.2f}")

            # Rule 4: Transfer Fee
            if r.get("transfer_fee_pct", 0) > 5:
                reasons.append(f"transfer_fee:{r.get('transfer_fee_pct')}%")

            # Rule 5: Holder Count
            # If fallback, we skip this because fetching holder count via RPC is expensive/slow
            if data_source != "rpc_fallback":
                if r.get("total_holders", 0) < 50:
                    reasons.append(f"holder_count:{r.get('total_holders')}_req_50")

            # Rule 6: LP Lock % (Conditional on Data Source)
            lp_locked_pct = r.get("overall_lp_locked_pct")
            
            if data_source == "rugcheck":
                # Strict check for RugCheck data
                if isinstance(lp_locked_pct, (int, float)) and lp_locked_pct < 80.0:
                    reasons.append(f"lp_locked:{lp_locked_pct:.1f}%_req_80%")
            elif data_source == "rpc_fallback":
                # We CANNOT check LP lock via simple RPC. 
                # We assume risk here to allow analysis to proceed to Overlap Check.
                # Just ensure basic Liquidity check (below) is respected.
                pass

            # Rule 7: Liquidity Amount (Available from both sources via DexScreener)
            total_lp = r.get("total_lp_usd")
            if total_lp != "unknown":
                if total_lp < 10000.0:
                    reasons.append(f"liquidity_usd:{total_lp:.2f}_req_10000")
            else:
                # If LP is unknown (DexScreener failed during fallback check), that's a risk
                if data_source == "rpc_fallback":
                     reasons.append("liquidity_usd:unknown")

        if reasons: 
            if self.debug:
                print(f"[Security] {mint} FAILED Gate ({r.get('data_source')}): {reasons}")
            overlap_result_stub = {"mint": mint, "grade": "NONE", "probation_meta": r.get("probation_meta")}
            await self._start_or_update_probation(mint, start, overlap_result_stub, reasons)
            
            self.scheduling_store.update_token_state(mint, {
                "status": "probation",
                "last_completed_check": now_ts,
                "next_scheduled_check": now_ts + self.repeat_interval_seconds,
                "total_checks_completed": check_count
            })
            return

        # === 3. HELIUS CALL (GATE PASSED) ===
        if self.debug:
            print(f"[Security] {mint} PASSED Gate ({r.get('data_source')}). Fetching holders...")
        
        try:
            overlap_result = await self._fetch_and_calculate_overlap(start)
        except Exception as e:
            if self.debug:
                print(f"[Analysis] Helius call failed for {mint}: {e}")
            self.scheduling_store.update_token_state(mint, {
                "last_error": f"Helius_fail: {e}",
                "last_error_at": datetime.now(timezone.utc).isoformat()
            })
            return

        # === 4. POST-HELIUS PROCESSING ===
        grade = overlap_result.get("grade", "NONE")
        if self.debug:
            print(f"[Analysis] {mint} Overlap Check: Grade={grade}, Count={overlap_result.get('overlap_count')}, Pct={overlap_result.get('overlap_percentage')}%")
        
        if grade == "NONE":
            # Save NONE result
            obj = safe_load_overlap(self.overlap_store)
            obj.setdefault(mint, []).append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "result": overlap_result,
                "security": f"passed_grade_none_{r.get('data_source')}",
                "rugcheck": self._extract_rugcheck_summary(r)
            })
            self.overlap_store.save(obj)
            
            self.scheduling_store.update_token_state(mint, {
                "status": "active",
                "last_completed_check": now_ts,
                "next_scheduled_check": now_ts + self.repeat_interval_seconds,
                "total_checks_completed": check_count
            })
            return

        # Grade PASSED -> DexScreener + ML + Final Save
        if self.debug:
            print(f"[Analysis] {mint} PASSED ALL CHECKS (Grade: {grade}).")

        try:
            dex_data = await retry_with_backoff(
                self._run_dexscreener_check, mint, retries=8, base_delay=1.5
            )
        except Exception:
            dex_data = {"ok": False, "price_usd": None, "market_cap_usd": None}

        # ML Prediction
        ml_prediction_result = None
        try:            
            ml_prediction_result = self.ml_classifier.predict(
                mint, 
                threshold=ML_PREDICTION_THRESHOLD,
                action_threshold=ML_ACTION_THRESHOLD,
                signal_type='discovery'
            )
        except Exception as e:
            ml_prediction_result = {'action': 'ERROR', 'error': str(e)}
        
        # Calculate ML_PASSED based on probability vs threshold
        ml_probability = ml_prediction_result.get("win_probability")
        ml_passed = False
        if ml_probability is not None:
            try:
                ml_passed = float(ml_probability) >= ML_PREDICTION_THRESHOLD
            except (ValueError, TypeError):
                ml_passed = False

        # Add ML_PASSED to overlap_result
        overlap_result["ML_PASSED"] = ml_passed
        if self.debug:
            print(f"[Analysis] ML Result for {mint}: PASSED={ml_passed}, Prob={ml_probability}")

        # Enrich metadata if missing from start or previous checks
        if "token_metadata" not in overlap_result:
            overlap_result["token_metadata"] = {}
        
        meta = overlap_result["token_metadata"]
        
        # 1. Fallback to RugCheck for Name/Symbol
        if not meta.get("name") or not meta.get("symbol"):
            rc_raw = r.get("raw", {})
            if isinstance(rc_raw, dict):
                token_info = rc_raw.get("token", {})
                if not meta.get("name"): meta["name"] = token_info.get("name")
                if not meta.get("symbol"): meta["symbol"] = token_info.get("symbol")
        
        # 2. Fallback to DexScreener for Name/Symbol
        if not meta.get("name") or not meta.get("symbol"):
            ds_raw = dex_data.get("raw", {})
            if isinstance(ds_raw, dict):
                base_token = ds_raw.get("baseToken", {})
                if not meta.get("name"): meta["name"] = base_token.get("name")
                if not meta.get("symbol"): meta["symbol"] = base_token.get("symbol")
        
        # 3. Fallback to start.extra if still missing
        if not meta.get("name") and start.extra: meta["name"] = start.extra.get("name")
        if not meta.get("symbol") and start.extra: meta["symbol"] = start.extra.get("symbol")

        # Save Final
        obj = safe_load_overlap(self.overlap_store)
        obj.setdefault(mint, []).append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "result": overlap_result,
            "security": f"passed_{r.get('data_source')}",
            "probation_meta": r.get("probation_meta"),
            "rugcheck": self._extract_rugcheck_summary(r),
            "dexscreener": {
                "current_price_usd": dex_data.get("price_usd"),
                "market_cap_usd": dex_data.get("market_cap_usd"),
            },
            "ml_prediction": {
                "probability": ml_prediction_result.get("win_probability"),
                "confidence": ml_prediction_result.get("confidence"),
                "risk_tier": ml_prediction_result.get("risk_tier"),
                "action": ml_prediction_result.get("action"),
                "error": ml_prediction_result.get("error"),
                "key_metrics": ml_prediction_result.get("key_metrics"),
                "warnings": ml_prediction_result.get("warnings")
            },
            "ML_PASSED": ml_passed
        })
        self.overlap_store.save(obj)
        if self.debug:
            print(f"[Analysis] {mint} SAVED to overlap_results.pkl (Grade: {grade}, ML_PASSED: {ml_passed})")
        
        if mint in self.pending_risky_tokens:
            self.pending_risky_tokens.pop(mint, None)
        
        self.scheduling_store.update_token_state(mint, {
            "status": "completed",
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "security_passed": True,
            "liquidity_usd": r.get("total_lp_usd", 0.0),
            "last_completed_check": now_ts,
            "total_checks_completed": check_count
        })
        if self.debug:
            print(f"[Analysis] {mint} COMPLETED via {r.get('data_source')}.")

    # --------------------------------------------------------------------------
    # --- 🚀 NEW: CONSOLIDATED ANALYSIS AND ROUTING LOGIC 🚀 ---
    # --------------------------------------------------------------------------

    def _extract_rugcheck_summary(self, r: Dict[str, Any]) -> Dict[str, Any]:
        """Helper to format RugCheck data for saving."""
        if not r.get("ok"):
            return {"error": r.get("error", "unknown"), "error_text": r.get("error_text")}
        return {
            "top1_holder_pct": r.get("top1_holder_pct", 0.0),
            "top_10_holders_pct": r.get("top_10_holders_pct", 0.0), # NEW
            "holder_count": r.get("total_holders", 0),
            "has_authorities": r.get("has_authorities", False),
            "creator_balance": r.get("creator_balance", 0),
            "transfer_fee_pct": r.get("transfer_fee_pct", 0),
            "lp_locked_pct": r.get("overall_lp_locked_pct", 0.0),
            "total_liquidity_usd": r.get("total_lp_usd", 0.0),
            "lp_lock_details": r.get("lp_lock_details", [])
        }
    
    async def _run_dexscreener_check(self, mint: str) -> Dict[str, Any]:
        """
        Fetches token data from DexScreener.
        Designed to be used with `retry_with_backoff`, so it raises exceptions on failure
        (HTTP errors, no pairs found, invalid price) to trigger retries.
        """
        url = f"https://api.dexscreener.com/latest/dex/tokens/{mint}"
        # ------------------ 🚀 CHANGE 5: Use self.http_session directly ------------------
        session = self.http_session
        async with self._api_sema:
            # Let retry_with_backoff handle exceptions from this block
            # <<< CORRECTION: Increased timeout
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

        # Extract FDV as market cap, converting to float safely
        try:
            fdv = float(p0.get("fdv", 0.0) or 0.0)
        except (ValueError, TypeError):
            fdv = 0.0

        return {
            "ok": True,
            "pair_exists": True,
            "liquidity_usd": liquidity,
            "price_usd": price_usd,
            "market_cap_usd": fdv,
            "raw": p0
        }

    async def _start_or_update_probation(self, mint: str, start: Any, overlap_result: Dict[str, Any], reasons: List[str]):
        now_iso = datetime.now(timezone.utc).isoformat()
        now_ts = int(datetime.now(timezone.utc).timestamp())
        if mint not in self.pending_risky_tokens:
            self.pending_risky_tokens[mint] = {
                "first_seen": now_iso,
                "first_seen_ts": now_ts,
                "last_checked": now_iso,
                "attempts": 1,
                "reasons": reasons.copy(),
                "overlap_result": overlap_result
            }
        else:
            entry = self.pending_risky_tokens[mint]
            entry["last_checked"] = now_iso
            entry["attempts"] = entry.get("attempts", 0) + 1
            entry["reasons"] = list(dict.fromkeys(entry.get("reasons", []) + reasons))

        # persist into scheduling store
        try:
            self.scheduling_store.update_token_state(mint, {
                "status": "probation",
                "probation_first_seen": self.pending_risky_tokens[mint]["first_seen"],
                "probation_last_checked": self.pending_risky_tokens[mint]["last_checked"],
                "probation_attempts": self.pending_risky_tokens[mint]["attempts"],
                "probation_reasons": self.pending_risky_tokens[mint]["reasons"],
                "launch_time": getattr(start, "block_time", int(datetime.now(timezone.utc).timestamp()))
            })
        except Exception:
            # scheduling store might not be critical; continue
            pass

        # ensure a single probation task per mint
        if mint in self._probation_tasks and not self._probation_tasks[mint].done():
            if self.debug:
                print(f"Probation: updated existing probation for {mint}; reasons={reasons}")
            return

        task = asyncio.create_task(self._probation_recheck_loop(mint, start))
        self._probation_tasks[mint] = task
        if self.debug:
            print(f"Probation: started probation for {mint}; reasons={reasons}")

    async def _probation_recheck_loop(self, mint: str, start: Any):
        first_seen_ts = int(self.pending_risky_tokens.get(mint, {}).get("first_seen_ts", int(datetime.now(timezone.utc).timestamp())))
        deadline = first_seen_ts + 24 * 3600
        interval = 5 * 60  # 5 minutes

        while True:
            now_ts = int(datetime.now(timezone.utc).timestamp())
            if now_ts >= deadline:
                if self.debug:
                    print(f"Probation: {mint} exceeded 6h probation -> dropping")
                self.pending_risky_tokens.pop(mint, None)
                try:
                    self.scheduling_store.update_token_state(mint, {
                        "status": "dropped",
                        "dropped_at": datetime.now(timezone.utc).isoformat(),
                        "probation_final": True
                    })
                    self._scheduled.discard(mint) # Cleanup from memory
                except Exception:
                    pass
                break
            
            # --- 🚀 MODIFIED: Call the new analysis function ---
            try:
                entry = self.pending_risky_tokens.get(mint)
                if not entry:
                    if self.debug:
                        print(f"Probation: {mint} no longer in pending list. Exiting loop.")
                    break # Token was promoted and removed
                
                if self.debug:
                    print(f"[Probation] Re-running analysis for {mint}...")
                    
                check_count = entry.get("attempts", 0)
                
                # This function will:
                # 1. Re-run RugCheck
                # 2. Re-run security gate
                # 3. If it fails again, call _start_or_update_probation (updating the entry)
                # 4. If it passes, run Helius, save, and remove from pending_risky_tokens
                await self.run_token_analysis_step(start, check_count)
                
                # Check if the token is still in probation after the analysis
                if mint not in self.pending_risky_tokens:
                    if self.debug:
                        print(f"Probation: {mint} passed during probation recheck -> exiting probation loop")
                    break # It passed and was promoted
                    
            except Exception as e:
                if self.debug:
                    print(f"Probation: error during recheck for {mint}: {e}")
                # Log error to scheduler
                try:
                    self.scheduling_store.update_token_state(mint, {
                        "last_error": str(e),
                        "last_error_at": datetime.now(timezone.utc).isoformat()
                    })
                except Exception:
                    pass
            
            await asyncio.sleep(interval)


    async def daily_dune_scheduler(self):
        """
        🚀 ASYNC Background task that runs Dune query once per day at a scheduled time.
        Runs at 2 AM UTC daily to build fresh cache with yesterday's data.
        """
        if self.debug:
            print("[DuneScheduler ASYNC] 🚀 Starting daily Dune scheduler (INFINITE POLLING)")
        
        while True:
            try:
                now = datetime.now(timezone.utc)
                
                # Calculate next 2 AM UTC
                next_run = now.replace(hour=2, minute=0, second=0, microsecond=0)
                if now >= next_run:
                    # If it's already past 2 AM today, schedule for tomorrow
                    next_run += timedelta(days=1)
                
                sleep_seconds = (next_run - now).total_seconds()
                
                if self.debug:
                    print(f"[DuneScheduler ASYNC] 🚀 Next Dune build scheduled for {next_run} (in {sleep_seconds/3600:.1f} hours)")
                
                # Sleep until scheduled time
                await asyncio.sleep(sleep_seconds)
                
                # Run the daily Dune cache build
                if self.debug:
                    print("[DuneScheduler ASYNC] 🚀 Starting scheduled daily Dune cache build")
                
                try:
                    new_token_holders = await self.dune_builder.build_today_from_dune(
                        self.token_discovery, self.holder_agg
                    )
                    self.last_dune_build = int(datetime.now(timezone.utc).timestamp())
                    
                    if self.debug:
                        print(f"[DuneScheduler ASYNC] ✅ Successfully built daily cache with {len(new_token_holders)} tokens")
                        
                except Exception as e:
                    if self.debug:
                        print(f"[DuneScheduler ASYNC] ❌ Failed to build daily Dune cache: {e}")
                        import traceback
                        traceback.print_exc()
                
            except Exception as e:
                if self.debug:
                    print(f"[DuneScheduler ASYNC] ❌ Scheduler error: {e}")
                # Sleep for 1 hour before retrying on error
                await asyncio.sleep(3600)

    async def ensure_dune_holders(self):
        """
        🚀 ASYNC: Ensure the 7-day Dune winners cache exists by loading last 7 days and
        building today's file if yesterday's data is missing or outdated.
        """
        if self.debug:
            print("[Monitor ASYNC] 🚀 Starting ensure_dune_holders()")
            
        # First, load existing 7-day cache
        token_to_top_holders, wallet_freq, loaded_days = self.dune_cache.load_last_7_days()
        
        today_key = datetime.now(timezone.utc).strftime("%Y%m%d")
        yesterday_key = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y%m%d")
        
        if self.debug:
            print(f"[Monitor ASYNC] Today: {today_key}, Yesterday: {yesterday_key}")
            print(f"[Monitor ASYNC] Loaded cache days: {loaded_days}")
            print(f"[Monitor ASYNC] Total cached tokens: {len(token_to_top_holders)}")
            print(f"[Monitor ASYNC] Total unique wallets: {len(wallet_freq)}")

        # Check if we need to build today's cache
        should_build_today = False
        
        # Build today's cache if it doesn't exist
        if today_key not in loaded_days:
            if self.debug:
                print(f"[Monitor ASYNC] 🚀 Today's cache missing ({today_key}), will build it")
            should_build_today = True
        else:
            if self.debug:
                print(f"[Monitor ASYNC] ✅ Today's cache already exists ({today_key})")

        # Also check if yesterday's data is missing (fallback)
        if yesterday_key not in loaded_days:
            if self.debug:
                print(f"[Monitor ASYNC] ⚠️ Yesterday's cache also missing ({yesterday_key}), will force build today's cache")
            should_build_today = True

        if should_build_today:
            try:
                if self.debug:
                    print("[Monitor ASYNC] 🚀 Building today's Dune holder cache with infinite polling...")
                new_token_holders = await self.dune_builder.build_today_from_dune(
                    self.token_discovery, self.holder_agg
                )
                if self.debug:
                    print(f"[Monitor ASYNC] ✅ Built today's cache with {len(new_token_holders)} tokens")
                    
                # Reload the cache after building
                token_to_top_holders, wallet_freq, loaded_days = self.dune_cache.load_last_7_days()
                if self.debug:
                    print(f"[Monitor ASYNC] ✅ After rebuild - loaded days: {loaded_days}")
                    print(f"[Monitor ASYNC] ✅ After rebuild - total tokens: {len(token_to_top_holders)}")
                    print(f"[Monitor ASYNC] ✅ After rebuild - total wallets: {len(wallet_freq)}")
                    
            except Exception as e:
                if self.debug:
                    print(f"[Monitor ASYNC] ❌ Failed to build today's Dune cache: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            if self.debug:
                print("[Monitor ASYNC] ✅ Skipping cache build - sufficient data already exists")

        return token_to_top_holders, wallet_freq, loaded_days

    async def startup_recovery(self):
        """
        🚀 REFACTORED: On startup, load scheduling state and re-launch
        the correct task for any pending, active, or probation tokens.
        """
        if self.debug:
            print("Monitor ASYNC: performing startup recovery")
            
        scheduling_state = self.scheduling_store.load()
        current_time = int(datetime.now(timezone.utc).timestamp())
        cutoff_time = current_time - (6 * 3600)
        
        self.scheduling_store.cleanup_old_states(cutoff_time)
        # Use the new async cleanup method
        await self.updater.cleanup_old_tokens_async()
        
        # Load all tracked tokens into a map for quick lookup
        try:
            # Use the new async getter
            tokens = await self.updater.get_tracked_tokens_async()
            token_map = {t.mint: t for t in tokens}
            if self.debug:
                print(f"[Recovery] Loaded {len(token_map)} tracked tokens into map.")
        except Exception as e:
            if self.debug:
                print(f"[Recovery] CRITICAL: Failed to load tracked tokens: {e}. Aborting recovery.")
            return

        recovery_tasks = []
        for token_mint, state in scheduling_state.items():
            if token_mint in self._scheduled:
                continue
            
            # Find the corresponding TradingStart object
            start_obj = token_map.get(token_mint)
            if not start_obj:
                if self.debug:
                    print(f"[Recovery] Skipping {token_mint}: Not found in tracked tokens (likely expired).")
                continue

            status = state.get("status", "unknown")
            
            if status == "pending_first" or status == "active":
                if self.debug:
                    print(f"[Recovery] Re-launching main check loop for {token_mint} (status: {status})")
                task = asyncio.create_task(self._schedule_overlap_checks_for_token(start_obj))
                recovery_tasks.append(task)
                
            elif status == "probation":
                if self.debug:
                    print(f"[Recovery] Re-launching probation loop for {token_mint}")
                # Re-hydrate the in-memory probation state to match
                self.pending_risky_tokens[token_mint] = {
                    "first_seen": state.get("probation_first_seen"),
                    "first_seen_ts": int(datetime.fromisoformat(state.get("probation_first_seen")).timestamp()),
                    "last_checked": state.get("probation_last_checked"),
                    "attempts": state.get("probation_attempts", 1),
                    "reasons": state.get("probation_reasons", ["recovered_from_probation"]),
                    "overlap_result": {"mint": token_mint, "grade": "NONE"} # Stub
                }
                task = asyncio.create_task(self._probation_recheck_loop(token_mint, start_obj))
                recovery_tasks.append(task)
                
            self._scheduled.add(token_mint)
            
        if recovery_tasks:
            if self.debug:
                print(f"Monitor ASYNC: started {len(recovery_tasks)} recovery tasks")

    async def poll_coingecko_loop(self):
        if self.debug:
            print("Monitor ASYNC: 🚀 starting CoinGecko poll loop")
        
        # Start the daily Dune scheduler as a background task
        asyncio.create_task(self.daily_dune_scheduler())
        
        # Start startup recovery
        await self.startup_recovery()
        
        while True:
            try:
                starts = await self.token_discovery.get_tokens_created_today(limit=500)
                if self.debug:
                    print(f"Monitor ASYNC: CoinGecko returned {len(starts)} tokens")
                new_tokens_scheduled = 0
                for s in starts:
                    if not s.mint:
                        continue
                    if s.mint in self._scheduled:
                        continue
                    existing_state = self.scheduling_store.get_token_state(s.mint)
                    if existing_state:
                        if self.debug:
                            print(f"Monitor ASYNC: token {s.mint} already has scheduling state, skipping")
                        continue
                    
                    # Launch the master task for this token
                    asyncio.create_task(self._schedule_overlap_checks_for_token(s))
                    self._scheduled.add(s.mint)
                    new_tokens_scheduled += 1
                    
                    # Save the initial "pending" state
                    current_time = int(datetime.now(timezone.utc).timestamp())
                    self.scheduling_store.update_token_state(s.mint, {
                        "launch_time": s.block_time or current_time,
                        "first_check_at": (s.block_time or current_time) + self.initial_check_delay_seconds,
                        "status": "pending_first",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "total_checks_completed": 0
                    })
                    
                if new_tokens_scheduled > 0 and self.debug:
                    print(f"Monitor ASYNC: scheduled overlap checks for {new_tokens_scheduled} new tokens")
                    
                try:
                    # Use the new async save method
                    await self.updater.save_trading_starts_async(starts, skip_existing=True)
                except Exception as e:
                    if self.debug:
                        print("Monitor ASYNC: updater save error", e)
                        
                current_time = time.time()
                if current_time - self.last_cleanup > 3600:
                    # Use the new async cleanup method
                    await self.updater.cleanup_old_tokens_async()
                    self.scheduling_store.cleanup_old_states()
                    await self._cleanup_finished_tasks() # Memory cleanup
                    self.last_cleanup = current_time
                    
            except Exception as e:
                if self.debug:
                    print("Monitor ASYNC: CoinGecko poll error", e)
                    traceback.print_exc()
                    
            await asyncio.sleep(self.coingecko_poll_interval_seconds)

    async def _schedule_overlap_checks_for_token(self, start: TradingStart):
        """
        🚀 REFACTORED: This is the master task for a single token's 24-hour lifecycle.
        It handles the initial delay and the repeating check loop, calling
        the unified `run_token_analysis_step` function.
        """
        now_ts = int(datetime.now(timezone.utc).timestamp())
        block_ts = int(start.block_time or now_ts)
        first_run_at = block_ts + self.initial_check_delay_seconds
        to_sleep = max(0, first_run_at - now_ts)
        
        if self.debug:
            print(f"_schedule ASYNC: token={start.mint} will first run in {to_sleep}s (at {datetime.fromtimestamp(first_run_at, timezone.utc)})")
            
        await asyncio.sleep(to_sleep)
        
        self.scheduling_store.update_token_state(start.mint, {"status": "running_first_check"})
        stop_after = block_ts + 6 * 3600
        check_count = 0
        
        while True:
            now_ts2 = int(datetime.now(timezone.utc).timestamp())
            if now_ts2 > stop_after:
                if self.debug:
                    print(f"_schedule ASYNC: token={start.mint} past 6h -> stopping scheduled checks")
                
                # Final status update
                current_state = self.scheduling_store.get_token_state(start.mint)
                if current_state.get("status") not in ["completed", "dropped"]:
                    self.scheduling_store.update_token_state(start.mint, {
                        "status": "dropped", # Dropped due to timeout, not failure
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "probation_final": True
                    })
                self._scheduled.discard(start.mint) # Cleanup from memory
                break
                
            try:
                # --- 🚀 CALL THE UNIFIED ANALYSIS FUNCTION ---
                await self.run_token_analysis_step(start, check_count)
                check_count += 1
                
                # --- Check status to see if we should stop the loop ---
                current_state = self.scheduling_store.get_token_state(start.mint)
                status = current_state.get("status")
                
                if status == "completed":
                    if self.debug:
                        print(f"_schedule ASYNC: token={start.mint} status is 'completed'. Stopping loop.")
                    self._scheduled.discard(start.mint) # Cleanup from memory
                    break # Token has passed and is finished
                
                if status == "dropped":
                    if self.debug:
                        print(f"_schedule ASYC: token={start.mint} status is 'dropped'. Stopping loop.")
                    self._scheduled.discard(start.mint) # Cleanup from memory
                    break # Token failed probation and is finished
                
                # If status is "active" or "probation", the loop continues
                if self.debug:
                    print(f"_schedule ASYNC: completed check #{check_count} for {start.mint}. Status: {status}. Next check in {self.repeat_interval_seconds}s")
                
            except Exception as e:
                if self.debug:
                    print(f"_schedule ASYNC: unhandled error in check loop for {start.mint}: {e}")
                    traceback.print_exc()
                self.scheduling_store.update_token_state(start.mint, {
                    "last_error": str(e),
                    "last_error_at": datetime.now(timezone.utc).isoformat()
                })
                
            await asyncio.sleep(self.repeat_interval_seconds)

    async def _fetch_and_calculate_overlap(self, start: TradingStart) -> Dict[str, Any]:
        """
        🚀 RENAMED: (Was check_holders_overlap)
        This function NOW ONLY does its core job: fetch holders (Helius) and
        calculate overlap against the Dune cache.
        It is ONLY called *after* the pre-Helius security gate has passed.
        """
        if self.debug:
            print(f"_fetch_and_calculate_overlap: computing for {start.mint}")

        # Ensure Dune winners cache available
        token_to_top_holders, wallet_freq, loaded_days = self.dune_cache.load_last_7_days()
        if self.debug:
            print(f"_fetch_and_calculate_overlap: using {len(loaded_days)} cached days, {len(token_to_top_holders)} tokens in memory")

        # Merge winners union and compute total frequency sum
        winners_union: Set[str] = set()
        for holders in token_to_top_holders.values():
            winners_union.update(holders)
        total_winner_wallets = len(winners_union)
        total_winner_weights = sum(wallet_freq.values()) if wallet_freq else 0

        # Fetch holders for the target token and apply hybrid sampling rule
        try:
            # --- THIS IS THE HELIUS CALL ---
            holders_list = await self.holder_agg.get_token_holders(start.mint, limit=1000, max_pages=2, decimals=None)
        except Exception as e:
            if self.debug:
                print(f"_fetch_and_calculate_overlap: failed to fetch holders for {start.mint}: {e}")
            return {"error": "fetch_holders_failed", "error_details": str(e), "grade": "NONE"}

        n = len(holders_list)
        if n <= 200:
            top_holders = [h.get("wallet") for h in holders_list if h.get("wallet")]
        else:
            top10 = max(1, int(n * 0.10))
            top_holders = [h.get("wallet") for h in holders_list[: min(500, top10)] if h.get("wallet")]

        top_set = set(top_holders)
        overlap = top_set.intersection(winners_union)
        overlap_count = len(overlap)
        top_count = len(top_set)

        # distinct concentration (overlap_count / total distinct winner wallets)
        concentration = (overlap_count / total_winner_wallets * 100.0) if total_winner_wallets else 0.0
        overlap_pct = (overlap_count / top_count * 100.0) if top_count > 0 else 0.0

        # weighted concentration (sum of wallet frequencies for overlapping wallets / total wallet frequencies)
        overlap_weight = sum(wallet_freq.get(w, 0) for w in overlap) if wallet_freq else 0
        weighted_concentration = (overlap_weight / total_winner_weights * 100.0) if total_winner_weights else 0.0

        grade = calculate_overlap_grade(
                overlap_count=overlap_count,
                overlap_percentage=overlap_pct, 
                concentration=concentration,
                weighted_concentration=weighted_concentration,
                total_new_holders=top_count,
                total_winner_wallets=total_winner_wallets
            )
        summary = {
            "token": start.mint,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "top_holders_checked": top_count,
            "overlap_count": overlap_count,
            "overlap_percentage": round(overlap_pct, 2),
            "concentration": round(concentration, 2),
            "weighted_concentration": round(weighted_concentration, 2),
            "total_winner_wallets": total_winner_wallets,
            "overlap_wallets_sample": list(overlap)[:20],
            "grade": grade,
            "detected_via": start.detected_via,
            "block_time": start.block_time,
            "token_metadata": {
                k: v for k, v in {
                    "name": start.extra.get("name") if start.extra else None,
                    "symbol": start.extra.get("symbol") if start.extra else None,
                    "fdv_usd": start.fdv_usd,
                    "volume_usd": start.volume_usd,
                    "source_dex": start.source_dex,
                    "price_change_percentage": start.price_change_percentage,
                }.items() if v is not None
            }
        }
        if self.debug:
            print(f"_fetch_and_calculate_overlap: {start.mint} overlap {overlap_count}/{top_count} ({overlap_pct:.2f}%) distinct_conc {concentration:.2f}% weighted_conc {weighted_concentration:.2f}% grade={grade}")
        return summary

# -----------------------
# Main loop wiring
# -----------------------
async def main_loop():
    """
    Initializes all components and starts the monitor's main polling loop.
    Manages the AIOHTTP session lifecycle.
    """
    # ------------------ 🚀 CHANGE 6: Manage session lifecycle here ------------------
    # The session is created here and passed into the Monitor,
    # which then uses it for all API calls (RugCheck, DexScreener).
    async with aiohttp.ClientSession() as http_session:
        try:
            sol_client = SolanaAlphaClient()
            ok = await sol_client.test_connection()
            print("🚀 Solana RPC ok:", ok)
            if not ok:
                print("⚠️ Solana RPC connection failed. Check Helius keys/status.")
                # We can continue, but holder_agg will likely fail.

            td = TokenDiscovery(
                client=sol_client,
                coingecko_pro_api_key=COINGECKO_PRO_API_KEY,
                dune_api_key=DUNE_API_KEY,
                dune_query_id=DUNE_QUERY_ID,
                dune_cache_file="./data/dune_recent.pkl",
                timestamp_cache_file="./data/last_timestamp.pkl",
                debug=True
            )
            holder_agg = HolderAggregator(sol_client, debug=True)
            
            # --- UPDATER INSTANCE: This now uses the new SQLite class ---
            updater = JobLibTokenUpdater(data_dir="./data/token_data", expiry_hours=6, debug=True)
            # ---
            
            dune_cache = DuneWinnersCache(cache_dir="./data/dune_cache", debug=True, supabase_bucket=BUCKET_NAME)
            dune_builder = DuneWinnersBuilder(cache=dune_cache, debug=True, max_concurrency=8)
            overlap_store = OverlapStore(filepath="./data/overlap_results.pkl", debug=True)
            scheduling_store = SchedulingStore(filepath="./data/scheduling_state.pkl", debug=True)

            try:
                # Assuming models are in the 'models' directory relative to execution
                ml_classifier = SolanaTokenPredictor(model_dir='models') 
            except Exception as e:
                print(f"❌ CRITICAL: Failed to load ML models: {e}")
                print("--- 💀 Token Monitor Halted ---")
                return

            # Initialize the main Monitor orchestrator
            monitor = Monitor(
                sol_client=sol_client,
                token_discovery=td,
                holder_agg=holder_agg,
                updater=updater,
                dune_cache=dune_cache,
                dune_builder=dune_builder,
                overlap_store=overlap_store,
                scheduling_store=scheduling_store,
                http_session=http_session, # Pass the managed session
                ml_classifier=ml_classifier, # Pass the ML classifier
                coingecko_poll_interval_seconds=30,
                initial_check_delay_seconds=600, # 10 minutes
                repeat_interval_seconds=1800, # 30 minutes
                debug=True
            )

            # Pre-load/build the Dune winners cache on startup
            print("🚀 Ensuring Dune winners cache is populated...")
            await monitor.ensure_dune_holders()
            print("✅ Dune winners cache is ready.")

            # Start the main polling loop (which also handles recovery)
            print("🚀 Starting main CoinGecko polling loop...")
            await monitor.poll_coingecko_loop()

        except RuntimeError as e:
            print(f"\n❌ CRITICAL RUNTIME ERROR: {e}")
            print("--- 💀 Token Monitor Halted ---")
        except Exception as e:
            print(f"\n❌ UNHANDLED EXCEPTION in main_loop: {e}")
            traceback.print_exc()
            print("--- 💀 Token Monitor Halted ---")

if __name__ == "__main__":
    print("--- 🚀 Starting Token Monitor ---")
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\n🛑 Token Monitor interrupted, exiting gracefully")
    except Exception as e:
        # This catches errors during asyncio.run() itself, if any
        print(f"\n❌ UNHANDLED TOP-LEVEL EXCEPTION: {e}")
        traceback.print_exc()
        print("--- 💀 Token Monitor Halted ---")