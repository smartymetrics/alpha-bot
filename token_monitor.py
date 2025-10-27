"""
Enhanced token monitor with persistent scheduling and hierarchical overlap scoring.
Now using CoinGecko Pro API for new token discovery.
Features:
 - SchedulingStore: Persistent scheduling state using joblib
 - Enhanced overlap grading with CRITICAL/HIGH/MEDIUM/LOW classifications
 - Concentration-based grading
 - Startup recovery logic
 - Persistent timestamp tracking to avoid refetching
 - 24-hour token expiration
 - Rolling 7-day Dune winners cache (per-day files)
 - Wallet frequency and weighted concentration
 - Hybrid holder sampling (<=200 all; >200 top10% capped 500)
 - Bounded concurrency for holder fetching (Semaphore)
 - FULLY ASYNC DUNE INTEGRATION with INFINITE POLLING (NO TIMEOUTS)
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

load_dotenv()

PROBATION_TOP_N = int(os.getenv("PROBATION_TOP_N", "10"))
PROBATION_THRESHOLD_PCT = float(os.getenv("PROBATION_THRESHOLD_PCT", "40"))

COINGECKO_PRO_API_KEY = os.environ.get("GECKO_API")
DUNE_API_KEY = os.environ.get("DUNE_API_KEY")
DUNE_QUERY_ID = int(os.environ.get("DUNE_QUERY_ID"))

def _load_keys_list(env_name: str) -> List[str]:
    """Load comma-separated API keys from environment variable"""
    value = os.getenv(env_name, "")
    if not value:
        return []
    return [k.strip() for k in value.split(",") if k.strip()]

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
    """Retry an async function with exponential backoff and jitter."""
    for attempt in range(1, retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            wait = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.3)
            print(f"[Retry] {func.__name__} failed (attempt {attempt}/{retries}): {e}. Retrying in {wait:.2f}s")
            await asyncio.sleep(wait)
    raise RuntimeError(f"{func.__name__} failed after {retries} retries")

# NOTE: keep dune_client import guarded
try:
    from dune_client.client import DuneClient
except Exception:
    DuneClient = None

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
# Token discovery (CoinGecko + FULLY ASYNC Dune)
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
        # CoinGecko
        self.coingecko_pro_api_key = coingecko_pro_api_key or os.environ.get("GECKO_API")
        self.coingecko_url = "https://pro-api.coingecko.com/api/v3/onchain/networks/solana/new_pools"
        self.last_processed_timestamp = self._load_last_timestamp(timestamp_cache_file)
        self.timestamp_cache_file = timestamp_cache_file
        # Dune
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
            print("TokenDiscovery initialized with CoinGecko Pro and ASYNC Dune (NO TIMEOUTS)")

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
        """
        Fetch the latest Dune query result (synchronous wrapper).
        Returns a list of row dicts from Dune.
        """
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
        🚀 FULLY ASYNC: Force a new execution of the Dune query and return fresh results.
        🚀 NO TIMEOUTS: Uses infinite polling with while True
        🚀 NON-BLOCKING: Uses await asyncio.sleep instead of time.sleep
        """
        if not self.dune_client or not self.dune_query_id:
            raise RuntimeError("Dune client or query_id not configured")

        query_id = int(self.dune_query_id)

        if self.debug:
            print(f"[Dune ASYNC] 🚀 Starting infinite polling for query {query_id} (NO TIMEOUTS)")

        try:
            from dune_client.query import QueryBase

            # Method 1: Try execute_query with INFINITE polling (NO TIMEOUT)
            try:
                query = QueryBase(query_id=query_id)
                execution = self.dune_client.execute_query(query)
                if self.debug:
                    print(f"[Dune ASYNC] 🚀 Started execution: {execution}")

                # Get execution ID
                execution_id = None
                if hasattr(execution, 'execution_id'):
                    execution_id = execution.execution_id
                elif hasattr(execution, 'id'):
                    execution_id = execution.id
                elif isinstance(execution, str):
                    execution_id = execution

                if execution_id:
                    if self.debug:
                        print(f"[Dune ASYNC] 🚀 Polling execution {execution_id} with INFINITE timeout")

                    attempt = 0
                    while True:  # 🚀 No timeout - runs until completion
                        try:
                            attempt += 1
                            status = self.dune_client.get_execution_status(execution_id)

                            # normalize state
                            raw_state = getattr(status, 'state',
                                        getattr(status, 'execution_status',
                                        getattr(status, 'execution_state', str(status))))
                            state_str = str(raw_state).lower()

                            if self.debug and attempt % 20 == 0:
                                print(f"[Dune ASYNC] 🚀 Attempt {attempt}: {state_str} (infinite polling)")

                            if "completed" in state_str:
                                if self.debug:
                                    print(f"[Dune ASYNC] ✅ Query completed after {attempt} attempts ({attempt * 3 / 60:.1f} minutes)")
                                break
                            elif "failed" in state_str:
                                raise RuntimeError(f"Query execution failed: {raw_state}")

                            await asyncio.sleep(3)

                        except Exception as e:
                            if self.debug:
                                print(f"[Dune ASYNC] ⚠️ Polling error: {e}")
                            break

                    # Get results
                    try:
                        payload = self.dune_client.get_result(execution_id)
                        rows = self._rows_from_dune_payload(payload)
                        if rows:
                            if self.debug:
                                print(f"[Dune ASYNC] ✅ Execute query successful: {len(rows)} rows after {attempt * 3 / 60:.1f} minutes")
                            return rows
                    except Exception as e:
                        if self.debug:
                            print(f"[Dune ASYNC] ❌ get_result failed: {e}")
            except Exception as e:
                if self.debug:
                    print(f"[Dune ASYNC] ❌ execute_query failed: {e}")

            # Method 2: Fallback to get_latest_result
            if self.debug:
                print("[Dune ASYNC] 🔄 Falling back to get_latest_result")
            payload = self.dune_client.get_latest_result(query_id)
            rows = self._rows_from_dune_payload(payload)
            if self.debug:
                print(f"[Dune ASYNC] 🔄 Fallback returned {len(rows)} rows")
                if rows:
                    print(f"[Dune ASYNC] Sample fallback row keys: {list(rows[0].keys())}")
            return rows

        except Exception as e:
            if self.debug:
                print(f"[Dune ASYNC] ❌ All methods failed: {e}")
                import traceback
                traceback.print_exc()
            return []

    async def get_tokens_launched_yesterday_cached(self, cache_max_age_days: int = 7) -> List[TradingStart]:
        """
        🚀 NOW ASYNC: Return list of TradingStart objects for tokens Dune reports as launched yesterday.
        This method retains compatibility with previous behavior (legacy single-file cache).
        """
        cache_path = self.dune_cache_file

        def rows_to_trading_starts(rows: List[Dict[str, Any]], target_yesterday: datetime.date) -> List[TradingStart]:
            if not rows:
                return []
            df = pd.DataFrame(rows)
            date_col = None
            mint_col = None
            for c in ("first_buy_date", "first_buy_date_utc", "block_date", "first_trade_date"):
                if c in df.columns:
                    date_col = c
                    break
            for c in ("mint_address", "mint", "token_bought_mint_address"):
                if c in df.columns:
                    mint_col = c
                    break
            if date_col is None or mint_col is None:
                return []
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            filtered = df[df[date_col].dt.date == target_yesterday]
            out = []
            for _, row in filtered.iterrows():
                try:
                    dt = pd.to_datetime(row[date_col])
                    if pd.isna(dt):
                        continue
                    if dt.tzinfo is None:
                        dt = dt.tz_localize("UTC")
                    ts = int(dt.tz_convert("UTC").timestamp())
                except Exception:
                    continue
                out.append(
                    TradingStart(mint=row[mint_col], block_time=ts, program_id="dune", detected_via="dune", extra={date_col: str(row[date_col])})
                )
            return out

        current_yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1))
        need_fetch = True

        # Check if we have valid cached data
        if os.path.exists(cache_path):
            try:
                cache_obj = joblib.load(cache_path)
                cached_rows = cache_obj.get("rows", [])
                fetched_at = None
                
                # Parse fetched_at timestamp
                try:
                    fetched_at = datetime.fromisoformat(cache_obj["fetched_at"])
                    if fetched_at.tzinfo is None:
                        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
                except Exception:
                    fetched_at = None

                # Primary validation: Check if cached data contains yesterday's tokens
                if cached_rows and fetched_at:
                    df = pd.DataFrame(cached_rows)
                    if "first_buy_date" in df.columns:
                        try:
                            first_buy = pd.to_datetime(df["first_buy_date"].iloc[0]).date()
                            if first_buy == current_yesterday:
                                # Secondary validation: Check if cache is not too old (fallback safety)
                                age_days = (datetime.now(timezone.utc) - fetched_at).days
                                if age_days <= cache_max_age_days:
                                    if self.debug:
                                        print(f"[Dune/cache] cached first_buy_date {first_buy} matches yesterday and age {age_days} <= {cache_max_age_days} days, using cache")
                                    need_fetch = False
                                    starts = rows_to_trading_starts(cached_rows, current_yesterday)
                                    if starts:  # Only use cache if it actually produces results
                                        return starts
                                    else:
                                        if self.debug:
                                            print("[Dune/cache] cache data didn't produce any tokens, fetching fresh")
                                        need_fetch = True
                                else:
                                    if self.debug:
                                        print(f"[Dune/cache] cached data too old: {age_days} > {cache_max_age_days} days")
                            else:
                                if self.debug:
                                    print(f"[Dune/cache] cached first_buy_date {first_buy} != yesterday {current_yesterday} -> need fresh data")
                        except Exception as e:
                            if self.debug:
                                print(f"[Dune/cache] failed to validate cached data: {e}")
                                
                # Fallback validation using fetched_at and target_yesterday (legacy compatibility)
                elif fetched_at and "target_yesterday" in cache_obj:
                    try:
                        cached_yesterday = datetime.fromisoformat(cache_obj["target_yesterday"]).date()
                        if cached_yesterday == current_yesterday:
                            age_days = (datetime.now(timezone.utc) - fetched_at).days
                            if age_days <= cache_max_age_days:
                                starts = rows_to_trading_starts(cached_rows, current_yesterday)
                                if starts:
                                    if self.debug:
                                        print(f"[Dune/cache] using cached data via fallback validation for yesterday={current_yesterday}")
                                    need_fetch = False
                                    return starts
                    except Exception as e:
                        if self.debug:
                            print(f"[Dune/cache] fallback validation failed: {e}")
                            
            except Exception as e:
                if self.debug:
                    print(f"[Dune/cache] error reading cache: {e}")

        # 🚀 ASYNC FETCH: Fetch fresh data if needed
        if need_fetch:
            try:
                if self.debug:
                    print("[Dune ASYNC] 🚀 Fetching fresh data from Dune API with infinite polling")
                rows = await self.fetch_dune_force_refresh()  # 🚀 NOW ASYNC
            except Exception as e:
                if self.debug:
                    print(f"[Dune ASYNC] ❌ fetch failure: {e}")
                return []

            # Save fresh data to cache
            try:
                cache_obj = {
                    "rows": rows,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "target_yesterday": current_yesterday.isoformat()
                }
                joblib.dump(cache_obj, cache_path)
                if self.debug:
                    print(f"[Dune/cache] ✅ cached fresh data for yesterday={current_yesterday}")
            except Exception as e:
                if self.debug:
                    print(f"[Dune/cache] ❌ write failed: {e}")

            starts = rows_to_trading_starts(rows, current_yesterday)
            if self.debug:
                print(f"[Dune ASYNC] ✅ found {len(starts)} tokens for yesterday={current_yesterday} after fresh fetch")
            return starts

        return []

    # ---------------- CoinGecko ----------------
    async def _fetch_coingecko_new_pools(self, limit: int = 500, timeout: int = 15) -> List[Dict[str, Any]]:
        """Fetch the latest 20 pools from CoinGecko and print their mint addresses, ignoring cache."""
        headers = {"accept": "application/json", "x-cg-pro-api-key": self.coingecko_pro_api_key}
        url = self.coingecko_url  # No pagination

        if self.debug:
            print("[CoinGecko] Fetching latest 20 pools (ignoring cache)")

        async with aiohttp.ClientSession() as sess:
            async with sess.get(url, headers=headers, timeout=timeout) as resp:
                resp.raise_for_status()
                data = await resp.json()
                pools = data.get("data", [])

        if not pools:
            if self.debug:
                print("[CoinGecko] No pools returned")
            return []

        all_pools = []
        for pool in pools:
            # Extract mint address
            base_token = pool["relationships"]["base_token"]["data"]
            mint = base_token["id"].replace("eth_", "").replace("solana_", "")

            # Extract timestamp
            block_time = self._parse_pool_created_at(pool["attributes"]["pool_created_at"])

            if self.debug:
                print(f"  - Mint: {mint} | Created: {pool['attributes']['pool_created_at']} | TS: {block_time}")

            all_pools.append(pool)

        if self.debug:
            print(f"[CoinGecko] Total pools fetched: {len(all_pools)}")

        return all_pools

    @staticmethod
    def _parse_pool_created_at(val: Any) -> Optional[int]:
        if not val:
            return None
        try:
            ts_str = str(val).replace("Z", "+00:00")
            dt = datetime.fromisoformat(ts_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except Exception as e:
            print(f"[ParseError] Could not parse timestamp {val}: {e}")
            return None

    @staticmethod
    def _utc_day_bounds_for_date(dt: Optional[datetime] = None) -> Tuple[int, int]:
        d = (dt or datetime.now(timezone.utc)).astimezone(timezone.utc)
        start = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(days=1) - timedelta(seconds=1)
        return int(start.timestamp()), int(end.timestamp())

    def _parse_coingecko_pool(self, pool: Dict[str, Any]) -> TradingStart:
        attributes = pool["attributes"]
        base_token = pool["relationships"]["base_token"]["data"]
        mint = base_token["id"].replace("eth_", "").replace("solana_", "")  # Handle both ETH and Solana
        block_time = self._parse_pool_created_at(attributes["pool_created_at"])
        return TradingStart(
            mint=mint,
            block_time=block_time,
            program_id="coingecko",
            detected_via="coingecko",
            extra={
                "name": attributes["name"].split(" / ")[0],
                "fdv_usd": attributes["fdv_usd"],
                "market_cap_usd": attributes.get("market_cap_usd") or attributes["fdv_usd"],
                "volume_usd": attributes["volume_usd"]["h24"],
                "source_dex": pool["relationships"]["dex"]["data"]["id"],
                "price_change_percentage": attributes["price_change_percentage"]["h24"],
            },
            fdv_usd=attributes["fdv_usd"],
            volume_usd=attributes["volume_usd"]["h24"],
            source_dex=pool["relationships"]["dex"]["data"]["id"],
            price_change_percentage=attributes["price_change_percentage"]["h24"],
        )

    async def get_tokens_created_today(self, limit: int = 500) -> List[TradingStart]:
        pools = await self._fetch_coingecko_new_pools(limit=limit)
        out = []
        now = int(datetime.now(timezone.utc).timestamp())
        cutoff = now - 24 * 3600  # only include pools launched in last 24 hours

        for pool in pools:
            block_time = self._parse_pool_created_at(pool["attributes"]["pool_created_at"])
            if not block_time:
                continue

            # ✅ filter: only pools created in last 24 hours
            if block_time < cutoff:
                continue

            ts = self._parse_coingecko_pool(pool)
            out.append(ts)

            # update last processed timestamp
            if self.last_processed_timestamp is None or block_time > self.last_processed_timestamp:
                self.last_processed_timestamp = block_time

        if out:
            self._save_last_timestamp()

        if self.debug:
            print(f"[CoinGecko] {len(out)} tokens launched")

        return out

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
class JobLibTokenUpdater:
    def __init__(self, data_dir: str = "./data/token_data", expiry_hours: int = 24, debug: bool = False):
        self.data_dir = os.path.abspath(data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        self.tokens_file = os.path.join(self.data_dir, "tokens.pkl")
        self.expiry_hours = expiry_hours
        self.debug = debug

    def _load_tokens(self) -> List[Any]:
        if os.path.exists(self.tokens_file):
            try:
                data = joblib.load(self.tokens_file)
                if isinstance(data, list):
                    return data
            except Exception as e:
                if self.debug:
                    print("JobLibTokenUpdater: load error", e)
        return []

    def _save_tokens(self, tokens: List[Any]):
        try:
            safe_tokens = [_normalize(t) for t in tokens]
            joblib.dump(safe_tokens, self.tokens_file)
        except Exception as e:
            print("JobLibTokenUpdater: save error", e)
            traceback.print_exc()
            debug_path = self.tokens_file + ".debug.json"
            try:
                with open(debug_path, "w", encoding="utf-8") as f:
                    json.dump([asdict(t) if isinstance(t, TradingStart) else str(t) for t in tokens], f, indent=2, default=str)
                print(f"Saved debug snapshot to {debug_path}")
            except Exception as ee:
                print("Also failed to dump debug snapshot:", ee)

    async def save_trading_starts_async(self, trading_starts: List[TradingStart], skip_existing: bool = True) -> Dict[str, int]:
        existing = self._load_tokens()
        existing_mints: Set[str] = set()
        for t in existing:
            if isinstance(t, dict):
                m = t.get("mint")
            elif isinstance(t, TradingStart):
                m = t.mint
            else:
                m = None
            if m:
                existing_mints.add(m)

        saved = 0
        skipped = 0
        errors = 0
        for s in trading_starts:
            try:
                if skip_existing and s.mint in existing_mints:
                    skipped += 1
                    continue
                existing.append(s)
                saved += 1
            except Exception:
                errors += 1
        self._save_tokens(existing)
        if self.debug:
            print(f"JobLibTokenUpdater: saved={saved} skipped={skipped} errors={errors} total_now={len(existing)}")
        return {"saved": saved, "skipped": skipped, "errors": errors}

    async def cleanup_old_tokens_async(self) -> int:
        tokens = self._load_tokens()
        if not tokens:
            return 0
        now = datetime.now(timezone.utc)
        cutoff = int((now - timedelta(hours=self.expiry_hours)).timestamp())
        kept: List[Any] = []
        for t in tokens:
            ts = 0
            if isinstance(t, dict):
                ts = int(t.get("block_time", 0) or 0)
            elif isinstance(t, TradingStart):
                ts = int(t.block_time or 0)
            if ts > cutoff:
                kept.append(t)
        deleted = len(tokens) - len(kept)
        if deleted:
            self._save_tokens(kept)
            if self.debug:
                print(f"JobLibTokenUpdater: cleaned {deleted} tokens older than {self.expiry_hours} hours")
        return deleted

    async def get_tracked_tokens_async(self, limit: Optional[int] = None) -> List[TradingStart]:
        tokens = self._load_tokens()
        norm: List[TradingStart] = []
        for t in tokens:
            if isinstance(t, TradingStart):
                norm.append(t)
            elif isinstance(t, dict):
                try:
                    norm.append(TradingStart(**t))
                except Exception:
                    allowed = {"mint","block_time","program_id","detected_via","extra","fdv_usd","volume_usd","source_dex","price_change_percentage"}
                    clean = {k:v for k,v in t.items() if k in allowed}
                    norm.append(TradingStart(**clean))
        norm.sort(key=lambda x: x.block_time or 0, reverse=True)
        if limit:
            norm = norm[:limit]
        return norm

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


def prune_old_overlap_entries(data: dict, expiry_hours: int = 24) -> dict:
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

    def save(self, obj: Dict[str, Any], expiry_hours: int = 24):
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

                # throttle uploads to Supabase (once every 2 minutes)
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
        cutoff = cutoff_timestamp or int((now - timedelta(hours=24)).timestamp())
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
        *,
        coingecko_poll_interval_seconds: int = 30,
        initial_check_delay_seconds: int = 1800, # 30 minutes
        repeat_interval_seconds: int = 3600, # 1 hour
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
        cutoff_ts = now_ts - (24 * 3600)  # 24 hours ago

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

    async def _run_rugcheck_check(self, mint: str) -> Dict[str, Any]:
        """
        Check token security using RugCheck API with comprehensive risk analysis.
        This version aggregates liquidity across all markets.
        """
        url = f"https://api.rugcheck.xyz/v1/tokens/{mint}/report"
        # ------------------ 🚀 CHANGE 4: Use self.http_session directly ------------------
        session = self.http_session
        async with self._api_sema:
            try:
                async with session.get(url, timeout=15) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        return {"ok": False, "error": f"rugcheck_status_{resp.status}", "error_text": text}
                    data = await resp.json()
            except Exception as e:
                return {"ok": False, "error": "rugcheck_exception", "error_text": str(e)}

        # Extract key security metrics
        # --- Probation Check ---
        probation_result = evaluate_probation_from_rugcheck(data)

        top_holders = data.get("topHolders", [])
        rugged = data.get("rugged", False)
        
        # Authorities and Creator Balance
        freeze_authority = data.get("freezeAuthority")
        mint_authority = data.get("mintAuthority")
        has_authorities = bool(freeze_authority or mint_authority)
        creator_balance = data.get("creatorBalance", 0)

        # Transfer fee check
        transfer_fee_pct = data.get("transferFee", {}).get("pct", 0)
        
        # Holder metrics
        total_holders = data.get("totalHolders", 0)
        top1_holder_pct = top_holders[0].get("pct", 0) if top_holders else 0.0

        # --- Liquidity Aggregation ---
        markets = data.get("markets", [])
        total_lp_locked_usd = 0.0
        total_lp_usd = 0.0
        lp_lock_details = []

        for market in markets:
            lp_data = market.get("lp", {})
            if lp_data:
                lp_locked_usd = float(lp_data.get("lpLockedUSD", 0))
                lp_unlocked_usd = float(lp_data.get("lpUnlocked", 0))
                
                # Total LP for this market is locked + unlocked
                lp_total_usd_market = lp_locked_usd + lp_unlocked_usd
                
                # Aggregate totals
                total_lp_locked_usd += lp_locked_usd
                total_lp_usd += lp_total_usd_market
                
                lp_lock_details.append({
                    "market_type": market.get("marketType"),
                    "locked_usd": lp_locked_usd,
                    "total_usd": lp_total_usd_market,
                    "locked_pct": lp_data.get("lpLockedPct", 0)
                })
        
        # Calculate overall LP lock percentage from aggregated values
        overall_lp_locked_pct = (total_lp_locked_usd / total_lp_usd * 100) if total_lp_usd > 0 else 0.0
        lp_lock_sufficient = overall_lp_locked_pct >= 95.0

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
            # Aggregated liquidity metrics
            "total_lp_usd": total_lp_usd,
            "overall_lp_locked_pct": overall_lp_locked_pct,
            "lp_lock_sufficient": lp_lock_sufficient,
            "probation": probation_result["probation"],
            "probation_meta": probation_result,
            "lp_lock_details": lp_lock_details,
            "raw": data # Keep raw data for debugging
        }


    async def _security_gate_and_route(self, start: Any, overlap_result: Dict[str, Any], overlap_store_obj: Dict[str, Any]):
        mint = getattr(start, "mint", None) or (overlap_result or {}).get("mint")
        grade = (overlap_result or {}).get("grade", "NONE")

        if grade == "NONE":
            overlap_store_obj.setdefault(mint, []).append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "result": overlap_result,
                "security": "skipped_grade_none"
            })
            try:
                self.overlap_store.save(overlap_store_obj)
            except Exception:
                pass
            return

        # Run RugCheck API call
        try:
            r = await retry_with_backoff(self._run_rugcheck_check, mint, retries=2, base_delay=0.8)
        except Exception as e:
            if self.debug:
                print(f"[Security] Exception fetching RugCheck data for {mint}: {e}")
            reasons = [f"rugcheck_exception:{str(e)}"]
            await self._start_or_update_probation(mint, start, overlap_result, reasons)
            return

        reasons = []
        
        if not r.get("ok"):
            reasons.append(f"rugcheck_error:{r.get('error')}")
            await self._start_or_update_probation(mint, start, overlap_result, reasons)
            return

        # --- Enforce Security and Liquidity Rules ---

        # Rule 0: Check for high holder concentration (probation) using the top N check.
        if r.get("probation"):
            probation_reason = r.get("probation_meta", {}).get("explanation", "Probation: High top holder concentration")
            reasons.append(probation_reason)
            if self.debug:
                print(f"[Security] {mint} FAILED probation check: {probation_reason}")

        # Rule 1: Check if token is marked as rugged
        if r.get("rugged"):
            reasons.append("rugged:true")
            if self.debug:
                print(f"[Security] {mint} FAILED check: Marked as RUGGED by RugCheck")

        # Rule 2: Check for mint/freeze authorities
        if r.get("has_authorities"):
            authorities = []
            if r.get("freeze_authority"): authorities.append("freeze")
            if r.get("mint_authority"): authorities.append("mint")
            reason_str = f"authorities:{','.join(authorities)}"
            reasons.append(reason_str)
            if self.debug:
                print(f"[Security] {mint} FAILED check: Has dangerous authorities: {', '.join(authorities)}")

        # Rule 3: Check creator token balance
        creator_balance = r.get("creator_balance", 0)
        if creator_balance > 0:
            reasons.append(f"creator_balance:{creator_balance}")
            if self.debug:
                print(f"[Security] {mint} FAILED check: Creator holds {creator_balance} tokens")

        # Rule 4: Check transfer fee (must be <= 5%)
        transfer_fee_pct = r.get("transfer_fee_pct", 0)
        if transfer_fee_pct > 5:
            reasons.append(f"transfer_fee:{transfer_fee_pct}%")
            if self.debug:
                print(f"[Security] {mint} FAILED check: Transfer fee is {transfer_fee_pct}% (max 5%)")

        # Rule 5 (REMOVED): The check for top 1 holder is now covered by the more general probation check (Rule 0).
        # top1_pct = r.get("top1_holder_pct", 0.0)
        # if top1_pct >= 40:
        #     reasons.append(f"top1_holder:{top1_pct:.1f}%")
        #     if self.debug:
        #         print(f"[Security] {mint} FAILED check: Top holder owns {top1_pct:.1f}% (max 40%)")

        # Rule 6: Check holder count (must be >= 500)
        holder_count = r.get("total_holders", 0)
        if holder_count < 500:
            reasons.append(f"holder_count:{holder_count}_req_500")
            if self.debug:
                print(f"[Security] {mint} FAILED check: Has {holder_count} holders (min 500)")

        # Rule 7: Check aggregated liquidity lock percentage (must be >= 95%)
        lp_locked_pct = r.get("overall_lp_locked_pct", 0.0)
        if lp_locked_pct < 95.0:
            reasons.append(f"lp_locked:{lp_locked_pct:.1f}%_req_95%")
            if self.debug:
                print(f"[Security] {mint} FAILED LP lock check: {lp_locked_pct:.1f}% locked (min 95%)")

        # Rule 8: Check aggregated total liquidity (must be >= $30,000)
        total_liquidity = r.get("total_lp_usd", 0.0)
        if total_liquidity < 30000.0:
            reasons.append(f"liquidity_usd:{total_liquidity:.2f}_req_30k")
            if self.debug:
                print(f"[Security] {mint} FAILED liquidity check: ${total_liquidity:,.2f} total liquidity (min $30,000)")
        
        # --- Decision Point ---
        if reasons:
            if r.get("probation_meta"):
                overlap_result["probation_meta"] = r.get("probation_meta")
            await self._start_or_update_probation(mint, start, overlap_result, reasons)
            return

        # Token passed all security checks, save result
        top1_pct = r.get("top1_holder_pct", 0.0)
        if self.debug:
            print(f"[Security] {mint} PASSED all checks: LP Locked={lp_locked_pct:.1f}%, Liquidity=${total_liquidity:,.2f}, Top Holder={top1_pct:.1f}%, Holders={holder_count}")
        
        try:
            # Get a fresh overlap analysis before saving the "passed" state
            fresh_overlap_result = await self.check_holders_overlap(start)
            
            # Fetch DexScreener data with robust retries
            dex_data = None
            try:
                dex_data = await retry_with_backoff(
                    self._run_dexscreener_check, mint, retries=8, base_delay=1.5
                )
            except Exception as e:
                if self.debug:
                    print(f"[Security] Dexscreener check for {mint} failed after all retries: {e}")
                # If it fails permanently, we'll save nulls
                dex_data = {"ok": False, "price_usd": None, "market_cap_usd": None}

            current_price_usd = dex_data.get("price_usd")
            market_cap_usd = dex_data.get("market_cap_usd")

            overlap_store_obj.setdefault(mint, []).append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "result": fresh_overlap_result,
                "security": "passed",
                "probation_meta": r.get("probation_meta"),
                "rugcheck": {
                    "top1_holder_pct": top1_pct,
                    "holder_count": holder_count,
                    "has_authorities": r.get("has_authorities"),
                    "creator_balance": creator_balance,
                    "transfer_fee_pct": transfer_fee_pct,
                    "lp_locked_pct": lp_locked_pct,
                    "total_liquidity_usd": total_liquidity,
                    "lp_lock_details": r.get("lp_lock_details", [])
                },
                "dexscreener": {
                    "current_price_usd": current_price_usd,
                    "market_cap_usd": market_cap_usd,
                }
            })
            
            if self.debug:
                print(f"Grade update for {mint}: {overlap_result.get('grade', 'N/A')} -> {fresh_overlap_result.get('grade', 'N/A')}")
                
        except Exception as e:
            if self.debug:
                print(f"Fresh overlap check failed for {mint}, using original: {e}")
            overlap_store_obj.setdefault(mint, []).append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "result": overlap_result,
                "security": "passed_stale_overlap"
            })

        self.overlap_store.save(overlap_store_obj)
        
        if mint in self.pending_risky_tokens:
            self.pending_risky_tokens.pop(mint, None)
        
        self.scheduling_store.update_token_state(mint, {
            "status": "completed",
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "security_passed": True,
            "liquidity_usd": total_liquidity,
            "lp_locked_pct": lp_locked_pct,
        })


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
            async with session.get(url, timeout=15) as resp:
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
        interval = 30 * 60  # 30 minutes

        while True:
            now_ts = int(datetime.now(timezone.utc).timestamp())
            if now_ts >= deadline:
                if self.debug:
                    print(f"Probation: {mint} exceeded 24h probation -> dropping")
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
            # run a security-only check by calling the orchestrator but reuse overlap_result
            try:
                # prepare a minimal object for overlap store usage
                obj = safe_load_overlap(self.overlap_store)
                await self._security_gate_and_route(start, self.pending_risky_tokens[mint]["overlap_result"], obj)
                if mint not in self.pending_risky_tokens:
                    if self.debug:
                        print(f"Probation: {mint} passed during probation -> exiting probation loop")
                    break
            except Exception as e:
                if self.debug:
                    print(f"Probation: error during recheck for {mint}: {e}")
                entry = self.pending_risky_tokens.get(mint)
                if entry:
                    entry["last_checked"] = datetime.now(timezone.utc).isoformat()
                    entry["attempts"] = entry.get("attempts", 0) + 1
                    try:
                        self.scheduling_store.update_token_state(mint, {
                            "probation_last_checked": entry["last_checked"],
                            "probation_attempts": entry["attempts"],
                            "last_error": str(e),
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
        if self.debug:
            print("Monitor ASYNC: performing startup recovery")
        scheduling_state = self.scheduling_store.load()
        current_time = int(datetime.now(timezone.utc).timestamp())
        cutoff_time = current_time - (24 * 3600)
        self.scheduling_store.cleanup_old_states(cutoff_time)
        await self.updater.cleanup_old_tokens_async()
        recovery_tasks = []
        for token_mint, state in scheduling_state.items():
            if token_mint in self._scheduled:
                continue
            launch_time = state.get("launch_time", 0)
            if current_time - launch_time > 24 * 3600:
                continue
            status = state.get("status", "unknown")
            if status == "pending_first":
                first_check_time = launch_time + self.initial_check_delay_seconds
                delay = max(0, first_check_time - current_time)
                if self.debug:
                    print(f"Recovery: scheduling first check for {token_mint} in {delay}s")
                task = asyncio.create_task(self._schedule_first_check_only(token_mint, delay))
                recovery_tasks.append(task)
            elif status == "active":
                next_scheduled = state.get("next_scheduled_check", 0)
                delay = max(0, next_scheduled - current_time)
                if delay <= 300:
                    delay = 0
                if self.debug:
                    print(f"Recovery: scheduling repeat check for {token_mint} in {delay}s")
                task = asyncio.create_task(self._schedule_repeat_check_only(token_mint, delay))
                recovery_tasks.append(task)
            elif status == "probation":
                # recreate minimal TradingStart and rehydrate probation loop
                try:
                    from_dataclass = globals().get("TradingStart")
                    start_obj = None
                    if from_dataclass:
                        start_obj = from_dataclass(mint=token_mint, block_time=state.get("launch_time"))
                    else:
                        # fallback minimal object
                        class _S: pass
                        start_obj = _S()
                        start_obj.mint = token_mint
                        start_obj.block_time = state.get("launch_time")
                    if self.debug:
                        print(f"Recovery: rehydrating probation for {token_mint}")
                    task = asyncio.create_task(self._probation_recheck_loop(token_mint, start_obj))
                    recovery_tasks.append(task)
                except Exception:
                    if self.debug:
                        print(f"Recovery: failed to rehydrate probation for {token_mint}")
            self._scheduled.add(token_mint)
        if recovery_tasks:
            if self.debug:
                print(f"Monitor ASYNC: started {len(recovery_tasks)} recovery tasks")

    async def _schedule_first_check_only(self, token_mint: str, delay_seconds: float):
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        tokens = await self.updater.get_tracked_tokens_async()
        token_start = None
        for t in tokens:
            if t.mint == token_mint:
                token_start = t
                break
        if not token_start:
            if self.debug:
                print(f"_schedule_first_check_only: token {token_mint} not found in tracked tokens")
            self._scheduled.discard(token_mint) # Cleanup from memory
            return
        try:
            res = await self.check_holders_overlap(token_start)
            obj = safe_load_overlap(self.overlap_store)
            # Route through security gate; it will save if passed or start probation otherwise
            await self._security_gate_and_route(token_start, res, obj)
            current_time = int(datetime.now(timezone.utc).timestamp())
            next_check = current_time + self.repeat_interval_seconds
            self.scheduling_store.update_token_state(token_mint, {
                "status": "active",
                "last_completed_check": current_time,
                "next_scheduled_check": next_check,
                "total_checks_completed": 1
            })
            asyncio.create_task(self._schedule_repeat_checks_for_token(token_start, next_check))
            if self.debug:
                print(f"_schedule_first_check_only: completed first check for {token_mint}, grade: {res.get('grade', 'N/A')}")
        
        except Exception as e:
            if self.debug:
                print(f"_schedule_first_check_only: error for {token_mint}: {e}")
                traceback.print_exc()
            self.scheduling_store.update_token_state(token_mint, {
                "status": "failed",
                "last_error": str(e),
                "last_error_at": datetime.now(timezone.utc).isoformat()
            })

    async def _schedule_repeat_check_only(self, token_mint: str, delay_seconds: float):
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        tokens = await self.updater.get_tracked_tokens_async()
        token_start = None
        for t in tokens:
            if t.mint == token_mint:
                token_start = t
                break
        if not token_start:
            if self.debug:
                print(f"_schedule_repeat_only: token {token_mint} not found in tracked tokens")
            self._scheduled.discard(token_mint) # Cleanup from memory
            return
        current_time = int(datetime.now(timezone.utc).timestamp())
        launch_time = token_start.block_time or current_time
        if current_time - launch_time > 24 * 3600:
            if self.debug:
                print(f"_schedule_repeat_check_only: token {token_mint} past 24h -> marking completed")
            self.scheduling_store.update_token_state(token_mint, {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat()
            })
            self._scheduled.discard(token_mint) # Cleanup from memory
            return
        try:
            res = await self.check_holders_overlap(token_start)
            obj = safe_load_overlap(self.overlap_store)
            obj.setdefault(token_start.mint, []).append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "result": res,
                "check_type": "repeat_check"
            })
            self.overlap_store.save(obj)
            next_check = current_time + self.repeat_interval_seconds
            state = self.scheduling_store.get_token_state(token_mint)
            check_count = state.get("total_checks_completed", 0) + 1
            self.scheduling_store.update_token_state(token_mint, {
                "last_completed_check": current_time,
                "next_scheduled_check": next_check,
                "total_checks_completed": check_count
            })
            asyncio.create_task(self._schedule_repeat_checks_for_token(token_start, next_check))
            if self.debug:
                print(f"_schedule_repeat_check_only: completed repeat check #{check_count} for {token_mint}, grade: {res.get('grade', 'N/A')}")
        except Exception as e:
            if self.debug:
                print(f"_schedule_repeat_check_only: error for {token_mint}: {e}")

    async def _schedule_repeat_checks_for_token(self, start: TradingStart, first_check_at: int):
        current_time = int(datetime.now(timezone.utc).timestamp())
        launch_time = start.block_time or current_time
        stop_after = launch_time + 24 * 3600
        check_time = first_check_at
        while check_time < stop_after:
            delay = max(0, check_time - int(datetime.now(timezone.utc).timestamp()))
            if delay > 0:
                await asyncio.sleep(delay)
            now = int(datetime.now(timezone.utc).timestamp())
            if now >= stop_after:
                if self.debug:
                    print(f"_schedule_repeat_checks: token {start.mint} past 24h -> stopping")
                self.scheduling_store.update_token_state(start.mint, {
                    "status": "completed",
                    "completed_at": datetime.now(timezone.utc).isoformat()
                })
                self._scheduled.discard(start.mint) # Cleanup from memory
                break
            try:
                res = await self.check_holders_overlap(start)
                obj = safe_load_overlap(self.overlap_store)
                obj.setdefault(start.mint, []).append({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "result": res,
                    "check_type": "repeat_check"
                })
                self.overlap_store.save(obj)
                state = self.scheduling_store.get_token_state(start.mint)
                check_count = state.get("total_checks_completed", 0) + 1
                next_check_time = now + self.repeat_interval_seconds
                self.scheduling_store.update_token_state(start.mint, {
                    "last_completed_check": now,
                    "next_scheduled_check": next_check_time,
                    "total_checks_completed": check_count
                })
                if self.debug:
                    print(f"_schedule_repeat_checks: completed check #{check_count} for {start.mint}, grade: {res.get('grade', 'N/A')}")
                check_time = next_check_time
            except Exception as e:
                if self.debug:
                    print(f"_schedule_repeat_checks: error for {start.mint}: {e}")
                check_time = now + self.repeat_interval_seconds

    async def poll_coingecko_loop(self):
        if self.debug:
            print("Monitor ASYNC: 🚀 starting CoinGecko poll loop")
        
        # Start the daily Dune scheduler as a background task
        asyncio.create_task(self.daily_dune_scheduler())
        
        # Start startup recovery (commented out as per original)
        # await self.startup_recovery()
        
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
                    asyncio.create_task(self._schedule_overlap_checks_for_token(s))
                    self._scheduled.add(s.mint)
                    new_tokens_scheduled += 1
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
                    await self.updater.save_trading_starts_async(starts, skip_existing=True)
                except Exception as e:
                    if self.debug:
                        print("Monitor ASYNC: updater save error", e)
                current_time = time.time()
                if current_time - self.last_cleanup > 3600:
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
        now_ts = int(datetime.now(timezone.utc).timestamp())
        block_ts = int(start.block_time or now_ts)
        first_run_at = block_ts + self.initial_check_delay_seconds
        to_sleep = max(0, first_run_at - now_ts)
        if self.debug:
            print(f"_schedule ASYNC: token={start.mint} will first run in {to_sleep}s (at {datetime.fromtimestamp(first_run_at, timezone.utc)})")
        await asyncio.sleep(to_sleep)
        self.scheduling_store.update_token_state(start.mint, {"status": "running_first_check"})
        stop_after = block_ts + 24 * 3600
        check_count = 0
        while True:
            now_ts2 = int(datetime.now(timezone.utc).timestamp())
            if now_ts2 > stop_after:
                if self.debug:
                    print(f"_schedule ASYNC: token={start.mint} past 24h -> stopping scheduled checks")
                self.scheduling_store.update_token_state(start.mint, {
                    "status": "completed",
                    "completed_at": datetime.now(timezone.utc).isoformat()
                })
                self._scheduled.discard(start.mint) # Cleanup from memory
                break
            try:
                res = await self.check_holders_overlap(start)
                check_count += 1
                obj = safe_load_overlap(self.overlap_store)
                # Route through security gate; it will save if passed or start probation otherwise
                await self._security_gate_and_route(start, res, obj)
                next_check_time = now_ts2 + self.repeat_interval_seconds
                self.scheduling_store.update_token_state(start.mint, {
                    "status": "active",
                    "last_completed_check": now_ts2,
                    "next_scheduled_check": next_check_time,
                    "total_checks_completed": check_count
                })
                if self.debug:
                    print(f"_schedule ASYNC: completed check #{check_count} for {start.mint}, grade: {res.get('grade', 'N/A')}, overlap: {res.get('overlap_count', 0)} wallets")
            except Exception as e:
                if self.debug:
                    print(f"_schedule ASYNC: overlap check error for {start.mint}: {e}")
                self.scheduling_store.update_token_state(start.mint, {
                    "last_error": str(e),
                    "last_error_at": datetime.now(timezone.utc).isoformat()
                })
            await asyncio.sleep(self.repeat_interval_seconds)

    async def check_holders_overlap(self, start: TradingStart, top_k_holders: int = 500) -> Dict[str, Any]:
        """
        Check overlap between the token's top holders and the merged 7-day Dune winners.
        Returns a summary dict with distinct and weighted concentration metrics.
        """
        if self.debug:
            print(f"check_holders_overlap ASYNC: computing for {start.mint}")

        # Ensure Dune winners cache available
        token_to_top_holders, wallet_freq, loaded_days = self.dune_cache.load_last_7_days()
        if self.debug:
            print(f"check_holders_overlap ASYNC: using {len(loaded_days)} cached days, {len(token_to_top_holders)} tokens in memory")

        # Merge winners union and compute total frequency sum
        winners_union: Set[str] = set()
        for holders in token_to_top_holders.values():
            winners_union.update(holders)
        total_winner_wallets = len(winners_union)
        total_winner_weights = sum(wallet_freq.values()) if wallet_freq else 0

        # Fetch holders for the target token and apply hybrid sampling rule
        try:
            holders_list = await self.holder_agg.get_token_holders(start.mint, limit=1000, max_pages=2, decimals=None)
        except Exception as e:
            if self.debug:
                print(f"check_holders_overlap ASYNC: failed to fetch holders for {start.mint}: {e}")
            return {"error": "fetch_holders_failed", "error_details": str(e)}

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
                    "fdv_usd": start.fdv_usd,
                    "volume_usd": start.volume_usd,
                    "source_dex": start.source_dex,
                    "price_change_percentage": start.price_change_percentage,
                }.items() if v is not None
            }
        }
        if self.debug:
            print(f"check_holders_overlap ASYNC: {start.mint} overlap {overlap_count}/{top_count} ({overlap_pct:.2f}%) distinct_conc {concentration:.2f}% weighted_conc {weighted_concentration:.2f}% grade={grade}")
        return summary

# -----------------------
# Main loop wiring
# -----------------------
async def main_loop():
    # ------------------ 🚀 CHANGE 6: Manage session lifecycle here ------------------
    async with aiohttp.ClientSession() as http_session:
        try:
            sol_client = SolanaAlphaClient()
            ok = await sol_client.test_connection()
            print("🚀 Solana RPC ok:", ok)

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
            updater = JobLibTokenUpdater(data_dir="./data/token_data", expiry_hours=24, debug=True)
            dune_cache = DuneWinnersCache(cache_dir="./data/dune_cache", debug=True)
            dune_builder = DuneWinnersBuilder(cache=dune_cache, debug=True, max_concurrency=8)
            overlap_store = OverlapStore(filepath="./data/overlap_results.pkl", debug=True)
            scheduling_store = SchedulingStore(filepath="./data/scheduling_state.pkl", debug=True)
            
            monitor = Monitor(
                sol_client=sol_client,
                token_discovery=td,
                holder_agg=holder_agg,
                updater=updater,
                dune_cache=dune_cache,
                dune_builder=dune_builder,
                overlap_store=overlap_store,
                scheduling_store=scheduling_store,
                http_session=http_session, # Pass the session here
                coingecko_poll_interval_seconds=30,
                initial_check_delay_seconds=30 * 60, # 30 minutes
                repeat_interval_seconds=1 * 3600, # 1 hour
                debug=True,
            )

            # Start both tasks concurrently
            dune_task = asyncio.create_task(monitor.ensure_dune_holders())
            coingecko_task = asyncio.create_task(monitor.poll_coingecko_loop())
            
            print("🚀 Starting CoinGecko polling and Dune cache building concurrently...")
            
            # Wait for both tasks (poll_coingecko_loop runs forever)
            await asyncio.gather(dune_task, coingecko_task)

        except asyncio.CancelledError:
            print("🚀 Main loop was cancelled. Shutting down gracefully.")
        finally:
            # The 'async with' block ensures the http_session is closed automatically.
            print("🚀 Main loop has finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("🚀 Interrupted, exiting")
