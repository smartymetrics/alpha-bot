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

# NOTE: keep dune_client import guarded
try:
    from dune_client.client import DuneClient
except Exception:
    DuneClient = None

BUCKET_NAME = os.getenv("SUPABASE_BUCKET", "monitor-data")



# At the top of token_monitor.py add:

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
    def __init__(self, rpc_url: str):
        self.rpc_url = rpc_url
        self.headers = {"Content-Type": "application/json"}

    async def make_rpc_call(self, method: str, params: List[Any]) -> Dict[str, Any]:
        payload = {"jsonrpc": "2.0", "id": "1", "method": method, "params": params}
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.rpc_url, json=payload, headers=self.headers, timeout=40) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except Exception as e:
                return {"error": str(e)}

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
# Token discovery (CoinGecko)
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
        self.coingecko_pro_api_key = "CG-nm5ynnbGBWTugQhVY2vNMWbP" #coingecko_pro_api_key or os.environ.get("GECKO_API")
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
            print("TokenDiscovery initialized with CoinGecko Pro")

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

    def fetch_dune_force_refresh(self) -> list[dict]:
        """Force a new execution of the Dune query and return fresh results."""
        if not self.dune_client or not self.dune_query_id:
            raise RuntimeError("Dune client or query_id not configured")
        if self.debug:
            print(f"[Dune] forcing new execution for query {self.dune_query_id}")
        
        try:
            # Import the Query class from dune_client
            from dune_client.types import QueryParameter
            from dune_client.query import QueryBase
            
            # Create a proper Query object - try different approaches based on dune_client version
            query = None
            
            # Method 1: Try using QueryBase directly
            try:
                query = QueryBase(query_id=self.dune_query_id)
                if self.debug:
                    print(f"[Dune] created QueryBase object: {query}")
            except Exception as e:
                if self.debug:
                    print(f"[Dune] QueryBase creation failed: {e}")
            
            # Method 2: Try using the client's refresh_into_dataframe approach
            if not query:
                try:
                    if self.debug:
                        print(f"[Dune] trying refresh_into_dataframe approach")
                    # Use refresh_into_dataframe which handles the Query object creation internally
                    df = self.dune_client.refresh_into_dataframe(self.dune_query_id)
                    if df is not None and not df.empty:
                        rows = df.to_dict('records')
                        if self.debug:
                            print(f"[Dune] refresh_into_dataframe returned {len(rows)} rows")
                        return rows
                except Exception as e:
                    if self.debug:
                        print(f"[Dune] refresh_into_dataframe failed: {e}")
            
            # Method 3: If we have a Query object, try execute_query
            if query:
                try:
                    execution = self.dune_client.execute_query(query)
                    
                    if self.debug:
                        print(f"[Dune] execution object: {execution}")
                    
                    # Extract execution ID
                    execution_id = None
                    if hasattr(execution, 'execution_id'):
                        execution_id = execution.execution_id
                    elif hasattr(execution, 'id'):
                        execution_id = execution.id
                    elif isinstance(execution, str):
                        execution_id = execution
                    
                    if execution_id:
                        # Poll for completion
                        max_attempts = 60
                        for attempt in range(max_attempts):
                            try:
                                status = self.dune_client.get_execution_status(execution_id)
                                
                                state = None
                                if hasattr(status, 'state'):
                                    state = status.state
                                elif hasattr(status, 'execution_status'):
                                    state = status.execution_status
                                elif isinstance(status, str):
                                    state = status
                                
                                if self.debug and attempt % 6 == 0:
                                    print(f"[Dune] status: {state}")
                                
                                if state in ("QUERY_STATE_COMPLETED", "completed", "EXECUTION_STATE_COMPLETED"):
                                    break
                                elif state in ("QUERY_STATE_FAILED", "failed", "EXECUTION_STATE_FAILED"):
                                    raise RuntimeError(f"Query failed: {state}")
                                
                                time.sleep(5)
                            except Exception as e:
                                if self.debug:
                                    print(f"[Dune] polling error: {e}")
                                break
                        
                        # Get results
                        try:
                            payload = self.dune_client.get_result(execution_id)
                            rows = self._rows_from_dune_payload(payload)
                            if rows:
                                if self.debug:
                                    print(f"[Dune] force refresh successful: {len(rows)} rows")
                                return rows
                        except Exception as e:
                            if self.debug:
                                print(f"[Dune] get_result failed: {e}")
                except Exception as e:
                    if self.debug:
                        print(f"[Dune] execute_query with Query object failed: {e}")
            
            # Method 4: Fallback to latest result
            if self.debug:
                print("[Dune] falling back to get_latest_result")
            payload = self.dune_client.get_latest_result(self.dune_query_id)
            rows = self._rows_from_dune_payload(payload)
            if self.debug:
                print(f"[Dune] fallback returned {len(rows)} rows")
            return rows
            
        except Exception as e:
            if self.debug:
                print(f"[Dune] all methods failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Final fallback - return empty list
            return []
            
    def get_tokens_launched_yesterday_cached(self, cache_max_age_days: int = 7) -> List[TradingStart]:
        """
        Return list of TradingStart objects for tokens Dune reports as launched yesterday.
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

        # Fetch fresh data if needed
        if need_fetch:
            try:
                if self.debug:
                    print("[Dune] Fetching fresh data from Dune API")
                rows = self.fetch_dune_force_refresh()
            except Exception as e:
                if self.debug:
                    print(f"[Dune] fetch failure: {e}")
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
                    print(f"[Dune/cache] cached fresh data for yesterday={current_yesterday}")
            except Exception as e:
                if self.debug:
                    print(f"[Dune/cache] write failed: {e}")

            starts = rows_to_trading_starts(rows, current_yesterday)
            if self.debug:
                print(f"[Dune] found {len(starts)} tokens for yesterday={current_yesterday} after fresh fetch")
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
# Dune 7-day rolling cache + builder (new)
# -----------------------
class DuneWinnersCache:
    """
    Rolling, per-day cache of Dune "winner" token holders.
    Files are stored under: ./data/dune_cache/dune_cache_YYYYMMDD.pkl

    load_last_7_days() returns:
      token_to_top_holders: dict[str, list[str]]
      wallet_freq: dict[str, int]  (how many times a wallet appeared across day/token slices)
      fetched_days: list[str]      (YYYYMMDD loaded)
    """
    def __init__(self, cache_dir: str = "./data/dune_cache", debug: bool = False):
        self.cache_dir = cache_dir
        self.debug = debug
        os.makedirs(self.cache_dir, exist_ok=True)

    def _today_key(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d")

    def _path_for(self, yyyymmdd: str) -> str:
        return os.path.join(self.cache_dir, f"dune_cache_{yyyymmdd}.pkl")

    def save_today(self, token_to_top_holders: Dict[str, List[str]]):
        """Save today's token->holders snapshot to a per-day file."""
        y = self._today_key()
        obj = {
            "token_to_top_holders": _sanitize_maybe(token_to_top_holders),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "day": y,
        }
        try:
            joblib.dump(obj, self._path_for(y))
            if self.debug:
                tot_wallets = len({w for v in token_to_top_holders.values() for w in v})
                print(f"[DuneCache] saved {len(token_to_top_holders)} tokens, ~{tot_wallets} unique wallets for {y}")
        except Exception as e:
            if self.debug:
                print(f"[DuneCache] save_today failed: {e}")

    def load_last_7_days(self) -> Tuple[Dict[str, List[str]], Dict[str, int], List[str]]:
        """Load and merge per-day files for the past 7 UTC dates. Deletes files older than 7 days."""
        days = [(datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y%m%d") for i in range(7)]
        token_to_top_holders: Dict[str, List[str]] = {}
        wallet_freq: Dict[str, int] = defaultdict(int)
        loaded_days: List[str] = []

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

        # cleanup older files
        for fname in os.listdir(self.cache_dir):
            if not fname.startswith("dune_cache_") or not fname.endswith(".pkl"):
                continue
            ymd = fname[len("dune_cache_"):-4]
            if ymd not in days:
                try:
                    os.remove(os.path.join(self.cache_dir, fname))
                    if self.debug:
                        print(f"[DuneCache] removed old cache file {fname}")
                except Exception as e:
                    if self.debug:
                        print(f"[DuneCache] cleanup failed for {fname}: {e}")

        if self.debug:
            uniq_wallets = len(wallet_freq)
            print(f"[DuneCache] Loaded {len(loaded_days)} days: {len(token_to_top_holders)} tokens, {uniq_wallets} unique wallets")

        return token_to_top_holders, dict(wallet_freq), loaded_days

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
        Fetch tokens from Dune (yesterday), fetch sampled holders concurrently, save today's per-day cache,
        and return the token->holders mapping for today.
        """
        if self.debug:
            print(f"[DuneBuilder] Starting build_today_from_dune")
            
        # Get tokens from yesterday's Dune data
        starts = token_discovery.get_tokens_launched_yesterday_cached()
        if self.debug:
            print(f"[DuneBuilder] Dune returned {len(starts)} tokens for yesterday")
            if starts:
                print(f"[DuneBuilder] Sample tokens: {[s.mint for s in starts[:3]]}")
            else:
                print(f"[DuneBuilder] No tokens returned by Dune - checking why...")
                # Debug why no tokens
                try:
                    # Try to load the raw cache to see what's in it
                    import os
                    cache_path = token_discovery.dune_cache_file
                    if os.path.exists(cache_path):
                        import joblib
                        cache_obj = joblib.load(cache_path)
                        rows = cache_obj.get("rows", [])
                        print(f"[DuneBuilder] Raw cache has {len(rows)} rows")
                        if rows:
                            print(f"[DuneBuilder] Sample row keys: {list(rows[0].keys()) if rows else 'none'}")
                            # Check dates in the data
                            import pandas as pd
                            df = pd.DataFrame(rows)
                            if "first_buy_date" in df.columns:
                                dates = pd.to_datetime(df["first_buy_date"], errors="coerce").dt.date.unique()
                                print(f"[DuneBuilder] Dates in cache: {sorted(dates)}")
                                yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1))
                                print(f"[DuneBuilder] Looking for yesterday: {yesterday}")
                    else:
                        print(f"[DuneBuilder] No cache file exists at {cache_path}")
                except Exception as e:
                    print(f"[DuneBuilder] Debug cache inspection failed: {e}")
        
        if not starts:
            if self.debug:
                print("[DuneBuilder] No tokens from Dune, returning empty mapping")
            # Still save an empty mapping for today to mark that we tried
            self.cache.save_today({})
            return {}
        
        if self.debug:
            print(f"[DuneBuilder] Processing {len(starts)} tokens to fetch holders")
        
        # Process tokens to get holders
        token_to_top_holders: Dict[str, List[str]] = {}
        successful_fetches = 0
        failed_fetches = 0
        
        # Process first few tokens sequentially for debugging
        for i, s in enumerate(starts[:5]):
            if not s.mint:
                if self.debug:
                    print(f"[DuneBuilder] Token {i}: No mint address, skipping")
                continue
            
            try:
                if self.debug:
                    print(f"[DuneBuilder] Token {i}: Processing {s.mint}")
                
                holders = await self._fetch_top_sampled_holders(holder_agg, s.mint)
                token_to_top_holders[s.mint] = holders
                
                if holders:
                    successful_fetches += 1
                    if self.debug:
                        print(f"[DuneBuilder] Token {i} ({s.mint}): Got {len(holders)} holders")
                else:
                    failed_fetches += 1
                    if self.debug:
                        print(f"[DuneBuilder] Token {i} ({s.mint}): No holders returned")
                        
                # Add a small delay to avoid overwhelming the RPC
                await asyncio.sleep(0.5)
                
            except Exception as e:
                failed_fetches += 1
                if self.debug:
                    print(f"[DuneBuilder] Token {i} ({s.mint}): Exception - {e}")
                    import traceback
                    traceback.print_exc()

        if self.debug:
            print(f"[DuneBuilder] Sequential batch complete: {successful_fetches} successful, {failed_fetches} failed")

        # If sequential processing worked, process remaining tokens concurrently
        if successful_fetches > 0 and len(starts) > 5:
            if self.debug:
                print("[DuneBuilder] Sequential processing worked, processing remaining tokens concurrently")
            
            remaining_starts = starts[5:]
            tasks = []
            mints = []
            
            for s in remaining_starts:
                if not s.mint:
                    continue
                mints.append(s.mint)
                tasks.append(asyncio.create_task(self._fetch_top_sampled_holders(holder_agg, s.mint)))

            if tasks:
                if self.debug:
                    print(f"[DuneBuilder] Starting {len(tasks)} concurrent holder fetches")
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                concurrent_successful = 0
                concurrent_failed = 0
                
                for mint, res in zip(mints, results):
                    if isinstance(res, Exception):
                        concurrent_failed += 1
                        if self.debug:
                            print(f"[DuneBuilder] Concurrent fetch failed for {mint}: {res}")
                        continue
                    
                    token_to_top_holders[mint] = res
                    if res:
                        concurrent_successful += 1
                    else:
                        concurrent_failed += 1
                        
                if self.debug:
                    print(f"[DuneBuilder] Concurrent batch complete: {concurrent_successful} successful, {concurrent_failed} failed")
        
        elif successful_fetches == 0:
            if self.debug:
                print("[DuneBuilder] Sequential processing failed completely - there might be an RPC issue")

        # Calculate final statistics
        total_tokens_with_holders = len([holders for holders in token_to_top_holders.values() if holders])
        total_unique_wallets = len({w for holders in token_to_top_holders.values() for w in holders})
        
        if self.debug:
            print(f"[DuneBuilder] Final result: {len(token_to_top_holders)} tokens processed")
            print(f"[DuneBuilder] - {total_tokens_with_holders} tokens have holders")
            print(f"[DuneBuilder] - {total_unique_wallets} unique wallet addresses")
            print(f"[DuneBuilder] - Average holders per token: {total_unique_wallets / max(1, total_tokens_with_holders):.1f}")
        
        # Save to today's per-day cache file
        try:
            self.cache.save_today(token_to_top_holders)
            if self.debug:
                print(f"[DuneBuilder] Successfully saved today's cache")
        except Exception as e:
            if self.debug:
                print(f"[DuneBuilder] Failed to save today's cache: {e}")
        
        return token_to_top_holders

# -----------------------
# Holder aggregation
# -----------------------
class HolderAggregator:
    def __init__(self, client: SolanaAlphaClient):
        self.client = client

    async def get_token_holders(self, token_mint: str, *, sleep_between: float = 0.15, limit: int = 1000, max_pages: Optional[int] = None, decimals: Optional[int] = None) -> List[Dict[str, Any]]:
        page = 1
        owner_balances = defaultdict(int)
        owner_token_account_counts = defaultdict(int)
        while True:
            payload_params = {"mint": token_mint, "page": page, "limit": limit, "displayOptions": {}}
            data = await self.client.make_rpc_call("getTokenAccounts", payload_params)
            token_accounts = data.get("result", {}).get("token_accounts", [])
            if not token_accounts:
                break
            for ta in token_accounts:
                owner = ta.get("owner") or ta.get("address")
                amt_raw = ta.get("amount", 0)
                if "account" in ta and isinstance(ta["account"], dict):
                    acct = ta["account"]
                    owner = owner or acct.get("owner")
                    amt_raw = acct.get("amount", 0)
                if isinstance(amt_raw, dict):
                    amt_raw = int(float(amt_raw.get("amount") or amt_raw.get("uiAmount", 0)))
                else:
                    try:
                        amt_raw = int(amt_raw)
                    except Exception:
                        amt_raw = int(float(amt_raw)) if amt_raw else 0
                if owner:
                    owner_balances[owner] += amt_raw
                    owner_token_account_counts[owner] += 1
            page += 1
            if max_pages and page > max_pages:
                break
            await asyncio.sleep(sleep_between)
        holders = []
        for owner, raw in owner_balances.items():
            human_balance = raw / (10 ** decimals) if decimals else None
            holders.append({"wallet": owner, "balance_raw": raw, "balance": human_balance, "balance_formatted": (f"{human_balance:,.{decimals}f}" if human_balance is not None and decimals is not None else str(raw)), "num_token_accounts": owner_token_account_counts[owner]})
        holders.sort(key=lambda x: x["balance_raw"], reverse=True)
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
        starts = token_discovery.get_tokens_launched_yesterday_cached()
        if self.debug:
            print(f"DuneHolderCache: found {len(starts)} dune tokens")
        mapping: Dict[str, Set[str]] = {}
        for s in starts:
            if not s.mint:
                continue
            try:
                holders = await holder_agg.get_token_holders(s.mint, limit=1000, max_pages=2, decimals=None)
                top_wallets = {h["wallet"] for h in holders[:top_n_per_token]}
                mapping[s.mint] = top_wallets
                if self.debug:
                    print(f"DuneHolderCache: token {s.mint} -> {len(top_wallets)} top holders")
            except Exception as e:
                if self.debug:
                    print(f"DuneHolderCache: error fetching holders for {s.mint}: {e}")
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

from datetime import datetime, timedelta, timezone

def prune_old_overlap_entries(data: dict, expiry_hours: int = 24) -> dict:
    """
    Removes individual overlap check entries older than expiry_hours.
    Keeps tokens if they have at least one recent check.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=expiry_hours)
    pruned = {}

    for mint, entries in data.items():
        new_entries = []
        for entry in entries:
            ts_str = entry.get("ts")
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                ts = None

            if ts and ts > cutoff:
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

    def save(self, obj: Dict[str, Any]):
        """Save overlap results to disk and periodically upload to Supabase."""
        now = time.time()
        with self._lock:  # ensure only one thread saves at a time
            try:
                # prune old entries before saving
                obj = prune_old_overlap_entries(obj, expiry_hours=24)

                # always save locally
                joblib.dump(_sanitize_maybe(obj), self.filepath)

                # throttle uploads to Supabase (once every 2 minutes)
                if now - self._last_upload < 120:
                    return

                upload_file(self.filepath, BUCKET_NAME)
                self._last_upload = now

                if self.debug:
                    print(f"OverlapStore: saved and uploaded at {time.ctime(now)}")

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

import pandas as pd

def safe_load_overlap(overlap_store):
    obj = overlap_store.load()
    if obj is None:
        return {}
    if isinstance(obj, pd.DataFrame):
        # Convert DataFrame to a dict with token_mint as keys
        return obj.to_dict(orient="list")
    if not isinstance(obj, dict):
        return {}
    return obj

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
    
    # CRITICAL: Extremely high overlap indicating potential manipulation
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
    
    # HIGH: Significant overlap suggesting coordinated activity
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
# Monitor
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
        *,
        coingecko_poll_interval_seconds: int = 30,
        initial_check_delay_seconds: int = 2 * 3600,
        repeat_interval_seconds: int = 6 * 3600,
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

    async def ensure_dune_holders(self):
        """
        Ensure the 7-day Dune winners cache exists by loading last 7 days and
        building today's file if yesterday's data is missing or outdated.
        """
        if self.debug:
            print("[Monitor] Starting ensure_dune_holders()")
            
        # First, load existing 7-day cache
        token_to_top_holders, wallet_freq, loaded_days = self.dune_cache.load_last_7_days()
        
        today_key = datetime.now(timezone.utc).strftime("%Y%m%d")
        yesterday_key = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y%m%d")
        
        if self.debug:
            print(f"[Monitor] Today: {today_key}, Yesterday: {yesterday_key}")
            print(f"[Monitor] Loaded cache days: {loaded_days}")
            print(f"[Monitor] Total cached tokens: {len(token_to_top_holders)}")
            print(f"[Monitor] Total unique wallets: {len(wallet_freq)}")

        # Check if we need to build today's cache
        should_build_today = False
        
        # Build today's cache if it doesn't exist
        if today_key not in loaded_days:
            if self.debug:
                print(f"[Monitor] Today's cache missing ({today_key}), will build it")
            should_build_today = True
        else:
            if self.debug:
                print(f"[Monitor] Today's cache already exists ({today_key})")

        # Also check if yesterday's data is missing (fallback)
        if yesterday_key not in loaded_days:
            if self.debug:
                print(f"[Monitor] Yesterday's cache also missing ({yesterday_key}), will force build today's cache")
            should_build_today = True

        if should_build_today:
            try:
                if self.debug:
                    print("[Monitor] Building today's Dune holder cache...")
                new_token_holders = await self.dune_builder.build_today_from_dune(
                    self.token_discovery, self.holder_agg
                )
                if self.debug:
                    print(f"[Monitor] Built today's cache with {len(new_token_holders)} tokens")
                    
                # Reload the cache after building
                token_to_top_holders, wallet_freq, loaded_days = self.dune_cache.load_last_7_days()
                if self.debug:
                    print(f"[Monitor] After rebuild - loaded days: {loaded_days}")
                    print(f"[Monitor] After rebuild - total tokens: {len(token_to_top_holders)}")
                    print(f"[Monitor] After rebuild - total wallets: {len(wallet_freq)}")
                    
            except Exception as e:
                if self.debug:
                    print(f"[Monitor] Failed to build today's Dune cache: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            if self.debug:
                print("[Monitor] Skipping cache build - sufficient data already exists")

        return token_to_top_holders, wallet_freq, loaded_days

    async def startup_recovery(self):
        if self.debug:
            print("Monitor: performing startup recovery")
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
            self._scheduled.add(token_mint)
        if recovery_tasks:
            if self.debug:
                print(f"Monitor: started {len(recovery_tasks)} recovery tasks")

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
            return
        try:
            res = await self.check_holders_overlap(token_start)
            obj = safe_load_overlap(self.overlap_store)
            obj.setdefault(token_start.mint, []).append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "result": res,
                "check_type": "first_check"
            })
            self.overlap_store.save(obj)
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
            print("Monitor: starting CoinGecko poll loop")
        await self.startup_recovery()
        while True:
            try:
                starts = await self.token_discovery.get_tokens_created_today(limit=500)
                if self.debug:
                    print(f"Monitor: CoinGecko returned {len(starts)} tokens")
                new_tokens_scheduled = 0
                for s in starts:
                    if not s.mint:
                        continue
                    if s.mint in self._scheduled:
                        continue
                    existing_state = self.scheduling_store.get_token_state(s.mint)
                    if existing_state:
                        if self.debug:
                            print(f"Monitor: token {s.mint} already has scheduling state, skipping")
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
                    print(f"Monitor: scheduled overlap checks for {new_tokens_scheduled} new tokens")
                try:
                    await self.updater.save_trading_starts_async(starts, skip_existing=True)
                except Exception as e:
                    if self.debug:
                        print("Monitor: updater save error", e)
                current_time = time.time()
                if current_time - self.last_cleanup > 3600:
                    await self.updater.cleanup_old_tokens_async()
                    self.scheduling_store.cleanup_old_states()
                    self.last_cleanup = current_time
            except Exception as e:
                if self.debug:
                    print("Monitor: CoinGecko poll error", e)
                    traceback.print_exc()
            await asyncio.sleep(self.coingecko_poll_interval_seconds)

    async def _schedule_overlap_checks_for_token(self, start: TradingStart):
        now_ts = int(datetime.now(timezone.utc).timestamp())
        block_ts = int(start.block_time or now_ts)
        first_run_at = block_ts + self.initial_check_delay_seconds
        to_sleep = max(0, first_run_at - now_ts)
        if self.debug:
            print(f"_schedule: token={start.mint} will first run in {to_sleep}s (at {datetime.fromtimestamp(first_run_at, timezone.utc)})")
        await asyncio.sleep(to_sleep)
        self.scheduling_store.update_token_state(start.mint, {"status": "running_first_check"})
        stop_after = block_ts + 24 * 3600
        check_count = 0
        while True:
            now_ts2 = int(datetime.now(timezone.utc).timestamp())
            if now_ts2 > stop_after:
                if self.debug:
                    print(f"_schedule: token={start.mint} past 24h -> stopping scheduled checks")
                self.scheduling_store.update_token_state(start.mint, {
                    "status": "completed",
                    "completed_at": datetime.now(timezone.utc).isoformat()
                })
                break
            try:
                res = await self.check_holders_overlap(start)
                check_count += 1
                obj = safe_load_overlap(self.overlap_store)
                obj.setdefault(start.mint, []).append({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "result": res,
                    "check_type": "first_check" if check_count == 1 else "repeat_check"
                })
                self.overlap_store.save(obj)
                next_check_time = now_ts2 + self.repeat_interval_seconds
                self.scheduling_store.update_token_state(start.mint, {
                    "status": "active",
                    "last_completed_check": now_ts2,
                    "next_scheduled_check": next_check_time,
                    "total_checks_completed": check_count
                })
                if self.debug:
                    print(f"_schedule: completed check #{check_count} for {start.mint}, grade: {res.get('grade', 'N/A')}, overlap: {res.get('overlap_count', 0)} wallets")
            except Exception as e:
                if self.debug:
                    print(f"_schedule: overlap check error for {start.mint}: {e}")
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
            print(f"check_holders_overlap: computing for {start.mint}")

        # Ensure Dune winners cache available
        token_to_top_holders, wallet_freq, loaded_days = self.dune_cache.load_last_7_days()
        if self.debug:
            print(f"check_holders_overlap: using {len(loaded_days)} cached days, {len(token_to_top_holders)} tokens in memory")

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
                print(f"check_holders_overlap: failed to fetch holders for {start.mint}: {e}")
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
            print(f"check_holders_overlap: {start.mint} overlap {overlap_count}/{top_count} ({overlap_pct:.2f}%) distinct_conc {concentration:.2f}% weighted_conc {weighted_concentration:.2f}% grade={grade}")
        return summary

# -----------------------
# Main loop wiring
# -----------------------
async def main_loop():
    from dotenv import load_dotenv
    load_dotenv()
    HELIUS_API_KEY = os.environ.get("HELIUS_API_KEY")
    COINGECKO_PRO_API_KEY = os.environ.get ("GECKO_API")
    DUNE_API_KEY = os.environ.get("DUNE_API_KEY")
    DUNE_QUERY_ID = int(os.environ.get("DUNE_QUERY_ID") or 5668844)
    BASE_URL = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"
    sol_client = SolanaAlphaClient(BASE_URL)
    ok = await sol_client.test_connection()
    print("Solana RPC ok:", ok)

    td = TokenDiscovery(
        client=sol_client,
        coingecko_pro_api_key=COINGECKO_PRO_API_KEY,
        dune_api_key=DUNE_API_KEY,
        dune_query_id=DUNE_QUERY_ID,
        dune_cache_file="./data/dune_recent.pkl",
        timestamp_cache_file="./data/last_timestamp.pkl",
        debug=True
    )
    holder_agg = HolderAggregator(sol_client)
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
        coingecko_poll_interval_seconds=30,
        initial_check_delay_seconds=2 * 3600,
        repeat_interval_seconds=6 * 3600,
        debug=True,
    )

    # Ensure we have today's Dune winners file and a 7-day rolling view
    await monitor.ensure_dune_holders()

    # Start monitoring loop
    await monitor.poll_coingecko_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("Interrupted, exiting")





