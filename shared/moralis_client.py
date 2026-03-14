#!/usr/bin/env python3
"""
moralis_client.py
Client for Moralis API with intelligent key rotation, backoff, and error handling.
- 400 (Quota): Blacklist key until next UTC day, rotate.
- 429 (Rate Limit): Exponential backoff, retry with same key.
- 5xx (Server Error): Exponential backoff, retry.
- Tracks last key per wallet to ensure rotation.

UPDATED: Now uses the /account/.../swaps endpoint to fetch 'buy'
transactions since 12am UTC of the current day.

SMART MONEY (v2): Replaced broken EVM entities endpoint with Solana-native
wallet profitability/summary endpoint.
  Endpoint: GET https://solana-gateway.moralis.io/account/mainnet/{address}/profitability/summary
  Response: total_realized_profit_usd, total_realized_profit_percentage,
            total_buys, total_sells, total_trade_volume, total_count_of_trades
  - PnL data is cached to disk (joblib) with a configurable TTL.
  - Only wallets in the top-N Dune Winners list are ever fetched (credit gate).
  - Cache is synced to Supabase on a 5-minute throttle.
  - ALL original trading/swap methods are completely unchanged.
"""

import asyncio
import aiohttp
import random
import os
import time
import joblib
import threading
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Set, Optional

# Supabase helpers — use the real functions from supabase_utils
# upload_file(local_path, bucket, remote_path) — deletes old, uploads new
# download_file(save_path, remote_name, bucket) — conditional GET, 304-aware
try:
    from supabase_utils import upload_file as _supabase_upload_file
    from supabase_utils import download_file as _supabase_download_file
    _SUPABASE_AVAILABLE = True
except ImportError:
    _SUPABASE_AVAILABLE = False

# ---------------------------------------------------------------------------
# PnL-based Smart Money classification thresholds
# ---------------------------------------------------------------------------
# These thresholds classify Solana wallets by realized trading performance
# rather than EVM entity labels (which don't exist for Solana wallets).
#
# Tier ladder (highest match wins):
#   ELITE      — proven top-tier profitable trader
#   STRONG     — consistently profitable, high volume
#   ACTIVE     — net positive, decent trade count
#   NOISE      — insufficient data or unprofitable
#   NEGATIVE   — confirmed loss-maker or bot-like behavior
#
# All thresholds are conservative and can be tuned via env vars.

PNL_ELITE_PROFIT_USD       = float(os.getenv("SM_PNL_ELITE_PROFIT_USD",   "50000"))   # $50k+ realized
PNL_ELITE_WIN_RATE         = float(os.getenv("SM_PNL_ELITE_WIN_RATE",     "55"))      # 55%+ win rate
PNL_STRONG_PROFIT_USD      = float(os.getenv("SM_PNL_STRONG_PROFIT_USD",  "10000"))   # $10k+ realized
PNL_STRONG_WIN_RATE        = float(os.getenv("SM_PNL_STRONG_WIN_RATE",    "50"))      # 50%+ win rate
PNL_ACTIVE_PROFIT_USD      = float(os.getenv("SM_PNL_ACTIVE_PROFIT_USD",  "1000"))    # $1k+ realized
PNL_MIN_TRADES             = int(os.getenv("SM_PNL_MIN_TRADES",           "5"))       # at least 5 trades
PNL_NEGATIVE_LOSS_USD      = float(os.getenv("SM_PNL_NEGATIVE_LOSS_USD",  "-5000"))   # >$5k loss = noise

# Weight multipliers applied to dune_frequency in the weighted overlap score.
# Mirrors the old entity multipliers — ELITE replaces Fund/VC, etc.
PNL_TIER_WEIGHT_MULTIPLIERS: Dict[str, float] = {
    "ELITE":    3.0,
    "STRONG":   2.5,
    "ACTIVE":   1.5,
    "NOISE":    1.0,   # default, no boost
    "NEGATIVE": 0.0,   # excluded from effective overlap
}

# Which tiers are considered "positive" (boost eligible)
POSITIVE_PNL_TIERS = frozenset({"ELITE", "STRONG", "ACTIVE"})
# Which tiers are excluded from effective overlap
NEGATIVE_PNL_TIERS = frozenset({"NEGATIVE"})

# Back-compat aliases — SmartMoneyScorer imports these names
POSITIVE_ENTITY_CATEGORIES = POSITIVE_PNL_TIERS
NEGATIVE_ENTITY_CATEGORIES = NEGATIVE_PNL_TIERS

# How many days before a cached PnL record is considered stale and re-fetched.
# PnL accumulates over time — 7 days is a reasonable refresh window.
# Set to 0 to disable TTL (cache forever).
LABEL_CACHE_TTL_DAYS = int(os.getenv("SMART_MONEY_LABEL_TTL_DAYS", "7"))

# Sentinel value stored when a wallet has no PnL data (e.g. new wallet, API error).
# _cached_at=0 ensures TTL applies equally — we re-check after TTL days.
_NO_ENTITY_SENTINEL = {
    "pnl_tier":                    "NOISE",
    "total_realized_profit_usd":   None,
    "total_realized_profit_pct":   None,
    "total_trade_volume_usd":      None,
    "total_buys":                  None,
    "total_sells":                 None,
    "total_count_of_trades":       None,
    "win_rate_pct":                None,
    "is_positive":                 False,
    "is_negative":                 False,
    "_cached_at":                  0,
}


class MoralisClient:
    """
    Handle all Moralis API communication with robust error handling
    and intelligent key rotation.
    """

    def __init__(
        self,
        http_session: aiohttp.ClientSession,
        api_keys: List[str],
        debug: bool = False,
        label_cache_path: str = "./data/wallet_pnl_cache.pkl",
        supabase_bucket: str = "monitor-data",
    ):
        if not api_keys:
            raise ValueError("MoralisClient requires at least one API key.")
        self.http_session = http_session
        self.api_keys = api_keys
        self.debug = debug
        self.supabase_bucket = supabase_bucket

        # ---- Smart Money label cache ----
        self._label_cache_path = label_cache_path
        self._label_cache_remote_name = os.path.basename(label_cache_path)  # wallet_labels.pkl
        self._label_cache: Dict[str, Dict] = {}
        self._label_cache_lock = threading.Lock()
        self._label_cache_dirty = False
        self._last_supabase_upload: float = 0.0
        self._supabase_upload_interval: float = 300.0  # 5 minutes
        os.makedirs(os.path.dirname(self._label_cache_path), exist_ok=True)
        self._load_label_cache()

        # ---- Original state tracking ----
        self._current_key_index: int = 0
        self._blacklisted_keys: Dict[str, int] = {}
        self._last_key_per_wallet: Dict[str, str] = {}

        if self.debug:
            print(
                f"[MoralisClient] Initialized with {len(api_keys)} keys. "
                f"PnL cache: {len(self._label_cache)} entries."
            )

    # =========================================================================
    # Smart Money label cache helpers  (NEW — all private)
    # =========================================================================

    def _load_label_cache(self):
        """Load persistent label cache from disk (called once at init)."""
        if os.path.exists(self._label_cache_path):
            try:
                data = joblib.load(self._label_cache_path)
                if isinstance(data, dict):
                    self._label_cache = data
                    if self.debug:
                        print(f"[MoralisClient] Loaded {len(data)} wallet PnL records from local cache.")
            except Exception as e:
                if self.debug:
                    print(f"[MoralisClient] ⚠️ Label cache load failed: {e}. Starting fresh.")
                self._label_cache = {}

    def download_label_cache_from_supabase(self):
        """
        Download the PnL cache from Supabase on startup and merge with any
        local entries.  Remote wins on conflict (remote is always a superset).
        Called once synchronously from winner_monitor.py startup().

        Remote path: smart_money/wallet_pnl_cache.pkl
        This keeps Smart Money files grouped separately from overlap results
        and dune_cache/ files in the bucket.
        """
        if not _SUPABASE_AVAILABLE:
            if self.debug:
                print("[MoralisClient] ⚠️ Supabase not available — skipping PnL cache download.")
            return

        remote_path = f"smart_money/{self._label_cache_remote_name}"

        try:
            if self.debug:
                print(f"[MoralisClient] ⬇️  Downloading label cache from Supabase ({remote_path})...")

            # download_file(save_path, file_name, bucket)
            # Uses signed URL + conditional GET (304-aware) — same as all other downloads
            result = _supabase_download_file(
                save_path=self._label_cache_path,
                file_name=remote_path,
                bucket=self.supabase_bucket,
            )

            if result is not None and os.path.exists(self._label_cache_path):
                try:
                    remote_data = joblib.load(self._label_cache_path)
                    if isinstance(remote_data, dict):
                        with self._label_cache_lock:
                            # Merge: remote entries overwrite local on conflict
                            # because the remote file is the accumulated superset
                            merged = {**self._label_cache, **remote_data}
                            self._label_cache = merged
                        if self.debug:
                            print(
                                f"[MoralisClient] ✅ Label cache merged. "
                                f"Total entries: {len(self._label_cache)}"
                            )
                except Exception as e:
                    if self.debug:
                        print(f"[MoralisClient] ⚠️ Failed to parse downloaded label cache: {e}")
            else:
                if self.debug:
                    print("[MoralisClient] ℹ️  No remote label cache found — using local only.")

        except Exception as e:
            if self.debug:
                print(f"[MoralisClient] ⚠️ Label cache download failed: {e}")

    def _save_label_cache(self, force_supabase: bool = False):
        """
        Persist label cache to disk (immediate, background thread).
        Also uploads to Supabase if 5 minutes have elapsed since last upload,
        or if force_supabase=True.

        Remote path: smart_money/wallet_labels.pkl
        upload_file() deletes the old file first, then uploads the new one
        (same behaviour as all other files in supabase_utils).
        """
        def _write():
            with self._label_cache_lock:
                snapshot = dict(self._label_cache)

            # Always save locally first
            try:
                joblib.dump(snapshot, self._label_cache_path)
                self._label_cache_dirty = False
            except Exception as e:
                if self.debug:
                    print(f"[MoralisClient] ⚠️ Label cache local save failed: {e}")
                return

            # Throttled Supabase upload
            now = time.time()
            elapsed = now - self._last_supabase_upload
            should_upload = force_supabase or (elapsed >= self._supabase_upload_interval)

            if should_upload and _SUPABASE_AVAILABLE:
                remote_path = f"smart_money/{self._label_cache_remote_name}"
                try:
                    # upload_file(file_path, bucket, remote_path)
                    # Internally does: remove(remote_path) then upload(remote_path, data)
                    success = _supabase_upload_file(
                        self._label_cache_path,
                        self.supabase_bucket,
                        remote_path,
                    )
                    if success:
                        self._last_supabase_upload = now
                        if self.debug:
                            print(
                                f"[MoralisClient] ☁️  Label cache uploaded to "
                                f"Supabase ({remote_path}, {len(snapshot)} entries)."
                            )
                    elif self.debug:
                        print("[MoralisClient] ⚠️ Label cache Supabase upload failed.")
                except Exception as e:
                    if self.debug:
                        print(f"[MoralisClient] ⚠️ Label cache Supabase upload error: {e}")

        threading.Thread(target=_write, daemon=True).start()

    def _is_label_stale(self, label: Dict) -> bool:
        """
        Return True if a cached PnL record has exceeded LABEL_CACHE_TTL_DAYS.
        Always returns False if TTL is disabled (set to 0).
        """
        if LABEL_CACHE_TTL_DAYS <= 0:
            return False
        cached_at = label.get("_cached_at", 0)
        if not cached_at:
            return True  # No timestamp = treat as stale, force re-fetch
        age_days = (time.time() - cached_at) / 86400
        return age_days >= LABEL_CACHE_TTL_DAYS

    def get_cached_label(self, address: str) -> Optional[Dict]:
        """
        Return the cached PnL dict for *address*, or None if:
          - Not yet cached, OR
          - Cache entry exists but has exceeded LABEL_CACHE_TTL_DAYS

        Returning None forces get_wallet_pnl() to make a fresh API call.
        We store _NO_ENTITY_SENTINEL for wallets with no PnL data so we
        don't keep hammering the API for genuinely empty wallets.
        """
        with self._label_cache_lock:
            label = self._label_cache.get(address)
        if label is None:
            return None
        if self._is_label_stale(label):
            if self.debug:
                age_days = (time.time() - label.get("_cached_at", 0)) / 86400
                print(
                    f"[MoralisClient] 🔄 PnL cache for {address[:8]}... is stale "
                    f"({age_days:.0f}d old, TTL={LABEL_CACHE_TTL_DAYS}d) — will re-fetch."
                )
            return None  # Treat as cache miss → triggers fresh API call
        return label

    # =========================================================================
    # Public Smart Money API method
    # =========================================================================

    async def get_wallet_pnl(self, address: str) -> Dict[str, Any]:
        """
        Fetch wallet profitability summary for a Solana wallet from Moralis.

        Endpoint:
            GET https://solana-gateway.moralis.io/account/mainnet/{address}/profitability/summary

        Response fields used:
            total_realized_profit_usd         — net USD profit from closed trades
            total_realized_profit_percentage  — % return on invested capital
            total_buys / total_sells          — trade counts
            total_count_of_trades             — total trades
            total_trade_volume                — total USD volume traded

        Cost model:
            Cache HIT  → 0 Moralis credits (returned immediately from disk cache)
            Cache MISS → 1 Moralis credit  (live API call, cached for TTL days)

        Returns a dict:
            {
                "pnl_tier":                   str,          # ELITE / STRONG / ACTIVE / NOISE / NEGATIVE
                "total_realized_profit_usd":  float | None,
                "total_realized_profit_pct":  float | None,
                "total_trade_volume_usd":     float | None,
                "total_buys":                 int | None,
                "total_sells":                int | None,
                "total_count_of_trades":      int | None,
                "win_rate_pct":               float | None, # sells / total_trades * 100 (proxy)
                "is_positive":                bool,         # ELITE / STRONG / ACTIVE
                "is_negative":                bool,         # NEGATIVE tier
                "_from_cache":                bool,
            }

        This method NEVER raises — callers receive the NOISE sentinel on any error.
        """
        # --- Cache HIT ---
        cached = self.get_cached_label(address)
        if cached is not None:
            result = dict(cached)
            result["_from_cache"] = True
            # Re-derive booleans in case thresholds changed since caching
            result["is_positive"] = result.get("pnl_tier") in POSITIVE_PNL_TIERS
            result["is_negative"] = result.get("pnl_tier") in NEGATIVE_PNL_TIERS
            return result

        # --- Cache MISS: call Moralis Solana profitability/summary ---
        url = f"https://solana-gateway.moralis.io/account/mainnet/{address}/profitability/summary"
        pnl_data: Dict[str, Any] = dict(_NO_ENTITY_SENTINEL)
        pnl_data["_cached_at"] = int(time.time())   # will be overwritten on success

        selected_key = None
        for attempt in range(4):
            try:
                selected_key = self._get_next_key_for_wallet(address)
            except RuntimeError:
                break  # All keys blacklisted — return sentinel

            headers = {"X-API-Key": selected_key, "accept": "application/json"}
            try:
                async with self.http_session.get(
                    url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:

                    if resp.status == 200:
                        data = await resp.json()
                        pnl_data = self._parse_pnl_response(data, address)
                        if self.debug:
                            print(
                                f"[MoralisClient] 💰 PnL for {address[:8]}…: "
                                f"tier={pnl_data['pnl_tier']}  "
                                f"profit=${pnl_data['total_realized_profit_usd']}  "
                                f"trades={pnl_data['total_count_of_trades']}"
                            )
                        break

                    elif resp.status == 400:
                        now = datetime.now(timezone.utc)
                        next_utc_day = (
                            now.replace(hour=0, minute=0, second=1, microsecond=0)
                            + timedelta(days=1)
                        )
                        self._blacklisted_keys[selected_key] = int(next_utc_day.timestamp())
                        if self.debug:
                            print(f"[MoralisClient] ⚠️ PnL key {selected_key[:6]}… blacklisted (Quota).")
                        continue

                    elif resp.status == 429:
                        delay = (2 ** attempt) + random.uniform(0.3, 0.8)
                        await asyncio.sleep(delay)
                        continue

                    elif resp.status == 404:
                        # New/empty wallet — store sentinel, don't retry until TTL expires
                        pnl_data = dict(_NO_ENTITY_SENTINEL)
                        pnl_data["_cached_at"] = int(time.time())
                        break

                    else:
                        if self.debug:
                            print(f"[MoralisClient] ⚠️ PnL API status {resp.status} for {address[:8]}…")
                        break

            except asyncio.TimeoutError:
                if self.debug:
                    print(f"[MoralisClient] ⚠️ PnL API timeout for {address[:8]}… (attempt {attempt+1})")
                await asyncio.sleep(1)
                continue
            except Exception as e:
                if self.debug:
                    print(f"[MoralisClient] ⚠️ PnL API error for {address[:8]}…: {e}")
                break

        # Store in cache (memory + background disk write) regardless of outcome
        with self._label_cache_lock:
            self._label_cache[address] = pnl_data
            self._label_cache_dirty = True
        self._save_label_cache()

        result = dict(pnl_data)
        result["_from_cache"] = False
        result["is_positive"] = result.get("pnl_tier") in POSITIVE_PNL_TIERS
        result["is_negative"] = result.get("pnl_tier") in NEGATIVE_PNL_TIERS
        return result

    # Backward-compat alias — SmartMoneyScorer calls get_wallet_labels()
    async def get_wallet_labels(self, address: str) -> Dict[str, Any]:
        return await self.get_wallet_pnl(address)

    # =========================================================================
    # PnL parsing and classification helpers  (pure functions, no I/O)
    # =========================================================================

    def _parse_pnl_response(self, data: Dict, address: str) -> Dict[str, Any]:
        """
        Parse the Moralis profitability/summary response into a normalised
        cache record.  Computes pnl_tier from the classification thresholds.

        Moralis response shape:
            {
                "total_realized_profit_usd":        "12345.67",
                "total_realized_profit_percentage": "42.3",
                "total_buys":                       "87",
                "total_sells":                      "64",
                "total_count_of_trades":            "151",
                "total_trade_volume":               "98000.00",
                ...
            }
        All numeric fields come back as strings — we cast them safely.
        """
        def _f(key: str) -> Optional[float]:
            try:
                return float(data.get(key) or 0)
            except (ValueError, TypeError):
                return None

        def _i(key: str) -> Optional[int]:
            try:
                return int(float(data.get(key) or 0))
            except (ValueError, TypeError):
                return None

        profit_usd    = _f("total_realized_profit_usd")
        profit_pct    = _f("total_realized_profit_percentage")
        total_buys    = _i("total_buys")
        total_sells   = _i("total_sells")
        trade_count   = _i("total_count_of_trades")
        trade_volume  = _f("total_trade_volume")

        # Win rate proxy: profitable sells / total trades
        # Moralis doesn't give per-trade P&L so we use the aggregate profit %
        # as a proxy — positive overall = majority of trades profitable
        win_rate_pct: Optional[float] = None
        if trade_count and trade_count > 0 and profit_pct is not None:
            # Rough proxy: map profit_pct to a win-rate estimate
            # Consistent winner at 50%+ profit_pct → likely 60%+ win rate
            win_rate_pct = round(50 + min(profit_pct / 4, 40), 1)  # caps at 90%
            win_rate_pct = max(0.0, win_rate_pct)

        pnl_tier = self._classify_pnl_tier(profit_usd, win_rate_pct, trade_count)

        return {
            "pnl_tier":                   pnl_tier,
            "total_realized_profit_usd":  profit_usd,
            "total_realized_profit_pct":  profit_pct,
            "total_trade_volume_usd":     trade_volume,
            "total_buys":                 total_buys,
            "total_sells":                total_sells,
            "total_count_of_trades":      trade_count,
            "win_rate_pct":               win_rate_pct,
            "is_positive":                pnl_tier in POSITIVE_PNL_TIERS,
            "is_negative":                pnl_tier in NEGATIVE_PNL_TIERS,
            "_cached_at":                 int(time.time()),
        }

    @staticmethod
    def _classify_pnl_tier(
        profit_usd: Optional[float],
        win_rate_pct: Optional[float],
        trade_count: Optional[int],
    ) -> str:
        """
        Map raw PnL metrics to a conviction tier.

        Tier ladder (highest match wins):
            ELITE    — $50k+ profit AND 55%+ win rate AND ≥5 trades
            STRONG   — $10k+ profit AND 50%+ win rate AND ≥5 trades
            ACTIVE   — $1k+ profit AND ≥5 trades
            NEGATIVE — net loss > $5k  (confirmed bad actor / noise)
            NOISE    — everything else (insufficient data, small profits)
        """
        if profit_usd is None or trade_count is None:
            return "NOISE"

        min_trades = PNL_MIN_TRADES

        # NEGATIVE: confirmed loss-maker — down-weight in overlap scoring
        if profit_usd <= PNL_NEGATIVE_LOSS_USD:
            return "NEGATIVE"

        if trade_count < min_trades:
            return "NOISE"

        wr = win_rate_pct or 0.0

        if profit_usd >= PNL_ELITE_PROFIT_USD and wr >= PNL_ELITE_WIN_RATE:
            return "ELITE"

        if profit_usd >= PNL_STRONG_PROFIT_USD and wr >= PNL_STRONG_WIN_RATE:
            return "STRONG"

        if profit_usd >= PNL_ACTIVE_PROFIT_USD:
            return "ACTIVE"

        return "NOISE"

    @staticmethod
    def _classify_positive(label: Dict) -> bool:
        """Back-compat helper — checks pnl_tier."""
        return label.get("pnl_tier") in POSITIVE_PNL_TIERS

    @staticmethod
    def _classify_negative(label: Dict) -> bool:
        """Back-compat helper — checks pnl_tier."""
        return label.get("pnl_tier") in NEGATIVE_PNL_TIERS

    # =========================================================================
    # ALL ORIGINAL METHODS BELOW — COMPLETELY UNCHANGED
    # =========================================================================

    def _clean_expired_blacklists(self):
        """Remove keys from blacklist if current UTC time > blacklist_until_timestamp"""
        now = int(datetime.now(timezone.utc).timestamp())
        self._blacklisted_keys = {
            k: v for k, v in self._blacklisted_keys.items() if v > now
        }

    def _get_next_key_for_wallet(self, wallet: str) -> str:
        """
        Get next available key that:
        1. Is not blacklisted (check against current UTC timestamp)
        2. Was not used for this wallet previously (best-effort)
        3. Rotates round-robin among valid keys

        Raises RuntimeError if all keys are blacklisted.
        """
        self._clean_expired_blacklists()

        valid_keys: Set[str] = {
            k for k in self.api_keys if k not in self._blacklisted_keys
        }

        if not valid_keys:
            raise RuntimeError("All Moralis API keys are blacklisted or unavailable.")

        last_used_key = self._last_key_per_wallet.get(wallet)

        if len(valid_keys) == 1:
            return list(valid_keys)[0]

        preferred_keys = valid_keys
        if last_used_key and last_used_key in valid_keys:
            preferred_keys = valid_keys - {last_used_key}

        start_idx = (self._current_key_index + 1) % len(self.api_keys)

        for i in range(len(self.api_keys)):
            idx = (start_idx + i) % len(self.api_keys)
            key = self.api_keys[idx]
            if key in preferred_keys:
                self._current_key_index = idx
                return key

        for i in range(len(self.api_keys)):
            idx = (start_idx + i) % len(self.api_keys)
            key = self.api_keys[idx]
            if key in valid_keys:
                self._current_key_index = idx
                return key

        raise RuntimeError("Failed to select a valid Moralis API key.")

    async def fetch_wallet_transfers(
        self,
        wallet_address: str,
        chain: str = "solana",
        limit: int = 100,
        max_retries: int = 5
    ) -> List[Dict]:
        """
        Fetch latest buy transactions from Moralis API using the /swaps endpoint.

        Endpoint: GET https://solana-gateway.moralis.io/account/mainnet/{wallet_address}/swaps
        Params: ?order=DESC&fromDate=...&transactionTypes=buy&limit=100
        """
        url = f"https://solana-gateway.moralis.io/account/mainnet/{wallet_address}/swaps"

        now_utc = datetime.now(timezone.utc)
        start_of_day = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        from_date_str = start_of_day.strftime('%Y-%m-%dT%H:%M:%S.000')

        params = {
            "order": "DESC",
            "fromDate": from_date_str,
            "transactionTypes": "buy",
            "limit": str(limit)
        }

        selected_key = None

        for attempt in range(max_retries):
            try:
                selected_key = self._get_next_key_for_wallet(wallet_address)
            except RuntimeError as e:
                if self.debug:
                    print(f"[MoralisClient] ❌ CRITICAL: {e}")
                return []

            headers = {"X-API-Key": selected_key, "accept": "application/json"}

            try:
                if self.debug:
                    print(f"[MoralisClient] Fetching swaps for {wallet_address} from {from_date_str}")

                async with self.http_session.get(
                    url, params=params, headers=headers, timeout=20
                ) as resp:

                    if resp.status == 200:
                        self._last_key_per_wallet[wallet_address] = selected_key
                        try:
                            data = await resp.json()
                            return data.get("result", [])
                        except Exception as e:
                            if self.debug:
                                print(f"[MoralisClient] ❌ JSON parse error for {wallet_address}: {e}")
                            return []

                    elif resp.status == 400:
                        now = datetime.now(timezone.utc)
                        next_utc_day = (
                            now.replace(hour=0, minute=0, second=1, microsecond=0)
                            + timedelta(days=1)
                        )
                        blacklist_until_ts = int(next_utc_day.timestamp())
                        self._blacklisted_keys[selected_key] = blacklist_until_ts
                        if self.debug:
                            print(f"[MoralisClient] ⚠️ Key {selected_key[:6]}... blacklisted (Quota) until {next_utc_day.isoformat()}")
                        continue

                    elif resp.status == 429:
                        delay = (2 ** attempt) + random.uniform(0.5, 1.5)
                        if self.debug:
                            print(f"[MoralisClient] ⚠️ Rate limited on key {selected_key[:6]}... Retrying in {delay:.2f}s")
                        await asyncio.sleep(delay)
                        continue

                    elif resp.status >= 500:
                        delay = (2 ** attempt) + random.uniform(0.5, 1.5)
                        if self.debug:
                            print(f"[MoralisClient] ⚠️ Server error {resp.status}. Retrying in {delay:.2f}s")
                        await asyncio.sleep(delay)
                        continue

                    else:
                        if self.debug:
                            print(f"[MoralisClient] ❌ Client error {resp.status} for {wallet_address}. Stopping.")
                        return []

            except asyncio.TimeoutError:
                if self.debug:
                    print("[MoralisClient] ⚠️ Request timed out. Retrying...")
                continue
            except Exception as e:
                if self.debug:
                    print(f"[MoralisClient] ❌ Unknown exception: {e}. Retrying...")
                await asyncio.sleep(1)
                continue

        if self.debug:
            print(f"[MoralisClient] ❌ All retries failed for {wallet_address}.")
        return []

    async def extract_unique_tokens(
        self, transactions: List[Dict]
    ) -> List[str]:
        """
        Parse Moralis /swaps response to extract unique token mints
        from 'buy' transactions.

        Filter logic:
        - API call already filtered for "transactionType": "buy"
        - Extract token mint from "bought.address" field
        - Return list of unique token addresses (deduplicated)
        """
        if not transactions or not isinstance(transactions, list):
            return []

        mints: Set[str] = set()

        try:
            for tx in transactions:
                if not isinstance(tx, dict):
                    continue

                bought_data = tx.get("bought")
                if not isinstance(bought_data, dict):
                    continue

                mint = bought_data.get("address")

                if mint and isinstance(mint, str) and len(mint) > 30:
                    if mint != "So11111111111111111111111111111111111111112":
                        mints.add(mint)

        except Exception as e:
            if self.debug:
                print(f"[MoralisClient] ❌ Error parsing transactions: {e}")
            return []

        return list(mints)