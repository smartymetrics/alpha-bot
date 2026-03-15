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

SMART MONEY (v3): Wallet PnL sourced from Birdeye API (confirmed working).
  Endpoint: GET https://public-api.birdeye.so/wallet/v2/pnl/summary
  Real response fields: data.summary.pnl.realized_profit_usd,
                        data.summary.counts.win_rate (decimal),
                        data.summary.counts.total_trade
  - BirdeyeClient handles all PnL API calls with its own key rotation.
  - This client owns the shared PnL cache (disk + Supabase sync).
  - PnL lookups are gated on ML_PASSED=True (enforced by SmartMoneyScorer).
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

        # ---- Smart Money PnL cache ----
        self._label_cache_path = label_cache_path
        self._label_cache_remote_name = os.path.basename(label_cache_path)  # wallet_pnl_cache.pkl
        self._label_cache: Dict[str, Dict] = {}
        self._label_cache_lock = threading.Lock()        # guards in-memory dict reads/writes
        self._supabase_upload_lock = threading.Lock()   # ensures only 1 thread does remove+upload at a time
        self._label_cache_dirty = False
        self._last_supabase_upload: float = 0.0
        self._supabase_upload_interval: float = 300.0  # 5 minutes
        os.makedirs(os.path.dirname(self._label_cache_path), exist_ok=True)
        self._load_label_cache()

        # ---- BirdeyeClient (attached after construction via set_birdeye_client) ----
        self._birdeye_client = None

        # Instance-level aliases for module constants — passed into BirdeyeClient
        self._NO_ENTITY_SENTINEL_INSTANCE = _NO_ENTITY_SENTINEL
        self.POSITIVE_PNL_TIERS_INSTANCE  = POSITIVE_PNL_TIERS
        self.NEGATIVE_PNL_TIERS_INSTANCE  = NEGATIVE_PNL_TIERS

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
        Persist PnL cache to disk (immediate, background thread).
        Also uploads to Supabase if 5 minutes have elapsed since last upload,
        or if force_supabase=True.

        Race-safety:
          - _label_cache_lock guards in-memory snapshot reads.
          - _supabase_upload_lock ensures only ONE thread runs remove()+upload()
            at a time, preventing 409 Duplicate errors on the free Supabase tier.
          - _last_supabase_upload is stamped BEFORE the thread is spawned so that
            rapid back-to-back calls don't each decide to upload concurrently.
        """
        now = time.time()
        elapsed = now - self._last_supabase_upload
        should_upload = force_supabase or (elapsed >= self._supabase_upload_interval)

        # Stamp immediately (in the calling thread) so no second caller
        # can pass the throttle check before the first upload finishes.
        if should_upload and _SUPABASE_AVAILABLE:
            self._last_supabase_upload = now

        def _write():
            with self._label_cache_lock:
                snapshot = dict(self._label_cache)

            # Always save locally first
            try:
                joblib.dump(snapshot, self._label_cache_path)
                self._label_cache_dirty = False
            except Exception as e:
                if self.debug:
                    print(f"[MoralisClient] ⚠️ PnL cache local save failed: {e}")
                return

            if not (should_upload and _SUPABASE_AVAILABLE):
                return

            remote_path = f"smart_money/{self._label_cache_remote_name}"

            # Only one thread may do remove()+upload() at a time.
            # If another upload is already in progress, skip — the cache
            # is already being written and we'll catch the next cycle.
            if not self._supabase_upload_lock.acquire(blocking=False):
                if self.debug:
                    print("[MoralisClient] ⏭️  PnL cache upload skipped — another upload in progress.")
                return

            try:
                success = _supabase_upload_file(
                    self._label_cache_path,
                    self.supabase_bucket,
                    remote_path,
                )
                if success:
                    if self.debug:
                        print(
                            f"[MoralisClient] ☁️  PnL cache uploaded to "
                            f"Supabase ({remote_path}, {len(snapshot)} entries)."
                        )
                elif self.debug:
                    print("[MoralisClient] ⚠️ PnL cache Supabase upload failed.")
            except Exception as e:
                if self.debug:
                    print(f"[MoralisClient] ⚠️ PnL cache Supabase upload error: {e}")
            finally:
                self._supabase_upload_lock.release()

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

    def set_birdeye_client(self, birdeye_client: Any) -> None:
        """
        Attach a BirdeyeClient instance after construction.
        Called from winner_monitor.py once both clients are initialised.
        SmartMoneyScorer calls get_wallet_pnl() which delegates here.
        """
        self._birdeye_client = birdeye_client

    async def get_wallet_pnl(self, address: str) -> Dict[str, Any]:
        """
        Fetch PnL summary for a Solana wallet via Birdeye.

        Delegates to BirdeyeClient which owns the actual API call,
        key rotation, and 429 handling. This method owns the cache.

        Cost model:
            Cache HIT  → 0 credits  (served from disk/memory cache)
            Cache MISS → 1 Birdeye API call (ML_PASSED gate enforced upstream)

        Returns a dict with keys:
            pnl_tier, total_realized_profit_usd, total_realized_profit_pct,
            total_trade_volume_usd, total_buys, total_sells,
            total_count_of_trades, total_win, total_loss,
            win_rate_pct, is_positive, is_negative, _from_cache
        """
        birdeye = getattr(self, "_birdeye_client", None)
        if birdeye is None:
            # BirdeyeClient not attached yet — return NOISE sentinel
            sentinel = dict(_NO_ENTITY_SENTINEL)
            sentinel["_cached_at"] = int(time.time())
            sentinel["_from_cache"] = False
            sentinel["is_positive"] = False
            sentinel["is_negative"] = False
            if self.debug:
                print(f"[MoralisClient] ⚠️ BirdeyeClient not attached — returning NOISE for {address[:8]}…")
            return sentinel

        return await birdeye.get_wallet_pnl(address)

    # Backward-compat alias — SmartMoneyScorer calls get_wallet_labels()
    async def get_wallet_labels(self, address: str) -> Dict[str, Any]:
        return await self.get_wallet_pnl(address)

    # =========================================================================
    # PnL classification  (BirdeyeClient calls this via injected classify_tier_fn)
    # =========================================================================

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