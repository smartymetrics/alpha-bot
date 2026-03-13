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

SMART MONEY PATCH: Added get_wallet_labels() with persistent local cache.
  - Discovery costs 1 Moralis credit; all subsequent lookups cost 0 credits.
  - Cache file: data/wallet_labels.pkl  (joblib format)
  - Only called for wallets in the top-500 of the Dune Winners list.
  - ALL existing methods are completely unchanged.
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
# Label constants — used by SmartMoneyScorer in smart_money_scorer.py
# ---------------------------------------------------------------------------
POSITIVE_ENTITY_CATEGORIES = frozenset({
    "fund", "venture capital", "defi whale", "high-frequency trader",
    "whale", "smart money", "institution", "market maker",
})
NEGATIVE_ENTITY_CATEGORIES = frozenset({
    "mev bot", "arbitrageur", "centralized exchange", "bridge",
    "bot", "arb bot", "spam", "scammer",
})

# How many days before a cached label is considered stale and re-fetched.
# Positive entity labels (Funds, VCs) are stable identities — 30 days is safe.
# Negative entity labels (MEV bots) are also stable — they don't reform.
# Set to 0 to disable TTL (cache forever).
LABEL_CACHE_TTL_DAYS = int(os.getenv("SMART_MONEY_LABEL_TTL_DAYS", "30"))

# Sentinel value stored in cache when a wallet has NO Moralis entity record.
# _cached_at is always set so TTL applies equally to "no entity" wallets —
# this prevents permanently skipping a wallet that later gets labeled by Moralis.
_NO_ENTITY_SENTINEL = {"entity_name": None, "category": None, "labels": [], "_cached_at": 0}


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
        label_cache_path: str = "./data/wallet_labels.pkl",
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
                f"Label cache: {len(self._label_cache)} entries."
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
                        print(f"[MoralisClient] Loaded {len(data)} wallet labels from local cache.")
            except Exception as e:
                if self.debug:
                    print(f"[MoralisClient] ⚠️ Label cache load failed: {e}. Starting fresh.")
                self._label_cache = {}

    def download_label_cache_from_supabase(self):
        """
        Download the label cache from Supabase on startup and merge with any
        local entries.  Remote wins on conflict (remote is always a superset).
        Called once synchronously from winner_monitor.py startup().

        Remote path: smart_money/wallet_labels.pkl
        This keeps Smart Money files grouped separately from overlap results
        and dune_cache/ files in the bucket.
        """
        if not _SUPABASE_AVAILABLE:
            if self.debug:
                print("[MoralisClient] ⚠️ Supabase not available — skipping label cache download.")
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
        Return True if a cached label has exceeded LABEL_CACHE_TTL_DAYS.
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
        Return the cached label dict for *address*, or None if:
          - Not yet cached, OR
          - Cache entry exists but has exceeded LABEL_CACHE_TTL_DAYS

        Returning None forces get_wallet_labels() to make a fresh API call.
        Callers can distinguish 'no entity found' from 'not yet looked up'
        because we store the _NO_ENTITY_SENTINEL for wallets with no record.
        """
        with self._label_cache_lock:
            label = self._label_cache.get(address)
        if label is None:
            return None
        if self._is_label_stale(label):
            if self.debug:
                age_days = (time.time() - label.get("_cached_at", 0)) / 86400
                print(
                    f"[MoralisClient] 🔄 Label for {address[:8]}... is stale "
                    f"({age_days:.0f}d old, TTL={LABEL_CACHE_TTL_DAYS}d) — will re-fetch."
                )
            return None  # Treat as cache miss → triggers fresh API call
        return label

    # =========================================================================
    # Public Smart Money API method  (NEW)
    # =========================================================================

    async def get_wallet_labels(self, address: str) -> Dict[str, Any]:
        """
        Fetch entity labels for a Solana wallet from the Moralis Entities API.

        Cost model:
            Cache HIT  → 0 Moralis credits (returned immediately from disk cache)
            Cache MISS → 1 Moralis credit  (live API call, then cached forever)

        Returns a dict:
            {
                "entity_name": str | None,   # e.g. "Jump Trading"
                "category":    str | None,   # e.g. "fund"  (lower-cased)
                "labels":      list[str],    # e.g. ["smart_money", "whale"]
                "is_positive": bool,         # True → Fund / VC / Whale
                "is_negative": bool,         # True → MEV bot / Bridge / CEX
                "_from_cache": bool,
            }

        This method NEVER raises — callers receive an empty/default result on
        any error so the main pipeline is never blocked.
        """
        # --- Cache HIT ---
        cached = self.get_cached_label(address)
        if cached is not None:
            result = dict(cached)
            result["_from_cache"] = True
            result["is_positive"] = self._classify_positive(result)
            result["is_negative"] = self._classify_negative(result)
            return result

        # --- Cache MISS: call Moralis Entities API ---
        url = f"https://deep-index.moralis.io/api/v2.2/wallets/{address}/entities"
        label_data: Dict[str, Any] = dict(_NO_ENTITY_SENTINEL)

        selected_key = None
        for attempt in range(4):
            try:
                selected_key = self._get_next_key_for_wallet(address)
            except RuntimeError:
                break  # All keys blacklisted — return empty result

            headers = {"X-API-Key": selected_key, "accept": "application/json"}
            try:
                async with self.http_session.get(
                    url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:

                    if resp.status == 200:
                        data = await resp.json()
                        # Moralis returns a list of entity objects or a single object
                        entities = data if isinstance(data, list) else [data]
                        if entities:
                            top = entities[0]  # primary entity
                            label_data = {
                                "entity_name": top.get("entityName") or top.get("entity_name"),
                                "category": (top.get("category") or "").lower() or None,
                                "labels": [
                                    lbl.get("name", lbl) if isinstance(lbl, dict) else str(lbl)
                                    for lbl in (top.get("labels") or [])
                                ],
                                "_cached_at": int(time.time()),
                            }
                        else:
                            label_data = dict(_NO_ENTITY_SENTINEL)
                            label_data["_cached_at"] = int(time.time())

                        if self.debug:
                            print(
                                f"[MoralisClient] 🏷️  Labels for {address[:8]}…: "
                                f"entity={label_data['entity_name']}, "
                                f"category={label_data['category']}"
                            )
                        break  # success

                    elif resp.status == 400:
                        # Quota exhausted — blacklist key, try next
                        now = datetime.now(timezone.utc)
                        next_utc_day = (
                            now.replace(hour=0, minute=0, second=1, microsecond=0)
                            + timedelta(days=1)
                        )
                        self._blacklisted_keys[selected_key] = int(next_utc_day.timestamp())
                        if self.debug:
                            print(f"[MoralisClient] ⚠️ Label key {selected_key[:6]}… blacklisted (Quota).")
                        continue

                    elif resp.status == 429:
                        delay = (2 ** attempt) + random.uniform(0.3, 0.8)
                        await asyncio.sleep(delay)
                        continue

                    elif resp.status == 404:
                        # No entity record → store sentinel so we never call again
                        label_data = dict(_NO_ENTITY_SENTINEL)
                        label_data["_cached_at"] = int(time.time())
                        break

                    else:
                        if self.debug:
                            print(f"[MoralisClient] ⚠️ Entity API status {resp.status} for {address[:8]}…")
                        break

            except asyncio.TimeoutError:
                if self.debug:
                    print(f"[MoralisClient] ⚠️ Entity API timeout for {address[:8]}… (attempt {attempt+1})")
                await asyncio.sleep(1)
                continue
            except Exception as e:
                if self.debug:
                    print(f"[MoralisClient] ⚠️ Entity API error for {address[:8]}…: {e}")
                break

        # Store in cache (disk + memory) regardless of outcome
        with self._label_cache_lock:
            self._label_cache[address] = label_data
            self._label_cache_dirty = True
        self._save_label_cache()

        result = dict(label_data)
        result["_from_cache"] = False
        result["is_positive"] = self._classify_positive(result)
        result["is_negative"] = self._classify_negative(result)
        return result

    # =========================================================================
    # Classification helpers  (NEW — pure functions, no I/O)
    # =========================================================================

    @staticmethod
    def _classify_positive(label: Dict) -> bool:
        cat = (label.get("category") or "").lower()
        labels_lower = {(lbl or "").lower() for lbl in (label.get("labels") or [])}
        return bool(
            cat in POSITIVE_ENTITY_CATEGORIES
            or labels_lower & POSITIVE_ENTITY_CATEGORIES
        )

    @staticmethod
    def _classify_negative(label: Dict) -> bool:
        cat = (label.get("category") or "").lower()
        labels_lower = {(lbl or "").lower() for lbl in (label.get("labels") or [])}
        return bool(
            cat in NEGATIVE_ENTITY_CATEGORIES
            or labels_lower & NEGATIVE_ENTITY_CATEGORIES
        )

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