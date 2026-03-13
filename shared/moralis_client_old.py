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

# Sentinel value stored in cache when a wallet has NO Moralis entity record
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
    ):
        if not api_keys:
            raise ValueError("MoralisClient requires at least one API key.")
        self.http_session = http_session
        self.api_keys = api_keys
        self.debug = debug

        # ---- Smart Money label cache ----
        self._label_cache_path = label_cache_path
        self._label_cache: Dict[str, Dict] = {}
        self._label_cache_lock = threading.Lock()
        self._label_cache_dirty = False           # write-back flag
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
                        print(f"[MoralisClient] Loaded {len(data)} wallet labels from cache.")
            except Exception as e:
                if self.debug:
                    print(f"[MoralisClient] ⚠️ Label cache load failed: {e}. Starting fresh.")
                self._label_cache = {}

    def _save_label_cache(self):
        """Persist label cache to disk (background thread, non-blocking)."""
        def _write():
            with self._label_cache_lock:
                try:
                    joblib.dump(dict(self._label_cache), self._label_cache_path)
                    self._label_cache_dirty = False
                except Exception as e:
                    if self.debug:
                        print(f"[MoralisClient] ⚠️ Label cache save failed: {e}")
        threading.Thread(target=_write, daemon=True).start()

    def get_cached_label(self, address: str) -> Optional[Dict]:
        """
        Return the cached label dict for *address*, or None if not cached yet.
        Callers can distinguish 'no entity found' from 'not yet looked up'
        because we store the _NO_ENTITY_SENTINEL for wallets with no record.
        """
        with self._label_cache_lock:
            return self._label_cache.get(address)

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