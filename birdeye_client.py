#!/usr/bin/env python3
"""
birdeye_client.py
=================
Birdeye wallet PnL client for Smart Money scoring.

Endpoint used:
    GET https://public-api.birdeye.so/wallet/v2/pnl/summary
    Header: X-API-KEY: {key}
    Header: x-chain: solana
    Param:  wallet={address}
    Param:  duration=7d   (7-day rolling window)

Real response shape (confirmed from live API):
    {
      "data": {
        "summary": {
          "unique_tokens": 22,
          "counts": {
            "total_buy": 35,
            "total_sell": 41,
            "total_trade": 76,
            "total_win": 1,
            "total_loss": 9,
            "win_rate": 0.066          ← decimal, NOT percentage
          },
          "cashflow_usd": {
            "total_invested": 25666.06,
            "total_sold": 32365.67,
            "current_value": 8800.28
          },
          "pnl": {
            "realized_profit_usd": -4794.63,
            "realized_profit_percent": -21.34,
            "unrealized_usd": 2612.78,
            "total_usd": -2181.85,
            "avg_profit_per_trade_usd": -28.71
          }
        }
      }
    }

Key rotation:
    - BIRDEYE_API_KEYS env var: comma-separated keys, e.g. "key1,key2,key3"
    - 429 Too Many Requests → blacklist key for 1 hour, rotate to next
    - 401 Unauthorized      → blacklist key permanently (bad key), rotate
    - All keys exhausted    → return NOISE sentinel, never raise

ML_PASSED gate (enforced by SmartMoneyScorer, not here):
    - PnL lookups only happen when ML_PASSED=True
    - This client is unaware of ML state — the gate lives upstream

Cache:
    - Shares the same disk cache as MoralisClient (wallet_pnl_cache.pkl)
    - TTL controlled by SMART_MONEY_LABEL_TTL_DAYS (default 7 days)
    - Cache is passed in from MoralisClient to avoid duplication
"""

import asyncio
import aiohttp
import os
import random
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BIRDEYE_BASE_URL = "https://public-api.birdeye.so"
BIRDEYE_PNL_ENDPOINT = "/wallet/v2/pnl/summary"
BIRDEYE_DURATION = os.getenv("BIRDEYE_PNL_DURATION", "7d")   # 7d, 30d, etc.

# How long to blacklist a key after 429 (seconds). Default: 1 hour.
BIRDEYE_RATE_LIMIT_BLACKLIST_SECS = int(os.getenv("BIRDEYE_RATE_LIMIT_BLACKLIST_SECS", "3600"))


class BirdeyeClient:
    """
    Birdeye wallet PnL client with key rotation and backoff.

    Designed to be used alongside MoralisClient — it shares the same
    PnL cache dict so wallet lookups are only ever made once per TTL
    regardless of which client fetched them.
    """

    def __init__(
        self,
        http_session: aiohttp.ClientSession,
        api_keys: List[str],
        pnl_cache: Dict[str, Dict],          # shared reference from MoralisClient._label_cache
        pnl_cache_lock,                       # shared threading.Lock from MoralisClient
        save_cache_fn,                        # callable: MoralisClient._save_label_cache()
        get_cached_fn,                        # callable: MoralisClient.get_cached_label(addr)
        classify_tier_fn,                     # callable: MoralisClient._classify_pnl_tier(...)
        no_entity_sentinel: Dict,             # _NO_ENTITY_SENTINEL from moralis_client
        positive_tiers: frozenset,            # POSITIVE_PNL_TIERS
        negative_tiers: frozenset,            # NEGATIVE_PNL_TIERS
        debug: bool = False,
    ):
        if not api_keys:
            raise ValueError("BirdeyeClient requires at least one API key.")

        self.http_session      = http_session
        self.api_keys          = list(api_keys)
        self._pnl_cache        = pnl_cache
        self._pnl_cache_lock   = pnl_cache_lock
        self._save_cache       = save_cache_fn
        self._get_cached       = get_cached_fn
        self._classify_tier    = classify_tier_fn
        self._sentinel         = no_entity_sentinel
        self._positive_tiers   = positive_tiers
        self._negative_tiers   = negative_tiers
        self.debug             = debug

        # Key state: {key: blacklisted_until_unix_ts}
        self._blacklisted_keys: Dict[str, float] = {}
        self._current_key_idx: int = 0

        if self.debug:
            print(f"[BirdeyeClient] Initialized with {len(self.api_keys)} key(s).")

    # =========================================================================
    # Public entry point
    # =========================================================================

    async def get_wallet_pnl(self, address: str) -> Dict[str, Any]:
        """
        Fetch PnL summary for a single Solana wallet.

        Cost model:
            Cache HIT  → 0 Birdeye credits
            Cache MISS → 1 Birdeye API call

        Returns same dict shape as MoralisClient.get_wallet_pnl() so
        SmartMoneyScorer needs no changes.
        """
        # --- Cache HIT ---
        cached = self._get_cached(address)
        if cached is not None:
            result = dict(cached)
            result["_from_cache"] = True
            result["is_positive"] = result.get("pnl_tier") in self._positive_tiers
            result["is_negative"] = result.get("pnl_tier") in self._negative_tiers
            return result

        # --- Cache MISS: call Birdeye ---
        pnl_data = await self._fetch_from_birdeye(address)

        # Write to shared cache
        with self._pnl_cache_lock:
            self._pnl_cache[address] = pnl_data
        self._save_cache()

        result = dict(pnl_data)
        result["_from_cache"] = False
        result["is_positive"] = result.get("pnl_tier") in self._positive_tiers
        result["is_negative"] = result.get("pnl_tier") in self._negative_tiers
        return result

    # =========================================================================
    # Internal fetch with key rotation
    # =========================================================================

    async def _fetch_from_birdeye(self, address: str) -> Dict[str, Any]:
        """Call Birdeye API with retry + key rotation. Never raises."""
        url = f"{BIRDEYE_BASE_URL}{BIRDEYE_PNL_ENDPOINT}"
        params = {"wallet": address, "duration": BIRDEYE_DURATION}

        sentinel = dict(self._sentinel)
        sentinel["_cached_at"] = int(time.time())

        for attempt in range(len(self.api_keys) + 2):   # try each key + 2 retries
            key = self._get_next_key()
            if key is None:
                if self.debug:
                    print(f"[BirdeyeClient] ⚠️ All keys exhausted for {address[:8]}… returning NOISE.")
                return sentinel

            headers = {
                "X-API-KEY": key,
                "accept":    "application/json",
                "x-chain":   "solana",
            }

            try:
                async with self.http_session.get(
                    url, params=params, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:

                    if resp.status == 200:
                        raw = await resp.json()
                        parsed = self._parse_response(raw, address)
                        if self.debug:
                            print(
                                f"[BirdeyeClient] 💰 {address[:8]}… "
                                f"tier={parsed['pnl_tier']}  "
                                f"profit=${parsed['total_realized_profit_usd']}  "
                                f"win_rate={parsed['win_rate_pct']}%  "
                                f"trades={parsed['total_count_of_trades']}"
                            )
                        return parsed

                    elif resp.status == 429:
                        # Rate limited — blacklist this key for BIRDEYE_RATE_LIMIT_BLACKLIST_SECS
                        until = time.time() + BIRDEYE_RATE_LIMIT_BLACKLIST_SECS
                        self._blacklisted_keys[key] = until
                        if self.debug:
                            print(
                                f"[BirdeyeClient] ⚠️ Key {key[:8]}… rate-limited. "
                                f"Blacklisted for {BIRDEYE_RATE_LIMIT_BLACKLIST_SECS//60} min."
                            )
                        # Small backoff before trying next key
                        await asyncio.sleep(random.uniform(0.5, 1.5))
                        continue

                    elif resp.status == 401:
                        # Bad key — blacklist permanently (until restart)
                        self._blacklisted_keys[key] = float("inf")
                        if self.debug:
                            print(f"[BirdeyeClient] ❌ Key {key[:8]}… is invalid (401). Removing.")
                        continue

                    elif resp.status == 404:
                        # Wallet not found / no data — store sentinel so we don't retry
                        if self.debug:
                            print(f"[BirdeyeClient] ℹ️  {address[:8]}… no PnL data (404).")
                        return sentinel

                    elif resp.status >= 500:
                        delay = (2 ** attempt) + random.uniform(0.3, 0.8)
                        if self.debug:
                            print(f"[BirdeyeClient] ⚠️ Server error {resp.status}. Retrying in {delay:.1f}s.")
                        await asyncio.sleep(delay)
                        continue

                    else:
                        if self.debug:
                            body = await resp.text()
                            print(
                                f"[BirdeyeClient] ⚠️ Unexpected status {resp.status} "
                                f"for {address[:8]}…: {body[:120]}"
                            )
                        return sentinel

            except asyncio.TimeoutError:
                if self.debug:
                    print(f"[BirdeyeClient] ⚠️ Timeout for {address[:8]}… (attempt {attempt + 1})")
                await asyncio.sleep(1)
                continue
            except Exception as e:
                if self.debug:
                    print(f"[BirdeyeClient] ⚠️ Error for {address[:8]}…: {e}")
                return sentinel

        return sentinel

    # =========================================================================
    # Response parsing — uses REAL field names from live API
    # =========================================================================

    def _parse_response(self, raw: Dict, address: str) -> Dict[str, Any]:
        """
        Parse the real Birdeye /wallet/v2/pnl/summary response.

        Confirmed field path from live API call:
            raw["data"]["summary"]["pnl"]["realized_profit_usd"]
            raw["data"]["summary"]["counts"]["win_rate"]      ← decimal (0.067 = 6.7%)
            raw["data"]["summary"]["counts"]["total_trade"]
            raw["data"]["summary"]["counts"]["total_buy"]
            raw["data"]["summary"]["counts"]["total_sell"]
            raw["data"]["summary"]["cashflow_usd"]["total_invested"]
        """
        sentinel = dict(self._sentinel)
        sentinel["_cached_at"] = int(time.time())

        try:
            data    = raw.get("data") or {}
            summary = data.get("summary") or {}

            if not summary:
                return sentinel

            pnl      = summary.get("pnl") or {}
            counts   = summary.get("counts") or {}
            cashflow = summary.get("cashflow_usd") or {}

            def _f(d: Dict, key: str) -> Optional[float]:
                try:
                    v = d.get(key)
                    return float(v) if v is not None else None
                except (ValueError, TypeError):
                    return None

            def _i(d: Dict, key: str) -> Optional[int]:
                try:
                    v = d.get(key)
                    return int(float(v)) if v is not None else None
                except (ValueError, TypeError):
                    return None

            realized_profit_usd = _f(pnl,    "realized_profit_usd")
            realized_profit_pct = _f(pnl,    "realized_profit_percent")
            total_invested_usd  = _f(cashflow,"total_invested")
            total_trade         = _i(counts,  "total_trade")
            total_buy           = _i(counts,  "total_buy")
            total_sell          = _i(counts,  "total_sell")
            total_win           = _i(counts,  "total_win")
            total_loss          = _i(counts,  "total_loss")

            # win_rate comes as a decimal (0.067), convert to percentage (6.7)
            win_rate_raw = _f(counts, "win_rate")
            win_rate_pct = round(win_rate_raw * 100, 1) if win_rate_raw is not None else None

            pnl_tier = self._classify_tier(realized_profit_usd, win_rate_pct, total_trade)

            return {
                "pnl_tier":                   pnl_tier,
                "total_realized_profit_usd":  realized_profit_usd,
                "total_realized_profit_pct":  realized_profit_pct,
                "total_trade_volume_usd":     total_invested_usd,
                "total_buys":                 total_buy,
                "total_sells":                total_sell,
                "total_count_of_trades":      total_trade,
                "total_win":                  total_win,
                "total_loss":                 total_loss,
                "win_rate_pct":               win_rate_pct,
                "is_positive":                pnl_tier in self._positive_tiers,
                "is_negative":                pnl_tier in self._negative_tiers,
                "_cached_at":                 int(time.time()),
            }

        except Exception as e:
            if self.debug:
                print(f"[BirdeyeClient] ⚠️ Parse error for {address[:8]}…: {e}")
            return sentinel

    # =========================================================================
    # Key rotation
    # =========================================================================

    def _get_next_key(self) -> Optional[str]:
        """
        Return the next available (non-blacklisted) API key, rotating
        round-robin. Returns None if all keys are currently blacklisted.
        """
        now = time.time()

        # Expire any blacklists that have passed
        self._blacklisted_keys = {
            k: v for k, v in self._blacklisted_keys.items() if v > now
        }

        available = [k for k in self.api_keys if k not in self._blacklisted_keys]
        if not available:
            return None

        # Round-robin across available keys
        self._current_key_idx = (self._current_key_idx + 1) % len(available)
        return available[self._current_key_idx % len(available)]

    @property
    def available_key_count(self) -> int:
        """Number of keys currently not blacklisted."""
        now = time.time()
        return sum(1 for k in self.api_keys
                   if self._blacklisted_keys.get(k, 0) <= now)