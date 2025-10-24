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
"""

import asyncio
import aiohttp
import random
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Set

class MoralisClient:
    """
    Handle all Moralis API communication with robust error handling
    and intelligent key rotation.
    """

    def __init__(
        self,
        http_session: aiohttp.ClientSession,
        api_keys: List[str],
        debug: bool = False
    ):
        if not api_keys:
            raise ValueError("MoralisClient requires at least one API key.")
        self.http_session = http_session
        self.api_keys = api_keys
        self.debug = debug
        
        # State Tracking
        self._current_key_index: int = 0
        # {key: blacklist_until_timestamp}
        self._blacklisted_keys: Dict[str, int] = {}
        # {wallet: last_key_used}
        self._last_key_per_wallet: Dict[str, str] = {}
        
        if self.debug:
            print(f"[MoralisClient] Initialized with {len(api_keys)} keys.")

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
        
        # If only one valid key left, we must use it
        if len(valid_keys) == 1:
            return list(valid_keys)[0]

        # Try to find a different key than last time
        preferred_keys = valid_keys
        if last_used_key and last_used_key in valid_keys:
            preferred_keys = valid_keys - {last_used_key}
        
        # Start search from the next index
        start_idx = (self._current_key_index + 1) % len(self.api_keys)
        
        for i in range(len(self.api_keys)):
            idx = (start_idx + i) % len(self.api_keys)
            key = self.api_keys[idx]
            if key in preferred_keys:
                self._current_key_index = idx
                return key
        
        # Fallback: all valid keys were the 'last_used_key',
        # so just return the next valid key in round-robin
        for i in range(len(self.api_keys)):
            idx = (start_idx + i) % len(self.api_keys)
            key = self.api_keys[idx]
            if key in valid_keys:
                self._current_key_index = idx
                return key

        # This should be unreachable if valid_keys is not empty
        raise RuntimeError("Failed to select a valid Moralis API key.")

    async def fetch_wallet_transfers(
        self, 
        wallet_address: str, 
        chain: str = "solana", # No longer used in URL, but kept for compatibility
        limit: int = 100,
        max_retries: int = 5
    ) -> List[Dict]:
        """
        Fetch latest buy transactions from Moralis API using the /swaps endpoint.
        
        Endpoint: GET https://solana-gateway.moralis.io/account/mainnet/{wallet_address}/swaps
        Params: ?order=DESC&fromDate=...&transactionTypes=buy&limit=100
        """
        
        # --- NEW: Use /swaps endpoint ---
        url = f"https://solana-gateway.moralis.io/account/mainnet/{wallet_address}/swaps"
        
        # --- NEW: Generate fromDate for 12am UTC of current day ---
        now_utc = datetime.now(timezone.utc)
        start_of_day = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        # Format as YYYY-MM-DDTHH:MM:SS.mmm
        from_date_str = start_of_day.strftime('%Y-%m-%dT%H:%M:%S.000')

        # --- NEW: Update params for /swaps endpoint ---
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
                return [] # All keys are blacklisted

            headers = {"X-API-Key": selected_key, "accept": "application/json"}
            
            try:
                if self.debug:
                    print(f"[MoralisClient] Fetching swaps for {wallet_address} from {from_date_str}")
                    
                async with self.http_session.get(
                    url, params=params, headers=headers, timeout=20
                ) as resp:
                    
                    # Success
                    if resp.status == 200:
                        self._last_key_per_wallet[wallet_address] = selected_key
                        try:
                            data = await resp.json()
                            # Moralis nests swaps results in 'result' key
                            return data.get("result", [])
                        except Exception as e:
                            if self.debug:
                                print(f"[MoralisClient] ❌ JSON parse error for {wallet_address}: {e}")
                            return []

                    # 400: Daily quota exhausted
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
                        continue # Retry with a new key

                    # 429: Rate limited
                    elif resp.status == 429:
                        delay = (2 ** attempt) + random.uniform(0.5, 1.5)
                        if self.debug:
                            print(f"[MoralisClient] ⚠️ Rate limited on key {selected_key[:6]}... Retrying in {delay:.2f}s")
                        await asyncio.sleep(delay)
                        continue # Retry with SAME key

                    # 5xx: Server error
                    elif resp.status >= 500:
                        delay = (2 ** attempt) + random.uniform(0.5, 1.5)
                        if self.debug:
                            print(f"[MoralisClient] ⚠️ Server error {resp.status}. Retrying in {delay:.2f}s")
                        await asyncio.sleep(delay)
                        continue # Retry with same key
                    
                    else:
                        # Other client error (e.g., 404, 401) - don't retry
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
                
                # --- NEW: Parse /swaps response structure ---
                # API call already filtered for "buy", so no need to check tx.type
                
                bought_data = tx.get("bought")
                if not isinstance(bought_data, dict):
                    continue
                    
                # Extract mint address
                mint = bought_data.get("address")
                
                if mint and isinstance(mint, str) and len(mint) > 30:
                    # Avoid adding SOL
                    if mint != "So11111111111111111111111111111111111111112":
                        mints.add(mint)
                        
        except Exception as e:
            if self.debug:
                print(f"[MoralisClient] ❌ Error parsing transactions: {e}")
            return []
            
        return list(mints)