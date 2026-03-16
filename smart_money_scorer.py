#!/usr/bin/env python3
"""
smart_money_scorer.py
=====================
3-Layer Smart Money Conviction Scorer — on-chain performance intelligence.

Layer A │ Analytical Performance  — Dune SQL frequency (rolling-7-day win-rate)
Layer B │ PnL Enrichment          — Moralis profitability/summary (ELITE/STRONG/ACTIVE/NOISE/NEGATIVE)
Layer C │ Behavioral Heuristics   — Helius RPC insider-rank + temporal clustering

Design principles
-----------------
• Credit-efficient: PnL data is fetched once and cached to disk (7-day TTL).
  Only wallets in the top-N Dune Winners list are ever fetched (credit gate).
• Non-destructive: Does NOT modify ML model features or retrain anything.
  The scorer runs POST-prediction and applies an "up-rank boost" to alerts.
• Fully async: All network I/O is async-safe and uses the shared aiohttp session.
• Fail-safe: Every method returns a safe default on any error.

Boost logic (applied in winner_monitor.py after ML prediction):
  Grade MEDIUM/HIGH/VERY_HIGH + Smart Money Cluster  →  SUPER-ALPHA  🔥
  Any grade + 2+ ELITE wallets                        →  SUPER-ALPHA  🔥
  Any grade + 1  ELITE wallet                         →  STRONG / ALPHA ⭐ + +1 tier
  Any grade + 3+ STRONG/ACTIVE wallets                →  STRONG / ALPHA ⭐
  All overlap wallets are NEGATIVE tier               →  FILTERED  (signal suppressed)

Usage
-----
    scorer = SmartMoneyScorer(
        moralis_client=moralis_client,
        wallet_ranker=wallet_ranker,
        top_n_label_threshold=500,
        cluster_window_seconds=1800,
        debug=True,
    )

    score = await scorer.score_token(
        mint="<token_mint>",
        overlap_wallets=["wallet1", "wallet2", ...],
        wallet_freq={"wallet1": 7, "wallet2": 3, ...},
        ml_passed=True,
        pair_created_at_ms=1234567890000,           # from DexScreener pairCreatedAt
        wallet_buy_timestamps={"wallet1": "2024-11-21T09:22:28.000Z"},
        rugcheck_raw=result["security"]["rugcheck_raw"]["raw"],
    )

    final_grade, alert_label, is_super_alpha = apply_smart_money_boost(
        current_grade=result["grade"],
        ml_prediction=result.get("ml_prediction"),
        smart_money_score=score,
    )
"""

import asyncio
import aiohttp
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from shared.moralis_client import (
    MoralisClient,
    POSITIVE_PNL_TIERS,
    NEGATIVE_PNL_TIERS,
    PNL_TIER_WEIGHT_MULTIPLIERS,
    # Back-compat aliases still exported from moralis_client
    POSITIVE_ENTITY_CATEGORIES,
    NEGATIVE_ENTITY_CATEGORIES,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# PnL tier weight multipliers for weighted overlap scoring.
# ELITE replaces the old Fund/VC label; STRONG replaces Whale/DeFi Whale, etc.
# Imported directly from moralis_client so there is a single source of truth:
#   ELITE    → 3.0x   (proven top performer, $50k+ profit, 55%+ win rate)
#   STRONG   → 2.5x   (consistently profitable, $10k+, 50%+ win rate)
#   ACTIVE   → 1.5x   (net positive, $1k+, ≥5 trades)
#   NOISE    → 1.0x   (insufficient data — default, no boost)
#   NEGATIVE → 0.0x   (loss-maker / bot — excluded from effective overlap)
ENTITY_WEIGHT_MULTIPLIERS = PNL_TIER_WEIGHT_MULTIPLIERS

# Minimum independent SM entities for a "cluster" signal
CLUSTER_MIN_WALLETS = 3

# Grade tier ordering for up-rank logic
GRADE_TIERS = ["NONE", "UNKNOWN", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WalletSmartMoneyProfile:
    """Full smart money assessment for a single wallet."""
    address: str

    # Layer A — Analytical Performance (Dune frequency)
    dune_frequency: int = 0
    alpha_winner_tier: str = "NONE"       # HIGH / MEDIUM / LOW / NONE

    # Layer B — PnL Enrichment (Moralis profitability/summary)
    pnl_tier: str = "NOISE"               # ELITE / STRONG / ACTIVE / NOISE / NEGATIVE
    total_realized_profit_usd: Optional[float] = None
    total_realized_profit_pct: Optional[float] = None
    total_trade_volume_usd: Optional[float] = None
    total_count_of_trades: Optional[int] = None
    win_rate_pct: Optional[float] = None
    is_positive_entity: bool = False      # True for ELITE / STRONG / ACTIVE
    is_negative_entity: bool = False      # True for NEGATIVE
    entity_weight_multiplier: float = 1.0 # derived from pnl_tier
    label_from_cache: bool = False

    # Layer C — Behavioral Heuristics (RugCheck + Moralis timestamps)
    insider_rank: Optional[int] = None    # minutes after launch (0 = RugCheck confirmed)
    is_insider: bool = False
    insider_type: Optional[str] = None   # "SNIPER" / "EARLY_BUYER" / "RUGCHECK"


@dataclass
class SmartMoneyTokenScore:
    """Aggregated Smart Money conviction score for a single token."""
    mint: str
    checked_at: str = ""

    # Wallet breakdown
    total_overlap_wallets: int = 0
    positive_entity_wallets: List[str] = field(default_factory=list)
    negative_entity_wallets: List[str] = field(default_factory=list)
    insider_wallets: List[str] = field(default_factory=list)
    effective_overlap_wallets: List[str] = field(default_factory=list)  # excl. negatives

    # Scores
    raw_overlap_count: int = 0
    effective_overlap_count: int = 0           # negatives excluded
    smart_money_weighted_score: float = 0.0    # sum(freq × entity_multiplier)
    has_smart_money_cluster: bool = False      # 3+ SM entities in 30-min window
    cluster_wallets: List[str] = field(default_factory=list)

    # Boost recommendation
    boost_tier: str = "NONE"     # SUPER_ALPHA / STRONG / STANDARD / FILTERED / NONE
    boost_reason: str = ""

    # Per-wallet profiles for logging / downstream consumers
    wallet_profiles: Dict[str, Dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------

class SmartMoneyScorer:
    """
    Evaluate Smart Money conviction for any token that has reached the
    overlap analysis stage in winner_monitor.py.

    Deliberately stateless with respect to the main pipeline —
    it only *reads* existing data and *appends* enrichment metadata.
    """

    def __init__(
        self,
        moralis_client: MoralisClient,
        wallet_ranker: Any,               # WinnerWalletRanker — avoids circular import
        top_n_label_threshold: int = 500,
        cluster_window_seconds: int = 1800,
        debug: bool = False,
    ):
        self.moralis_client = moralis_client
        self.wallet_ranker = wallet_ranker
        self.top_n_label_threshold = top_n_label_threshold
        self.cluster_window_seconds = cluster_window_seconds
        self.debug = debug

        # In-memory buy-time store for cluster detection: mint → [(wallet, ts), …]
        self._buy_time_cache: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

    # =========================================================================
    # Public entry point
    # =========================================================================

    async def score_token(
        self,
        mint: str,
        overlap_wallets: List[str],
        wallet_freq: Dict[str, int],
        ml_passed: bool = False,
        pair_created_at_ms: Optional[int] = None,      # unix ms from DexScreener pairCreatedAt
        wallet_buy_timestamps: Optional[Dict[str, str]] = None,  # {wallet: ISO timestamp} from Moralis
        rugcheck_raw: Optional[Dict] = None,            # raw RugCheck data for topHolders insider flag
    ) -> SmartMoneyTokenScore:
        """
        Main method called by AlphaTokenAnalyzer for every token that passes
        the security gate and has overlap_count > 0.

        Parameters
        ----------
        mint                 : token mint address
        overlap_wallets      : wallets from the Dune Winners list that hold this token
        wallet_freq          : {wallet: dune_appearance_frequency}
        ml_passed            : if False, skip Birdeye PnL lookups (save API credits).
                               Layer A (Dune frequency) and Layer C (insider) still run.
        pair_created_at_ms   : token pair launch time in unix milliseconds (from DexScreener).
                               Used to detect early buyers — wallets that bought within
                               INSIDER_WINDOW_MINUTES of launch.
        wallet_buy_timestamps: {wallet_address: ISO_timestamp} of each wallet's earliest
                               buy of this token (from Moralis swap history).
        rugcheck_raw         : raw RugCheck response dict. Used to cross-reference
                               topHolders[].insider flag with overlap wallets.

        Returns
        -------
        SmartMoneyTokenScore with all three layers populated
        """
        score = SmartMoneyTokenScore(
            mint=mint,
            checked_at=datetime.now(timezone.utc).isoformat(),
            raw_overlap_count=len(overlap_wallets),
            total_overlap_wallets=len(overlap_wallets),
        )

        if not overlap_wallets:
            score.boost_tier = "NONE"
            score.boost_reason = "no_overlap_wallets"
            return score

        # Determine which wallets are within the Birdeye PnL gate
        top_n_wallet_set = self._get_top_n_wallet_set()

        # Build per-wallet profiles across all 3 layers concurrently.
        # ml_passed controls whether Layer B (Birdeye PnL) is called.
        profiles = await self._build_wallet_profiles(
            overlap_wallets, wallet_freq, top_n_wallet_set, mint, ml_passed,
            pair_created_at_ms, wallet_buy_timestamps, rugcheck_raw,
        )

        # Aggregate
        effective_wallets: List[str] = []
        sm_weighted_score: float = 0.0
        positive_entities: List[str] = []
        negative_entities: List[str] = []
        insider_wallets: List[str] = []

        for wallet, profile in profiles.items():
            if profile.is_negative_entity:
                negative_entities.append(wallet)
                if self.debug:
                    print(
                        f"[SmartMoney] 🚫 {wallet[:8]}... EXCLUDED "
                        f"(pnl_tier: {profile.pnl_tier}, "
                        f"profit: ${profile.total_realized_profit_usd})"
                    )
                continue

            effective_wallets.append(wallet)
            freq = wallet_freq.get(wallet, 1)
            sm_weighted_score += freq * profile.entity_weight_multiplier

            if profile.is_positive_entity:
                positive_entities.append(wallet)
                if self.debug:
                    print(
                        f"[SmartMoney] ✅ {wallet[:8]}... POSITIVE "
                        f"pnl_tier={profile.pnl_tier}, "
                        f"profit=${profile.total_realized_profit_usd}, "
                        f"multiplier={profile.entity_weight_multiplier}x, "
                        f"freq={freq}"
                    )

            if profile.is_insider:
                insider_wallets.append(wallet)
                if self.debug:
                    print(
                        f"[SmartMoney] 🎯 {wallet[:8]}... {profile.insider_type or 'INSIDER'} "
                        f"({profile.insider_rank} min after launch)"
                    )

        # Cluster detection
        cluster_wallets = self._detect_smart_money_cluster(
            mint, positive_entities, profiles
        )
        has_cluster = len(cluster_wallets) >= CLUSTER_MIN_WALLETS

        # Populate score object
        score.effective_overlap_wallets = effective_wallets
        score.effective_overlap_count = len(effective_wallets)
        score.positive_entity_wallets = positive_entities
        score.negative_entity_wallets = negative_entities
        score.insider_wallets = insider_wallets
        score.smart_money_weighted_score = round(sm_weighted_score, 3)
        score.has_smart_money_cluster = has_cluster
        score.cluster_wallets = cluster_wallets
        score.wallet_profiles = {w: asdict(p) for w, p in profiles.items()}

        score.boost_tier, score.boost_reason = self._calculate_boost(score)

        if self.debug:
            print(
                f"[SmartMoney] 📊 {mint[:8]}... | "
                f"eff={score.effective_overlap_count} "
                f"(+{len(positive_entities)} pos, -{len(negative_entities)} neg) | "
                f"insiders={len(insider_wallets)} | "
                f"cluster={has_cluster} | "
                f"weighted={score.smart_money_weighted_score:.2f} | "
                f"boost={score.boost_tier}"
            )

        return score

    # =========================================================================
    # Layer A: Analytical Performance
    # =========================================================================

    @staticmethod
    def _get_alpha_winner_tier(frequency: int) -> str:
        """Classify wallet by 7-day Dune appearance frequency."""
        if frequency >= 10:
            return "HIGH"
        if frequency >= 5:
            return "MEDIUM"
        if frequency >= 2:
            return "LOW"
        return "NONE"

    # =========================================================================
    # Layer B: Categorical Enrichment  (Moralis entities)
    # =========================================================================

    def _get_top_n_wallet_set(self) -> Set[str]:
        """Return the top-N wallet addresses from the Dune Winners ranking."""
        try:
            top_n = self.wallet_ranker.get_top_n_wallets(n=self.top_n_label_threshold)
            return {w for w, _ in top_n}
        except Exception:
            return set()

    async def _fetch_entity_label(self, wallet: str, in_top_n: bool, ml_passed: bool = False) -> Dict[str, Any]:
        """
        Credit gate — two conditions must both be true to call Birdeye:
          1. Wallet is in the top-N Dune Winners list
          2. ML_PASSED=True for this token (set by caller)

        If either condition fails → return NOISE sentinel (0 credits).
        Cache HITs always return immediately regardless of ml_passed.
        """
        # Always check cache first — never skip a cached result
        cached = self.moralis_client.get_cached_label(wallet)
        if cached is not None:
            result = dict(cached)
            result["_from_cache"] = True
            result["is_positive"] = result.get("pnl_tier") in POSITIVE_PNL_TIERS
            result["is_negative"] = result.get("pnl_tier") in NEGATIVE_PNL_TIERS
            return result

        # Gate: skip API call if outside top-N OR ML didn't pass
        if not in_top_n or not ml_passed:
            reason = "not_in_top_n" if not in_top_n else "ml_not_passed"
            if self.debug and not in_top_n is False:
                pass  # don't spam logs for every non-top-N wallet
            return {
                "pnl_tier":                  "NOISE",
                "total_realized_profit_usd": None,
                "total_realized_profit_pct": None,
                "total_trade_volume_usd":    None,
                "total_count_of_trades":     None,
                "win_rate_pct":              None,
                "is_positive":               False,
                "is_negative":               False,
                "_from_cache":               True,   # treated as "no API needed"
                "_skipped":                  True,
                "_skip_reason":              reason,
            }

        return await self.moralis_client.get_wallet_pnl(wallet)

    @staticmethod
    def _get_entity_multiplier(pnl_tier: str) -> float:
        """Map PnL tier to a weight multiplier for the weighted overlap score."""
        return PNL_TIER_WEIGHT_MULTIPLIERS.get(pnl_tier, 1.0)

    # =========================================================================
    # Layer C: Insider Detection  (RugCheck + Moralis timestamps, zero extra credits)
    # =========================================================================

    # Tier 1 — SNIPER: bought within this many minutes of pair creation
    SNIPER_WINDOW_MINUTES:  int = int(os.getenv("SM_SNIPER_WINDOW_MINUTES",  "5"))
    # Tier 2 — EARLY_BUYER: bought within this many minutes (wider window)
    INSIDER_WINDOW_MINUTES: int = int(os.getenv("SM_INSIDER_WINDOW_MINUTES", "30"))

    async def _check_insider(
        self,
        wallet: str,
        pair_created_at_ms: Optional[int],
        wallet_buy_timestamps: Optional[Dict[str, str]],
        rc_insider_owners: Set[str],
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Determine if a wallet is an early insider using two free signals.
        Returns (minutes_after_launch, insider_type) or (None, None).

        Signal 1 — RugCheck topHolders[].insider:
            Strongest signal — RugCheck's own graph analysis flagged this wallet.
            Returns (0, "RUGCHECK").

        Signal 2 — Timestamp comparison (Moralis swap ts vs DexScreener pairCreatedAt):
            ≤ SM_SNIPER_WINDOW_MINUTES (default 5 min)  → (minutes, "SNIPER")
            ≤ SM_INSIDER_WINDOW_MINUTES (default 30 min) → (minutes, "EARLY_BUYER")
            > SM_INSIDER_WINDOW_MINUTES                  → (None, None)

        Zero extra API calls. Both sources already fetched upstream.
        """
        # Signal 1: RugCheck graph insider — strongest, check first
        if wallet in rc_insider_owners:
            if self.debug:
                print(f"[SmartMoney] 🎯 {wallet[:8]}... RUGCHECK INSIDER")
            return 0, "RUGCHECK"

        # Signal 2: Timestamp-based early buyer detection
        if not pair_created_at_ms or not wallet_buy_timestamps:
            return None, None

        buy_ts_iso = wallet_buy_timestamps.get(wallet)
        if not buy_ts_iso:
            return None, None

        try:
            buy_dt  = datetime.fromisoformat(buy_ts_iso.replace("Z", "+00:00"))
            pair_dt = datetime.fromtimestamp(pair_created_at_ms / 1000, tz=timezone.utc)
            mins    = (buy_dt - pair_dt).total_seconds() / 60

            if mins < 0:
                # Bought before pair existed — data anomaly, skip
                return None, None

            if mins <= self.SNIPER_WINDOW_MINUTES:
                if self.debug:
                    print(f"[SmartMoney] 🎯 {wallet[:8]}... SNIPER ({mins:.1f} min after launch)")
                return round(mins), "SNIPER"

            if mins <= self.INSIDER_WINDOW_MINUTES:
                if self.debug:
                    print(f"[SmartMoney] 🎯 {wallet[:8]}... EARLY BUYER ({mins:.1f} min after launch)")
                return round(mins), "EARLY_BUYER"

        except Exception as e:
            if self.debug:
                print(f"[SmartMoney] ⚠️ Insider timestamp parse error for {wallet[:8]}...: {e}")

        return None, None

    def _record_buy_time(self, mint: str, wallet: str):
        """Record current timestamp as the buy-time for cluster detection."""
        now_ts = int(time.time())
        self._buy_time_cache[mint].append((wallet, now_ts))
        cutoff = now_ts - (self.cluster_window_seconds * 2)
        self._buy_time_cache[mint] = [
            (w, t) for w, t in self._buy_time_cache[mint] if t >= cutoff
        ]

    def _detect_smart_money_cluster(
        self,
        mint: str,
        positive_wallets: List[str],
        profiles: Dict[str, WalletSmartMoneyProfile],
    ) -> List[str]:
        """
        Detect 3+ independent Smart Money entities buying the same token
        within cluster_window_seconds.  'Independent' = different entity names
        or categories (prevents one fund with many wallets triggering a cluster).
        """
        for w in positive_wallets:
            self._record_buy_time(mint, w)

        buy_events = self._buy_time_cache.get(mint, [])
        profiled_positives = {w for w in positive_wallets if w in profiles}
        relevant_events = [
            (w, t) for w, t in buy_events if w in profiled_positives
        ]

        if len(relevant_events) < CLUSTER_MIN_WALLETS:
            return []

        relevant_events.sort(key=lambda x: x[1])
        best_cluster: List[str] = []

        for i in range(len(relevant_events)):
            window_wallets: List[str] = []
            window_entity_ids: Set[str] = set()
            t_start = relevant_events[i][1]

            for w, t in relevant_events[i:]:
                if t - t_start > self.cluster_window_seconds:
                    break
                profile = profiles.get(w)
                # Deduplicate by pnl_tier bucket — prevents the same tier of
                # wallet trivially counting as "independent" entities.
                # Use address as tiebreaker when pnl_tier is NOISE/None.
                pnl_tier = getattr(profile, "pnl_tier", None) if profile else None
                entity_id = (
                    f"{pnl_tier}:{w}" if pnl_tier in ("NOISE", None)
                    else pnl_tier + ":" + w   # still unique per wallet, but tier-grouped
                )
                if entity_id not in window_entity_ids:
                    window_wallets.append(w)
                    window_entity_ids.add(entity_id)

            if len(window_wallets) >= CLUSTER_MIN_WALLETS:
                if len(window_wallets) > len(best_cluster):
                    best_cluster = window_wallets

        return best_cluster

    # =========================================================================
    # Combined profile builder
    # =========================================================================

    async def _build_wallet_profiles(
        self,
        wallets: List[str],
        wallet_freq: Dict[str, int],
        top_n_wallet_set: Set[str],
        mint: str,
        ml_passed: bool = False,
        pair_created_at_ms: Optional[int] = None,
        wallet_buy_timestamps: Optional[Dict[str, str]] = None,
        rugcheck_raw: Optional[Dict] = None,
    ) -> Dict[str, WalletSmartMoneyProfile]:
        """Build a WalletSmartMoneyProfile for each wallet (all 3 layers in parallel)."""

        # Pre-compute RugCheck insider set from topHolders for O(1) lookup
        rc_insider_owners: Set[str] = set()
        if rugcheck_raw:
            for holder in (rugcheck_raw.get("topHolders") or []):
                if holder.get("insider") is True:
                    rc_insider_owners.add(holder.get("owner", ""))

        async def _profile_one(wallet: str) -> Tuple[str, WalletSmartMoneyProfile]:
            freq     = wallet_freq.get(wallet, 1)
            in_top_n = wallet in top_n_wallet_set

            pnl_task = asyncio.create_task(
                self._fetch_entity_label(wallet, in_top_n, ml_passed)
            )
            insider_task = asyncio.create_task(
                self._check_insider(
                    wallet,
                    pair_created_at_ms,
                    wallet_buy_timestamps,
                    rc_insider_owners,
                )
            )
            pnl, insider_result = await asyncio.gather(pnl_task, insider_task)

            pnl_tier = pnl.get("pnl_tier", "NOISE")

            # _check_insider returns (minutes_after_launch, insider_type) or (None, None)
            insider_rank, insider_type = insider_result if insider_result else (None, None)

            profile = WalletSmartMoneyProfile(
                address=wallet,
                # Layer A
                dune_frequency=freq,
                alpha_winner_tier=self._get_alpha_winner_tier(freq),
                # Layer B
                pnl_tier=pnl_tier,
                total_realized_profit_usd=pnl.get("total_realized_profit_usd"),
                total_realized_profit_pct=pnl.get("total_realized_profit_pct"),
                total_trade_volume_usd=pnl.get("total_trade_volume_usd"),
                total_count_of_trades=pnl.get("total_count_of_trades"),
                win_rate_pct=pnl.get("win_rate_pct"),
                is_positive_entity=pnl.get("is_positive", False),
                is_negative_entity=pnl.get("is_negative", False),
                entity_weight_multiplier=self._get_entity_multiplier(pnl_tier),
                label_from_cache=pnl.get("_from_cache", False),
                # Layer C
                insider_rank=insider_rank,       # minutes after launch (0 = RugCheck)
                is_insider=insider_rank is not None,
                insider_type=insider_type,       # "SNIPER" / "EARLY_BUYER" / "RUGCHECK"
            )
            return wallet, profile

        tasks = [asyncio.create_task(_profile_one(w)) for w in wallets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        profiles: Dict[str, WalletSmartMoneyProfile] = {}
        for res in results:
            if isinstance(res, Exception):
                if self.debug:
                    print(f"[SmartMoney] ⚠️ Profile build error: {res}")
                continue
            w, profile = res
            profiles[w] = profile

        return profiles

    # =========================================================================
    # Boost tier calculation
    # =========================================================================

    def _calculate_boost(
        self, score: SmartMoneyTokenScore
    ) -> Tuple[str, str]:
        """
        Assign a boost tier based on the aggregated smart money evidence.

        Tier ladder (highest match wins):
            SUPER_ALPHA  → SM cluster present  OR  2+ ELITE wallets
            STRONG       → 1 ELITE wallet      OR  3+ STRONG/ACTIVE wallets
            STANDARD     → any positive PnL wallet (ACTIVE or above)
            FILTERED     → ALL overlap wallets are NEGATIVE tier
            NONE         → no PnL data / all NOISE
        """
        pos = len(score.positive_entity_wallets)
        neg = len(score.negative_entity_wallets)
        eff = score.effective_overlap_count

        if neg > 0 and eff == 0:
            return (
                "FILTERED",
                f"all_{neg}_overlap_wallets_are_negative_pnl_tier",
            )

        if score.has_smart_money_cluster:
            cluster_abbrev = ",".join(
                w[:8] for w in score.cluster_wallets[:5]
            )
            return (
                "SUPER_ALPHA",
                f"smart_money_cluster_{len(score.cluster_wallets)}_wallets:[{cluster_abbrev}]",
            )

        # Count ELITE wallets (replaces old fund/VC count)
        elite_count = sum(
            1
            for w in score.positive_entity_wallets
            if (score.wallet_profiles.get(w) or {}).get("pnl_tier") == "ELITE"
        )

        if elite_count >= 2:
            return "SUPER_ALPHA", f"{elite_count}_elite_pnl_wallets"

        if elite_count == 1:
            return "STRONG", "1_elite_pnl_wallet"

        if pos >= 3:
            return "STRONG", f"{pos}_positive_pnl_wallets"

        if pos >= 1:
            return "STANDARD", f"{pos}_positive_pnl_wallet(s)"

        if neg > 0:
            return (
                "STANDARD",
                f"overlap_contains_{neg}_negative_pnl_wallets_excluded",
            )

        return "NONE", "no_pnl_data_available"


# ---------------------------------------------------------------------------
# Post-prediction grade booster  (pure function — called in winner_monitor.py)
# ---------------------------------------------------------------------------

def apply_smart_money_boost(
    current_grade: str,
    ml_prediction: Optional[Dict],
    smart_money_score: SmartMoneyTokenScore,
) -> Tuple[str, str, bool]:
    """
    Overlay Smart Money conviction on top of the existing grade + ML prediction.

    This is a POST-PREDICTION overlay — model features and training are
    never touched.  Results are appended to the token result dict and
    forwarded to the Telegram alert formatter.

    Parameters
    ----------
    current_grade      : e.g. "MEDIUM"
    ml_prediction      : dict with keys probability / confidence / risk_tier
    smart_money_score  : SmartMoneyTokenScore from scorer.score_token()

    Returns
    -------
    (final_grade, alert_label, is_super_alpha)

    alert_label values:
        "SUPER-ALPHA 🔥"  — top-tier institutional / cluster signal
        "ALPHA ⭐"         — strong smart money backing
        "STANDARD 📊"      — regular overlap signal
        "FILTERED ⚠️"      — signal suppressed (bots / CEX wallets)
    """
    boost_tier = smart_money_score.boost_tier
    is_super = False

    # Suppressed signal
    if boost_tier == "FILTERED":
        return current_grade, "FILTERED ⚠️", False

    ml_prob = 0.0
    if ml_prediction:
        try:
            ml_prob = float(ml_prediction.get("probability") or 0.0)
        except (ValueError, TypeError):
            ml_prob = 0.0

    if boost_tier == "SUPER_ALPHA":
        is_super = True
        return _upgrade_grade(current_grade, steps=1), "SUPER-ALPHA 🔥", True

    # Grade MEDIUM+ + SM cluster = SUPER-ALPHA (spec rule)
    if (
        boost_tier in ("STRONG", "STANDARD")
        and current_grade in ("MEDIUM", "HIGH", "VERY_HIGH")
        and smart_money_score.has_smart_money_cluster
    ):
        return _upgrade_grade(current_grade, steps=1), "SUPER-ALPHA 🔥", True

    if boost_tier == "STRONG":
        return _upgrade_grade(current_grade, steps=1), "ALPHA ⭐", False

    if boost_tier == "STANDARD":
        label = "ALPHA ⭐" if ml_prob >= 0.65 else "STANDARD 📊"
        return current_grade, label, False

    return current_grade, "STANDARD 📊", False


def _upgrade_grade(grade: str, steps: int = 1) -> str:
    """Upgrade a grade by N tier steps, capped at VERY_HIGH."""
    try:
        idx = GRADE_TIERS.index(grade)
        return GRADE_TIERS[min(idx + steps, len(GRADE_TIERS) - 1)]
    except ValueError:
        return grade#!/usr/bin/env python3
"""
smart_money_scorer.py
=====================
3-Layer Smart Money Conviction Scorer — on-chain performance intelligence.

Layer A │ Analytical Performance  — Dune SQL frequency (rolling-7-day win-rate)
Layer B │ PnL Enrichment          — Moralis profitability/summary (ELITE/STRONG/ACTIVE/NOISE/NEGATIVE)
Layer C │ Behavioral Heuristics   — Helius RPC insider-rank + temporal clustering

Design principles
-----------------
• Credit-efficient: PnL data is fetched once and cached to disk (7-day TTL).
  Only wallets in the top-N Dune Winners list are ever fetched (credit gate).
• Non-destructive: Does NOT modify ML model features or retrain anything.
  The scorer runs POST-prediction and applies an "up-rank boost" to alerts.
• Fully async: All network I/O is async-safe and uses the shared aiohttp session.
• Fail-safe: Every method returns a safe default on any error.

Boost logic (applied in winner_monitor.py after ML prediction):
  Grade MEDIUM/HIGH/VERY_HIGH + Smart Money Cluster  →  SUPER-ALPHA  🔥
  Any grade + 2+ ELITE wallets                        →  SUPER-ALPHA  🔥
  Any grade + 1  ELITE wallet                         →  STRONG / ALPHA ⭐ + +1 tier
  Any grade + 3+ STRONG/ACTIVE wallets                →  STRONG / ALPHA ⭐
  All overlap wallets are NEGATIVE tier               →  FILTERED  (signal suppressed)

Usage
-----
    scorer = SmartMoneyScorer(
        moralis_client=moralis_client,
        wallet_ranker=wallet_ranker,
        top_n_label_threshold=500,
        cluster_window_seconds=1800,
        debug=True,
    )

    score = await scorer.score_token(
        mint="<token_mint>",
        overlap_wallets=["wallet1", "wallet2", ...],
        wallet_freq={"wallet1": 7, "wallet2": 3, ...},
        ml_passed=True,
        pair_created_at_ms=1234567890000,           # from DexScreener pairCreatedAt
        wallet_buy_timestamps={"wallet1": "2024-11-21T09:22:28.000Z"},
        rugcheck_raw=result["security"]["rugcheck_raw"]["raw"],
    )

    final_grade, alert_label, is_super_alpha = apply_smart_money_boost(
        current_grade=result["grade"],
        ml_prediction=result.get("ml_prediction"),
        smart_money_score=score,
    )
"""

import asyncio
import aiohttp
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from shared.moralis_client import (
    MoralisClient,
    POSITIVE_PNL_TIERS,
    NEGATIVE_PNL_TIERS,
    PNL_TIER_WEIGHT_MULTIPLIERS,
    # Back-compat aliases still exported from moralis_client
    POSITIVE_ENTITY_CATEGORIES,
    NEGATIVE_ENTITY_CATEGORIES,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# PnL tier weight multipliers for weighted overlap scoring.
# ELITE replaces the old Fund/VC label; STRONG replaces Whale/DeFi Whale, etc.
# Imported directly from moralis_client so there is a single source of truth:
#   ELITE    → 3.0x   (proven top performer, $50k+ profit, 55%+ win rate)
#   STRONG   → 2.5x   (consistently profitable, $10k+, 50%+ win rate)
#   ACTIVE   → 1.5x   (net positive, $1k+, ≥5 trades)
#   NOISE    → 1.0x   (insufficient data — default, no boost)
#   NEGATIVE → 0.0x   (loss-maker / bot — excluded from effective overlap)
ENTITY_WEIGHT_MULTIPLIERS = PNL_TIER_WEIGHT_MULTIPLIERS

# Minimum independent SM entities for a "cluster" signal
CLUSTER_MIN_WALLETS = 3

# Grade tier ordering for up-rank logic
GRADE_TIERS = ["NONE", "UNKNOWN", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WalletSmartMoneyProfile:
    """Full smart money assessment for a single wallet."""
    address: str

    # Layer A — Analytical Performance (Dune frequency)
    dune_frequency: int = 0
    alpha_winner_tier: str = "NONE"       # HIGH / MEDIUM / LOW / NONE

    # Layer B — PnL Enrichment (Moralis profitability/summary)
    pnl_tier: str = "NOISE"               # ELITE / STRONG / ACTIVE / NOISE / NEGATIVE
    total_realized_profit_usd: Optional[float] = None
    total_realized_profit_pct: Optional[float] = None
    total_trade_volume_usd: Optional[float] = None
    total_count_of_trades: Optional[int] = None
    win_rate_pct: Optional[float] = None
    is_positive_entity: bool = False      # True for ELITE / STRONG / ACTIVE
    is_negative_entity: bool = False      # True for NEGATIVE
    entity_weight_multiplier: float = 1.0 # derived from pnl_tier
    label_from_cache: bool = False

    # Layer C — Behavioral Heuristics (RugCheck + Moralis timestamps)
    insider_rank: Optional[int] = None    # minutes after launch (0 = RugCheck confirmed)
    is_insider: bool = False
    insider_type: Optional[str] = None   # "SNIPER" / "EARLY_BUYER" / "RUGCHECK"


@dataclass
class SmartMoneyTokenScore:
    """Aggregated Smart Money conviction score for a single token."""
    mint: str
    checked_at: str = ""

    # Wallet breakdown
    total_overlap_wallets: int = 0
    positive_entity_wallets: List[str] = field(default_factory=list)
    negative_entity_wallets: List[str] = field(default_factory=list)
    insider_wallets: List[str] = field(default_factory=list)
    effective_overlap_wallets: List[str] = field(default_factory=list)  # excl. negatives

    # Scores
    raw_overlap_count: int = 0
    effective_overlap_count: int = 0           # negatives excluded
    smart_money_weighted_score: float = 0.0    # sum(freq × entity_multiplier)
    has_smart_money_cluster: bool = False      # 3+ SM entities in 30-min window
    cluster_wallets: List[str] = field(default_factory=list)

    # Boost recommendation
    boost_tier: str = "NONE"     # SUPER_ALPHA / STRONG / STANDARD / FILTERED / NONE
    boost_reason: str = ""

    # Per-wallet profiles for logging / downstream consumers
    wallet_profiles: Dict[str, Dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------

class SmartMoneyScorer:
    """
    Evaluate Smart Money conviction for any token that has reached the
    overlap analysis stage in winner_monitor.py.

    Deliberately stateless with respect to the main pipeline —
    it only *reads* existing data and *appends* enrichment metadata.
    """

    def __init__(
        self,
        moralis_client: MoralisClient,
        wallet_ranker: Any,               # WinnerWalletRanker — avoids circular import
        top_n_label_threshold: int = 500,
        cluster_window_seconds: int = 1800,
        debug: bool = False,
    ):
        self.moralis_client = moralis_client
        self.wallet_ranker = wallet_ranker
        self.top_n_label_threshold = top_n_label_threshold
        self.cluster_window_seconds = cluster_window_seconds
        self.debug = debug

        # In-memory buy-time store for cluster detection: mint → [(wallet, ts), …]
        self._buy_time_cache: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

    # =========================================================================
    # Public entry point
    # =========================================================================

    async def score_token(
        self,
        mint: str,
        overlap_wallets: List[str],
        wallet_freq: Dict[str, int],
        ml_passed: bool = False,
        pair_created_at_ms: Optional[int] = None,      # unix ms from DexScreener pairCreatedAt
        wallet_buy_timestamps: Optional[Dict[str, str]] = None,  # {wallet: ISO timestamp} from Moralis
        rugcheck_raw: Optional[Dict] = None,            # raw RugCheck data for topHolders insider flag
    ) -> SmartMoneyTokenScore:
        """
        Main method called by AlphaTokenAnalyzer for every token that passes
        the security gate and has overlap_count > 0.

        Parameters
        ----------
        mint                 : token mint address
        overlap_wallets      : wallets from the Dune Winners list that hold this token
        wallet_freq          : {wallet: dune_appearance_frequency}
        ml_passed            : if False, skip Birdeye PnL lookups (save API credits).
                               Layer A (Dune frequency) and Layer C (insider) still run.
        pair_created_at_ms   : token pair launch time in unix milliseconds (from DexScreener).
                               Used to detect early buyers — wallets that bought within
                               INSIDER_WINDOW_MINUTES of launch.
        wallet_buy_timestamps: {wallet_address: ISO_timestamp} of each wallet's earliest
                               buy of this token (from Moralis swap history).
        rugcheck_raw         : raw RugCheck response dict. Used to cross-reference
                               topHolders[].insider flag with overlap wallets.

        Returns
        -------
        SmartMoneyTokenScore with all three layers populated
        """
        score = SmartMoneyTokenScore(
            mint=mint,
            checked_at=datetime.now(timezone.utc).isoformat(),
            raw_overlap_count=len(overlap_wallets),
            total_overlap_wallets=len(overlap_wallets),
        )

        if not overlap_wallets:
            score.boost_tier = "NONE"
            score.boost_reason = "no_overlap_wallets"
            return score

        # Determine which wallets are within the Birdeye PnL gate
        top_n_wallet_set = self._get_top_n_wallet_set()

        # Build per-wallet profiles across all 3 layers concurrently.
        # ml_passed controls whether Layer B (Birdeye PnL) is called.
        profiles = await self._build_wallet_profiles(
            overlap_wallets, wallet_freq, top_n_wallet_set, mint, ml_passed,
            pair_created_at_ms, wallet_buy_timestamps, rugcheck_raw,
        )

        # Aggregate
        effective_wallets: List[str] = []
        sm_weighted_score: float = 0.0
        positive_entities: List[str] = []
        negative_entities: List[str] = []
        insider_wallets: List[str] = []

        for wallet, profile in profiles.items():
            if profile.is_negative_entity:
                negative_entities.append(wallet)
                if self.debug:
                    print(
                        f"[SmartMoney] 🚫 {wallet[:8]}... EXCLUDED "
                        f"(pnl_tier: {profile.pnl_tier}, "
                        f"profit: ${profile.total_realized_profit_usd})"
                    )
                continue

            effective_wallets.append(wallet)
            freq = wallet_freq.get(wallet, 1)
            sm_weighted_score += freq * profile.entity_weight_multiplier

            if profile.is_positive_entity:
                positive_entities.append(wallet)
                if self.debug:
                    print(
                        f"[SmartMoney] ✅ {wallet[:8]}... POSITIVE "
                        f"pnl_tier={profile.pnl_tier}, "
                        f"profit=${profile.total_realized_profit_usd}, "
                        f"multiplier={profile.entity_weight_multiplier}x, "
                        f"freq={freq}"
                    )

            if profile.is_insider:
                insider_wallets.append(wallet)
                if self.debug:
                    print(
                        f"[SmartMoney] 🎯 {wallet[:8]}... {profile.insider_type or 'INSIDER'} "
                        f"({profile.insider_rank} min after launch)"
                    )

        # Cluster detection
        cluster_wallets = self._detect_smart_money_cluster(
            mint, positive_entities, profiles
        )
        has_cluster = len(cluster_wallets) >= CLUSTER_MIN_WALLETS

        # Populate score object
        score.effective_overlap_wallets = effective_wallets
        score.effective_overlap_count = len(effective_wallets)
        score.positive_entity_wallets = positive_entities
        score.negative_entity_wallets = negative_entities
        score.insider_wallets = insider_wallets
        score.smart_money_weighted_score = round(sm_weighted_score, 3)
        score.has_smart_money_cluster = has_cluster
        score.cluster_wallets = cluster_wallets
        score.wallet_profiles = {w: asdict(p) for w, p in profiles.items()}

        score.boost_tier, score.boost_reason = self._calculate_boost(score)

        if self.debug:
            print(
                f"[SmartMoney] 📊 {mint[:8]}... | "
                f"eff={score.effective_overlap_count} "
                f"(+{len(positive_entities)} pos, -{len(negative_entities)} neg) | "
                f"insiders={len(insider_wallets)} | "
                f"cluster={has_cluster} | "
                f"weighted={score.smart_money_weighted_score:.2f} | "
                f"boost={score.boost_tier}"
            )

        return score

    # =========================================================================
    # Layer A: Analytical Performance
    # =========================================================================

    @staticmethod
    def _get_alpha_winner_tier(frequency: int) -> str:
        """Classify wallet by 7-day Dune appearance frequency."""
        if frequency >= 10:
            return "HIGH"
        if frequency >= 5:
            return "MEDIUM"
        if frequency >= 2:
            return "LOW"
        return "NONE"

    # =========================================================================
    # Layer B: Categorical Enrichment  (Moralis entities)
    # =========================================================================

    def _get_top_n_wallet_set(self) -> Set[str]:
        """Return the top-N wallet addresses from the Dune Winners ranking."""
        try:
            top_n = self.wallet_ranker.get_top_n_wallets(n=self.top_n_label_threshold)
            return {w for w, _ in top_n}
        except Exception:
            return set()

    async def _fetch_entity_label(self, wallet: str, in_top_n: bool, ml_passed: bool = False) -> Dict[str, Any]:
        """
        Credit gate — two conditions must both be true to call Birdeye:
          1. Wallet is in the top-N Dune Winners list
          2. ML_PASSED=True for this token (set by caller)

        If either condition fails → return NOISE sentinel (0 credits).
        Cache HITs always return immediately regardless of ml_passed.
        """
        # Always check cache first — never skip a cached result
        cached = self.moralis_client.get_cached_label(wallet)
        if cached is not None:
            result = dict(cached)
            result["_from_cache"] = True
            result["is_positive"] = result.get("pnl_tier") in POSITIVE_PNL_TIERS
            result["is_negative"] = result.get("pnl_tier") in NEGATIVE_PNL_TIERS
            return result

        # Gate: skip API call if outside top-N OR ML didn't pass
        if not in_top_n or not ml_passed:
            reason = "not_in_top_n" if not in_top_n else "ml_not_passed"
            if self.debug and not in_top_n is False:
                pass  # don't spam logs for every non-top-N wallet
            return {
                "pnl_tier":                  "NOISE",
                "total_realized_profit_usd": None,
                "total_realized_profit_pct": None,
                "total_trade_volume_usd":    None,
                "total_count_of_trades":     None,
                "win_rate_pct":              None,
                "is_positive":               False,
                "is_negative":               False,
                "_from_cache":               True,   # treated as "no API needed"
                "_skipped":                  True,
                "_skip_reason":              reason,
            }

        return await self.moralis_client.get_wallet_pnl(wallet)

    @staticmethod
    def _get_entity_multiplier(pnl_tier: str) -> float:
        """Map PnL tier to a weight multiplier for the weighted overlap score."""
        return PNL_TIER_WEIGHT_MULTIPLIERS.get(pnl_tier, 1.0)

    # =========================================================================
    # Layer C: Insider Detection  (RugCheck + Moralis timestamps, zero extra credits)
    # =========================================================================

    # Tier 1 — SNIPER: bought within this many minutes of pair creation
    SNIPER_WINDOW_MINUTES:  int = int(os.getenv("SM_SNIPER_WINDOW_MINUTES",  "5"))
    # Tier 2 — EARLY_BUYER: bought within this many minutes (wider window)
    INSIDER_WINDOW_MINUTES: int = int(os.getenv("SM_INSIDER_WINDOW_MINUTES", "30"))

    async def _check_insider(
        self,
        wallet: str,
        pair_created_at_ms: Optional[int],
        wallet_buy_timestamps: Optional[Dict[str, str]],
        rc_insider_owners: Set[str],
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Determine if a wallet is an early insider using two free signals.
        Returns (minutes_after_launch, insider_type) or (None, None).

        Signal 1 — RugCheck topHolders[].insider:
            Strongest signal — RugCheck's own graph analysis flagged this wallet.
            Returns (0, "RUGCHECK").

        Signal 2 — Timestamp comparison (Moralis swap ts vs DexScreener pairCreatedAt):
            ≤ SM_SNIPER_WINDOW_MINUTES (default 5 min)  → (minutes, "SNIPER")
            ≤ SM_INSIDER_WINDOW_MINUTES (default 30 min) → (minutes, "EARLY_BUYER")
            > SM_INSIDER_WINDOW_MINUTES                  → (None, None)

        Zero extra API calls. Both sources already fetched upstream.
        """
        # Signal 1: RugCheck graph insider — strongest, check first
        if wallet in rc_insider_owners:
            if self.debug:
                print(f"[SmartMoney] 🎯 {wallet[:8]}... RUGCHECK INSIDER")
            return 0, "RUGCHECK"

        # Signal 2: Timestamp-based early buyer detection
        if not pair_created_at_ms or not wallet_buy_timestamps:
            return None, None

        buy_ts_iso = wallet_buy_timestamps.get(wallet)
        if not buy_ts_iso:
            return None, None

        try:
            buy_dt  = datetime.fromisoformat(buy_ts_iso.replace("Z", "+00:00"))
            pair_dt = datetime.fromtimestamp(pair_created_at_ms / 1000, tz=timezone.utc)
            mins    = (buy_dt - pair_dt).total_seconds() / 60

            if mins < 0:
                # Bought before pair existed — data anomaly, skip
                return None, None

            if mins <= self.SNIPER_WINDOW_MINUTES:
                if self.debug:
                    print(f"[SmartMoney] 🎯 {wallet[:8]}... SNIPER ({mins:.1f} min after launch)")
                return round(mins), "SNIPER"

            if mins <= self.INSIDER_WINDOW_MINUTES:
                if self.debug:
                    print(f"[SmartMoney] 🎯 {wallet[:8]}... EARLY BUYER ({mins:.1f} min after launch)")
                return round(mins), "EARLY_BUYER"

        except Exception as e:
            if self.debug:
                print(f"[SmartMoney] ⚠️ Insider timestamp parse error for {wallet[:8]}...: {e}")

        return None, None

    def _record_buy_time(self, mint: str, wallet: str):
        """Record current timestamp as the buy-time for cluster detection."""
        now_ts = int(time.time())
        self._buy_time_cache[mint].append((wallet, now_ts))
        cutoff = now_ts - (self.cluster_window_seconds * 2)
        self._buy_time_cache[mint] = [
            (w, t) for w, t in self._buy_time_cache[mint] if t >= cutoff
        ]

    def _detect_smart_money_cluster(
        self,
        mint: str,
        positive_wallets: List[str],
        profiles: Dict[str, WalletSmartMoneyProfile],
    ) -> List[str]:
        """
        Detect 3+ independent Smart Money entities buying the same token
        within cluster_window_seconds.  'Independent' = different entity names
        or categories (prevents one fund with many wallets triggering a cluster).
        """
        for w in positive_wallets:
            self._record_buy_time(mint, w)

        buy_events = self._buy_time_cache.get(mint, [])
        profiled_positives = {w for w in positive_wallets if w in profiles}
        relevant_events = [
            (w, t) for w, t in buy_events if w in profiled_positives
        ]

        if len(relevant_events) < CLUSTER_MIN_WALLETS:
            return []

        relevant_events.sort(key=lambda x: x[1])
        best_cluster: List[str] = []

        for i in range(len(relevant_events)):
            window_wallets: List[str] = []
            window_entity_ids: Set[str] = set()
            t_start = relevant_events[i][1]

            for w, t in relevant_events[i:]:
                if t - t_start > self.cluster_window_seconds:
                    break
                profile = profiles.get(w)
                # Deduplicate by pnl_tier bucket — prevents the same tier of
                # wallet trivially counting as "independent" entities.
                # Use address as tiebreaker when pnl_tier is NOISE/None.
                pnl_tier = getattr(profile, "pnl_tier", None) if profile else None
                entity_id = (
                    f"{pnl_tier}:{w}" if pnl_tier in ("NOISE", None)
                    else pnl_tier + ":" + w   # still unique per wallet, but tier-grouped
                )
                if entity_id not in window_entity_ids:
                    window_wallets.append(w)
                    window_entity_ids.add(entity_id)

            if len(window_wallets) >= CLUSTER_MIN_WALLETS:
                if len(window_wallets) > len(best_cluster):
                    best_cluster = window_wallets

        return best_cluster

    # =========================================================================
    # Combined profile builder
    # =========================================================================

    async def _build_wallet_profiles(
        self,
        wallets: List[str],
        wallet_freq: Dict[str, int],
        top_n_wallet_set: Set[str],
        mint: str,
        ml_passed: bool = False,
        pair_created_at_ms: Optional[int] = None,
        wallet_buy_timestamps: Optional[Dict[str, str]] = None,
        rugcheck_raw: Optional[Dict] = None,
    ) -> Dict[str, WalletSmartMoneyProfile]:
        """Build a WalletSmartMoneyProfile for each wallet (all 3 layers in parallel)."""

        # Pre-compute RugCheck insider set from topHolders for O(1) lookup
        rc_insider_owners: Set[str] = set()
        if rugcheck_raw:
            for holder in (rugcheck_raw.get("topHolders") or []):
                if holder.get("insider") is True:
                    rc_insider_owners.add(holder.get("owner", ""))

        async def _profile_one(wallet: str) -> Tuple[str, WalletSmartMoneyProfile]:
            freq     = wallet_freq.get(wallet, 1)
            in_top_n = wallet in top_n_wallet_set

            pnl_task = asyncio.create_task(
                self._fetch_entity_label(wallet, in_top_n, ml_passed)
            )
            insider_task = asyncio.create_task(
                self._check_insider(
                    wallet,
                    pair_created_at_ms,
                    wallet_buy_timestamps,
                    rc_insider_owners,
                )
            )
            pnl, insider_result = await asyncio.gather(pnl_task, insider_task)

            pnl_tier = pnl.get("pnl_tier", "NOISE")

            # _check_insider returns (minutes_after_launch, insider_type) or (None, None)
            insider_rank, insider_type = insider_result if insider_result else (None, None)

            profile = WalletSmartMoneyProfile(
                address=wallet,
                # Layer A
                dune_frequency=freq,
                alpha_winner_tier=self._get_alpha_winner_tier(freq),
                # Layer B
                pnl_tier=pnl_tier,
                total_realized_profit_usd=pnl.get("total_realized_profit_usd"),
                total_realized_profit_pct=pnl.get("total_realized_profit_pct"),
                total_trade_volume_usd=pnl.get("total_trade_volume_usd"),
                total_count_of_trades=pnl.get("total_count_of_trades"),
                win_rate_pct=pnl.get("win_rate_pct"),
                is_positive_entity=pnl.get("is_positive", False),
                is_negative_entity=pnl.get("is_negative", False),
                entity_weight_multiplier=self._get_entity_multiplier(pnl_tier),
                label_from_cache=pnl.get("_from_cache", False),
                # Layer C
                insider_rank=insider_rank,       # minutes after launch (0 = RugCheck)
                is_insider=insider_rank is not None,
                insider_type=insider_type,       # "SNIPER" / "EARLY_BUYER" / "RUGCHECK"
            )
            return wallet, profile

        tasks = [asyncio.create_task(_profile_one(w)) for w in wallets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        profiles: Dict[str, WalletSmartMoneyProfile] = {}
        for res in results:
            if isinstance(res, Exception):
                if self.debug:
                    print(f"[SmartMoney] ⚠️ Profile build error: {res}")
                continue
            w, profile = res
            profiles[w] = profile

        return profiles

    # =========================================================================
    # Boost tier calculation
    # =========================================================================

    def _calculate_boost(
        self, score: SmartMoneyTokenScore
    ) -> Tuple[str, str]:
        """
        Assign a boost tier based on the aggregated smart money evidence.

        Tier ladder (highest match wins):
            SUPER_ALPHA  → SM cluster present  OR  2+ ELITE wallets
            STRONG       → 1 ELITE wallet      OR  3+ STRONG/ACTIVE wallets
            STANDARD     → any positive PnL wallet (ACTIVE or above)
            FILTERED     → ALL overlap wallets are NEGATIVE tier
            NONE         → no PnL data / all NOISE
        """
        pos = len(score.positive_entity_wallets)
        neg = len(score.negative_entity_wallets)
        eff = score.effective_overlap_count

        if neg > 0 and eff == 0:
            return (
                "FILTERED",
                f"all_{neg}_overlap_wallets_are_negative_pnl_tier",
            )

        if score.has_smart_money_cluster:
            cluster_abbrev = ",".join(
                w[:8] for w in score.cluster_wallets[:5]
            )
            return (
                "SUPER_ALPHA",
                f"smart_money_cluster_{len(score.cluster_wallets)}_wallets:[{cluster_abbrev}]",
            )

        # Count ELITE wallets (replaces old fund/VC count)
        elite_count = sum(
            1
            for w in score.positive_entity_wallets
            if (score.wallet_profiles.get(w) or {}).get("pnl_tier") == "ELITE"
        )

        if elite_count >= 2:
            return "SUPER_ALPHA", f"{elite_count}_elite_pnl_wallets"

        if elite_count == 1:
            return "STRONG", "1_elite_pnl_wallet"

        if pos >= 3:
            return "STRONG", f"{pos}_positive_pnl_wallets"

        if pos >= 1:
            return "STANDARD", f"{pos}_positive_pnl_wallet(s)"

        if neg > 0:
            return (
                "STANDARD",
                f"overlap_contains_{neg}_negative_pnl_wallets_excluded",
            )

        return "NONE", "no_pnl_data_available"


# ---------------------------------------------------------------------------
# Post-prediction grade booster  (pure function — called in winner_monitor.py)
# ---------------------------------------------------------------------------

def apply_smart_money_boost(
    current_grade: str,
    ml_prediction: Optional[Dict],
    smart_money_score: SmartMoneyTokenScore,
) -> Tuple[str, str, bool]:
    """
    Overlay Smart Money conviction on top of the existing grade + ML prediction.

    This is a POST-PREDICTION overlay — model features and training are
    never touched.  Results are appended to the token result dict and
    forwarded to the Telegram alert formatter.

    Parameters
    ----------
    current_grade      : e.g. "MEDIUM"
    ml_prediction      : dict with keys probability / confidence / risk_tier
    smart_money_score  : SmartMoneyTokenScore from scorer.score_token()

    Returns
    -------
    (final_grade, alert_label, is_super_alpha)

    alert_label values:
        "SUPER-ALPHA 🔥"  — top-tier institutional / cluster signal
        "ALPHA ⭐"         — strong smart money backing
        "STANDARD 📊"      — regular overlap signal
        "FILTERED ⚠️"      — signal suppressed (bots / CEX wallets)
    """
    boost_tier = smart_money_score.boost_tier
    is_super = False

    # Suppressed signal
    if boost_tier == "FILTERED":
        return current_grade, "FILTERED ⚠️", False

    ml_prob = 0.0
    if ml_prediction:
        try:
            ml_prob = float(ml_prediction.get("probability") or 0.0)
        except (ValueError, TypeError):
            ml_prob = 0.0

    if boost_tier == "SUPER_ALPHA":
        is_super = True
        return _upgrade_grade(current_grade, steps=1), "SUPER-ALPHA 🔥", True

    # Grade MEDIUM+ + SM cluster = SUPER-ALPHA (spec rule)
    if (
        boost_tier in ("STRONG", "STANDARD")
        and current_grade in ("MEDIUM", "HIGH", "VERY_HIGH")
        and smart_money_score.has_smart_money_cluster
    ):
        return _upgrade_grade(current_grade, steps=1), "SUPER-ALPHA 🔥", True

    if boost_tier == "STRONG":
        return _upgrade_grade(current_grade, steps=1), "ALPHA ⭐", False

    if boost_tier == "STANDARD":
        label = "ALPHA ⭐" if ml_prob >= 0.65 else "STANDARD 📊"
        return current_grade, label, False

    return current_grade, "STANDARD 📊", False


def _upgrade_grade(grade: str, steps: int = 1) -> str:
    """Upgrade a grade by N tier steps, capped at VERY_HIGH."""
    try:
        idx = GRADE_TIERS.index(grade)
        return GRADE_TIERS[min(idx + steps, len(GRADE_TIERS) - 1)]
    except ValueError:
        return grade