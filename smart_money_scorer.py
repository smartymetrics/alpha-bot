#!/usr/bin/env python3
"""
smart_money_scorer.py
=====================
3-Layer Smart Money Conviction Scorer — Arkham / Nansen-style wallet intelligence.

Layer A │ Analytical Performance  — Dune SQL frequency (rolling-7-day win-rate)
Layer B │ Categorical Enrichment  — Moralis entity labeling (Fund, VC, Whale, Bot…)
Layer C │ Behavioral Heuristics   — Helius RPC insider-rank + temporal clustering

Design principles
-----------------
• Credit-efficient: Moralis labels are fetched once and cached to disk forever.
  Only wallets in the top-500 Dune Winners list are ever labeled (credit gate).
• Non-destructive: Does NOT modify ML model features or retrain anything.
  The scorer runs POST-prediction and applies an "up-rank boost" to alerts.
• Fully async: All network I/O is async-safe and uses the shared aiohttp session.
• Fail-safe: Every method returns a safe default on any error.

Boost logic (applied in winner_monitor.py after ML prediction):
  Grade MEDIUM/HIGH/VERY_HIGH + Smart Money Cluster  →  SUPER-ALPHA  🔥
  Any grade + 2+ Fund/VC wallets                     →  SUPER-ALPHA  🔥
  Any grade + 1  Fund/VC wallet                      →  STRONG / ALPHA ⭐ + +1 tier
  Any grade + 3+ positive entity wallets             →  STRONG / ALPHA ⭐
  All overlap wallets are negative entities          →  FILTERED  (signal suppressed)

Usage
-----
    scorer = SmartMoneyScorer(
        moralis_client=moralis_client,
        helius_api_key=HELIUS_API_KEY,
        http_session=http_session,
        wallet_ranker=wallet_ranker,
        top_n_label_threshold=500,
        cluster_window_seconds=1800,
        debug=True,
    )

    score = await scorer.score_token(
        mint="<token_mint>",
        overlap_wallets=["wallet1", "wallet2", ...],
        wallet_freq={"wallet1": 7, "wallet2": 3, ...},
    )

    final_grade, alert_label, is_super_alpha = apply_smart_money_boost(
        current_grade=result["grade"],
        ml_prediction=result.get("ml_prediction"),
        smart_money_score=score,
    )
"""

import asyncio
import aiohttp
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from shared.moralis_client import (
    MoralisClient,
    POSITIVE_ENTITY_CATEGORIES,
    NEGATIVE_ENTITY_CATEGORIES,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Entity quality multipliers for weighted overlap scoring
ENTITY_WEIGHT_MULTIPLIERS: Dict[str, float] = {
    "fund":                  3.0,
    "venture capital":       3.0,
    "defi whale":            2.5,
    "whale":                 2.0,
    "high-frequency trader": 2.0,
    "smart money":           2.5,
    "institution":           3.0,
    "market maker":          1.5,
    # Negative entities are excluded entirely (weight = 0.0)
    "mev bot":               0.0,
    "arbitrageur":           0.0,
    "centralized exchange":  0.0,
    "bridge":                0.0,
    "bot":                   0.0,
    "arb bot":               0.0,
    "spam":                  0.0,
    "scammer":               0.0,
}

# Insider rank threshold: wallet buy must appear in first N mint signatures
INSIDER_RANK_THRESHOLD = 50

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

    # Layer A — Analytical Performance
    dune_frequency: int = 0
    alpha_winner_tier: str = "NONE"       # HIGH / MEDIUM / LOW / NONE

    # Layer B — Categorical Enrichment
    entity_name: Optional[str] = None
    entity_category: Optional[str] = None
    entity_labels: List[str] = field(default_factory=list)
    is_positive_entity: bool = False
    is_negative_entity: bool = False
    entity_weight_multiplier: float = 1.0
    label_from_cache: bool = False

    # Layer C — Behavioral Heuristics
    insider_rank: Optional[int] = None    # 1-based position in mint tx history
    is_insider: bool = False


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
        helius_api_key: str,
        http_session: aiohttp.ClientSession,
        wallet_ranker: Any,               # WinnerWalletRanker — avoids circular import
        top_n_label_threshold: int = 500,
        cluster_window_seconds: int = 1800,
        debug: bool = False,
    ):
        self.moralis_client = moralis_client
        self.helius_api_key = helius_api_key
        self.http_session = http_session
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
    ) -> SmartMoneyTokenScore:
        """
        Main method called by AlphaTokenAnalyzer for every token that passes
        the security gate and has overlap_count > 0.

        Parameters
        ----------
        mint            : token mint address
        overlap_wallets : wallets from the Dune Winners list that hold this token
        wallet_freq     : {wallet: dune_appearance_frequency}

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

        # Determine which wallets are within the Moralis labeling gate
        top_n_wallet_set = self._get_top_n_wallet_set()

        # Build per-wallet profiles across all 3 layers concurrently
        profiles = await self._build_wallet_profiles(
            overlap_wallets, wallet_freq, top_n_wallet_set, mint
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
                        f"(category: {profile.entity_category})"
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
                        f"entity={profile.entity_name}, "
                        f"cat={profile.entity_category}, "
                        f"multiplier={profile.entity_weight_multiplier}x, "
                        f"freq={freq}"
                    )

            if profile.is_insider:
                insider_wallets.append(wallet)
                if self.debug:
                    print(
                        f"[SmartMoney] 🎯 {wallet[:8]}... INSIDER "
                        f"(rank #{profile.insider_rank})"
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

    async def _fetch_entity_label(self, wallet: str, in_top_n: bool) -> Dict[str, Any]:
        """
        Credit gate:
          • Wallet NOT in top-N  → skip API, return empty label (0 credits)
          • Wallet in top-N      → check persistent cache, then API (0 or 1 credit)
        """
        if not in_top_n:
            return {
                "entity_name": None,
                "category": None,
                "labels": [],
                "is_positive": False,
                "is_negative": False,
                "_from_cache": True,
                "_skipped": True,
            }
        return await self.moralis_client.get_wallet_labels(wallet)

    @staticmethod
    def _get_entity_multiplier(category: Optional[str], labels: List[str]) -> float:
        """Map entity category/labels to a weight multiplier."""
        cat = (category or "").lower()
        if cat in ENTITY_WEIGHT_MULTIPLIERS:
            return ENTITY_WEIGHT_MULTIPLIERS[cat]
        for lbl in labels:
            lbl_l = (lbl or "").lower()
            if lbl_l in ENTITY_WEIGHT_MULTIPLIERS:
                return ENTITY_WEIGHT_MULTIPLIERS[lbl_l]
        return 1.0   # default: standard winner wallet

    # =========================================================================
    # Layer C: Behavioral Heuristics  (Helius RPC)
    # =========================================================================

    async def _get_insider_rank(self, mint: str, wallet: str) -> Optional[int]:
        """
        Check whether *wallet* is among the first INSIDER_RANK_THRESHOLD
        fee-payers in the token's mint transaction history via
        getSignaturesForAddress on Helius (which returns feePayer in metadata).

        Returns 1-based rank, or None if not found / API unavailable.
        """
        if not self.helius_api_key:
            return None

        url = f"https://mainnet.helius-rpc.com/?api-key={self.helius_api_key}"
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getSignaturesForAddress",
            "params": [
                mint,
                {"limit": INSIDER_RANK_THRESHOLD, "commitment": "confirmed"},
            ],
        }

        try:
            async with self.http_session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                sigs = data.get("result") or []

                # Earliest transactions are at the end of the list
                for rank, sig_info in enumerate(reversed(sigs), start=1):
                    if isinstance(sig_info, dict):
                        fee_payer = (
                            sig_info.get("feePayer")
                            or sig_info.get("fee_payer")
                        )
                        if fee_payer == wallet:
                            return rank
                return None

        except Exception as e:
            if self.debug:
                print(
                    f"[SmartMoney] ⚠️ Helius insider check failed "
                    f"for {wallet[:8]}...: {e}"
                )
            return None

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
                entity_id = (
                    (profile.entity_name or "")
                    or (profile.entity_category or "")
                    or w
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
    ) -> Dict[str, WalletSmartMoneyProfile]:
        """Build a WalletSmartMoneyProfile for each wallet (all 3 layers in parallel)."""

        async def _profile_one(wallet: str) -> Tuple[str, WalletSmartMoneyProfile]:
            freq = wallet_freq.get(wallet, 1)
            in_top_n = wallet in top_n_wallet_set

            label_task = asyncio.create_task(
                self._fetch_entity_label(wallet, in_top_n)
            )
            insider_task = asyncio.create_task(
                self._get_insider_rank(mint, wallet)
            )
            label, insider_rank = await asyncio.gather(label_task, insider_task)

            category = label.get("category")
            labels_list = label.get("labels") or []

            profile = WalletSmartMoneyProfile(
                address=wallet,
                # Layer A
                dune_frequency=freq,
                alpha_winner_tier=self._get_alpha_winner_tier(freq),
                # Layer B
                entity_name=label.get("entity_name"),
                entity_category=category,
                entity_labels=labels_list,
                is_positive_entity=label.get("is_positive", False),
                is_negative_entity=label.get("is_negative", False),
                entity_weight_multiplier=self._get_entity_multiplier(
                    category, labels_list
                ),
                label_from_cache=label.get("_from_cache", False),
                # Layer C
                insider_rank=insider_rank,
                is_insider=(
                    insider_rank is not None
                    and insider_rank <= INSIDER_RANK_THRESHOLD
                ),
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
            SUPER_ALPHA  → SM cluster present  OR  2+ Fund/VC wallets
            STRONG       → 1 Fund/VC wallet   OR  3+ positive entity wallets
            STANDARD     → any positive entity wallet present
            FILTERED     → ALL overlap wallets are negative entities
            NONE         → no entity data / no positives / no negatives
        """
        pos = len(score.positive_entity_wallets)
        neg = len(score.negative_entity_wallets)
        eff = score.effective_overlap_count

        if neg > 0 and eff == 0:
            return (
                "FILTERED",
                f"all_{neg}_overlap_wallets_are_negative_entities",
            )

        if score.has_smart_money_cluster:
            cluster_abbrev = ",".join(
                w[:8] for w in score.cluster_wallets[:5]
            )
            return (
                "SUPER_ALPHA",
                f"smart_money_cluster_{len(score.cluster_wallets)}_wallets:[{cluster_abbrev}]",
            )

        fund_vc_count = sum(
            1
            for w in score.positive_entity_wallets
            if (score.wallet_profiles.get(w) or {}).get("entity_category")
            in {"fund", "venture capital"}
        )

        if fund_vc_count >= 2:
            return "SUPER_ALPHA", f"{fund_vc_count}_fund_vc_wallets"

        if fund_vc_count == 1:
            return "STRONG", "1_fund_or_vc_wallet"

        if pos >= 3:
            return "STRONG", f"{pos}_positive_entity_wallets"

        if pos >= 1:
            return "STANDARD", f"{pos}_positive_entity_wallet(s)"

        if neg > 0:
            return (
                "STANDARD",
                f"overlap_contains_{neg}_negative_entities_excluded",
            )

        return "NONE", "no_entity_data_available"


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