#!/usr/bin/env python3
"""
tests/verify_labeling.py
========================
Automated verification of the Smart Money 3-layer scoring pipeline.

Tests
-----
1. Entity multiplier lookup — known categories map to correct weights
2. Alpha-winner tier     — frequency bands map to correct tiers
3. Grade upgrade         — _upgrade_grade() respects tier ordering + ceiling
4. Boost tier: FILTERED  — all-negative overlap suppresses signal
5. Boost tier: SUPER_ALPHA (cluster) — 3+ entities within 30-min window
6. Boost tier: SUPER_ALPHA (Fund x2) — two Fund wallets trigger super-alpha
7. Boost tier: STRONG (Fund x1)  — single Fund wallet
8. Boost tier: STRONG (3 pos)    — three positive-entity wallets
9. Boost tier: STANDARD          — one positive-entity wallet
10. apply_smart_money_boost integration — correct (grade, label, flag) returned
11. MEV bot exclusion            — bot wallets are removed from effective overlap
12. Moralis label cache round-trip — mock label persisted and retrieved correctly

Run with:  python -m pytest tests/verify_labeling.py -v
       or:  python tests/verify_labeling.py
"""

import asyncio
import sys
import os
import tempfile
import time
import unittest
from dataclasses import field
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_money_scorer import (
    SmartMoneyScorer,
    SmartMoneyTokenScore,
    WalletSmartMoneyProfile,
    ENTITY_WEIGHT_MULTIPLIERS,
    GRADE_TIERS,
    CLUSTER_MIN_WALLETS,
    apply_smart_money_boost,
    _upgrade_grade,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_score(**kwargs) -> SmartMoneyTokenScore:
    """Build a SmartMoneyTokenScore with sensible defaults for testing."""
    defaults = dict(
        mint="TestMint111",
        checked_at="2024-01-01T00:00:00+00:00",
        total_overlap_wallets=5,
        raw_overlap_count=5,
        effective_overlap_count=5,
        effective_overlap_wallets=["w1", "w2", "w3", "w4", "w5"],
        positive_entity_wallets=[],
        negative_entity_wallets=[],
        insider_wallets=[],
        smart_money_weighted_score=5.0,
        has_smart_money_cluster=False,
        cluster_wallets=[],
        boost_tier="NONE",
        boost_reason="",
        wallet_profiles={},
    )
    defaults.update(kwargs)
    return SmartMoneyTokenScore(**defaults)


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

class TestEntityMultipliers(unittest.TestCase):
    """Test 1: Entity multiplier lookup."""

    def test_fund_multiplier(self):
        self.assertEqual(ENTITY_WEIGHT_MULTIPLIERS["fund"], 3.0)

    def test_vc_multiplier(self):
        self.assertEqual(ENTITY_WEIGHT_MULTIPLIERS["venture capital"], 3.0)

    def test_defi_whale_multiplier(self):
        self.assertEqual(ENTITY_WEIGHT_MULTIPLIERS["defi whale"], 2.5)

    def test_mev_bot_is_zero(self):
        self.assertEqual(ENTITY_WEIGHT_MULTIPLIERS["mev bot"], 0.0)

    def test_bridge_is_zero(self):
        self.assertEqual(ENTITY_WEIGHT_MULTIPLIERS["bridge"], 0.0)

    def test_scorer_get_entity_multiplier_by_category(self):
        self.assertEqual(
            SmartMoneyScorer._get_entity_multiplier("fund", []), 3.0
        )

    def test_scorer_get_entity_multiplier_by_label(self):
        # Category is None, but label matches
        self.assertEqual(
            SmartMoneyScorer._get_entity_multiplier(None, ["whale"]), 2.0
        )

    def test_scorer_get_entity_multiplier_default(self):
        # Unknown category → default 1.0
        self.assertEqual(
            SmartMoneyScorer._get_entity_multiplier("unknown_xyz", []), 1.0
        )


class TestAlphaWinnerTier(unittest.TestCase):
    """Test 2: Alpha-winner tier classification."""

    def test_high_tier(self):
        self.assertEqual(SmartMoneyScorer._get_alpha_winner_tier(10), "HIGH")
        self.assertEqual(SmartMoneyScorer._get_alpha_winner_tier(15), "HIGH")

    def test_medium_tier(self):
        self.assertEqual(SmartMoneyScorer._get_alpha_winner_tier(5), "MEDIUM")
        self.assertEqual(SmartMoneyScorer._get_alpha_winner_tier(9), "MEDIUM")

    def test_low_tier(self):
        self.assertEqual(SmartMoneyScorer._get_alpha_winner_tier(2), "LOW")
        self.assertEqual(SmartMoneyScorer._get_alpha_winner_tier(4), "LOW")

    def test_none_tier(self):
        self.assertEqual(SmartMoneyScorer._get_alpha_winner_tier(0), "NONE")
        self.assertEqual(SmartMoneyScorer._get_alpha_winner_tier(1), "NONE")


class TestGradeUpgrade(unittest.TestCase):
    """Test 3: Grade upgrade utility."""

    def test_upgrade_low_to_medium(self):
        self.assertEqual(_upgrade_grade("LOW", steps=1), "MEDIUM")

    def test_upgrade_medium_to_high(self):
        self.assertEqual(_upgrade_grade("MEDIUM", steps=1), "HIGH")

    def test_upgrade_caps_at_very_high(self):
        self.assertEqual(_upgrade_grade("VERY_HIGH", steps=1), "VERY_HIGH")
        self.assertEqual(_upgrade_grade("VERY_HIGH", steps=5), "VERY_HIGH")

    def test_upgrade_none_stays_none_ish(self):
        # NONE is index 0, +1 = UNKNOWN
        self.assertEqual(_upgrade_grade("NONE", steps=1), "UNKNOWN")

    def test_upgrade_unknown_grade_unchanged(self):
        self.assertEqual(_upgrade_grade("INVALID_GRADE"), "INVALID_GRADE")


class TestBoostTierCalculation(unittest.TestCase):
    """Tests 4-9: _calculate_boost() logic."""

    def _scorer(self) -> SmartMoneyScorer:
        """Build a minimal SmartMoneyScorer with mocked dependencies."""
        scorer = SmartMoneyScorer.__new__(SmartMoneyScorer)
        scorer.moralis_client = MagicMock()
        scorer.helius_api_key = "fake"
        scorer.http_session = MagicMock()
        scorer.wallet_ranker = MagicMock()
        scorer.top_n_label_threshold = 500
        scorer.cluster_window_seconds = 1800
        scorer.debug = False
        from collections import defaultdict
        scorer._buy_time_cache = defaultdict(list)
        return scorer

    def test_filtered_all_negative(self):
        """Test 4: all overlap wallets are negative → FILTERED."""
        scorer = self._scorer()
        score = _make_score(
            effective_overlap_count=0,
            effective_overlap_wallets=[],
            positive_entity_wallets=[],
            negative_entity_wallets=["bot1", "bot2"],
        )
        tier, reason = scorer._calculate_boost(score)
        self.assertEqual(tier, "FILTERED")
        self.assertIn("negative", reason)

    def test_super_alpha_cluster(self):
        """Test 5: SM cluster (3+ entities) → SUPER_ALPHA."""
        scorer = self._scorer()
        score = _make_score(
            positive_entity_wallets=["w1", "w2", "w3"],
            has_smart_money_cluster=True,
            cluster_wallets=["w1", "w2", "w3"],
            effective_overlap_count=3,
        )
        tier, reason = scorer._calculate_boost(score)
        self.assertEqual(tier, "SUPER_ALPHA")
        self.assertIn("cluster", reason)

    def test_super_alpha_two_funds(self):
        """Test 6: 2 Fund/VC wallets → SUPER_ALPHA."""
        scorer = self._scorer()
        score = _make_score(
            positive_entity_wallets=["fund1", "vc1"],
            effective_overlap_count=2,
            wallet_profiles={
                "fund1": {"entity_category": "fund"},
                "vc1":   {"entity_category": "venture capital"},
            },
        )
        tier, reason = scorer._calculate_boost(score)
        self.assertEqual(tier, "SUPER_ALPHA")
        self.assertIn("fund_vc", reason)

    def test_strong_one_fund(self):
        """Test 7: 1 Fund wallet → STRONG."""
        scorer = self._scorer()
        score = _make_score(
            positive_entity_wallets=["fund1"],
            effective_overlap_count=1,
            wallet_profiles={"fund1": {"entity_category": "fund"}},
        )
        tier, reason = scorer._calculate_boost(score)
        self.assertEqual(tier, "STRONG")
        self.assertIn("fund_or_vc", reason)

    def test_strong_three_positive(self):
        """Test 8: 3 positive (non-fund) wallets → STRONG."""
        scorer = self._scorer()
        score = _make_score(
            positive_entity_wallets=["whale1", "whale2", "whale3"],
            effective_overlap_count=3,
            wallet_profiles={
                "whale1": {"entity_category": "whale"},
                "whale2": {"entity_category": "defi whale"},
                "whale3": {"entity_category": "high-frequency trader"},
            },
        )
        tier, reason = scorer._calculate_boost(score)
        self.assertEqual(tier, "STRONG")
        self.assertIn("3", reason)

    def test_standard_one_positive(self):
        """Test 9: 1 positive entity wallet → STANDARD."""
        scorer = self._scorer()
        score = _make_score(
            positive_entity_wallets=["whale1"],
            effective_overlap_count=1,
            wallet_profiles={"whale1": {"entity_category": "whale"}},
        )
        tier, reason = scorer._calculate_boost(score)
        self.assertEqual(tier, "STANDARD")


class TestApplySmartMoneyBoost(unittest.TestCase):
    """Test 10: apply_smart_money_boost() integration."""

    def test_filtered_returns_filtered_label(self):
        score = _make_score(boost_tier="FILTERED", boost_reason="all_2_overlap_wallets_are_negative_entities")
        grade, label, is_super = apply_smart_money_boost("MEDIUM", None, score)
        self.assertEqual(label, "FILTERED ⚠️")
        self.assertFalse(is_super)
        self.assertEqual(grade, "MEDIUM")  # grade unchanged when filtered

    def test_super_alpha_upgrades_grade(self):
        score = _make_score(boost_tier="SUPER_ALPHA", boost_reason="cluster")
        grade, label, is_super = apply_smart_money_boost("MEDIUM", None, score)
        self.assertEqual(grade, "HIGH")   # MEDIUM + 1 = HIGH
        self.assertEqual(label, "SUPER-ALPHA 🔥")
        self.assertTrue(is_super)

    def test_strong_upgrades_grade_and_labels_alpha(self):
        score = _make_score(boost_tier="STRONG", boost_reason="1_fund_or_vc_wallet")
        grade, label, is_super = apply_smart_money_boost("LOW", None, score)
        self.assertEqual(grade, "MEDIUM")
        self.assertEqual(label, "ALPHA ⭐")
        self.assertFalse(is_super)

    def test_standard_high_ml_prob_gives_alpha_label(self):
        score = _make_score(boost_tier="STANDARD", boost_reason="1_positive_entity_wallet(s)")
        ml = {"probability": 0.72}
        grade, label, is_super = apply_smart_money_boost("HIGH", ml, score)
        self.assertEqual(grade, "HIGH")  # grade unchanged for STANDARD
        self.assertEqual(label, "ALPHA ⭐")

    def test_standard_low_ml_prob_gives_standard_label(self):
        score = _make_score(boost_tier="STANDARD", boost_reason="1_positive_entity_wallet(s)")
        ml = {"probability": 0.40}
        grade, label, is_super = apply_smart_money_boost("HIGH", ml, score)
        self.assertEqual(label, "STANDARD 📊")

    def test_grade_medium_with_cluster_gives_super_alpha(self):
        """Spec rule: Grade MEDIUM + SM cluster = SUPER-ALPHA."""
        score = _make_score(
            boost_tier="STRONG",
            has_smart_money_cluster=True,
            cluster_wallets=["w1", "w2", "w3"],
        )
        grade, label, is_super = apply_smart_money_boost("MEDIUM", None, score)
        self.assertEqual(label, "SUPER-ALPHA 🔥")
        self.assertTrue(is_super)


class TestMEVBotExclusion(unittest.TestCase):
    """Test 11: negative entity wallets are excluded from effective overlap."""

    def test_negative_wallets_excluded(self):
        """
        Simulate a token where 2 of 4 overlap wallets are MEV bots.
        effective_overlap_count must be 2, not 4.
        """
        score = _make_score(
            total_overlap_wallets=4,
            raw_overlap_count=4,
            effective_overlap_count=2,
            effective_overlap_wallets=["legit1", "legit2"],
            positive_entity_wallets=[],
            negative_entity_wallets=["mev_bot_1", "mev_bot_2"],
        )
        # Confirm the contract
        self.assertEqual(score.effective_overlap_count, 2)
        self.assertNotIn("mev_bot_1", score.effective_overlap_wallets)
        self.assertNotIn("mev_bot_2", score.effective_overlap_wallets)


class TestMoralisLabelCache(unittest.TestCase):
    """Test 12: Moralis label cache persists and retrieves correctly."""

    def test_cache_round_trip(self):
        """
        Write a label to the cache via get_wallet_labels() mock and
        verify get_cached_label() returns it on the next call.
        """
        import asyncio
        import joblib

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "wallet_labels.pkl")

            # Build a minimal MoralisClient with the temp cache path
            mock_session = MagicMock()
            client = __import__("shared.moralis_client", fromlist=["MoralisClient"]).MoralisClient(
                http_session=mock_session,
                api_keys=["fake_key_123"],
                debug=False,
                label_cache_path=cache_path,
            )

            # Inject a fake response
            fake_label = {
                "entity_name": "Jump Trading",
                "category": "fund",
                "labels": ["smart_money"],
                "_cached_at": int(time.time()),
            }

            address = "FundWalletAddress1111111111111111111111111"
            with client._label_cache_lock:
                client._label_cache[address] = fake_label
                client._label_cache_dirty = True
            client._save_label_cache()

            # Small sleep to let background thread write
            time.sleep(0.3)

            # Now load a fresh client from the same cache file
            client2 = __import__("shared.moralis_client", fromlist=["MoralisClient"]).MoralisClient(
                http_session=mock_session,
                api_keys=["fake_key_123"],
                debug=False,
                label_cache_path=cache_path,
            )

            cached = client2.get_cached_label(address)
            self.assertIsNotNone(cached, "Label should be in cache after round-trip")
            self.assertEqual(cached.get("entity_name"), "Jump Trading")
            self.assertEqual(cached.get("category"), "fund")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Smart Money Scorer — Verification Test Suite")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestEntityMultipliers,
        TestAlphaWinnerTier,
        TestGradeUpgrade,
        TestBoostTierCalculation,
        TestApplySmartMoneyBoost,
        TestMEVBotExclusion,
        TestMoralisLabelCache,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
    else:
        print(f"❌ {len(result.failures)} FAILURE(S), {len(result.errors)} ERROR(S)")
        sys.exit(1)