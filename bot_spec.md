# Telegram Alert Format — Redesign Brief

## Overview

The current alert format has several issues that make it look unprofessional and confusing to degen traders. This document specifies the exact changes needed to bring it up to the standard of top-tier bots like GMGN and AlphaScan.

---

## Current Alert (Example)

```
🔥 BEARISH ❌: $MOGGED 🔥
MOGGED
CwdVZL7bVt37eyvHvrUZtbo5d3wVBnaLqT6WXZP4pump
Status: Graduated ✅
Overlap Grade: LOW
--- 📈 Market Data ---
💰 Price: $0.00005818
📊 Market Cap: $58.18K
💧 Liquidity: $22.14K
📉 LP/MC Ratio: 38.06%
⏰ Age: 1d 3h ago
--- 📊 Activity ---
Buy/Sell: 54% / 46%
👥 Holders: 1,548
--- 🛡️ Safety ---
Score: EXCELLENT ✅ (1/100)
🔒 LP Locked: 100.00%
Mint: ✅ Renounced
Freeze: ✅ Renounced
--- ⚠️ Top Risks ---
✅ No significant risks
--- 🤖 ML Insight ---
🟠 Win Chance: 50.1% ██░░░
🟠 Signal: SKIP
📉 Confidence: LOW
🚨 Risk: High Risk
--- 🧠 SENTIMENT: BEARISH ❌ ---
Alpha Score: 22/100 🟩🟩⬜⬜⬜⬜⬜⬜⬜⬜
--- 🔗 Links ---
Solscan | GMGN | DexScreener
```

---

## Problems to Fix

### 1. Contradictory headline
`🔥 BEARISH ❌: $MOGGED 🔥` uses the fire emoji (which means SUPER-ALPHA) alongside BEARISH. These come from two different systems and directly contradict each other. A trader reading this immediately loses trust in the bot.

**Fix:** The headline should display only the `trader_sentiment` value. The SUPER-ALPHA badge should only appear if `is_super_alpha = True`.

### 2. Low-score alerts should not be sent
An Alpha Score of 22/100 with BEARISH sentiment should never reach the user. Sending weak alerts trains users to ignore the bot entirely.

**Fix:** Only send alerts when `alpha_score >= 40`. Below that threshold, either suppress the alert entirely or send it to a separate "watchlist" channel.

### 3. Smart Money section is completely missing
The bot now computes a full Smart Money analysis including:
- Number of winner wallets holding the token
- PnL tier breakdown (Elite / Strong / Active)
- Combined cluster profit
- Average win rate across the cluster
- Sniper detection (wallets that bought within 5 minutes of launch)
- Early buyer detection (wallets that bought within 30 minutes of launch)

None of this appears in the current alert. This is the most valuable signal the bot produces and it is completely invisible to users.

**Fix:** Add a dedicated Smart Money section (see Target Format below).

### 4. ML section is cluttered — 4 lines for 1 idea
Win Chance, Signal, Confidence, and Risk are four lines all saying the same thing. A trader only needs one line.

**Fix:** Collapse to a single line: `🤖 ML: 64% WIN CHANCE  |  Action: WATCH`

### 5. LP/MC Ratio adds no value in an alert
No trader acts on LP/MC ratio from a Telegram alert. It is noise.

**Fix:** Remove it entirely.

### 6. Sentiment is both the headline AND a buried section
The `--- 🧠 SENTIMENT: BEARISH ❌ ---` section at the bottom duplicates the headline. Pick one location.

**Fix:** Sentiment and Alpha Score belong at the top as the headline signal. Remove the duplicate section at the bottom.

### 7. Token address shown in full — wastes space
The full mint address `CwdVZL7bVt37eyvHvrUZtbo5d3wVBnaLqT6WXZP4pump` takes up a full line and cannot be tapped usefully in Telegram. The links at the bottom already cover this.

**Fix:** Shorten to first 8 + last 4 characters: `CwdVZL7b...pump`

---

## Target Alert Format

Below is the exact format to implement. Every field maps to data already available in the `conviction_summary` dict produced by the bot.

```
━━━━━━━━━━━━━━━━━━━━━━━━━
⭐ STRONG BULLISH  |  Grade: MEDIUM
$MOGGED  •  Graduated ✅  •  1d 3h old
━━━━━━━━━━━━━━━━━━━━━━━━━

🧠 Alpha Score: 64/100  ████████░░
Sentiment: STRONG BULLISH ⭐  (HIGH confidence)

🏆 SMART MONEY
Winners holding: 18  |  Conviction: 75%
🏅 2 Elite  💪 3 Strong  ✅ 1 Active
Combined PnL: +$142,000  |  Avg Win Rate: 67%
🎯 2 Snipers bought in first 5 mins

🤖 ML:  64% WIN CHANCE  |  Action: WATCH

💰 MARKET
Price: $0.000058  |  MCap: $58K  |  Liq: $22K
Vol 1h: $12,400  |  Buy pressure: 54%  |  Age: 1d 3h

🔒 SAFETY
LP: 100% locked  |  Holders: 1,548  |  Risk score: 1/100
Mint ✅  Freeze ✅  No risks

🔗  Solscan  |  GMGN  |  DexScreener
━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Field Mapping

Every field in the target format maps directly to a key in the `conviction_summary` dict. No new data needs to be fetched.

| Alert Field | conviction_summary key |
|---|---|
| Sentiment label in headline | `trader_sentiment` |
| Grade | `grade` |
| Token symbol | `token_symbol` |
| Token status (Graduated etc.) | From existing token metadata |
| Token age | `pair_age_mins` (convert to human-readable) |
| Alpha Score | `alpha_score` |
| Alpha Score bar | Compute from `alpha_score` (filled blocks = score/10) |
| Sentiment confidence | `sentiment_confidence` |
| Winners holding | `overlap_count` |
| Wallet conviction % | `wallet_conviction_pct` |
| Elite / Strong / Active counts | `pnl_tier_breakdown.ELITE`, `.STRONG`, `.ACTIVE` |
| Combined PnL | `cluster_combined_profit_usd` |
| Avg Win Rate | `cluster_avg_win_rate_pct` |
| Sniper count | `sniper_count` |
| Early buyer count | `early_buyer_count` |
| ML Win Chance | `ml_probability_pct` |
| ML Action | `ml.action` |
| Price | `price_usd` |
| Market Cap | `market_cap_usd` |
| Liquidity | `liquidity_usd` |
| Volume 1h | `volume_1h_usd` |
| Buy pressure | `buy_pressure_1h_pct` |
| LP Locked | `lp_locked_pct` |
| Holders | `holder_count` |
| Risk score | `security.lp_locked_pct` / rugcheck score |
| DexScreener URL | `dex_url` |

---

## Conditional Display Rules

These sections should only appear when data is available and meaningful:

| Section | Show condition |
|---|---|
| Entire alert | `alpha_score >= 40` |
| SUPER-ALPHA badge 🔥 | `is_super_alpha == True` |
| Smart Money section | `overlap_count > 0` |
| PnL tier line | At least one wallet has `pnl_tier` in ELITE/STRONG/ACTIVE |
| Combined PnL line | `cluster_combined_profit_usd is not None` |
| Avg Win Rate line | `cluster_avg_win_rate_pct is not None` |
| Snipers line | `sniper_count > 0` |
| Early buyers line | `early_buyer_count > 0` (only if no snipers, else merge) |
| No risks line | `risks list is empty` |
| Top Risks | Only if 1+ risks exist — show max 2 |

---

## Alpha Score Progress Bar

Replace the current emoji block art with a clean 10-block bar:

```python
def alpha_bar(score: int) -> str:
    filled = score // 10
    empty  = 10 - filled
    return "█" * filled + "░" * empty
```

Examples:
- Score 22 → `██░░░░░░░░`
- Score 64 → `██████░░░░`  
- Score 85 → `████████░░`

---

## Sentiment → Emoji Mapping

| trader_sentiment value | Display |
|---|---|
| `EXTREME BULLISH 🔥` | `🔥 EXTREME BULLISH` |
| `STRONG BULLISH ⭐` | `⭐ STRONG BULLISH` |
| `BULLISH ✅` | `✅ BULLISH` |
| `CAUTIOUS ⚠️` | `⚠️ CAUTIOUS` |
| `BEARISH ❌` | `❌ BEARISH` — **do not send unless alpha_score >= 40** |

---

## What NOT to include

Remove these fields entirely from the alert:

- LP/MC Ratio
- Full mint address (replace with shortened version in header only)
- Separate `SENTIMENT` section at bottom (it's now the headline)
- `Confidence: LOW` and `Risk: High Risk` as separate ML lines (fold into one line)
- `Overlap Grade: LOW` as a standalone line (it's shown in the headline)

---

## Summary of Changes

1. **Headline** — Use `trader_sentiment` only. No conflicting emojis.
2. **Minimum threshold** — Only send alerts with `alpha_score >= 40`.
3. **Add Smart Money section** — Winner wallet count, PnL tiers, combined profit, avg win rate, sniper count.
4. **Collapse ML section** — One line only.
5. **Remove LP/MC ratio** — No value in an alert.
6. **Remove duplicate sentiment section** — It belongs in the headline, not at the bottom.
7. **Shorten token address** — `first8...last4` format only.
8. **Conditional Smart Money lines** — Only show PnL/sniper data when it exists.