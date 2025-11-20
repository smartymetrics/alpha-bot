# 🚀 Solana Memecoin Intelligence System

An advanced token monitoring and prediction system that combines machine learning classification with real-time tracking and Telegram alerting. The system analyzes Solana memecoins to predict winners and detect high-risk tokens before launch.

---

## 🎯 Problem Statement

The Solana memecoin market is extremely volatile and risky:
- **High failure rate**: Most tokens lose 50%+ value within 24 hours
- **Insider manipulation**: Coordinated networks control token supply
- **Information asymmetry**: Retail traders lack tools to assess token quality
- **Time-sensitive**: Token age significantly impacts success probability
- **Complex signals**: Multiple factors (liquidity, holders, momentum) must be evaluated simultaneously

**Traditional approaches fail because:**
- Manual analysis is too slow for the fast-moving memecoin market
- Single metrics (e.g., liquidity alone) are insufficient predictors
- Pattern recognition across dozens of features requires ML

---

## 🧠 Machine Learning Solution

### The Classification Problem
**Goal**: Predict whether a Solana memecoin will be a "winner" (gains) or "loser" (losses) based on early signals.

**Input**: 75+ features including:
- Market metrics (liquidity, volume, FDV)
- Holder concentration (top 10%, creator balance)
- Insider networks (size, token amounts, coordination)
- Time factors (token age, trading hours, weekends)
- Momentum indicators (24h price change, volume quality)
- Risk scores (pump/dump risk, concentration risk)

**Output**: Binary classification with probability scores
- `1` = Win (token expected to gain value)
- `0` = Loss (token expected to lose value)

### Model Architecture

**Ensemble Approach** combining three gradient boosting models:
1. **CatBoost** (70% weight) - Best for categorical features and class imbalance
2. **LightGBM** (30% weight) - Fast, efficient with large feature sets
3. **XGBoost** (optional, 0-10% weight) - Additional diversity

**Key Technical Decisions:**
```python
# Class imbalance handling
scale_pos_weight = (losses / wins)  # Typically 2-3x
auto_class_weights = 'Balanced'

# Regularization to prevent overfitting
- L2 regularization (reg_lambda=5-15)
- Max depth limited to 4
- Min samples per leaf: 10-15
- Feature subsampling: 80%

# Training strategy
- Stratified K-fold cross-validation (5 folds)
- Feature selection via F-statistic (SelectKBest)
- Optimal K determined by CV performance
```

### Feature Engineering Highlights

**🆕 Novel Derived Features:**
```python
# Insider concentration (KEY METRIC)
insider_supply_pct = (insider_tokens / total_supply) * 100

# Token freshness with exponential decay
freshness_score = exp(-token_age_hours / 6)  # 6hr half-life

# Market health composite
market_health = (
    liquidity_depth * 0.30 +
    lp_lock_quality * 0.30 +
    holder_quality * 0.20 +
    volume_efficiency * 0.20
)

# Concentration risk composite
concentration_risk = (
    top_10_holders_pct * 0.5 +
    creator_balance_pct * 0.3 +
    insider_supply_pct * 0.2
)
```

**Time-based Features:**
- Ultra fresh (<1h), very fresh (<4h), fresh (<12h) flags
- Trading session indicators (Asian/European/US hours)
- Peak trading time detection (14:00-20:00 UTC)

**Risk Indicators:**
- Creator dumped flag (creator balance <1% within 24h)
- Whale dominated (top 10 holders >70%)
- Extreme concentration (>60%)

### Performance Metrics

**Current Results:**
- **Test AUC**: 0.75-0.80 (strong discrimination)
- **Cross-validation**: 5-fold stratified, consistent performance
- **Optimal threshold**: 0.70 for high-precision BUY signals

**Confusion Matrix Interpretation:**
```
Predicted:        Loss    Win
Actual Loss:      TN      FP   ← False positives (acceptable risk)
Actual Win:       FN      TP   ← False negatives (missed gains)
```

**Threshold Strategy:**
- **≥0.70**: BUY signal (high precision, fewer false positives)
- **0.40-0.70**: MONITOR (uncertain)
- **<0.40**: AVOID (high recall for losses)

### Top Predictive Features

Based on CatBoost feature importance:

1. **🔴 insider_supply_pct** - What % of supply do insider networks control?
2. **⭐ price_change_h24_pct** - Recent momentum
3. **⭐ top_10_holders_pct** - Concentration risk
4. **🆕 liquidity_depth_pct** - Liquidity relative to FDV
5. **🆕 market_health_score** - Composite quality indicator
6. **⭐ total_insider_networks** - Coordination level
7. **🆕 freshness_score** - Token age decay
8. **⭐ is_lp_locked_95_plus** - LP security
9. **🆕 concentration_risk** - Combined holder/insider risk
10. **⭐ pump_dump_risk_score** - Rugcheck API score

---

## ✨ System Features

### 🔎 Token Monitor (`token_monitor.py`)
* **Real-time tracking**: Monitors token holder distributions continuously
* **Concentration metrics**: Calculates distinct and weighted concentration
* **Winner overlap analysis**: Compares current holders with historically successful wallets
* **ML integration**: Fetches features and runs ensemble predictions
* **Persistent storage**: Saves overlap results to `overlap_results.pkl`
* **Cloud sync**: Periodically syncs data with Supabase

### 🤖 Telegram Bot (`bot.py`)
* **User interface**: Telegram-based access to monitoring data and predictions
* **Subscription system**: Time-based access control with expiry tracking
* **Admin controls**:
   * Add new users with custom subscription duration
   * View user statistics and activity
   * Manage alert preferences
* **Smart alerts**: Only sends notifications to active subscribers
* **Grade-based filtering**: Shows tokens by ML-predicted grade

### 🧮 ML Training Pipeline (`train_model.py`)
* **Feature engineering**: Creates 75+ features from raw token data
* **Automated feature selection**: Finds optimal K features via cross-validation
* **Ensemble training**: Trains and validates XGBoost, LightGBM, CatBoost
* **Hyperparameter optimization**: Tests multiple configurations
* **Performance analysis**: Detailed metrics, confusion matrix, feature importance
* **Model persistence**: Saves models and metadata for production inference

---

## 📂 Data Storage

### Local Persistence
* `overlap_results.pkl` → Overlap metrics and token grades
* `bot_user_prefs.pkl` → User preferences and settings
* `bot_user_stats.pkl` → Usage statistics
* `bot_alerts_state.pkl` → Alert history and delivery status

### ML Artifacts
* `models/xgboost_model.pkl` → Trained XGBoost classifier
* `models/lightgbm_model.pkl` → Trained LightGBM classifier
* `models/catboost_model.pkl` → Trained CatBoost classifier
* `models/feature_selector.pkl` → SelectKBest feature selector
* `models/model_metadata.json` → Training metadata, feature lists, performance metrics

### Cloud Storage (Supabase)
All critical data is synced to Supabase for:
- Multi-instance deployment
- Backup and recovery
- Cross-service data sharing

---

## 📊 Overlap Analysis

For each monitored token:

* **Overlap %** → Share of top holders who appear in the winner union
* **Distinct concentration** → Overlap wallets ÷ total distinct winners
* **Weighted concentration** → Overlap wallet frequency ÷ total frequencies
* **ML Grade** → Ensemble model prediction (A/B/C/D/F)
  - **Grade A** (≥0.80): Strong buy signal
  - **Grade B** (0.70-0.79): Buy signal
  - **Grade C** (0.50-0.69): Monitor
  - **Grade D** (0.40-0.49): Caution
  - **Grade F** (<0.40): Avoid

This multi-factor approach highlights tokens with:
1. High similarity to historically successful wallets
2. Favorable ML prediction scores
3. Low insider/concentration risk

---

## 👥 User Management

* **Admins**: Never expire, full access to all features
* **Normal users**: Require active subscription
   * Admin sets duration when adding user
   * Expired users receive monthly reminders
   * Non-subscribed users cannot receive alerts or access predictions
* **Statistics tracking**: Usage, alert engagement, feature access

---

## 🛠 Tech Stack

**Core Libraries:**
- `pandas` + `numpy` → Data manipulation
- `scikit-learn` → ML preprocessing, feature selection, metrics
- `xgboost`, `lightgbm`, `catboost` → Gradient boosting models
- `joblib` → Model serialization

**Infrastructure:**
- `python-telegram-bot` → Bot interface
- `supabase-py` → Cloud storage
- `asyncio` → Background scheduling and concurrent tasks

**APIs:**
- DexScreener → Token market data
- Rugcheck → Risk scoring
- CoinGecko Pro → Price feeds
- Helius → Solana blockchain data

---

## 🚦 Usage

### 1. Train the ML Models

```bash
# Prepare your dataset in data/token_datasets.csv
# Required columns: label_status, liquidity_usd, volume_h24_usd, etc.

python train_model.py
```

**Output:**
- Trained models in `models/`
- Feature importance analysis
- Cross-validation scores
- Optimal threshold recommendations

### 2. Run the Token Monitor

```bash
python token_monitor.py
```

**Monitor will:**
- Fetch token data from APIs
- Calculate overlap with winner wallets
- Run ML predictions on new tokens
- Grade tokens A-F based on ensemble scores
- Update `overlap_results.pkl` continuously

### 3. Start the Telegram Bot

```bash
python bot.py
```

**Bot features:**
- `/start` - Welcome and subscription status
- `/stats` - View user statistics (admin only)
- `/add_user <telegram_id> <days>` - Add new subscriber (admin only)
- View tokens by grade with inline buttons

### 4. Environment Setup

Create a `.env` file:

```bash
export TELEGRAM_TOKEN="your-telegram-bot-token"
export SUPABASE_URL="your-supabase-url"
export SUPABASE_KEY="your-supabase-service-key"
export COINGECKO_PRO_API="your-coingecko-api-key"
export HELIUS_API="your-helius-api-key"
```

---

## 📌 Production Recommendations

### Model Retraining
- **Frequency**: Weekly or bi-weekly
- **Trigger**: When new data accumulates (500+ new samples)
- **Validation**: Always check CV scores before deployment
- **A/B testing**: Run new model in parallel before switching

### Alert Thresholds
```python
# Conservative (fewer false positives)
BUY_THRESHOLD = 0.75
AVOID_THRESHOLD = 0.35

# Balanced (recommended)
BUY_THRESHOLD = 0.70
AVOID_THRESHOLD = 0.40

# Aggressive (more signals)
BUY_THRESHOLD = 0.65
AVOID_THRESHOLD = 0.45
```

### Monitoring
- Track model prediction distribution over time
- Monitor alert click-through rates
- Log false positives/negatives for retraining
- Watch for concept drift (market regime changes)

### Risk Management
⚠️ **Important**: This system provides signals, NOT financial advice.
- Always verify predictions with manual research
- Never invest more than you can afford to lose
- Consider multiple signals beyond ML scores
- Memecoins are inherently high-risk

---

## 🏗️ Architecture Overview

```
┌─────────────────┐
│  DexScreener    │
│  Rugcheck APIs  │
└────────┬────────┘
         │
         v
┌─────────────────┐      ┌──────────────┐
│ token_monitor   │─────>│  Supabase    │
│  - Fetch data   │<─────│  (sync data) │
│  - ML inference │      └──────────────┘
│  - Grade tokens │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ overlap_results │
│      .pkl       │
└────────┬────────┘
         │
         v
┌─────────────────┐      ┌──────────────┐
│   bot.py        │─────>│   Telegram   │
│  - User mgmt    │<─────│    Users     │
│  - Alerts       │      └──────────────┘
│  - Display data │
└─────────────────┘
```

---

## 🤝 Contributing

When improving the ML model:
1. Document new features in the feature engineering section
2. Update metadata.json with new feature names
3. Run cross-validation to verify improvements
4. Test on held-out data before production deployment

---

## ⚠️ Disclaimer

This system is for educational and research purposes. Cryptocurrency trading involves substantial risk of loss. The ML predictions are probabilistic and not guarantees. Always conduct your own research and consult financial advisors before making investment decisions.
