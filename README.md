# 🚀 Token Monitor & Telegram Bot

This project combines a **token monitoring engine** with a **Telegram bot interface**.  
It tracks top token holders, detects overlaps with winning wallets, and delivers real-time alerts to subscribed users.  

---

## ✨ Features

### 🔎 Token Monitor (`token_monitor.py`)
- Tracks token holder distributions over time.  
- Computes **distinct concentration** and **weighted concentration** metrics.  
- Grades tokens based on overlap with recent "winning" wallets.  
- Stores overlap results in `overlap_results.pkl`.  
- Syncs data periodically with **Supabase storage**.  

### 🤖 Telegram Bot (`bot.py`)
- Provides users with a **Telegram interface** to token monitoring data.  
- Supports **subscriptions** with expiry tracking.  
- Admins can:
  - Add new users with a set subscription duration.  
  - View user statistics.  
- Alerts are only sent to subscribed users.  
- Unsubscribed users see the welcome screen but cannot access features.  

---

## 📂 Data Storage

The project persists state in both **local files** and **Supabase** for syncing:

- `overlap_results.pkl` → Overlap metrics  
- `bot_user_prefs.pkl` → User preferences  
- `bot_user_stats.pkl` → Usage statistics  
- `bot_alerts_state.pkl` → Alert history  

---

## 📊 Overlap Analysis

For each token:
- **Overlap %** → share of top holders who also appear in the winner union.  
- **Distinct concentration** → overlap wallets ÷ total distinct winners.  
- **Weighted concentration** → overlap wallet frequency ÷ total wallet frequencies.  
- **Grade** → calculated risk/reward category.  

This allows the bot to highlight tokens with high similarity to historically successful wallets.  

---

## 👥 User Management

- **Admins** never expire and can always use the bot.  
- **Normal users** require a subscription:  
  - Admin sets duration in days when adding them.  
  - Expired users are notified **monthly**.  
  - Non-subscribed users cannot receive alerts.  

---

## 🛠 Tech Stack

- **Python**  
- **pandas** + **joblib** → data persistence  
- **Supabase** → shared storage  
- **python-telegram-bot** → Telegram integration  
- **asyncio** → background scheduling  

---

## 🚦 Usage

1. Run the monitor to start tracking token data:  
   ```bash
   python token_monitor.py
Run the Telegram bot for user interaction:

2. ```bash
    python bot.py
Ensure environment variables are set before running:

```bash
export TELEGRAM_TOKEN="your-telegram-token"
export SUPABASE_URL="your-supabase-url"
export SUPABASE_KEY="your-supabase-key"
export COINGECKO_PRO_API="your-api-key"
export HELIUS_API="your-api-key"

📌 Notes
- token_monitor.py should run continuously in the background to keep overlap data updated.
- bot.py depends on the overlap results and user data, so both services complement each other.
- The system is designed for 24/7 monitoring and alerting.
