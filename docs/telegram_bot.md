# 🤖 iSmartyBot: Comprehensive Documentation

Welcome to the official documentation for the iSmartyBot. This guide covers every menu, setting, and feature available in the bot, explaining how to use them to optimize your trading and notification experience.

---

## 📱 Main Menu
The **Main Menu** is your central hub. From here, you can access all major functional areas.

- **📊 Dashboard & Trading**: View your portfolio, open positions, and trading performance.
- **🔔 Notifications**: Configure which token alerts you receive.
- **⚙️ Settings**: Customize bot behavior, trading parameters, and modes.
- **🤖 ML Predictions**: Access the machine learning analyzer for specific tokens.
- **ℹ️ Help**: Quick reference for commands and topics.

> [!TIP]
> Your subscription status and active bot modes (Alerts/Trading) are always displayed at the top of the Main Menu for quick reference.

---

## 📊 Dashboard & Trading
This section is for monitoring your active paper trading portfolio.

- **💼 View Portfolio**: Lists all your currently open positions, showing entry price, current price, and ROI.
- **📈 View P&L**: A detailed breakdown of your realized and unrealized Profit & Loss.
- **📜 Trade History**: A chronological log of your completed trades.
- **📈 Performance Stats**: Visual and numeric summary of your win rate, average ROI, and trade count.
- **👀 Watchlist**: Tokens you are currently tracking but haven't entered yet.

> [!NOTE]
> If trading is disabled, this menu will prompt you to **Enable Paper Trading** and set your initial capital.

---

## 🔔 Notifications (Alerts)
Configure how the bot notifies you of new opportunities.

### 🎯 Discovery Grades
Filter alerts based on their "Grade" (CRITICAL, HIGH, MEDIUM, LOW).
- **CRITICAL**: The highest quality signals with major metrics aligned.
- **LOW**: Includes all detected tokens, even those with minor metrics.
*Click buttons to toggle each grade on/off.*

### 🌟 Alpha Notifications
Premium, high-priority alerts with deep security analysis and ML insights.
- **Subscription**: Requires an active subscription to receive these curated signals.
- **Insights**: Includes top 5 risks, market health scores, and insider holdings analysis.

### 🧠 Min Probability Filters
Set a minimum "predicted win rate" to reduce noise.
- **Discovery Alerts**: Filter out discovery signals below a certain ML win %.
- **Alpha Alerts**: Filter out alpha signals below a certain ML win %.
*Example: Setting Alpha to 80% means you'll only see signals the ML model is 80% confident in.*

---

## ⚙️ Settings
The brain of the bot. Here you define exactly how the bot behaves.

### 🔄 Bot Modes
Choose how you want to interact with the bot:
- **🔔 Alerts Only**: Receive notifications only (no automatic or manual paper trading).
- **📈 Trading Only**: Use the bot for paper trading without receiving alert messages.
- **🚀 Both Modes**: Recommended. Get alerts and have the bot automatically (or manually) trade them.

### 📈 Paper Trading Settings
Deep configuration for your trading strategy:

- **💰 Reset Capital**: Start fresh with a new balance (e.g., $10,000).
- **💵 Reserve Balance**: Keeps a portion of your capital "untouchable" by auto-trading.
- **📏 Min Trade Size**: Prevents the bot from entering trades too small (e.g., < $20).
- **📊 Trade Size**:
    - **Percentage-Based**: Use a % of your available portfolio per trade.
    - **Fixed Amount**: Use a specific $ amount for every trade.
- **🤖 Auto-Trade**: Toggle whether the bot should automatically enter trades based on alerts.
- **🚜 Auto-Trade Filters**: 
    - **Grades/Alpha Selection**: Choose exactly which signal sources trigger the auto-trader.
    - **Min Probability (Auto)**: Separate from notification filters. Set the minimum win probability for *automatic* execution.
- **🎯 Take Profit (TP)**: Set your target exit strategy (Detailed guide below).
- **🛑 Stop Loss (SL)**: Set a global safety net to exit trades if they drop (e.g., -20%).
- **🔀 Confluence Settings**:
    - **Confluence**: When Discovery and Alpha signals fire on the same token, the bot "pyramids" (adds to) the position instead of closing it.
    - **Add-On Size**: How much to add to the existing trade (% of original).
    - **Max Exposure**: Maximum total capital allowed in a single token.

---

## 🎯 Take Profit (TP) Guide
The bot uses a sophisticated "Smart ATH" system to calculate exit points.

- **Global TP**: Your default target for all trades.
- **Overrides**: Set specific TP targets for Discovery vs. Alpha signals.

**TP Options:**
- **median**: Uses the middle value of historical All-Time Highs (Balanced).
- **mean**: Uses the average ATH (Aggressive/Higher target).
- **mode**: Uses the most frequent ATH reached.
- **smart**: Statistically calculates targets reached 75% of the time (Consistent).
- **Custom %**: Set a fixed target (e.g., 50%).

---

## 🤖 ML Predictions
Analyze any token manually by sending its mint address.

- **Win Probability**: The model's confidence in the token's success.
- **Risk Tier**: Assessment of security and market risks.
- **Market Health**: A score based on liquidity, volume, and holder distribution.
- **Insights**: Direct analysis of dev holdings, liquidity locks, and "pump & dump" risk.

---

## 📜 Full Command Reference

While the menu system covers almost everything, these commands provide direct access for power users:

| Command | Description |
| :--- | :--- |
| `/start` | Open the Main Menu & configure modes |
| `/help` | Detailed help menu for all features |
| `/myalerts` | View your current active filters & stats |
| `/setalerts` | Quick set alert grades (e.g. `/setalerts CRITICAL HIGH`) |
| `/papertrade` | Enable trading & set capital (e.g. `/papertrade 5000`) |
| `/portfolio` | Display current open positions & ROI |
| `/pnl` | View realized and unrealized profit/loss |
| `/performance` | Detailed trading statistics & win rates |
| `/watchlist` | List tokens currently being monitored |
| `/resetcapital`| Reset your trading account balance |
| `/predict` | Analyze a specific token mint address |
| `/set_tp` | Set global TP preference |
| `/set_min_prob` | Set minimum probability filters |

---

## ℹ️ Support
If you encounter any issues or have questions, use the **Help** menu or contact the admin directly: [**@smartymetrics**](https://t.me/smartymetrics)
