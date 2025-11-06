# User Guide: Telegram Bot

The Telegram bot is your main interface for receiving real-time alerts and managing your paper trading portfolio.

## 🚀 Getting Started

1.  **Find the Bot:** Open Telegram and start a conversation with the bot.
2.  **Run `/start`:** Use the `/start` command to activate the bot.
3.  **Subscription:** The bot will tell you if your subscription is active. If not, you must contact an administrator to be added.
4.  **Choose Your Mode:** You will be prompted to select how you want to use the bot:
    * **🔔 Alerts Only:** You will only receive alert messages for new tokens.
    * **📈 Paper Trading Only:** The bot will automatically execute paper trades based on signals but will *not* send you alert messages.
    * **🚀 Both Modes:** You will receive alerts *and* the bot will auto-trade for you.

## 🔔 Alert Commands

These commands control the alerts you receive.

| Command | Description |
| :--- | :--- |
| **`/start`** | Brings up the initial menu to change your bot mode (Alerts, Paper Trading, or Both). |
| **`/myalerts`** | View your current settings, subscribed grades, and alert statistics. |
| **`/setalerts [GRADES]`** | Configure which alert grades you want to receive. Grades are `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`. <br> **Examples:** <br> • `/setalerts CRITICAL` (Only get the highest priority alerts) <br> • `/setalerts CRITICAL HIGH` <br> • `/setalerts CRITICAL HIGH MEDIUM LOW` (Get all alerts) |
| **`/alpha_subscribe`** | **(Recommended)** Opt-in to receive high-priority "Alpha" alerts. These are generated from monitoring the activity of known "winner" wallets. |
| **`/alpha_unsubscribe`** | Opt-out of receiving Alpha Alerts. |

## 📈 Paper Trading Commands

When paper trading mode is active, the bot automatically manages a virtual portfolio for you. You can use these commands to track its performance.

| Command | Description |
| :--- | :--- |
| **`/papertrade [amount]`** | Enable paper trading mode and set your starting capital. This will create a new virtual portfolio. <br> **Example:** `/papertrade 1000` (Starts a new portfolio with $1,000). |
| **`/portfolio`** | View a detailed summary of your paper trading portfolio, including open positions, available capital, invested capital, and total P/L. |
| **`/pnl`** | Get a quick, real-time update on the unrealized Profit/Loss of all your open positions. |
| **`/history [limit]`** | View your closed trade history. If no limit is provided, it shows the last 10 trades. <br> **Example:** `/history 20` (Shows the last 20 trades) |
| **`/performance`** | View detailed trading statistics, including total P/L, win rate, best/worst trade, average win/loss, and more. |
| **`/watchlist`** | View tokens that the bot has received a signal for but is still watching for a good entry price. |
| **`/resetcapital [amount]`** | **(Warning)** Resets your entire portfolio. This closes all open positions, clears your watchlist, and starts you fresh with a new capital amount. Your trade history is preserved. <br> **Example:** `/resetcapital 5000` |

## ⚙️ General Commands

| Command | Description |
| :--- | :--- |
| **`/help`** | Displays a full list of available commands. |
| **`/stop`** | Deactivates your account and stops all alerts and services. You can re-enable it with `/start`. |
| **`/stats`** | View your personal usage statistics, such as total alerts received. |