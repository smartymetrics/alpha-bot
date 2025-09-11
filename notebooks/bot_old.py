#!/usr/bin/env python3
"""
Enhanced Telegram alert bot for token-grade notifications with improved user management.

Environment variables required:
- BOT_TOKEN           (Telegram bot token)
- SUPABASE_URL        (e.g. https://abcdxyz.supabase.co)
- SUPABASE_KEY        (service role key or anon/public key with storage permissions)
- SUPABASE_BUCKET     (storage bucket name; default: monitor-data)
- SUPABASE_FILENAME   (filename in bucket; default: monitor_results.pkl)

Optional:
- STALE_LIMIT_HOURS   (how old is "stale"; default 2)
- POLL_INTERVAL_SECS  (how often bot checks; default 1800 = 30m)
- ADMIN_USER_ID       (Telegram user ID for admin commands)

Run as a worker on Render (Procfile: `worker: python bot.py`)
"""

import os
import io
import time
import asyncio
import datetime
import logging
import joblib
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    Defaults,
)
from dotenv import load_dotenv

load_dotenv()

# ---------- Config & paths ----------
BOT_TOKEN = os.getenv("BOT_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = os.getenv("SUPABASE_BUCKET", "monitor-data")
FILE_NAME = os.getenv("SUPABASE_FILENAME", "monitor_results.pkl")
ADMIN_USER_ID = os.getenv("ADMIN_USER_ID")  # admin user ID

STALE_LIMIT_HOURS = float(os.getenv("STALE_LIMIT_HOURS", "2"))
POLL_INTERVAL_SECS = int(os.getenv("POLL_INTERVAL_SECS", "1800"))  # 30 minutes by default

DATA_DIR = Path(os.getenv("DATA_DIR", "."))  # where .pkl files will be stored in container
MONITOR_FILE = DATA_DIR / "monitor_results.pkl"  # downloaded file
ALERTS_STATE_FILE = DATA_DIR / "alerts_state.pkl"
USER_PREFS_FILE = DATA_DIR / "user_prefs.pkl"
USER_STATS_FILE = DATA_DIR / "user_stats.pkl"

# sanity checks
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable is required")

if not SUPABASE_URL or not SUPABASE_KEY:
    logging.warning("SUPABASE_URL or SUPABASE_KEY not provided. Bot will try to run but cannot download monitor data.")

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------- Thread-safe file access ----------
from threading import Lock
file_lock = Lock()

def safe_load(path: Path, default):
    with file_lock:
        if not path.exists():
            return default
        try:
            return joblib.load(path)
        except Exception as e:
            logging.exception("Failed loading %s: %s", path, e)
            return default

def safe_save(path: Path, data):
    with file_lock:
        try:
            joblib.dump(data, path)
        except Exception as e:
            logging.exception("Failed saving %s: %s", path, e)

# ---------- Enhanced user management ----------
class UserManager:
    def __init__(self, prefs_file: Path, stats_file: Path):
        self.prefs_file = prefs_file
        self.stats_file = stats_file
    
    def get_user_prefs(self, chat_id: str) -> Dict[str, Any]:
        """Get user preferences with defaults"""
        prefs = safe_load(self.prefs_file, {})
        return prefs.get(chat_id, {
            "grades": ["CRITICAL", "HIGH"],
            "created_at": self.now_iso(),
            "updated_at": self.now_iso(),
            "active": True,
            "total_alerts_received": 0,
            "last_alert_at": None
        })
    
    def update_user_prefs(self, chat_id: str, updates: Dict[str, Any]) -> bool:
        """Update user preferences"""
        try:
            prefs = safe_load(self.prefs_file, {})
            if chat_id not in prefs:
                prefs[chat_id] = {
                    "grades": ["CRITICAL", "HIGH"],
                    "created_at": self.now_iso(),
                    "active": True,
                    "total_alerts_received": 0
                }
            
            prefs[chat_id].update(updates)
            prefs[chat_id]["updated_at"] = self.now_iso()
            safe_save(self.prefs_file, prefs)
            # Upload to Supabase after local save
            upload_bot_data_to_supabase()
            return True
        except Exception as e:
            logging.exception(f"Failed to update user prefs for {chat_id}: {e}")
            return False
    
    def deactivate_user(self, chat_id: str) -> bool:
        """Deactivate user without deleting their data"""
        return self.update_user_prefs(chat_id, {"active": False, "deactivated_at": self.now_iso()})
    
    def activate_user(self, chat_id: str) -> bool:
        """Reactivate user"""
        return self.update_user_prefs(chat_id, {"active": True, "reactivated_at": self.now_iso()})
    
    def get_active_users(self) -> Dict[str, Dict[str, Any]]:
        """Get all active users"""
        prefs = safe_load(self.prefs_file, {})
        return {k: v for k, v in prefs.items() if v.get("active", True)}
    
    def get_user_stats(self, chat_id: str) -> Dict[str, Any]:
        """Get user statistics"""
        stats = safe_load(self.stats_file, {})
        return stats.get(chat_id, {
            "alerts_received": 0,
            "last_alert_at": None,
            "joined_at": None,
            "grade_breakdown": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        })
    
    def update_user_stats(self, chat_id: str, grade: str = None):
        """Update user statistics when they receive an alert"""
        try:
            stats = safe_load(self.stats_file, {})
            if chat_id not in stats:
                stats[chat_id] = {
                    "alerts_received": 0,
                    "last_alert_at": None,
                    "joined_at": self.now_iso(),
                    "grade_breakdown": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
                }
            
            stats[chat_id]["alerts_received"] += 1
            stats[chat_id]["last_alert_at"] = self.now_iso()
            if grade in stats[chat_id]["grade_breakdown"]:
                stats[chat_id]["grade_breakdown"][grade] += 1
            
            safe_save(self.stats_file, stats)
            # Upload to Supabase after local save
            upload_bot_data_to_supabase()
        except Exception as e:
            logging.exception(f"Failed to update stats for {chat_id}: {e}")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get platform-wide statistics"""
        prefs = safe_load(self.prefs_file, {})
        stats = safe_load(self.stats_file, {})
        
        total_users = len(prefs)
        active_users = len([u for u in prefs.values() if u.get("active", True)])
        total_alerts = sum(s.get("alerts_received", 0) for s in stats.values())
        
        grade_totals = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for user_stats in stats.values():
            for grade, count in user_stats.get("grade_breakdown", {}).items():
                if grade in grade_totals:
                    grade_totals[grade] += count
        
        return {
            "total_users": total_users,
            "active_users": active_users,
            "total_alerts_sent": total_alerts,
            "grade_breakdown": grade_totals,
            "generated_at": self.now_iso()
        }
    
    @staticmethod
    def now_iso():
        return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    
    def save_to_supabase(self):
        """Upload user data to Supabase after changes"""
        upload_bot_data_to_supabase()

# Initialize user manager
user_manager = UserManager(USER_PREFS_FILE, USER_STATS_FILE)

# ---------- Supabase helpers ----------
def download_results_from_supabase(local_path: Path = MONITOR_FILE) -> bool:
    """
    Downloads the monitor_results.pkl from Supabase storage public endpoint.
    Returns True on success, False otherwise.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        logging.error("Supabase config is missing.")
        return False

    url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{FILE_NAME}"
    headers = {"apikey": SUPABASE_KEY}
    try:
        logging.info("Downloading monitor results from Supabase: %s", url)
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200:
            with file_lock:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, "wb") as f:
                    f.write(r.content)
            logging.info("Downloaded monitor_results to %s", local_path)
            return True
        else:
            logging.warning("Failed to download monitor_results: %s %s", r.status_code, r.text)
            return False
    except Exception as e:
        logging.exception("Error downloading from Supabase: %s", e)
        return False

def upload_file_to_supabase(local_path: Path, remote_filename: str) -> bool:
    """
    Upload a file to Supabase storage bucket.
    Returns True on success, False otherwise.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        logging.error("Supabase config is missing for upload.")
        return False

    url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET_NAME}/{remote_filename}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/octet-stream"
    }
    
    try:
        with open(local_path, "rb") as f:
            response = requests.post(url, headers=headers, data=f, timeout=30)
        
        if response.status_code in [200, 201]:
            logging.info(f"Uploaded {local_path} to {remote_filename}")
            return True
        else:
            logging.warning(f"Upload failed: {response.status_code} {response.text}")
            return False
    except Exception as e:
        logging.exception(f"Error uploading {local_path}: {e}")
        return False

def upload_bot_data_to_supabase():
    """Upload all bot data files to Supabase"""
    files_to_upload = [
        ("bot_user_prefs.pkl", USER_PREFS_FILE),
        ("bot_user_stats.pkl", USER_STATS_FILE), 
        ("bot_alerts_state.pkl", ALERTS_STATE_FILE)
    ]
    
    for remote_name, local_path in files_to_upload:
        if local_path.exists():
            upload_file_to_supabase(local_path, remote_name)

def download_bot_data_from_supabase():
    """Download bot data files from Supabase on startup"""
    files_to_download = [
        ("bot_user_prefs.pkl", USER_PREFS_FILE),
        ("bot_user_stats.pkl", USER_STATS_FILE), 
        ("bot_alerts_state.pkl", ALERTS_STATE_FILE)
    ]
    
    for remote_name, local_path in files_to_download:
        url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{remote_name}"
        headers = {"apikey": SUPABASE_KEY}
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(r.content)
                logging.info(f"Downloaded {remote_name} to {local_path}")
        except Exception as e:
            logging.debug(f"Could not download {remote_name}: {e}")

# ---------- Utils ----------
def now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def parse_iso(ts: str) -> datetime.datetime:
    # Accept ISO with or without trailing 'Z'
    if ts.endswith("Z"):
        ts = ts[:-1]
    return datetime.datetime.fromisoformat(ts)

def is_monitor_stale(local_path: Path = MONITOR_FILE, stale_hours: float = STALE_LIMIT_HOURS) -> bool:
    if not local_path.exists():
        logging.warning("Monitor file does not exist locally: %s", local_path)
        return True
    try:
        data = joblib.load(local_path)
        last_updated = data.get("last_updated")
        if not last_updated:
            logging.warning("monitor_results missing last_updated field")
            return True
        last_dt = parse_iso(last_updated)
        now = datetime.datetime.utcnow()
        diff_hours = (now - last_dt).total_seconds() / 3600.0
        logging.debug("Monitor last_updated: %s (%.2f hours ago)", last_updated, diff_hours)
        return diff_hours > stale_hours
    except Exception as e:
        logging.exception("Error inspecting monitor file for staleness: %s", e)
        return True

def format_alert_html(token_data: Dict[str, Any], alert_type: str, previous_grade: str = None) -> str:
    """
    Build an HTML-formatted alert string for a token.
    """
    name = token_data.get("token_metadata", {}).get("name") or token_data.get("name") or "Unknown"
    symbol = token_data.get("token_metadata", {}).get("symbol") or token_data.get("symbol") or ""
    grade = token_data.get("grade", "UNKNOWN")
    liquidity = token_data.get("token_metadata", {}).get("liquidity")
    overlap = token_data.get("overlap_percentage")
    concentration = token_data.get("concentration_percentage")
    mint = token_data.get("mint")

    liquidity_str = f"${liquidity:,}" if liquidity is not None else "N/A"
    overlap_str = f"{overlap:.2f}%" if isinstance(overlap, (int, float)) else "N/A"
    conc_str = f"{concentration:.2f}%" if isinstance(concentration, (int, float)) else "N/A"

    solscan = f"https://solscan.io/token/{mint}"
    gmgn = f"https://gmgn.ai/sol/token/{mint}"

    header = ""
    if alert_type == "NEW":
        header = "🚀 <b>New Token Detected</b>"
    elif alert_type == "CHANGE":
        if previous_grade:
            direction = "⬆️ Upgraded" if grade and previous_grade and grade != previous_grade else "⬆️/⬇️ Grade Changed"
            header = f"{direction}"
        else:
            header = "🔁 <b>Grade Changed</b>"
    else:
        header = "🔔 <b>Alert</b>"

    lines = [
        header,
        f"<b>{name}</b> ({symbol})",
        f"<b>Grade:</b> {grade}" + (f" (was {previous_grade})" if previous_grade else ""),
        f"<b>Liquidity:</b> {liquidity_str}",
        f"<b>Overlap:</b> {overlap_str}",
        f"<b>Concentration:</b> {conc_str}",
        "",
        f'<a href="{solscan}">Solscan</a> | <a href="{gmgn}">GMGN</a>'
    ]
    return "\n".join(lines)

def is_admin(update: Update) -> bool:
    """Check if user is admin"""
    if not ADMIN_USER_ID:
        return False
    return str(update.effective_user.id) == ADMIN_USER_ID

# ---------- Enhanced Command handlers ----------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    user_prefs = user_manager.get_user_prefs(chat_id)
    
    # Initialize or reactivate user
    if not user_prefs.get("created_at"):
        user_manager.update_user_prefs(chat_id, {
            "grades": ["CRITICAL", "HIGH"],
            "active": True,
            "created_at": now_iso()
        })
        user_prefs = user_manager.get_user_prefs(chat_id)
    else:
        user_manager.activate_user(chat_id)
    
    # Create inline keyboard for quick setup
    keyboard = [
        [
            InlineKeyboardButton("🔴 CRITICAL Only", callback_data="preset_critical"),
            InlineKeyboardButton("🔥 CRITICAL + HIGH", callback_data="preset_critical_high")
        ],
        [
            InlineKeyboardButton("📊 All Grades", callback_data="preset_all"),
            InlineKeyboardButton("⚙️ Custom Setup", callback_data="custom_setup")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    welcome_msg = (
        f"👋 <b>Welcome to Token Grade Alerts!</b>\n\n"
        f"🎯 Current subscription: <b>{', '.join(user_prefs.get('grades', []))}</b>\n\n"
        f"This bot monitors new Solana tokens and alerts you when they show "
        f"overlap with yesterday's winning tokens based on holder analysis.\n\n"
        f"<b>Grade Meanings:</b>\n"
        f"🔴 <b>CRITICAL</b> - Very high overlap (50%+ or strong concentration)\n"
        f"🟠 <b>HIGH</b> - Significant overlap (30%+ overlap)\n"
        f"🟡 <b>MEDIUM</b> - Notable overlap (15%+ overlap)\n"
        f"🟢 <b>LOW</b> - Some overlap (5%+ overlap)\n\n"
        f"Choose your alert preferences:"
    )
    
    await update.message.reply_html(welcome_msg, reply_markup=reply_markup)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline button presses"""
    query = update.callback_query
    chat_id = str(query.from_user.id)
    await query.answer()
    
    if query.data == "preset_critical":
        user_manager.update_user_prefs(chat_id, {"grades": ["CRITICAL"]})
        await query.edit_message_text(
            "✅ <b>Preferences Updated!</b>\n\n"
            "You will receive alerts for: <b>CRITICAL</b> grade tokens only.\n\n"
            "Use /myalerts to check your settings or /help for more commands.",
            parse_mode="HTML"
        )
    
    elif query.data == "preset_critical_high":
        user_manager.update_user_prefs(chat_id, {"grades": ["CRITICAL", "HIGH"]})
        await query.edit_message_text(
            "✅ <b>Preferences Updated!</b>\n\n"
            "You will receive alerts for: <b>CRITICAL, HIGH</b> grade tokens.\n\n"
            "Use /myalerts to check your settings or /help for more commands.",
            parse_mode="HTML"
        )
    
    elif query.data == "preset_all":
        user_manager.update_user_prefs(chat_id, {"grades": ["CRITICAL", "HIGH", "MEDIUM", "LOW"]})
        await query.edit_message_text(
            "✅ <b>Preferences Updated!</b>\n\n"
            "You will receive alerts for: <b>ALL</b> grades (CRITICAL, HIGH, MEDIUM, LOW).\n\n"
            "Use /myalerts to check your settings or /help for more commands.",
            parse_mode="HTML"
        )
    
    elif query.data == "custom_setup":
        await query.edit_message_text(
            "⚙️ <b>Custom Setup</b>\n\n"
            "Use the command: <code>/setalerts GRADE1 GRADE2 ...</code>\n\n"
            "Available grades: CRITICAL, HIGH, MEDIUM, LOW\n\n"
            "Examples:\n"
            "• <code>/setalerts CRITICAL</code>\n"
            "• <code>/setalerts CRITICAL HIGH MEDIUM</code>\n"
            "• <code>/setalerts HIGH MEDIUM LOW</code>",
            parse_mode="HTML"
        )

async def setalerts_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    args = context.args or []
    valid = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
    chosen = [a.upper() for a in args if a.upper() in valid]
    
    if not chosen:
        keyboard = [
            [
                InlineKeyboardButton("🔴 CRITICAL", callback_data="preset_critical"),
                InlineKeyboardButton("🔥 CRITICAL + HIGH", callback_data="preset_critical_high")
            ],
            [InlineKeyboardButton("📊 All Grades", callback_data="preset_all")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_html(
            "⚠️ <b>Usage:</b> <code>/setalerts GRADE1 GRADE2 ...</code>\n\n"
            "<b>Available grades:</b> CRITICAL, HIGH, MEDIUM, LOW\n\n"
            "<b>Examples:</b>\n"
            "• <code>/setalerts CRITICAL HIGH</code>\n"
            "• <code>/setalerts MEDIUM LOW</code>\n\n"
            "Or use quick presets:",
            reply_markup=reply_markup
        )
        return
    
    success = user_manager.update_user_prefs(chat_id, {"grades": chosen, "active": True})
    
    if success:
        await update.message.reply_html(
            f"✅ <b>Alert preferences updated!</b>\n\n"
            f"You will receive alerts for: <b>{', '.join(chosen)}</b>\n\n"
            f"Use /myalerts to view your current settings."
        )
    else:
        await update.message.reply_text("❌ Failed to save preferences. Please try again.")

async def myalerts_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    user_prefs = user_manager.get_user_prefs(chat_id)
    user_stats = user_manager.get_user_stats(chat_id)
    
    if not user_prefs.get("active", False):
        await update.message.reply_text("You are not currently subscribed. Use /start to subscribe.")
        return
    
    # Format stats
    total_alerts = user_stats.get("alerts_received", 0)
    last_alert = user_stats.get("last_alert_at")
    grade_breakdown = user_stats.get("grade_breakdown", {})
    
    last_alert_str = "Never" if not last_alert else f"<i>{last_alert[:10]}</i>"
    
    breakdown_lines = []
    for grade in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        count = grade_breakdown.get(grade, 0)
        if count > 0:
            breakdown_lines.append(f"  • {grade}: {count}")
    
    breakdown_text = "\n".join(breakdown_lines) if breakdown_lines else "  • No alerts received yet"
    
    msg = (
        f"📊 <b>Your Alert Settings</b>\n\n"
        f"🎯 <b>Subscribed to:</b> {', '.join(user_prefs.get('grades', []))}\n"
        f"📈 <b>Total alerts received:</b> {total_alerts}\n"
        f"🕐 <b>Last alert:</b> {last_alert_str}\n\n"
        f"<b>Breakdown by grade:</b>\n{breakdown_text}\n\n"
        f"Use /setalerts to change your preferences."
    )
    
    await update.message.reply_html(msg)

async def stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    success = user_manager.deactivate_user(chat_id)
    
    if success:
        await update.message.reply_html(
            "😔 <b>You have been unsubscribed</b>\n\n"
            "Your preferences are saved. Use /start anytime to reactivate alerts.\n\n"
            "Thank you for using Token Grade Alerts!"
        )
    else:
        await update.message.reply_text("❌ Failed to unsubscribe. Please try again.")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "🤖 <b>Token Grade Alerts - Help</b>\n\n"
        
        "<b>🔧 Commands:</b>\n"
        "• /start - Subscribe and set preferences\n"
        "• /setalerts GRADE1 GRADE2 - Set alert grades\n"
        "• /myalerts - View your settings and stats\n"
        "• /stop - Unsubscribe (keeps your data)\n"
        "• /help - Show this help message\n\n"
        
        "<b>📊 Grade Explanation:</b>\n"
        "• <b>CRITICAL</b> - 50%+ overlap or high concentration\n"
        "• <b>HIGH</b> - 30%+ overlap, significant activity\n"
        "• <b>MEDIUM</b> - 15%+ overlap, worth monitoring\n"
        "• <b>LOW</b> - 5%+ overlap, minor interest\n\n"
        
        "<b>📈 How it works:</b>\n"
        "The bot monitors new Solana tokens and compares their top holders "
        "with holders of yesterday's winning tokens. Higher overlap percentages "
        "suggest potential coordinated activity or insider information.\n\n"
        
        "<b>⏰ Timing:</b>\n"
        "New tokens are checked 2 hours after launch, then every 6 hours "
        "for the first 24 hours to track evolving patterns."
    )
    
    await update.message.reply_html(help_text)

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show platform statistics (admin only for detailed stats)"""
    chat_id = str(update.effective_chat.id)
    user_stats = user_manager.get_user_stats(chat_id)
    
    # Basic user stats for everyone
    msg = (
        f"📊 <b>Your Statistics</b>\n\n"
        f"📬 Total alerts received: <b>{user_stats.get('alerts_received', 0)}</b>\n"
        f"📅 Member since: <i>{user_stats.get('joined_at', 'Unknown')[:10] if user_stats.get('joined_at') else 'Unknown'}</i>\n"
    )
    
    # Admin gets platform-wide stats
    if is_admin(update):
        platform_stats = user_manager.get_all_stats()
        msg += (
            f"\n🏢 <b>Platform Statistics (Admin)</b>\n"
            f"👥 Total users: <b>{platform_stats['total_users']}</b>\n"
            f"✅ Active users: <b>{platform_stats['active_users']}</b>\n"
            f"📨 Total alerts sent: <b>{platform_stats['total_alerts_sent']}</b>\n\n"
            f"<b>Alerts by grade:</b>\n"
        )
        for grade, count in platform_stats['grade_breakdown'].items():
            if count > 0:
                msg += f"  • {grade}: {count}\n"
    
    await update.message.reply_html(msg)

# ---------- Alert processing ----------
def detect_alerts(results_tokens: Dict[str, Any], alerts_state: Dict[str, Any]):
    """
    returns list of tuples: (mint, token_data, alert_type, previous_grade_or_none)
    alert_type is "NEW" or "CHANGE"
    """
    alerts = []
    for mint, token_data in results_tokens.items():
        current_grade = token_data.get("grade")
        if mint not in alerts_state:
            alerts.append((mint, token_data, "NEW", None))
        else:
            prev = alerts_state[mint].get("last_grade")
            if prev != current_grade:
                alerts.append((mint, token_data, "CHANGE", prev))
    return alerts

async def send_digest_for_user(app: Application, chat_id: str, messages: List[str], grades_sent: List[str]):
    """
    Send a single digest message to a user and update their stats.
    """
    if not messages:
        return
    
    full = "📢 <b>Token Alerts (recent)</b>\n\n" + "\n\n".join(messages)
    try:
        await app.bot.send_message(
            chat_id=int(chat_id), 
            text=full, 
            parse_mode="HTML", 
            disable_web_page_preview=False
        )
        
        # Update user stats for each grade sent
        for grade in grades_sent:
            user_manager.update_user_stats(chat_id, grade)
            
        logging.info(f"Sent {len(messages)} alerts to user {chat_id}")
        
    except Exception as e:
        logging.exception("Failed to send message to %s: %s", chat_id, e)

# ---------- Background loop ----------
async def background_loop(app: Application):
    logging.info("Starting background loop (poll interval %s secs)", POLL_INTERVAL_SECS)
    while True:
        start_ts = time.time()
        # 1. Download latest monitor file
        downloaded = download_results_from_supabase(MONITOR_FILE)
        if not downloaded:
            logging.warning("Could not download monitor results; will skip this cycle.")
            await asyncio.sleep(POLL_INTERVAL_SECS)
            continue

        # 2. Check stale
        if is_monitor_stale(MONITOR_FILE, STALE_LIMIT_HOURS):
            logging.warning("Monitor is stale. Skipping alerting until monitor updates.")
            await asyncio.sleep(POLL_INTERVAL_SECS)
            continue

        # 3. Load data
        try:
            monitor_pkg = joblib.load(MONITOR_FILE)
            results_tokens = monitor_pkg.get("tokens", {}) if isinstance(monitor_pkg, dict) else {}
        except Exception as e:
            logging.exception("Failed reading monitor file: %s", e)
            await asyncio.sleep(POLL_INTERVAL_SECS)
            continue

        alerts_state = safe_load(ALERTS_STATE_FILE, {})
        
        # 4. Get only active users
        active_users = user_manager.get_active_users()

        # 5. Detect alerts
        new_alerts = detect_alerts(results_tokens, alerts_state)
        logging.info("Detected %d alertable tokens in this cycle", len(new_alerts))

        # 6. Build per-user digests
        digests = {chat_id: {"messages": [], "grades": []} for chat_id in active_users.keys()}
        for mint, token_data, alert_type, prev_grade in new_alerts:
            grade = token_data.get("grade")
            for chat_id, prefs in active_users.items():
                if grade in prefs.get("grades", []):
                    formatted = format_alert_html(token_data, alert_type, previous_grade=prev_grade)
                    digests[chat_id]["messages"].append(formatted)
                    digests[chat_id]["grades"].append(grade)

        # 7. Send digests
        tasks = []
        for chat_id, digest_data in digests.items():
            if digest_data["messages"]:
                tasks.append(send_digest_for_user(
                    app, chat_id, digest_data["messages"], digest_data["grades"]
                ))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # 8. Update alerts_state for alerted tokens (preserve history)
        now_ts = now_iso()
        for mint, token_data, alert_type, prev_grade in new_alerts:
            entry = alerts_state.get(mint)
            record = {"grade": token_data.get("grade"), "alert_time": now_ts, "alert_type": alert_type}
            if not entry:
                alerts_state[mint] = {
                    "last_grade": token_data.get("grade"),
                    "first_detected": token_data.get("block_time") or now_ts,
                    "last_alert_time": now_ts,
                    "grade_history": [record]
                }
            else:
                entry.setdefault("grade_history", []).append(record)
                entry["last_grade"] = token_data.get("grade")
                entry["last_alert_time"] = now_ts

        # persist state
        safe_save(ALERTS_STATE_FILE, alerts_state)
        # Upload updated alerts state to Supabase
        upload_file_to_supabase(ALERTS_STATE_FILE, "bot_alerts_state.pkl")

        # 9. Sleep until next cycle (but account for how long work took)
        elapsed = time.time() - start_ts
        to_sleep = max(0, POLL_INTERVAL_SECS - elapsed)
        logging.info("Cycle complete. Sleeping %.1f seconds", to_sleep)
        await asyncio.sleep(to_sleep)

# ---------- Data integration from monitor.py ----------
def load_monitor_results() -> Dict[str, Any]:
    """
    Load and process results from the monitor system.
    This integrates with your monitor.py overlap detection results.
    """
    try:
        # Try to load from overlap store (from monitor.py)
        overlap_file = Path("./data/overlap_results.pkl")
        if overlap_file.exists():
            overlap_data = joblib.load(overlap_file)
            
            # Convert overlap results to our expected format
            tokens = {}
            for mint, checks in overlap_data.items():
                if not checks:
                    continue
                
                # Get the most recent check
                latest_check = max(checks, key=lambda x: x.get("ts", ""))
                result = latest_check.get("result", {})
                
                tokens[mint] = {
                    "mint": mint,
                    "grade": result.get("grade", "UNKNOWN"),
                    "overlap_percentage": result.get("overlap_percentage", 0),
                    "concentration_percentage": result.get("concentration", 0),
                    "overlap_count": result.get("overlap_count", 0),
                    "checked_at": result.get("checked_at"),
                    "block_time": result.get("block_time"),
                    "token_metadata": {
                        "name": result.get("token_metadata", {}).get("name"),
                        "symbol": result.get("token_metadata", {}).get("symbol"),
                        "liquidity": result.get("token_metadata", {}).get("liquidity"),
                    }
                }
            
            return {"tokens": tokens, "last_updated": now_iso(), "source": "overlap_store"}
    
    except Exception as e:
        logging.exception("Error loading overlap results: %s", e)
    
    # Fallback to empty results
    return {"tokens": {}, "last_updated": now_iso(), "source": "fallback"}

# ---------- Admin commands ----------
async def admin_stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Detailed admin statistics"""
    if not is_admin(update):
        await update.message.reply_text("Access denied.")
        return
    
    platform_stats = user_manager.get_all_stats()
    prefs = safe_load(USER_PREFS_FILE, {})
    
    # Calculate additional metrics
    inactive_users = len([u for u in prefs.values() if not u.get("active", True)])
    recent_users = len([u for u in prefs.values() 
                      if u.get("created_at") and 
                      (datetime.datetime.utcnow() - parse_iso(u["created_at"])).days <= 7])
    
    
    msg = (
        f"👑 <b>Admin Dashboard</b>\n\n"
        f"📊 <b>User Statistics:</b>\n"
        f"• Total registered: {platform_stats['total_users']}\n"
        f"• Active users: {platform_stats['active_users']}\n"
        f"• Inactive users: {inactive_users}\n"
        f"• New users (7 days): {recent_users}\n\n"
        f"📨 <b>Alert Statistics:</b>\n"
        f"• Total alerts sent: {platform_stats['total_alerts_sent']}\n"
        f"• CRITICAL alerts: {platform_stats['grade_breakdown']['CRITICAL']}\n"
        f"• HIGH alerts: {platform_stats['grade_breakdown']['HIGH']}\n"
        f"• MEDIUM alerts: {platform_stats['grade_breakdown']['MEDIUM']}\n"
        f"• LOW alerts: {platform_stats['grade_breakdown']['LOW']}\n\n"
        f"🔧 <b>System Status:</b>\n"
        f"• Monitor file exists: {MONITOR_FILE.exists()}\n"
        f"• Monitor stale: {is_monitor_stale()}\n"
    )
    
    await update.message.reply_html(msg)

async def broadcast_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Broadcast message to all active users (admin only)"""
    if not is_admin(update):
        await update.message.reply_text("Access denied.")
        return
    
    if not context.args:
        await update.message.reply_text("Usage: /broadcast <message>")
        return
    
    message = " ".join(context.args)
    active_users = user_manager.get_active_users()
    
    sent = 0
    failed = 0
    
    for chat_id in active_users.keys():
        try:
            await context.bot.send_message(
                chat_id=int(chat_id),
                text=f"📢 <b>Announcement</b>\n\n{message}",
                parse_mode="HTML"
            )
            sent += 1
        except Exception as e:
            logging.warning(f"Failed to send broadcast to {chat_id}: {e}")
            failed += 1
        
        await asyncio.sleep(0.1)  # Rate limiting
    
    await update.message.reply_html(
        f"✅ Broadcast complete!\n\n"
        f"• Sent: {sent}\n"
        f"• Failed: {failed}"
    )

# ---------- Enhanced startup ----------
async def on_startup(app: Application):
    # create data dir
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download existing bot data from Supabase first
    download_bot_data_from_supabase()
    
    # ensure files exist with defaults if not downloaded
    safe_save(USER_PREFS_FILE, safe_load(USER_PREFS_FILE, {}))
    safe_save(ALERTS_STATE_FILE, safe_load(ALERTS_STATE_FILE, {}))
    safe_save(USER_STATS_FILE, safe_load(USER_STATS_FILE, {}))
    
    # Try to integrate with monitor.py results if available
    if not MONITOR_FILE.exists():
        logging.info("Monitor file doesn't exist, trying to load from overlap store")
        try:
            monitor_data = load_monitor_results()
            if monitor_data.get("tokens"):
                safe_save(MONITOR_FILE, monitor_data)
                logging.info("Loaded %d tokens from overlap store", len(monitor_data["tokens"]))
        except Exception as e:
            logging.warning("Could not load from overlap store: %s", e)
    
    # start background loop task
    app.create_task(background_loop(app))
    logging.info("Bot startup complete. Monitoring for token alerts...")

# ---------- Main ----------
def main():
    # set defaults for nice behaviour (optional)
    # Defaults.text_parse_mode("HTML")
    Defaults.text_parse_mode = "HTML"

    app = Application.builder().token(BOT_TOKEN).build()

    # Basic commands
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("setalerts", setalerts_cmd))
    app.add_handler(CommandHandler("myalerts", myalerts_cmd))
    app.add_handler(CommandHandler("stop", stop_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("stats", stats_cmd))
    
    # Admin commands
    app.add_handler(CommandHandler("admin", admin_stats_cmd))
    app.add_handler(CommandHandler("broadcast", broadcast_cmd))
    
    # Button handler
    app.add_handler(CallbackQueryHandler(button_handler))

    app.post_init = None
    # app.start(on_startup)
    app.post_init = on_startup

    logging.info("Starting enhanced telegram bot with user management...")
    app.run_polling(allowed_updates=None)  # runs until stopped

if __name__ == "__main__":
    main()