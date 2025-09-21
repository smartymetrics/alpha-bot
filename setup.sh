#!/bin/bash
# setup.sh - One-command deploy for token_monitor.py

# ====== CONFIGURE THESE ======
PROJECT_DIR="$HOME/token_monitor_project"
SCRIPT_NAME="token_monitor.py"
REQ_FILE="requirements.txt"
SCREEN_NAME="tokenmonitor"

# ====== CREATE PROJECT FOLDER ======
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# ====== COPY SCRIPT & REQUIREMENTS ======
# Assumes you already SCP'd token_monitor.py and requirements.txt into this directory

# ====== INSTALL PYTHON PACKAGES ======
echo "Installing Python packages..."
pip3 install --upgrade pip --user
pip3 install --user -r $REQ_FILE

# ====== SET ENVIRONMENT VARIABLES ======
echo "Setting up environment variables..."
ENV_FILE="$HOME/.bashrc"

# Add your variables here (edit as needed):
if ! grep -q "TELEGRAM_TOKEN" $ENV_FILE; then
  echo 'export TELEGRAM_TOKEN="YOUR_TELEGRAM_TOKEN"' >> $ENV_FILE
  echo 'export SUPABASE_URL="YOUR_SUPABASE_URL"' >> $ENV_FILE
  echo 'export SUPABASE_KEY="YOUR_SUPABASE_KEY"' >> $ENV_FILE
fi

# Apply changes
source $ENV_FILE

# ====== START SCRIPT IN SCREEN SESSION ======
echo "Starting $SCRIPT_NAME in a detached screen session..."
screen -dmS $SCREEN_NAME python3 $PROJECT_DIR/$SCRIPT_NAME

echo "✅ Setup complete! Use 'screen -r $SCREEN_NAME' to view logs."
