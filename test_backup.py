import pickle
import os

base_dir = r"c:\Users\HP USER\Documents\Data Analyst\degen smart\data"
user_prefs = os.path.join(base_dir, "bot_user_prefs.pkl")
portfolios = os.path.join(base_dir, "bot_portfolios.pkl")

print("--- DEGEN SMART DATA DIAGNOSTIC ---")

if os.path.exists(user_prefs):
    try:
        with open(user_prefs, 'rb') as f:
            data = pickle.load(f)
            print(f"Users found in degen smart/bot_user_prefs.pkl: {len(data)}")
    except Exception as e:
        print(f"Failed to read users: {e}")
else:
    print("No user prefs file found.")

if os.path.exists(portfolios):
    try:
        with open(portfolios, 'rb') as f:
            data = pickle.load(f)
            print(f"Portfolios found in degen smart/bot_portfolios.pkl: {len(data)}")
    except Exception as e:
        print(f"Failed to read portfolios: {e}")
else:
    print("No portfolios file found.")
