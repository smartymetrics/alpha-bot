# key_check.py
from dotenv import load_dotenv
import os, requests, re
load_dotenv()

def split_clean(raw):
    if not raw:
        return []
    parts = re.split(r"[,\n\r]+", raw)
    cleaned = []
    for p in parts:
        s = (p or "").strip()
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1].strip()
        s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s).strip()
        if s:
            cleaned.append(s)
    return cleaned

m = split_clean(os.getenv("MORALIS_API_KEY") or os.getenv("MORALIS_KEYS") or "")
h = split_clean(os.getenv("HELIUS_API_KEY") or os.getenv("HELIUS_KEYS") or "")

print("Moralis keys (count):", len(m))
for i,k in enumerate(m):
    print("  ", i, repr(k[:32]), "... length:", len(k))

print("Helius keys (count):", len(h))
for i,k in enumerate(h):
    print("  ", i, repr(k[:32]), "... length:", len(k))

# Optional: do a single test request per Moralis key to the swaps endpoint:
for i,k in enumerate(m):
    try:
        r = requests.get(
            "https://solana-gateway.moralis.io/token/mainnet/zgQnq6GEUWuEEa2QvqT69amJtKaj7oU4nKDP4cTpump/swaps?limit=1",
            headers={"X-API-Key": k, "Accept": "application/json"},
            timeout=8
        )
        print("Moralis key", i, "status:", r.status_code)
    except Exception as e:
        print("Moralis key", i, "error:", e)

# Optional: test helius keys
for i,k in enumerate(h):
    try:
        payload = {"jsonrpc":"2.0","id":1,"method":"getHealth","params":[]}
        r = requests.post(f"https://mainnet.helius-rpc.com/?api-key={k}", json=payload, timeout=8)
        print("Helius key", i, "status:", r.status_code)
    except Exception as e:
        print("Helius key", i, "error:", e)
