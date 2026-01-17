
import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

@dataclass
class TradingStart:
    mint: Optional[str] = None
    block_time: Optional[int] = None
    program_id: Optional[str] = None
    detected_via: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None
    fdv_usd: Optional[float] = None
    volume_usd: Optional[float] = None
    source_dex: Optional[str] = None
    price_change_percentage: Optional[float] = None

def _parse_iso_timestamp(val: Any) -> Optional[int]:
    if not val: return None
    try:
        ts_str = str(val).replace("Z", "+00:00")
        if "T" in ts_str and len(ts_str) == 19:
            ts_str += "+00:00"
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception: return None

def parse_boosted_tokens(data: List[Dict[str, Any]]) -> List[TradingStart]:
    out = []
    now_ts = int(time.time())
    for item in data:
        if item.get("chainId") != "solana":
            continue
        
        mint = item.get("tokenAddress")
        if not mint:
            continue
            
        # Boosted tokens don't always have a creation time in the boosts API,
        # but the request says to parse it similar to profiles (which use now_ts).
        # Some items might have a timestamp or creation time buried in links or description,
        # but typically this endpoint is for currently boosted tokens.
        
        out.append(TradingStart(
            mint=mint,
            block_time=now_ts,
            program_id="dexscreener_boost",
            detected_via="dexscreener_boost",
            extra={
                "url": item.get("url"),
                "description": item.get("description"),
                "icon": item.get("icon"),
                "header": item.get("header"),
                "links": item.get("links"),
                "totalAmount": item.get("totalAmount"),
                "amount": item.get("amount")
            },
            fdv_usd=0.0,
            volume_usd=0.0,
            source_dex="dexscreener",
            price_change_percentage=0.0
        ))
    return out

# Example data from user request
mock_data = [
    {"url":"https://dexscreener.com/solana/9pgx8fuywg4wffrcpahjquseenvdccbbnqq9rzvgpump","chainId":"solana","tokenAddress":"9pgx8fuYwG4wFFrcPAhJquSEenvDCCbbnqQ9RzVgpump","description":"Everyone knows Pepe as a 4chan meme...","icon":"3df5d7503febe1b5cd7982e7661b223df95d370cd454e701a879d3c8565c52a0","header":"https://cdn.dexscreener.com/cms/images/d9892165e08d1097656cef207bbd02701d468ce9623dd7117fcd956ecd322a55?width=900&height=300&fit=crop&quality=95&format=auto","openGraph":"https://cdn.dexscreener.com/token-images/og/solana/9pgx8fuYwG4wFFrcPAhJquSEenvDCCbbnqQ9RzVgpump?timestamp=1768671900000","links":[{"url":"https://yaruocoin.xyz/"},{"url":"https://ja.wikipedia.org/wiki/%E3%82%84%E3%82%8B%E5%A4%AB"},{"url":"https://en.namu.wiki/w/%EC%95%BC%EB%A3%A8%EC%98%A4"},{"type":"twitter","url":"https://x.com/YaruoSOL"}],"totalAmount":10,"amount":10},
    {"url":"https://dexscreener.com/ethereum/0x123","chainId":"ethereum","tokenAddress":"0x123","description":"Not Solana","totalAmount":10,"amount":10}
]

if __name__ == "__main__":
    results = parse_boosted_tokens(mock_data)
    print(f"Parsed {len(results)} tokens")
    for r in results:
        print(f"Mint: {r.mint}, Source: {r.detected_via}, Links: {len(r.extra['links'])}")
