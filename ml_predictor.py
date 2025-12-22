"""
Production Inference Script for Solana Memecoin Classifier
Fetches data from DexScreener and Rugcheck APIs in real-time
"""

import pandas as pd
import numpy as np
import joblib
import json
import requests
from datetime import datetime
import time
import random
from typing import Dict, Optional
import logging
from sklearn.ensemble import RandomForestClassifier, IsolationForest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SolanaTokenPredictor:
    def __init__(self, model_dir='models'):
        """Load all models and preprocessing artifacts"""
        logger.info("Loading models...")
        self.selector = joblib.load(f'{model_dir}/feature_selector.pkl')
        self.xgb_model = joblib.load(f'{model_dir}/xgboost_model.pkl')
        self.lgb_model = joblib.load(f'{model_dir}/lightgbm_model.pkl')
        self.cat_model = joblib.load(f'{model_dir}/catboost_model.pkl')
        self.rf_model = joblib.load(f'{model_dir}/rf_model.pkl')
        self.iso_forest = joblib.load(f'{model_dir}/isolation_forest.pkl')
        
        with open(f'{model_dir}/model_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        self.ensemble_weights = self.metadata['ensemble_weights']
        self.selected_features = self.metadata['selected_features']
        self.all_features = self.metadata['all_features']
        
        logger.info(f"✅ Loaded {self.metadata['model_type']} model")
        logger.info(f"   Test AUC: {self.metadata['performance']['test_auc']:.4f}")
        logger.info(f"   Selected features: {len(self.selected_features)}")
    
    def fetch_dexscreener_data(self, mint: str) -> Optional[Dict]:
        """
        Fetch token data from DexScreener API with robust Retry/Backoff logic.
        Retries for approx 2 minutes on 429s/5xx errors.
        """
        url = f"https://api.dexscreener.com/latest/dex/tokens/{mint}"
        
        # Retry Configuration
        max_retries = 8       # Will span > 2 minutes with exponential backoff
        base_delay = 2.0      # Start with 2 seconds
        max_delay = 45.0      # Cap delay at 45 seconds per sleep
        
        for attempt in range(max_retries):
            try:
                # 1. Make Request (increased timeout to 15s)
                response = requests.get(url, timeout=15)
                
                # 2. Handle Success
                if response.status_code == 200:
                    data = response.json()
                    
                    if not data.get('pairs') or len(data['pairs']) == 0:
                        logger.warning(f"No pairs found for {mint}")
                        return None
                    
                    # Get the first/main pair (usually highest liquidity)
                    pair = data['pairs'][0]
                    
                    return {
                        'price_usd': float(pair.get('priceUsd', 0)),
                        'liquidity_usd': float(pair.get('liquidity', {}).get('usd', 0)),
                        'volume_h24_usd': float(pair.get('volume', {}).get('h24', 0)),
                        'fdv_usd': float(pair.get('fdv', 0) or pair.get('marketCap', 0)),
                        'price_change_h24_pct': float(pair.get('priceChange', {}).get('h24', 0)),
                        'pair_created_at': pair.get('pairCreatedAt', 0),
                    }

                # 3. Handle Rate Limiting (429)
                elif response.status_code == 429:
                    logger.warning(f"⚠️ DexScreener 429 Rate Limit for {mint}. Retrying...")
                    # Check headers for specific advice, else fall back to calc
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        sleep_time = int(retry_after) + 1
                    else:
                        # Exponential Backoff: 2s, 4s, 8s, 16s...
                        sleep_time = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, 1)

                # 4. Handle Not Found (404) - Do not retry
                elif response.status_code == 404:
                    logger.error(f"Token not found on DexScreener: {mint}")
                    return None

                # 5. Handle Server Errors (5xx)
                else:
                    logger.warning(f"DexScreener HTTP {response.status_code} for {mint}. Retrying...")
                    sleep_time = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, 1)

            except requests.exceptions.RequestException as e:
                # Handle network level errors (DNS, connection refused, etc)
                logger.warning(f"DexScreener Network Error for {mint}: {e}. Retrying...")
                sleep_time = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, 1)

            # Execute the wait (unless it was the last attempt)
            if attempt < max_retries - 1:
                logger.info(f"Waiting {sleep_time:.1f}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(sleep_time)

        # If we reach here, all retries failed
        logger.error(f"❌ Failed to fetch DexScreener data for {mint} after {max_retries} attempts.")
        return None
    
    def fetch_rugcheck_data(self, mint: str) -> Optional[Dict]:
        """Fetch security data from Rugcheck API"""
        url = f"https://api.rugcheck.xyz/v1/tokens/{mint}/report"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract key security metrics from the nested structure
            token = data.get('token', {})
            token_meta = token if isinstance(token, dict) else {}
            
            # FIXED: Use 'or []' to handle cases where API returns null (None) instead of a list
            # .get('key', []) only works if key is missing, not if value is null
            top_holders = data.get('topHolders') or []
            markets = data.get('markets') or []
            risks = data.get('risks') or []
            
            # Calculate top 10 holders percentage
            # This was previously crashing when top_holders was None
            top_10_pct = sum(h.get('pct', 0) for h in top_holders[:10])
            
            # Calculate total LP locked and liquidity from markets
            total_lp_locked = 0
            total_liquidity = 0
            
            for market in markets:
                # Guard against None entries in markets list
                if not market: 
                    continue
                    
                lp_data = market.get('lp', {})
                if not lp_data:
                    continue
                    
                total_lp_locked += lp_data.get('lpLockedUSD', 0) or 0
                
                # Calculate liquidity from reserves
                base_usd = lp_data.get('baseUSD', 0) or 0
                quote_usd = lp_data.get('quoteUSD', 0) or 0
                total_liquidity += (base_usd + quote_usd)
            
            # LP lock percentage
            lp_locked_pct = (total_lp_locked / total_liquidity * 100) if total_liquidity > 0 else 0
            
            # Insider networks
            insider_networks = data.get('insiderNetworks') or []
            
            # Calculate insider metrics safely
            if insider_networks:
                largest_network = max((n.get('size', 0) or 0) for n in insider_networks)
                total_insider_tokens = sum((n.get('tokenAmount', 0) or 0) for n in insider_networks)
            else:
                largest_network = 0
                total_insider_tokens = 0
            
            # Get token supply
            token_supply = token_meta.get('supply', 0)
            
            # Creator balance
            creator_balance = data.get('creatorBalance', 0)
            creator_balance_pct = (float(creator_balance) / float(token_supply) * 100) if token_supply > 0 else 0
            
            return {
                'creator_address': data.get('creator', 'UNKNOWN'),
                'creator_balance_pct': creator_balance_pct,
                'top_10_holders_pct': float(top_10_pct),
                'total_lp_locked_usd': float(total_lp_locked),
                'is_lp_locked_95_plus': int(lp_locked_pct >= 95),
                'has_mint_authority': int(token_meta.get('mintAuthority') is not None),
                'has_freeze_authority': int(token_meta.get('freezeAuthority') is not None),
                'token_supply': float(token_supply),
                'total_insider_networks': len(insider_networks),
                'largest_insider_network_size': largest_network,
                'total_insider_token_amount': total_insider_tokens,
                'rugcheck_risks': risks,
                'lp_locked_pct': lp_locked_pct,  # For debugging
            }
        except Exception as e:
            logger.error(f"Error fetching Rugcheck data: {e}")
            # Print traceback for easier debugging
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def calculate_pump_dump_risk(self, token_data: Dict) -> float:
        """Calculate pump & dump risk score"""
        risk_components = []
        
        # Insider risk
        insider_size = token_data.get('largest_insider_network_size', 0)
        if insider_size > 100:
            risk_components.append(30)
        elif insider_size > 50:
            risk_components.append(20)
        elif insider_size > 20:
            risk_components.append(10)
        
        # Top holder concentration risk
        top_10 = token_data.get('top_10_holders_pct', 0)
        top_1_pct = top_10 / 10  # Approximate top 1 holder
        if top_1_pct > 30:
            risk_components.append(25)
        elif top_1_pct > 20:
            risk_components.append(15)
        elif top_1_pct > 10:
            risk_components.append(8)
        
        # LP lock risk
        is_locked = token_data.get('is_lp_locked_95_plus', 0)
        if not is_locked:
            lp_usd = token_data.get('total_lp_locked_usd', 0)
            total_liq = token_data.get('liquidity_usd', 1)
            lp_pct = (lp_usd / total_liq * 100) if total_liq > 0 else 0
            
            if lp_pct < 50:
                risk_components.append(20)
            elif lp_pct < 80:
                risk_components.append(12)
            elif lp_pct < 95:
                risk_components.append(5)
        
        # Authority risk
        has_mint = token_data.get('has_mint_authority', 0)
        has_freeze = token_data.get('has_freeze_authority', 0)
        auth_score = (50 if has_mint else 0) + (50 if has_freeze else 0)
        risk_components.append(min(auth_score / 100 * 15, 15))
        
        # Creator risk
        creator_pct = token_data.get('creator_balance_pct', 0)
        token_age_hours = token_data.get('token_age_hours', 0)
        creator_dumped = (creator_pct == 0) and (token_age_hours < 24)
        
        if creator_dumped:
            risk_components.append(10)
        elif creator_pct > 10:
            risk_components.append(7)
        elif creator_pct > 5:
            risk_components.append(4)
        
        return sum(risk_components)
    
    def engineer_features(self, raw_data: Dict) -> Dict:
        """Engineer all features from raw API data"""
        now = datetime.utcnow()
        
        # === TIME FEATURES ===
        pair_created_ms = raw_data.get('pair_created_at', 0)
        if pair_created_ms:
            pair_created_ts = pair_created_ms / 1000
            token_age_seconds = time.time() - pair_created_ts
        else:
            token_age_seconds = 0
        
        token_age_hours = token_age_seconds / 3600
        
        features = {
            # Core time features
            'token_age_at_signal_seconds': token_age_seconds,
            'token_age_hours': token_age_hours,
            'time_of_day_utc': now.hour,
            'day_of_week_utc': now.weekday(),
            'is_weekend_utc': int(now.weekday() >= 5),
            'is_public_holiday_any': 0,  # Would need holiday API
            
            # Time-based derived features
            'is_ultra_fresh': int(token_age_hours < 1),
            'is_very_fresh': int(token_age_hours < 4),
            'is_fresh': int(token_age_hours < 12),
            'freshness_score': np.exp(-token_age_hours / 6),
            'age_penalty': 1 / (1 + token_age_hours),
            
            # Trading session indicators
            'asian_hours': int(0 <= now.hour < 8),
            'european_hours': int(8 <= now.hour < 16),
            'us_hours': int(16 <= now.hour < 24),
            'peak_hours': int(14 <= now.hour <= 20),
            'off_peak_day': int(now.weekday() >= 5),
        }
        
        # === MARKET DATA ===
        price = raw_data.get('price_usd', 0)
        liquidity = raw_data.get('liquidity_usd', 0)
        volume = raw_data.get('volume_h24_usd', 0)
        fdv = raw_data.get('fdv_usd', 0)
        price_change = raw_data.get('price_change_h24_pct', 0)
        
        features.update({
            'price_usd': price,
            'liquidity_usd': liquidity,
            'volume_h24_usd': volume,
            'fdv_usd': fdv,
            'price_change_h24_pct': price_change,
            
            # Log transforms
            'log_liquidity': np.log1p(liquidity),
            'log_volume': np.log1p(volume),
            'log_fdv': np.log1p(fdv),
        })
        
        # === RATIOS ===
        features['volume_to_liquidity_ratio'] = volume / liquidity if liquidity > 0 else 0
        features['fdv_to_liquidity_ratio'] = fdv / liquidity if liquidity > 0 else 0
        features['liquidity_to_volume_ratio'] = liquidity / volume if volume > 0 else 0
        features['liquidity_depth_pct'] = (liquidity / fdv * 100) if fdv > 0 else 0
        features['volume_efficiency'] = volume / fdv if fdv > 0 else 0
        features['market_efficiency'] = (features['volume_to_liquidity_ratio'] / features['liquidity_to_volume_ratio']) if features['liquidity_to_volume_ratio'] > 0 else features['volume_to_liquidity_ratio']
        features['liquidity_coverage'] = liquidity / volume if volume > 0 else 0
        
        # === SECURITY DATA ===
        creator_pct = raw_data.get('creator_balance_pct', 0)
        top_10_pct = raw_data.get('top_10_holders_pct', 0)
        lp_locked = raw_data.get('total_lp_locked_usd', 0)
        token_supply = raw_data.get('token_supply', 0)
        
        features.update({
            'creator_address': raw_data.get('creator_address', 'UNKNOWN'),
            'creator_balance_pct': creator_pct,
            'top_10_holders_pct': top_10_pct,
            'total_lp_locked_usd': lp_locked,
            'is_lp_locked_95_plus': raw_data.get('is_lp_locked_95_plus', 0),
            'token_supply': token_supply,
            'log_lp_locked': np.log1p(lp_locked),
        })
        
        # LP lock quality
        features['lp_lock_quality'] = (lp_locked / liquidity * 100) if liquidity > 0 else 0
        
        # === HOLDER FEATURES ===
        features.update({
            'holder_quality_score': 100 - top_10_pct,
            'extreme_concentration': int(top_10_pct > 60),
            'whale_dominated': int(top_10_pct > 70),
            'creator_dumped': int((creator_pct < 1) and (token_age_hours < 24)),
            'creator_holds_significant': int(creator_pct > 10),
            'concentration_risk': top_10_pct * 0.5 + creator_pct * 0.3,
        })
        
        # === INSIDER FEATURES ===
        total_networks = raw_data.get('total_insider_networks', 0)
        largest_network = raw_data.get('largest_insider_network_size', 0)
        total_insider_tokens = raw_data.get('total_insider_token_amount', 0)
        
        features.update({
            'total_insider_networks': total_networks,
            'largest_insider_network_size': largest_network,
            'total_insider_token_amount': total_insider_tokens,
        })
        
        # 🔴 KEY DERIVED INSIDER FEATURES
        features['insider_supply_pct'] = (total_insider_tokens / token_supply * 100) if token_supply > 0 else 0
        features['avg_insider_network_size'] = largest_network / total_networks if total_networks > 0 else 0
        features['insider_density'] = total_insider_tokens / total_networks if total_networks > 0 else 0
        features['insider_risk_composite'] = (
            features['insider_supply_pct'] * 0.4 +
            largest_network * 0.3 +
            (total_networks * 2) * 0.3
        )
        
        # Update concentration risk to include insiders
        features['concentration_risk'] = (
            top_10_pct * 0.5 +
            creator_pct * 0.3 +
            features['insider_supply_pct'] * 0.2
        )
        
        # === RISK SCORES ===
        features['pump_dump_risk_score'] = self.calculate_pump_dump_risk(raw_data)
        features['liquidity_risk_score'] = (100 - features['lp_lock_quality']) * 0.5 if features['is_lp_locked_95_plus'] == 0 else 0
        features['insider_holder_risk'] = (
            features['insider_risk_composite'] * 0.5 +
            features['concentration_risk'] * 0.5
        )
        
        # === MARKET HEALTH ===
        features['market_health_score'] = (
            (features['liquidity_depth_pct'] / 100) * 30 +
            (features['lp_lock_quality'] / 100) * 30 +
            (features['holder_quality_score'] / 100) * 20 +
            (features['volume_efficiency'] * 100) * 20
        )
        
        # === MOMENTUM FEATURES ===
        features.update({
            'strong_uptrend': int(price_change > 50),
            'moderate_uptrend': int(20 < price_change <= 50),
            'weak_momentum': int(price_change < 10),
            'volume_quality': volume / (1 + abs(price_change)) if price_change != 0 else volume,
        })
        
        # === ANOMALY DETECTION ===
        # Prepare features for isolation forest (must match training)
        anomaly_input = pd.DataFrame([{
            'log_liquidity': features.get('log_liquidity', 0),
            'log_volume': features.get('log_volume', 0),
            'price_change_h24_pct': features.get('price_change_h24_pct', 0),
            'top_10_holders_pct': features.get('top_10_holders_pct', 0),
            'creator_balance_pct': features.get('creator_balance_pct', 0)
        }])
        
        # Calculate anomaly score (higher is more anomalous)
        # Note: score_samples returns negative values, we negated it in training
        features['anomaly_score'] = -self.iso_forest.score_samples(anomaly_input)[0]
        features['is_anomaly'] = int(self.iso_forest.predict(anomaly_input)[0] == -1)
        
        return features
    
    def predict(self, mint: str, threshold: float = 0.50, action_threshold: float = 0.70) -> Dict:
        """
        Predict if token will reach 50% gain
        
        Args:
            mint: Token mint address
            threshold: Probability threshold for filtering/upload (default 0.50)
            action_threshold: Probability threshold for BUY/CONSIDER actions (default 0.70)
        
        Returns:
            dict with prediction, probability, and recommendation
        """
        logger.info(f"Analyzing token: {mint}")
        
        # Fetch data from APIs
        logger.info("Fetching DexScreener data...")
        dex_data = self.fetch_dexscreener_data(mint)
        if not dex_data:
            return {'error': 'Could not fetch DexScreener data'}
        
        logger.info("Fetching Rugcheck data...")
        rug_data = self.fetch_rugcheck_data(mint)
        if not rug_data:
            return {'error': 'Could not fetch Rugcheck data'}
        
        # Combine data
        raw_data = {**dex_data, **rug_data}
        
        # Engineer features
        logger.info("Engineering features...")
        features = self.engineer_features(raw_data)
        
        # Create DataFrame with all features
        df = pd.DataFrame([features])
        
        # Ensure all expected features exist
        for feat in self.all_features:
            if feat not in df.columns:
                df[feat] = 0
        
        # Select features in correct order
        X = df[self.all_features]
        
        # Apply feature selection (returns DataFrame with selected features)
        X_selected = self.selector.transform(X)
        
        # Convert back to DataFrame with selected feature names for models
        X_selected_df = pd.DataFrame(
            X_selected, 
            columns=self.selected_features
        )
        
        # Get predictions from all models (now with proper feature names)
        xgb_proba = self.xgb_model.predict_proba(X_selected_df)[0, 1]
        lgb_proba = self.lgb_model.predict_proba(X_selected_df)[0, 1]
        cat_proba = self.cat_model.predict_proba(X_selected_df)[0, 1]
        rf_proba = self.rf_model.predict_proba(X_selected_df)[0, 1]
        
        # Ensemble prediction
        ensemble_proba = (
            self.ensemble_weights.get('catboost', 0.25) * cat_proba +
            self.ensemble_weights.get('lightgbm', 0.25) * lgb_proba +
            self.ensemble_weights.get('xgboost', 0.25) * xgb_proba +
            self.ensemble_weights.get('random_forest', 0.25) * rf_proba
        )
        
        # Decision logic (using action_threshold, not filtering threshold)
        if ensemble_proba >= action_threshold:
            action = "BUY"
            confidence = "HIGH"
            risk_tier = "LOW RISK"
        elif ensemble_proba >= 0.60:
            action = "CONSIDER"
            confidence = "MEDIUM"
            risk_tier = "MEDIUM RISK"
        elif ensemble_proba >= 0.45:
            action = "SKIP"
            confidence = "LOW"
            risk_tier = "HIGH RISK"
        else:
            action = "AVOID"
            confidence = "VERY LOW"
            risk_tier = "VERY HIGH RISK"
        
        # Model agreement score
        model_predictions = [xgb_proba, lgb_proba, cat_proba, rf_proba]
        agreement = 1 - (np.std(model_predictions) / np.mean(model_predictions)) if np.mean(model_predictions) > 0 else 0
        
        return {
            'mint': mint,
            'action': action,
            'win_probability': float(ensemble_proba),
            'confidence': confidence,
            'risk_tier': risk_tier,
            'model_agreement': float(agreement),
            'individual_predictions': {
                'xgboost': float(xgb_proba),
                'lightgbm': float(lgb_proba),
                'catboost': float(cat_proba),
                'random_forest': float(rf_proba)
            },
            'key_metrics': {
                'token_age_hours': float(features.get('token_age_hours', 0)),
                'liquidity_usd': float(raw_data.get('liquidity_usd', 0)),
                'volume_h24_usd': float(raw_data.get('volume_h24_usd', 0)),
                'price_change_h24_pct': float(raw_data.get('price_change_h24_pct', 0)),
                'insider_supply_pct': float(features.get('insider_supply_pct', 0)),
                'top_10_holders_pct': float(raw_data.get('top_10_holders_pct', 0)),
                'pump_dump_risk_score': float(features.get('pump_dump_risk_score', 0)),
                'market_health_score': float(features.get('market_health_score', 0)),
            },
            'warnings': self._generate_warnings(features, raw_data)
        }
    
    def _generate_warnings(self, features: Dict, raw_data: Dict) -> list:
        """Generate warning messages based on risk factors"""
        warnings = []
        
        if features.get('pump_dump_risk_score', 0) > 50:
            warnings.append("⚠️ HIGH pump & dump risk detected")
        
        if features.get('insider_supply_pct', 0) > 20:
            warnings.append("🔴 Insiders control >20% of supply")
        
        if raw_data.get('top_10_holders_pct', 0) > 70:
            warnings.append("⚠️ Extreme holder concentration (whale risk)")
        
        if raw_data.get('is_lp_locked_95_plus', 0) == 0:
            warnings.append("⚠️ Liquidity not sufficiently locked")
        
        if features.get('token_age_hours', 0) > 48:
            warnings.append("⏰ Token is >48h old (momentum may have passed)")
        
        if raw_data.get('liquidity_usd', 0) < 10000:
            warnings.append("💧 Low liquidity (<$10k)")
        
        # Only warn about creator dump if it's NOT a very fresh token (<2h)
        # Fresh Pump.Fun tokens often have 0% creator balance by design
        if features.get('creator_dumped', 0) == 1 and features.get('token_age_hours', 0) > 2:
            warnings.append("🚨 Creator appears to have dumped")
        
        return warnings


# Example usage
if __name__ == "__main__":
    import sys
    
    predictor = SolanaTokenPredictor()
    
    # Get mint from command line or use example
    if len(sys.argv) > 1:
        mint_address = sys.argv[1]
    else:
        # Example mint (replace with actual)
        mint_address = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC as example
        print(f"No mint provided, using example: {mint_address}")
    
    # Get prediction
    result = predictor.predict(mint_address, threshold=0.60)
    
    if 'error' in result:
        print(f"\n❌ Error: {result['error']}")
    else:
        print("\n" + "="*80)
        print("🎯 PREDICTION RESULT")
        print("="*80)
        print(f"\nMint: {result['mint']}")
        print(f"\n🎲 Action: {result['action']}")
        print(f"📊 Win Probability: {result['win_probability']:.1%}")
        print(f"🎯 Confidence: {result['confidence']}")
        print(f"⚠️  Risk Tier: {result['risk_tier']}")
        print(f"🤝 Model Agreement: {result['model_agreement']:.1%}")
        
        print(f"\n📈 Key Metrics:")
        for key, value in result['key_metrics'].items():
            print(f"   • {key}: {value:.2f}")
        
        print(f"\n🤖 Individual Model Predictions:")
        for model, prob in result['individual_predictions'].items():
            print(f"   • {model:10s}: {prob:.1%}")
        
        if result['warnings']:
            print(f"\n⚠️  Warnings:")
            for warning in result['warnings']:
                print(f"   {warning}")
        
        print("\n" + "="*80)