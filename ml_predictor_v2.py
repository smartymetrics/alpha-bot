"""
Production Inference Script for Solana Memecoin Classifier
Uses ensemble of XGBoost, LightGBM, and CatBoost models
"""

import pandas as pd
import numpy as np
import joblib
import json

class SolanaMemeTokenClassifier:
    def __init__(self, model_dir='models', debug=False):
        """Load all models and preprocessing artifacts"""
        self.debug = debug
        self.selector = joblib.load(f'{model_dir}/feature_selector.pkl')
        self.xgb_model = joblib.load(f'{model_dir}/xgboost_model.pkl')
        self.lgb_model = joblib.load(f'{model_dir}/lightgbm_model.pkl')
        self.cat_model = joblib.load(f'{model_dir}/catboost_model.pkl')
        self.label_encoders = joblib.load(f'{model_dir}/label_encoders.pkl')
        
        with open(f'{model_dir}/model_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        self.ensemble_weights = self.metadata['ensemble_weights']
        self.selected_features = self.metadata['selected_features']
        self.all_features = self.metadata['all_features']
        
        print(f"✅ Loaded {self.metadata['model_type']} model")
        print(f"   Test AUC: {self.metadata['performance']['test_auc']:.4f}")
        print(f"   Selected features: {len(self.selected_features)}")
    
    def engineer_features(self, df):
        """Apply same feature engineering as training"""
        df = df.copy()
        
        # Basic features
        df['token_age_hours'] = df['token_age_at_signal_seconds'] / 3600
        df['token_age_hours_capped'] = df['token_age_hours'].clip(0, 24)
        df['is_ultra_fresh'] = (df['token_age_at_signal_seconds'] < 3600).astype(int)
        
        # Log transforms
        df['log_liquidity'] = np.log1p(df['liquidity_usd'])
        df['log_volume'] = np.log1p(df['volume_h24_usd'])
        df['log_fdv'] = np.log1p(df['fdv_usd'])
        
        # Basic ratios
        df['volume_to_liquidity_ratio'] = np.where(
            df['liquidity_usd'] > 0,
            df['volume_h24_usd'] / df['liquidity_usd'],
            0
        )
        df['fdv_to_liquidity_ratio'] = np.where(
            df['liquidity_usd'] > 0,
            df['fdv_usd'] / df['liquidity_usd'],
            0
        )
        
        # Risk signals
        df['high_risk_signal'] = (
            (df['pump_dump_risk_score'] > 30) & 
            (df['authority_risk_score'] > 15)
        ).astype(int)
        df['premium_signal'] = (
            (df['grade'] == 'CRITICAL') & 
            (df['signal_source'] == 'alpha')
        ).astype(int)
        df['extreme_concentration'] = (df['top_10_holders_pct'] > 60).astype(int)
        
        # ADVANCED COMPOSITE FEATURES
        df['liquidity_per_holder'] = np.where(
            df['top_10_holders_pct'] > 0,
            df['liquidity_usd'] / (df['top_10_holders_pct'] / 10),
            0
        )
        df['smart_money_score'] = (
            df['overlap_quality_score'] * df['winner_wallet_density']
        )
        df['risk_adjusted_volume'] = np.where(
            df['pump_dump_risk_score'] > 0,
            df['volume_h24_usd'] / (1 + df['pump_dump_risk_score']),
            df['volume_h24_usd']
        )
        df['liquidity_velocity'] = np.where(
            df['liquidity_usd'] > 0,
            df['volume_h24_usd'] / df['liquidity_usd'],
            0
        )
        df['holder_quality_score'] = 100 - df['top_10_holders_pct']
        df['market_efficiency'] = np.where(
            (df['liquidity_usd'] > 0) & (df['fdv_usd'] > 0),
            (df['volume_h24_usd'] / df['liquidity_usd']) / (df['fdv_usd'] / df['liquidity_usd']),
            0
        )
        df['risk_weighted_smart_money'] = np.where(
            df['pump_dump_risk_score'] + df['authority_risk_score'] > 0,
            df['smart_money_score'] / (1 + df['pump_dump_risk_score'] + df['authority_risk_score']),
            df['smart_money_score']
        )
        df['concentration_risk'] = (
            df['top_10_holders_pct'] * 0.4 +
            df['creator_balance_pct'] * 0.3 +
            df['whale_concentration_score'] * 0.3
        )
        df['liquidity_depth_score'] = np.where(
            df['fdv_usd'] > 0,
            (df['liquidity_usd'] / df['fdv_usd']) * 100,
            0
        )
        df['freshness_score'] = np.exp(-df['token_age_hours'] / 6)
        
        return df
    
    def predict(self, token_data, threshold=0.70):
        """
        Predict if token will reach 50% gain
        
        Args:
            token_data: dict or DataFrame with token features
            threshold: probability threshold for BUY signal (default 0.70 for high precision)
        
        Returns:
            dict with prediction, probability, and recommendation
        """
        # Convert to DataFrame if dict
        if isinstance(token_data, dict):
            df = pd.DataFrame([token_data])
        else:
            df = token_data.copy()
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Select features
        X = df[self.all_features].copy()
        
        # Encode categorical features with handling for unseen labels
        for col in self.metadata['categorical_features']:
            if col in X.columns:
                try:
                    # Check if values are in the encoder's classes
                    values = X[col].astype(str)
                    unseen_mask = ~values.isin(self.label_encoders[col].classes_)
                    
                    if unseen_mask.any():
                        # Replace unseen labels with the most common class from training
                        most_common = self.label_encoders[col].classes_[0]
                        X.loc[unseen_mask, col] = most_common
                        if self.debug:
                            print(f"Warning: Unseen values in '{col}' replaced with '{most_common}'")
                    
                    X[col] = self.label_encoders[col].transform(values)
                except Exception as e:
                    if self.debug:
                        print(f"Error encoding '{col}': {e}. Using default value.")
                    X[col] = 0
        
        # Apply feature selection - this returns a numpy array
        X_selected_array = self.selector.transform(X)
        
        # Convert back to DataFrame with proper column names for the models
        # Get the selected feature names from the selector
        selected_feature_mask = self.selector.get_support()
        selected_feature_names = [name for name, selected in zip(self.all_features, selected_feature_mask) if selected]
        X_selected = pd.DataFrame(X_selected_array, columns=selected_feature_names, index=X.index)
        
        # Get predictions from all models
        xgb_proba = self.xgb_model.predict_proba(X_selected)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(X_selected)[:, 1]
        cat_proba = self.cat_model.predict_proba(X_selected)[:, 1]
        
        # Ensemble prediction
        if self.ensemble_weights:
            ensemble_proba = (
                self.ensemble_weights['catboost'] * cat_proba +
                self.ensemble_weights['lightgbm'] * lgb_proba +
                self.ensemble_weights['xgboost'] * xgb_proba
            )
        else:
            ensemble_proba = cat_proba  # Use best single model
        
        # Decision logic with risk tiers
        results = []
        for i, prob in enumerate(ensemble_proba):
            if prob >= threshold:
                action = "BUY"
                confidence = "HIGH"
                risk_tier = "LOW RISK"
            elif prob >= 0.60:
                action = "CONSIDER"
                confidence = "MEDIUM"
                risk_tier = "MEDIUM RISK"
            elif prob >= 0.45:
                action = "SKIP"
                confidence = "LOW"
                risk_tier = "HIGH RISK"
            else:
                action = "AVOID"
                confidence = "VERY LOW"
                risk_tier = "VERY HIGH RISK"
            
            results.append({
                'action': action,
                'win_probability': float(prob),
                'confidence': confidence,
                'risk_tier': risk_tier,
                'individual_predictions': {
                    'xgboost': float(xgb_proba[i]),
                    'lightgbm': float(lgb_proba[i]),
                    'catboost': float(cat_proba[i])
                }
            })
        
        return results[0] if len(results) == 1 else results
    
    def batch_predict(self, tokens_df, threshold=0.70):
        """Predict for multiple tokens at once"""
        results = self.predict(tokens_df, threshold)
        
        # Add to DataFrame
        tokens_df = tokens_df.copy()
        tokens_df['win_probability'] = [r['win_probability'] for r in results]
        tokens_df['action'] = [r['action'] for r in results]
        tokens_df['confidence'] = [r['confidence'] for r in results]
        tokens_df['risk_tier'] = [r['risk_tier'] for r in results]
        
        return tokens_df.sort_values('win_probability', ascending=False)


# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = SolanaMemeTokenClassifier()
    
    # Example token data
    example_token = {
        'signal_source': 'alpha',
        'grade': 'CRITICAL',
        'liquidity_usd': 50000,
        'volume_h24_usd': 150000,
        'fdv_usd': 100000,
        'price_change_h24_pct': 25.5,
        'creator_balance_pct': 5.0,
        'top_10_holders_pct': 35.0,
        'total_lp_locked_usd': 30000,
        'whale_concentration_score': 40.0,
        'pump_dump_risk_score': 15.0,
        'authority_risk_score': 5.0,
        'overlap_quality_score': 0.8,
        'winner_wallet_density': 0.6,
        'token_age_at_signal_seconds': 1800,  # 30 minutes old
    }
    
    # Get prediction
    result = classifier.predict(example_token, threshold=0.70)
    
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"Action: {result['action']}")
    print(f"Win Probability: {result['win_probability']:.2%}")
    print(f"Confidence: {result['confidence']}")
    print(f"Risk Tier: {result['risk_tier']}")
    print("\nIndividual Model Predictions:")
    for model, prob in result['individual_predictions'].items():
        print(f"  {model:10s}: {prob:.2%}")