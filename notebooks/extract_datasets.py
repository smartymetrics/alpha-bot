#!/usr/bin/env python3
"""
extract_features_from_datasets.py

Extracts training data from your Supabase datasets (created by collector.py)
and prepares it in the EXACT format needed for model training.

This script:
1. Downloads labeled datasets from Supabase
2. Extracts ONLY features that exist in collector.py
3. Creates derived features
4. Saves to CSV for training

Usage:
    # Extract ALL available data (automatic)
    python extract_features_from_datasets.py
    
    # Extract with date range
    python extract_features_from_datasets.py --start-date 2025-10-01 --end-date 2025-11-12
    
    # Custom output path
    python extract_features_from_datasets.py --output data/my_training_data.csv
"""

import os
import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import argparse
from supabase import create_client
from dotenv import load_dotenv
import logging
from typing import Optional, List, Dict

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract ML features from collector.py snapshots."""
    
    # CRITICAL: These must match collector.py exactly!
    FEATURE_MAPPING = {
        # Market features (from collector.py)
        'price_usd': 'price_usd',
        'fdv_usd': 'fdv_usd',
        'liquidity_usd': 'liquidity_usd',
        'volume_h24_usd': 'volume_h24_usd',
        'price_change_h24_pct': 'price_change_h24_pct',
        
        # Security features
        'creator_balance_pct': 'creator_balance_pct',
        'top_10_holders_pct': 'top_10_holders_pct',
        'total_lp_locked_usd': 'total_lp_locked_usd',
        'has_mint_authority': 'has_mint_authority',
        'has_freeze_authority': 'has_freeze_authority',
        'is_lp_locked_95_plus': 'is_lp_locked_95_plus',
        
        # Risk encoding
        'rugcheck_risk_level': 'rugcheck_risk_level',
        
        # Time features
        'time_of_day_utc': 'time_of_day_utc',
        'day_of_week_utc': 'day_of_week_utc',
        'is_weekend_utc': 'is_weekend_utc',
        'is_public_holiday_any': 'is_public_holiday_any',
        
        # Signal features
        'signal_source': 'signal_source',
        'grade': 'grade',
        
        # Token age
        'token_age_at_signal_seconds': 'token_age_at_signal_seconds',
        
        # Timestamp for sorting
        'checked_at_timestamp': 'checked_at_timestamp',
        
        # Identifier
        'mint': 'mint',
    }
    
    def __init__(self):
        self.supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        self.bucket = os.getenv("SUPABASE_BUCKET", "monitor-data")
        
    async def download_dataset_file(self, remote_path: str) -> Optional[Dict]:
        """Download a single dataset JSON from Supabase."""
        try:
            response = await asyncio.to_thread(
                self.supabase.storage.from_(self.bucket).download,
                remote_path
            )
            return json.loads(response)
        except Exception as e:
            logger.debug(f"Failed to download {remote_path}: {e}")
            return None
    
    async def list_all_dataset_folders(self, pipeline: str) -> List[str]:
        """List all date folders in the datasets directory for a pipeline."""
        try:
            folders = await asyncio.to_thread(
                self.supabase.storage.from_(self.bucket).list,
                f"datasets/{pipeline}"
            )
            # Extract folder names that look like dates (YYYY-MM-DD)
            date_folders = []
            for folder in folders:
                name = folder.get('name', '')
                # Skip 'expired_no_label' folder
                if name == 'expired_no_label':
                    continue
                # Check if it matches YYYY-MM-DD pattern
                if len(name) == 10 and name[4] == '-' and name[7] == '-':
                    try:
                        datetime.strptime(name, '%Y-%m-%d')
                        date_folders.append(name)
                    except ValueError:
                        continue
            return sorted(date_folders)
        except Exception as e:
            logger.error(f"Failed to list folders for {pipeline}: {e}")
            return []
    
    async def list_dataset_files(self, pipeline: str, date_str: str) -> List[str]:
        """List all dataset files for a date."""
        folder = f"datasets/{pipeline}/{date_str}"
        try:
            files = await asyncio.to_thread(
                self.supabase.storage.from_(self.bucket).list,
                folder
            )
            return [f['name'] for f in files if f['name'].endswith('.json')]
        except Exception as e:
            return []
    
    def extract_features_from_snapshot(self, snapshot: dict) -> dict:
        """
        Extract ML features from a single snapshot.
        Returns a flat dict with all features needed for training.
        """
        features_dict = {}
        
        # Get features from snapshot
        snapshot_features = snapshot.get('features', {})
        
        # Extract all mapped features
        for ml_feature, snapshot_key in self.FEATURE_MAPPING.items():
            value = snapshot_features.get(snapshot_key)
            
            # Handle boolean -> int conversion
            if isinstance(value, bool):
                value = int(value)
            
            # Handle None
            if value is None:
                if ml_feature in ['price_usd', 'fdv_usd', 'liquidity_usd', 
                                 'volume_h24_usd', 'price_change_h24_pct',
                                 'creator_balance_pct', 'top_10_holders_pct',
                                 'total_lp_locked_usd', 'token_age_at_signal_seconds',
                                 'time_of_day_utc', 'day_of_week_utc',
                                 'checked_at_timestamp']:
                    value = 0.0
                elif ml_feature in ['has_mint_authority', 'has_freeze_authority',
                                   'is_lp_locked_95_plus', 'is_weekend_utc',
                                   'is_public_holiday_any']:
                    value = 0
                elif ml_feature in ['rugcheck_risk_level', 'signal_source', 
                                   'grade', 'mint']:
                    value = 'UNKNOWN'
            
            features_dict[ml_feature] = value
        
        # Add derived features (computed from existing features)
        features_dict['volume_to_liquidity_ratio'] = (
            features_dict['volume_h24_usd'] / features_dict['liquidity_usd']
            if features_dict['liquidity_usd'] > 0 else 0
        )
        
        features_dict['fdv_to_liquidity_ratio'] = (
            features_dict['fdv_usd'] / features_dict['liquidity_usd']
            if features_dict['liquidity_usd'] > 0 else 0
        )
        
        features_dict['is_new_token'] = int(
            features_dict['token_age_at_signal_seconds'] < 43200  # <12 hours
        )
        
        # Add label if available
        label = snapshot.get('label', {})
        if label:
            features_dict['label_status'] = label.get('status', 'unknown')
            features_dict['label_ath_roi'] = label.get('ath_roi', 0)
            features_dict['label_final_roi'] = label.get('final_roi', 0)
            features_dict['label_hit_50_percent'] = int(label.get('hit_50_percent', False))
            features_dict['label_token_age_hours'] = label.get('token_age_hours', 0)
            features_dict['label_tracking_duration_hours'] = label.get('tracking_duration_hours', 0)
        else:
            # Expired/unlabeled
            features_dict['label_status'] = 'expired'
            features_dict['label_ath_roi'] = 0
            features_dict['label_final_roi'] = 0
            features_dict['label_hit_50_percent'] = 0
            features_dict['label_token_age_hours'] = 0
            features_dict['label_tracking_duration_hours'] = 0
        
        return features_dict
    
    async def extract_all_data(self, pipeline: str) -> List[Dict]:
        """
        Extract all datasets for a pipeline (auto-discover all dates).
        Returns list of feature dicts.
        """
        logger.info(f"Discovering all date folders for {pipeline}...")
        date_folders = await self.list_all_dataset_folders(pipeline)
        
        if not date_folders:
            logger.warning(f"No date folders found for {pipeline}")
            return []
        
        logger.info(f"Found {len(date_folders)} date folders: {date_folders[0]} to {date_folders[-1]}")
        
        all_features = []
        
        for date_str in date_folders:
            # List files
            files = await self.list_dataset_files(pipeline, date_str)
            
            if files:
                logger.info(f"Processing {pipeline}/{date_str}: {len(files)} files")
                
                for filename in files:
                    remote_path = f"datasets/{pipeline}/{date_str}/{filename}"
                    snapshot = await self.download_dataset_file(remote_path)
                    
                    if snapshot:
                        try:
                            features = self.extract_features_from_snapshot(snapshot)
                            all_features.append(features)
                        except Exception as e:
                            logger.error(f"Failed to extract features from {filename}: {e}")
        
        logger.info(f"Extracted {len(all_features)} samples from {len(date_folders)} dates")
        return all_features
    
    async def extract_date_range(self, pipeline: str, start_date: str, end_date: str) -> List[Dict]:
        """
        Extract all datasets for a pipeline between two dates.
        Returns list of feature dicts.
        """
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        all_features = []
        dates_processed = 0
        
        current_date = start_dt
        while current_date <= end_dt:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # List files
            files = await self.list_dataset_files(pipeline, date_str)
            
            if files:
                logger.info(f"Processing {pipeline}/{date_str}: {len(files)} files")
                
                for filename in files:
                    remote_path = f"datasets/{pipeline}/{date_str}/{filename}"
                    snapshot = await self.download_dataset_file(remote_path)
                    
                    if snapshot:
                        try:
                            features = self.extract_features_from_snapshot(snapshot)
                            all_features.append(features)
                        except Exception as e:
                            logger.error(f"Failed to extract features from {filename}: {e}")
                
                dates_processed += 1
            
            current_date += timedelta(days=1)
        
        logger.info(f"Extracted {len(all_features)} samples from {dates_processed} dates")
        return all_features
    
    async def create_training_dataset(self, start_date: Optional[str], end_date: Optional[str], output_file: str):
        """
        Create complete training dataset from both pipelines.
        If start_date/end_date are None, extracts ALL available data.
        """
        if start_date and end_date:
            logger.info(f"Creating training dataset: {start_date} to {end_date}")
            # Extract from both pipelines with date range
            logger.info("\n📥 Extracting discovery signals...")
            discovery_features = await self.extract_date_range('discovery', start_date, end_date)
            
            logger.info("\n📥 Extracting alpha signals...")
            alpha_features = await self.extract_date_range('alpha', start_date, end_date)
        else:
            logger.info("Creating training dataset: ALL AVAILABLE DATA")
            # Extract ALL data from both pipelines
            logger.info("\n📥 Extracting ALL discovery signals...")
            discovery_features = await self.extract_all_data('discovery')
            
            logger.info("\n📥 Extracting ALL alpha signals...")
            alpha_features = await self.extract_all_data('alpha')
        
        # Combine
        all_features = discovery_features + alpha_features
        logger.info(f"\n✅ Combined: {len(all_features)} total samples")
        
        if len(all_features) == 0:
            logger.error("No data extracted! Check your Supabase bucket and datasets folder.")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        
        # Remove duplicates (same mint + timestamp)
        if 'mint' in df.columns and 'checked_at_timestamp' in df.columns:
            df = df.drop_duplicates(subset=['mint', 'checked_at_timestamp'])
            logger.info(f"After deduplication: {len(df)} samples")
        
        # Remove unlabeled samples
        df = df[df['label_status'].isin(['win', 'loss'])]
        logger.info(f"After removing unlabeled: {len(df)} samples")
        
        if len(df) == 0:
            logger.warning("No labeled samples (win/loss) found in the data!")
            return None
        
        # Sort by timestamp
        if 'checked_at_timestamp' in df.columns:
            df = df.sort_values('checked_at_timestamp')
        
        # Print statistics
        logger.info("\n📊 Dataset Statistics:")
        logger.info(f"  Total samples: {len(df)}")
        logger.info(f"  Wins: {(df['label_status'] == 'win').sum()} ({(df['label_status'] == 'win').mean()*100:.2f}%)")
        logger.info(f"  Losses: {(df['label_status'] == 'loss').sum()} ({(df['label_status'] == 'loss').mean()*100:.2f}%)")
        
        logger.info("\n📊 By Signal Source:")
        if 'signal_source' in df.columns:
            for source in df['signal_source'].unique():
                source_df = df[df['signal_source'] == source]
                win_rate = (source_df['label_status'] == 'win').mean()
                logger.info(f"  {source}: {len(source_df)} samples, {win_rate*100:.2f}% win rate")
        
        logger.info("\n📊 By Grade:")
        if 'grade' in df.columns:
            for grade in sorted(df['grade'].unique()):
                grade_df = df[df['grade'] == grade]
                win_rate = (grade_df['label_status'] == 'win').mean()
                logger.info(f"  {grade}: {len(grade_df)} samples, {win_rate*100:.2f}% win rate")
        
        # Save
        df.to_csv(output_file, index=False)
        logger.info(f"\n✅ Saved to: {output_file}")
        
        # Print feature list
        logger.info("\n📋 Features extracted:")
        feature_cols = [c for c in df.columns if c not in ['label_status', 'label_ath_roi', 
                                                            'label_final_roi', 'label_hit_50_percent',
                                                            'label_token_age_hours', 
                                                            'label_tracking_duration_hours']]
        for i, col in enumerate(feature_cols, 1):
            logger.info(f"  {i:2}. {col}")
        
        return df


async def main():
    parser = argparse.ArgumentParser(
        description="Extract training features from Supabase datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract ALL available data (automatic discovery)
  python extract_features_from_datasets.py
  
  # Extract with custom output path
  python extract_features_from_datasets.py --output data/my_dataset.csv
  
  # Extract specific date range
  python extract_features_from_datasets.py --start-date 2025-10-01 --end-date 2025-11-12
        """
    )
    parser.add_argument('--output', type=str, default='data/token_datasets.csv', 
                       help='Output CSV file (default: data/token_datasets.csv)')
    parser.add_argument('--start-date', type=str, default=None, 
                       help='Optional: Start date YYYY-MM-DD (extracts all if not provided)')
    parser.add_argument('--end-date', type=str, default=None, 
                       help='Optional: End date YYYY-MM-DD (extracts all if not provided)')
    
    args = parser.parse_args()
    
    # Validate date arguments
    if (args.start_date and not args.end_date) or (args.end_date and not args.start_date):
        logger.error("Both --start-date and --end-date must be provided together, or neither.")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    # Extract features
    extractor = FeatureExtractor()
    df = await extractor.create_training_dataset(
        args.start_date,
        args.end_date,
        args.output
    )
    
    if df is not None and len(df) > 0:
        print("\n" + "="*60)
        print("✅ FEATURE EXTRACTION COMPLETE!")
        print("="*60)
        print(f"\n📁 Dataset saved: {args.output}")
        print(f"📊 Total samples: {len(df)}")
        print(f"🎯 Win rate: {(df['label_status'] == 'win').mean()*100:.2f}%")
        print("\nNext step: Run the training notebook!")
    else:
        print("\n" + "="*60)
        print("⚠️  NO DATA EXTRACTED")
        print("="*60)
        print("\nPlease check:")
        print("  1. Supabase credentials in .env file")
        print("  2. datasets/ folder exists in your bucket")
        print("  3. Date folders contain .json files")
        print("  4. Files have been labeled (status: win/loss)")


if __name__ == "__main__":
    asyncio.run(main())