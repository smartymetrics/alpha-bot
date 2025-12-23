# Signal-Type-Aware ML Prediction (Option 2: Feature-Based)

## Overview
The ML model now learns to recognize and adapt to **different signal types** (ALPHA vs DISCOVERY) by including `signal_type_alpha` as an engineered feature. This allows the model to learn different win-rate patterns for winner wallet overlap signals vs fresh token discovery signals.

## Architecture

### Single Unified Model with Signal-Type Feature

The model includes `signal_type_alpha` as one of the engineered features:
- **1.0** = ALPHA signal (winner wallet overlap)
- **0.0** = DISCOVERY signal (fresh token detection)
- **0.5** = Unknown/unspecified signal type

This lets the ensemble learn different decision boundaries for each signal type while maintaining **backward compatibility**.

## How It Works

### 1. Feature Engineering
During training and inference, the `signal_type_alpha` feature is automatically computed:

```python
if signal_type and signal_type.upper() == 'ALPHA':
    features['signal_type_alpha'] = 1.0
elif signal_type and signal_type.upper() == 'DISCOVERY':
    features['signal_type_alpha'] = 0.0
else:
    features['signal_type_alpha'] = 0.5  # Default for unknown/none
```

### 2. Model Training
The feature is included in `DERIVED_FEATURES` in train_model.py:
```python
DERIVED_FEATURES = [
    ...
    'signal_type_alpha',  # 1=alpha, 0=discovery, 0.5=unknown
    ...
]
```

During feature selection, the model automatically determines the importance of signal_type relative to other features.

### 3. Prediction Routing

**ALPHA signals (winner wallet overlap):**
```python
ml_prediction_result = predictor.predict(
    mint, 
    threshold=0.50,
    action_threshold=0.70,
    signal_type='alpha'  # Sets signal_type_alpha=1.0
)
```

**DISCOVERY signals (fresh tokens):**
```python
ml_prediction_result = predictor.predict(
    mint, 
    threshold=0.50,
    action_threshold=0.70,
    signal_type='discovery'  # Sets signal_type_alpha=0.0
)
```

**Legacy services (no signal type):**
```python
ml_prediction_result = predictor.predict(
    mint, 
    threshold=0.50,
    action_threshold=0.70
    # signal_type not specified = sets signal_type_alpha=0.5
)
```

## Benefits

✅ **Backward compatible** - Existing services work without changes  
✅ **Single model** - Cleaner infrastructure, easier maintenance  
✅ **Automatic learning** - Model learns signal-type patterns during training  
✅ **Flexible** - Can call with or without signal_type parameter  
✅ **Transparent** - Feature importance shows if signal_type matters  

## Implementation Details

### For Existing Unified Model
Your current `models/` directory works **as-is**. Signal_type feature will be added on next training run.

### For Next Training Run
1. Ensure your `token_datasets.csv` includes a `signal_type` or similar column, OR
2. The model will auto-fill with 0.5 (unknown) if column is missing
3. Run `train_model.py` as usual - signal_type feature is automatically included
4. Retraining will learn different patterns for each signal type

## Threshold Management

**Two thresholds control behavior:**

1. **ML_PREDICTION_THRESHOLD** (0.50)
   - Filters which tokens get included in results
   - `ml_passed = probability >= 0.50`

2. **ML_ACTION_THRESHOLD** (0.70)
   - Determines the action label
   - BUY: >= 0.70
   - CONSIDER: 0.60-0.69
   - SKIP: 0.45-0.59
   - AVOID: < 0.45

Both can be adjusted without retraining or changing code.

## Migration Path

### Phase 1: Current (No Changes Needed)
- Existing services call predictor without signal_type
- signal_type_alpha defaults to 0.5 (neutral)
- Everything works as before

### Phase 2: Next Training (Add Signal Type)
- If you have signal type labels in historical data:
  - Add `signal_type` column to training CSV
  - Set to 'alpha' or 'discovery' based on source
- Run `train_model.py`
- Model learns different patterns automatically

### Phase 3: Optional - Update Callers
- Update winner_monitor.py to pass `signal_type='alpha'`
- Update token_monitor.py to pass `signal_type='discovery'`
- Gets better predictions with explicit signal type

## Why Option 2?

| Aspect | Option 1 (Dual Models) | Option 2 (Signal Feature) |
|--------|------------------------|---------------------------|
| Breaking Changes | ⚠️ Yes | ✅ No |
| Infrastructure | Complex | Simple |
| Backward Compat | No | Yes |
| Single Model | No | Yes |
| Learning | Separate | Unified |
| Flexibility | Fixed | Adaptive |

**Option 2 is cleaner** because the ensemble can learn trade-offs between signals automatically, rather than requiring explicit model separation.

