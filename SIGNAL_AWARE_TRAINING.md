# Signal-Type-Aware ML Training & Inference

## Overview

The system now supports **two separate model sets** for maximum flexibility:

1. **Standard Model** (`models/`) - Your original unified model
2. **Signal-Aware Model** (`models/signal_aware/`) - NEW, learns alpha vs discovery patterns separately

This allows you to:
- Keep your existing pipeline working without any changes
- Train a signal-type-aware model when your dataset is ready
- Switch between models seamlessly with automatic detection

---

## Model Sets

### 🔵 Standard Model (models/)

**Status**: Backward compatible, always works  
**Training Script**: `train_model.py` (unchanged)  
**Features**: All features EXCEPT signal_type_alpha  
**signal_type Parameter**: Optional (defaults to 0.5 = neutral)

**When to use**:
- You don't have signal_type labels in your dataset
- You want to keep using existing trained models
- Your service doesn't call with signal_type

**Example**:
```python
predictor = SolanaTokenPredictor()
result = predictor.predict(mint)  # Works fine, no signal_type needed
```

---

### 🟢 Signal-Aware Model (models/signal_aware/)

**Status**: Requires complete setup  
**Training Script**: `train_model_signal_aware.py` (NEW)  
**Features**: Includes `signal_type_alpha` feature (1.0=alpha, 0.0=discovery)  
**signal_type Parameter**: **REQUIRED** ('alpha' or 'discovery')

**When to use**:
- You have signal_type labels for all tokens in your dataset
- You want better performance on alpha vs discovery signals
- You're ready to use explicit signal classification

**Requirements**:
- Training dataset must have 'signal_type' column
- Values must be 'alpha' or 'discovery' (NO missing values)
- NO default values or NaN allowed

**Example**:
```python
predictor = SolanaTokenPredictor()
result = predictor.predict(mint, signal_type='alpha')  # REQUIRED
result = predictor.predict(mint, signal_type='discovery')  # REQUIRED
```

---

## How It Works

### Automatic Detection

When you initialize `SolanaTokenPredictor()`, it:

1. **Checks** if `models/signal_aware/` exists with trained models
2. **If YES** → Loads signal-aware models, requires signal_type
3. **If NO** → Falls back to `models/`, signal_type is optional

```python
class SolanaTokenPredictor:
    def __init__(self):
        if os.path.exists('models/signal_aware/xgboost_model.pkl'):
            self.use_signal_aware = True
            self.requires_signal_type = True
        else:
            self.use_signal_aware = False
            self.requires_signal_type = False
```

---

## Training Instructions

### Train Standard Model (Keep Using Existing)

```bash
python train_model.py
```

- Uses dataset without signal_type column
- Trains single unified ensemble
- Saves to `models/`
- No changes needed, backward compatible

### Train Signal-Aware Model (NEW)

```bash
python train_model_signal_aware.py
```

**Requirements**:
1. Your `data/token_datasets.csv` must have `signal_type` column
2. Valid values: `'alpha'` or `'discovery'` only
3. No NaN values allowed
4. All rows must be labeled

**Process**:
- Validates dataset
- Creates `signal_type_alpha` feature (1.0/0.0)
- Trains on stratified split (preserves signal_type distribution)
- Saves to `models/signal_aware/`
- Generates metadata with signal_type awareness flag

**Output**:
```
models/signal_aware/
├── xgboost_model.pkl
├── lightgbm_model.pkl
├── catboost_model.pkl
├── rf_model.pkl
├── feature_selector.pkl
├── scaler.pkl
├── isolation_forest.pkl
└── model_metadata.json
```

---

## Feature Selection

### Both models use SelectKBest (k=20)

The top 20 features are automatically selected during training.

**Signal-Aware Specific**: If `signal_type_alpha` is selected, the model learns different decision boundaries for alpha vs discovery signals.

Check metadata to see if it was selected:
```python
metadata = json.load(open('models/signal_aware/model_metadata.json'))
if metadata['signal_type_alpha_selected']:
    print("✅ Model learned signal-type patterns!")
else:
    print("⚠️ signal_type_alpha not selected, model may not differentiate")
```

---

## Migration Path

### Phase 1: Current State
- Using `models/` with standard model
- Services call `predict(mint)` without signal_type
- Everything works, no changes needed

### Phase 2: Add Signal Types to Dataset
- Collect signal_type labels for your tokens
- 'alpha' = from winner wallet overlap (winner_monitor.py)
- 'discovery' = from fresh token detection (token_monitor.py)
- Add column to `data/token_datasets.csv`

### Phase 3: Train Signal-Aware Model
- Run `python train_model_signal_aware.py`
- Creates `models/signal_aware/` directory
- System auto-detects and switches to signal-aware mode

### Phase 4: Update Integration Points
- Update `winner_monitor.py`: pass `signal_type='alpha'`
- Update `token_monitor.py`: pass `signal_type='discovery'`
- External services: automatically switch once signal-aware models detected

---

## Integration Examples

### winner_monitor.py

```python
# Current (works with both models)
ml_prediction_result = self.ml_classifier.predict(
    mint, 
    threshold=ML_PREDICTION_THRESHOLD,
    action_threshold=ML_ACTION_THRESHOLD,
    signal_type='alpha'  # Pass explicit signal type
)

# If using signal-aware models: signal_type is REQUIRED
# If using standard models: signal_type is used as feature (optional)
```

### token_monitor.py

```python
# Current (works with both models)
ml_prediction_result = self.ml_classifier.predict(
    mint, 
    threshold=ML_PREDICTION_THRESHOLD,
    action_threshold=ML_ACTION_THRESHOLD,
    signal_type='discovery'  # Pass explicit signal type
)

# If using signal-aware models: signal_type is REQUIRED
# If using standard models: signal_type is used as feature (optional)
```

### External Service (No Changes Needed)

```python
# Works with standard models (signal_type optional)
predictor = SolanaTokenPredictor()
result = predictor.predict(mint)  # ✅ Works

# If you switch to signal-aware models later...
# You'll get error requiring signal_type
# Update to: result = predictor.predict(mint, signal_type='alpha')
```

---

## Technical Details

### signal_type_alpha Feature

**Encoding**:
- ALPHA tokens: `signal_type_alpha = 1.0`
- DISCOVERY tokens: `signal_type_alpha = 0.0`
- Unknown (standard model): `signal_type_alpha = 0.5`

**Why This Works**:
- Lets ensemble learn different win-rate patterns
- Alpha (winner wallets) may have different characteristics
- Discovery (fresh tokens) may have different risk/reward
- Single model learns both patterns with this feature flag

### Model Selection (SelectKBest)

Each training run selects top 20 features via f_classif scoring:

```python
selector = SelectKBest(score_func=f_classif, k=20)
selected_features = selector.fit_transform(X_train, y_train)
```

**For signal-aware models**: If `signal_type_alpha` gets selected (top 20 score), the model will definitely use signal-type patterns. If not selected, it's still available but not in top features.

---

## Comparison Table

| Aspect | Standard Model | Signal-Aware Model |
|--------|---------------|--------------------|
| **Directory** | `models/` | `models/signal_aware/` |
| **Training Script** | `train_model.py` | `train_model_signal_aware.py` |
| **Requires signal_type** | ❌ No | ✅ Yes |
| **signal_type Values** | N/A | 'alpha', 'discovery' |
| **signal_type_alpha Feature** | ❌ Missing | ✅ Included (1.0/0.0) |
| **Learns Signal Patterns** | ❌ No | ✅ Yes |
| **Breaking Changes** | ❌ None | ✅ Requires signal_type param |
| **Dataset Requirement** | No signal_type column | signal_type column required |
| **Auto-Fallback** | (default) | Falls back to standard if not found |

---

## Troubleshooting

### Error: "signal_type required: use 'alpha' or 'discovery'"

**Cause**: Using signal-aware models without providing signal_type

**Fix**:
```python
# ❌ This fails with signal-aware models
result = predictor.predict(mint)

# ✅ This works
result = predictor.predict(mint, signal_type='alpha')
```

### Error: "Could not load scaler from signal_aware model"

**Cause**: Signal-aware model missing scaler.pkl (shouldn't happen)

**Fix**: Retrain signal-aware model:
```bash
python train_model_signal_aware.py
```

### signal_type_alpha Not Selected

**Check metadata**:
```python
metadata = json.load(open('models/signal_aware/model_metadata.json'))
print(metadata['signal_type_alpha_selected'])  # False
```

**Why**: Feature selection algorithm didn't include it in top 20

**Is this a problem?**: No, just means other features were more predictive. Signal-type still helps via the 0.5 default encoding.

---

## Best Practices

1. **Keep standard model trained**: You'll always have a fallback
2. **Update dataset gradually**: Label signal_types incrementally
3. **Test signal-aware model**: Compare AUC before deploying
4. **Monitor both paths**: Have fallback in case signal-aware performs worse
5. **Retrain regularly**: Both models benefit from fresh data

---

## Next Steps

1. ✅ Run `train_model.py` → creates standard `models/`
2. 📝 Add signal_type labels to your dataset
3. 🚀 Run `train_model_signal_aware.py` → creates `models/signal_aware/`
4. 🔄 System auto-detects and uses signal-aware models
5. 📊 Monitor performance improvements

---

*Last Updated: December 2025*  
*Supports both standard and signal-aware model sets seamlessly*
