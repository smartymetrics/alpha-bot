# Quick Start: Signal-Type-Aware Models

## Two Model Systems

### 🟦 Standard Model (Current)
```bash
python train_model.py
```
- Saves to: `models/`
- No signal_type needed
- Works with existing code

### 🟩 Signal-Aware Model (New)
```bash
python train_model_signal_aware.py
```
- Saves to: `models/signal_aware/`
- Requires signal_type in dataset
- Better alpha vs discovery performance

---

## Usage (Auto-Switches)

```python
from ml_predictor import SolanaTokenPredictor

predictor = SolanaTokenPredictor()

# If using signal-aware models (models/signal_aware/ exists):
result = predictor.predict(mint, signal_type='alpha')      # REQUIRED
result = predictor.predict(mint, signal_type='discovery')  # REQUIRED

# If using standard models (models/ only):
result = predictor.predict(mint)                           # Optional
result = predictor.predict(mint, signal_type='alpha')      # Works too
```

---

## Dataset Requirements for Signal-Aware Training

**Column**: `signal_type`  
**Valid Values**: `'alpha'` or `'discovery'` only  
**Missing Values**: NOT allowed  
**Default Values**: NOT allowed  

```python
# ✅ Valid
df['signal_type'] = 'alpha'    # All alpha
df['signal_type'] = 'discovery'  # All discovery
df['signal_type'] = ['alpha', 'discovery', 'alpha', ...]  # Mixed

# ❌ Invalid
df['signal_type'] = 'alpha_v2'   # Wrong value
df['signal_type'] = None         # NaN not allowed
df['signal_type'] = [None, 'alpha', 'discovery']  # Has NaN
```

---

## Training Flowchart

```
Start: Do you have signal_type labels?
  ├─ NO  → python train_model.py → models/
  │        (standard model, works fine)
  │
  └─ YES → Add signal_type column to dataset
             ├─ python train_model_signal_aware.py
             └─ models/signal_aware/ created
                (system auto-uses this if it exists)
```

---

## How System Decides

```python
if os.path.exists('models/signal_aware/xgboost_model.pkl'):
    → Use signal-aware models (REQUIRES signal_type)
else:
    → Use standard models (signal_type optional)
```

---

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| "signal_type required: 'alpha' or 'discovery'" | Using signal-aware models without param | Add `signal_type='alpha'` to call |
| "Invalid signal_type values" | Dataset has 'alpha_v2', 'new', etc | Only use 'alpha' or 'discovery' |
| "NaN signal_type" | Dataset has missing values | Fill all signal_type values |

---

## Integration Points

### winner_monitor.py (Alpha Signals)
```python
result = self.ml_classifier.predict(
    mint,
    signal_type='alpha'  # Winner wallet overlap
)
```

### token_monitor.py (Discovery Signals)
```python
result = self.ml_classifier.predict(
    mint,
    signal_type='discovery'  # Fresh token detection
)
```

---

## Comparison

| Feature | Standard | Signal-Aware |
|---------|----------|--------------|
| Works without signal_type | ✅ Yes | ❌ No |
| Learns alpha patterns | ❌ No | ✅ Yes |
| Learns discovery patterns | ❌ No | ✅ Yes |
| Training time | ~2-3 min | ~2-3 min |
| Fallback available | N/A | ✅ Yes (to standard) |

---

## One-Minute Setup

1. **Add signal_type to dataset**:
   ```python
   df['signal_type'] = 'alpha'  # or 'discovery'
   df.to_csv('data/token_datasets.csv', index=False)
   ```

2. **Train signal-aware model**:
   ```bash
   python train_model_signal_aware.py
   ```

3. **Update calls**:
   ```python
   # Now requires signal_type
   result = predictor.predict(mint, signal_type='alpha')
   ```

Done! System auto-detects and uses signal-aware models.

---

## Monitoring

Check which model is active:

```python
predictor = SolanaTokenPredictor()
print(f"Signal-aware: {predictor.use_signal_aware}")
print(f"Model dir: {predictor.model_dir}")
print(f"Requires signal_type: {predictor.requires_signal_type}")
```

Check if signal_type_alpha was selected:

```python
import json
metadata = json.load(open('models/signal_aware/model_metadata.json'))
print(f"signal_type_alpha selected: {metadata['signal_type_alpha_selected']}")
```

---

## Summary

- ✅ Standard model still works (no changes needed)
- ✅ Signal-aware model auto-detects when available
- ✅ Both work with same predictor class
- ✅ Easy to switch between them
- ✅ Backward compatible with existing code

No breaking changes. Just add signal_type to dataset when ready!
