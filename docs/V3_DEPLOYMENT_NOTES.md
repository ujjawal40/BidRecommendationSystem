# v3 Fee-Sensitive Win Probability — Deployment Notes

## What v3 Is

A second Phase 1B win-probability model trained alongside v2. It exists
to provide a real fee-elasticity curve when v2 saturates. It does **not**
replace v2 as the primary classifier.

## Why It Exists

v2 (AUC 0.948, isotonic-calibrated) is saturated at inference: raw model
output is ~0.97 for any realistic input, calibrator clips to 0.95. As a
result the EV optimizer always picked the highest fee in the sweep.

The previous fix was a hand-tuned heuristic sigmoid (in
`api/enhanced_prediction_service.py::_heuristic_win_prob`). It worked but
was anchored on hard-coded `P_HIGH=0.80 / P_LOW=0.22 / k=3.5`.

v3_fee_sensitive replaces that hand-tuned curve with a learned model.

## Three v3 Variants Trained (2026-05-13)

| Variant | AUC | Span at Inference | Status |
|---------|-----|-------------------|--------|
| v3 default | 0.936 | ~0pp (saturated) | not deployed |
| v3 leaky (run #2) | 0.9998 | 0pp | archived — leakage via `land_acres_log` |
| v3_fee_sensitive | 0.883 | 24-37pp | **DEPLOYED as saturation fallback** |

v3_fee_sensitive was trained with `V3_DROP_CONTEXT_FEE=1`, which excludes
13 absolute-fee context aggregates that crowd out `BidFee` as a signal.

## Files

```
outputs/models/
  lightgbm_win_probability_v3_fee_sensitive.txt          (booster)
  lightgbm_win_probability_v3_fee_sensitive_metadata.json
  win_probability_v3_fee_sensitive_calibrator.pkl        (gitignored)

outputs/reports/
  win_rate_lookups_v3.json                               (API inference lookups)
  win_probability_v3_fee_sensitive_feature_importance.csv

outputs/figures/
  win_probability_v3_fee_sensitivity.png                 (training curves)
  v2_vs_v3_fee_curve.png                                 (4-scenario comparison)
```

## How It Plugs Into the API

`api/enhanced_prediction_service.py`:

1. `_load_v3_fee_sensitive()` — loads model + calibrator + lookups at startup
2. `_populate_v3_features()` — looks up the six LOO win-rate features at
   predict time using the categoricals from the request
3. `_predict_v3_fee_sensitive()` — single-fee inference; recomputes ALL
   fee-relative features per fee (this was the bug fix that finally got
   the curve non-flat at inference)
4. `_predict_win_probability()` — when v2 raw >= SATURATION_THRESHOLD,
   calls v3 instead of `_heuristic_win_prob`
5. `get_fee_sensitivity_curve()` — when the per-point curve is still flat
   after the per-call saturation routing, rebuilds it via v3
6. `_run_startup_canary()` — extended with a v3 verification pass

## Regenerating

Locally, after fresh data preprocessing:

```bash
# v3 default (saved as lightgbm_win_probability_v3.txt)
python scripts/29_fee_sensitive_win_prob.py

# v3 fee-sensitive — the deployed variant
V3_DROP_CONTEXT_FEE=1 python scripts/29_fee_sensitive_win_prob.py
```

Both runs read `data/features/BidData_features_v2.csv` (NOT the raw
enriched CSV — that was the original script bug that ate 24 v2 features).

To compare curves side-by-side after training:

```bash
python scripts/29b_v2_vs_v3_curve_comparison.py
```

## Render Deployment Checklist

1. v3 model files must be committed to the repo
   (`lightgbm_win_probability_v3_fee_sensitive.txt` + metadata JSON)
2. `outputs/reports/win_rate_lookups_v3.json` must be committed
3. The `.pkl` calibrator is gitignored — Render will work without it
   (uses raw model output, slightly less calibrated, still elastic)
4. `/api/health` should return `v3_fee_sensitive_loaded: true`
5. `/api/debug` exposes file presence for v3 dependencies
6. Watch startup logs for `[EnhancedPredictor] v3_fee_sensitive canary OK`

## Failure Modes

- **v3 lookups JSON missing** → v3 still loads, but every category maps
  to the global win rate. Fee sensitivity remains but loses category
  specificity.
- **v3 .pkl calibrator missing** → uses raw booster output. Still clipped
  to `[0.05, 0.95]`. Probably reasonable, but Brier may be slightly worse.
- **v3 .txt model missing** → silently falls back to the heuristic
  sigmoid (same as before this change). Startup canary logs which
  fallback is active.
- **Feature alignment break** → v3 canary will log "v3_fee_sensitive
  curve is FLAT" warning at startup. This was the bug that bit v2 (24 of
  44 features silently zeroed); same guardrail now exists for v3.

## Known Limitations

- Some inputs still saturate v3 (Litigation/CA, Consulting/NY with
  certain office locations). These reflect a real high-baseline data
  pattern — the segment + region has ~90% historical win rate, so the
  model predicts ~95% regardless of fee. Not a bug.
- v3 was trained on the same time-based 60/20/20 split as v2. Same
  drift assumptions apply.
- AUC of 0.883 means v3 is a worse ranker than v2. We accept this for
  the EV-curve use case; the primary classifier (used for the top-line
  win probability shown to users) remains v2 when v2 isn't saturated.
