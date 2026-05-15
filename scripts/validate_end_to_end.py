"""
End-to-End Validation & Debug Script
=====================================
Comprehensive checks across API, model files, and prediction behaviour.
Runs without needing a live API server (direct model calls).
Also optionally hits a live server if --url is provided.

Usage:
    # Direct model validation (no server needed):
    python scripts/validate_end_to_end.py

    # Also test against a live API:
    python scripts/validate_end_to_end.py --url http://localhost:5001

    # Verbose mode (print prediction details for every check):
    python scripts/validate_end_to_end.py --verbose

Exit code: 0 if all checks pass, 1 if any FAIL.

Author: Raj Tiwari
Date: 2026-02-25
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

# ============================================================================
# HELPERS
# ============================================================================

PASS  = "\033[92m✓ PASS\033[0m"
FAIL  = "\033[91m✗ FAIL\033[0m"
WARN  = "\033[93m! WARN\033[0m"
INFO  = "\033[96m  INFO\033[0m"


class Results:
    def __init__(self):
        self.passed  = 0
        self.failed  = 0
        self.warned  = 0
        self._log    = []

    def ok(self, name, detail=""):
        self.passed += 1
        self._log.append((PASS, name, detail))
        print(f"  {PASS}  {name}" + (f" — {detail}" if detail else ""))

    def fail(self, name, detail=""):
        self.failed += 1
        self._log.append((FAIL, name, detail))
        print(f"  {FAIL}  {name}" + (f" — {detail}" if detail else ""))

    def warn(self, name, detail=""):
        self.warned += 1
        self._log.append((WARN, name, detail))
        print(f"  {WARN}  {name}" + (f" — {detail}" if detail else ""))

    def info(self, detail):
        print(f"  {INFO}  {detail}")

    def summary(self):
        print(f"\n{'='*60}")
        print(f"  PASSED:  {self.passed}")
        print(f"  FAILED:  {self.failed}")
        print(f"  WARNED:  {self.warned}")
        print(f"{'='*60}")
        return self.failed == 0


R = Results()


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ============================================================================
# SECTION 1 — MODEL FILES
# ============================================================================

def check_model_files():
    section("1. MODEL FILES")
    from config.model_config import MODELS_DIR, REPORTS_DIR

    required = {
        # v2 models (production)
        "lightgbm_bidfee_v2_model.txt":                  MODELS_DIR,
        "lightgbm_win_probability_v2.txt":               MODELS_DIR,
        "lightgbm_win_probability_v2_metadata.json":     MODELS_DIR,
        "win_probability_v2_calibrator.pkl":             MODELS_DIR,
        # Reports consumed by API
        "api_precomputed_stats_v2.json":                 REPORTS_DIR,
        "feature_defaults_v2.json":                      REPORTS_DIR,
        "empirical_bands.json":                          REPORTS_DIR,
    }
    optional = {
        "lightgbm_win_probability_v3.txt":               MODELS_DIR,
        "win_rate_lookups_v3.json":                      REPORTS_DIR,
    }

    for fname, base_dir in required.items():
        path = base_dir / fname
        if path.exists():
            size_kb = path.stat().st_size / 1024
            R.ok(fname, f"{size_kb:.0f} KB")
        else:
            R.fail(fname, f"NOT FOUND at {path}")

    for fname, base_dir in optional.items():
        path = base_dir / fname
        if path.exists():
            R.ok(fname, "optional — present")
        else:
            R.warn(fname, "optional — not found")


# ============================================================================
# SECTION 2 — MODEL LOADING
# ============================================================================

def check_model_loading():
    section("2. MODEL LOADING")
    try:
        from api.enhanced_prediction_service import EnhancedBidPredictor
        predictor = EnhancedBidPredictor()
        R.ok("EnhancedBidPredictor init")
        return predictor
    except Exception as e:
        R.fail("EnhancedBidPredictor init", str(e))
        traceback.print_exc()
        return None


# ============================================================================
# SECTION 3 — BASIC PREDICTIONS
# ============================================================================

BASIC_CASES = [
    {
        "name": "Financing / Illinois / Multifamily",
        "kwargs": dict(business_segment="Financing", property_type="Multifamily",
                       property_state="Illinois", target_time=30),
        "expect": {"predicted_fee": (1000, 8000), "win_prob_pct": (20, 95)},
    },
    {
        "name": "Consulting / California / Commercial",
        "kwargs": dict(business_segment="Consulting", property_type="Commercial",
                       property_state="California", target_time=45),
        "expect": {"predicted_fee": (500, 15000), "win_prob_pct": (20, 95)},
    },
    {
        "name": "Estate / New York / Residential",
        "kwargs": dict(business_segment="Estate/Gift Tax", property_type="Residential",
                       property_state="New York", target_time=21),
        "expect": {"predicted_fee": (500, 10000), "win_prob_pct": (10, 95)},
    },
    {
        "name": "Financing / Texas / Multifamily + Chicago office",
        "kwargs": dict(business_segment="Financing", property_type="Multifamily",
                       property_state="Texas", target_time=30,
                       office_location="Chicago", sub_property_type="Conventional"),
        "expect": {"predicted_fee": (1000, 10000), "win_prob_pct": (20, 95)},
    },
    {
        "name": "Financial Reporting / New Jersey / Office",
        "kwargs": dict(business_segment="Financial Reporting", property_type="Commercial",
                       property_state="New Jersey", target_time=60),
        "expect": {"predicted_fee": (500, 12000), "win_prob_pct": (10, 95)},
    },
    {
        "name": "Short turnaround — 7 days",
        "kwargs": dict(business_segment="Financing", property_type="Multifamily",
                       property_state="Illinois", target_time=7),
        "expect": {"predicted_fee": (500, 10000), "win_prob_pct": (10, 95)},
    },
    {
        "name": "Long turnaround — 120 days",
        "kwargs": dict(business_segment="Financing", property_type="Multifamily",
                       property_state="Illinois", target_time=120),
        "expect": {"predicted_fee": (1000, 15000), "win_prob_pct": (10, 95)},
    },
    {
        "name": "With open_date (Dec 2023 — winter seasonality)",
        "kwargs": dict(business_segment="Financing", property_type="Multifamily",
                       property_state="Illinois", target_time=30, open_date="2023-12-15"),
        "expect": {"predicted_fee": (1000, 8000), "win_prob_pct": (10, 95)},
    },
]


def check_basic_predictions(predictor, verbose=False):
    section("3. BASIC PREDICTIONS")

    results = []
    for case in BASIC_CASES:
        try:
            t0 = time.time()
            result = predictor.predict(**case["kwargs"])
            elapsed_ms = (time.time() - t0) * 1000

            fee = result["predicted_fee"]
            wp  = result["win_probability"]["probability_pct"]
            low = result["confidence_interval"]["low"]
            high = result["confidence_interval"]["high"]
            ev  = result["expected_value"]

            detail = f"fee=${fee:,.0f}  win={wp:.0f}%  EV=${ev:,.0f}  [{elapsed_ms:.0f}ms]"

            fee_ok = case["expect"]["predicted_fee"][0] <= fee <= case["expect"]["predicted_fee"][1]
            wp_ok  = case["expect"]["win_prob_pct"][0] <= wp <= case["expect"]["win_prob_pct"][1]
            band_ok = low <= fee <= high  # predicted fee should be inside its own confidence band

            if fee_ok and wp_ok and band_ok:
                R.ok(case["name"], detail)
            else:
                issues = []
                if not fee_ok:
                    issues.append(f"fee={fee:.0f} outside [{case['expect']['predicted_fee'][0]}-{case['expect']['predicted_fee'][1]}]")
                if not wp_ok:
                    issues.append(f"win_prob={wp:.0f}% outside [{case['expect']['win_prob_pct'][0]}-{case['expect']['win_prob_pct'][1]}]%")
                if not band_ok:
                    issues.append(f"fee not in CI [{low:.0f}-{high:.0f}]")
                R.fail(case["name"], "; ".join(issues))

            results.append(result)

            if verbose:
                R.info(f"  raw result: {json.dumps({k: result[k] for k in ['predicted_fee','win_probability','expected_value','confidence_level']}, default=str)}")

        except Exception as e:
            R.fail(case["name"], f"Exception: {e}")
            traceback.print_exc()
            results.append(None)

    return results


# ============================================================================
# SECTION 4 — WIN PROBABILITY DISTRIBUTION
# ============================================================================

def check_win_prob_distribution(predictor):
    """
    Win probabilities should be distributed across a reasonable range —
    NOT all the same value (which was the symptom of the 24-feature skew bug).
    """
    section("4. WIN PROBABILITY DISTRIBUTION")

    segments = ["Financing", "Consulting", "Financial Reporting", "Estate/Gift Tax", "Litigation"]
    states   = ["Illinois", "California", "Texas", "New York", "Florida"]
    ptypes   = ["Multifamily", "Commercial", "Residential"]

    probs = []
    for seg in segments:
        for state in states:
            for ptype in ptypes:
                try:
                    r = predictor.predict(
                        business_segment=seg, property_type=ptype,
                        property_state=state, target_time=30
                    )
                    probs.append(r["win_probability"]["probability_pct"])
                except Exception:
                    pass

    if not probs:
        R.fail("Win prob distribution", "No predictions succeeded")
        return

    probs = np.array(probs)
    spread = float(probs.max() - probs.min())
    std    = float(probs.std())
    pct_at_ceiling = float((probs >= 94.9).mean() * 100)
    pct_at_floor   = float((probs <= 5.1).mean()  * 100)

    R.info(f"n={len(probs)}  min={probs.min():.1f}%  max={probs.max():.1f}%  "
           f"mean={probs.mean():.1f}%  std={std:.1f}pp")
    R.info(f"At ceiling (≥95%): {pct_at_ceiling:.0f}%  At floor (≤5%): {pct_at_floor:.0f}%")

    # PASS criteria
    if spread < 10:
        R.fail("Win prob spread", f"Only {spread:.1f}pp spread — likely feature alignment bug (expect ≥30pp)")
    elif spread < 20:
        R.warn("Win prob spread", f"{spread:.1f}pp spread — low, check feature alignment")
    else:
        R.ok("Win prob spread", f"{spread:.1f}pp spread across {len(probs)} combinations")

    if pct_at_ceiling > 50:
        R.fail("Win prob ceiling", f"{pct_at_ceiling:.0f}% of predictions at 95% ceiling — feature alignment bug")
    elif pct_at_ceiling > 20:
        R.warn("Win prob ceiling", f"{pct_at_ceiling:.0f}% at ceiling")
    else:
        R.ok("Win prob ceiling fraction", f"Only {pct_at_ceiling:.0f}% at ceiling")

    if std < 5:
        R.warn("Win prob std dev", f"{std:.1f}pp — very low, model may not be differentiating inputs")
    else:
        R.ok("Win prob std dev", f"{std:.1f}pp")


# ============================================================================
# SECTION 5 — FEE SENSITIVITY
# ============================================================================

def check_fee_sensitivity(predictor):
    """
    Win probability should change meaningfully when fee is varied.
    The empirical truth: Q4 fees (top 25%) have materially lower win rates (32% vs 55-57%).
    We check that win prob at 2× market average is lower than at 0.5× market average.
    """
    section("5. FEE SENSITIVITY")

    test_cases = [
        ("Financing", "Multifamily", "Illinois", 30),
        ("Consulting", "Commercial", "California", 45),
        ("Financial Reporting", "Commercial", "New York", 60),
    ]

    for seg, ptype, state, ttime in test_cases:
        try:
            # Get base result to find segment average and fee curve
            result = predictor.predict(
                business_segment=seg, property_type=ptype,
                property_state=state, target_time=ttime
            )
            seg_avg   = result["segment_benchmark"]
            base_fee  = result["predicted_fee"]
            curve     = result.get("fee_curve", {}).get("curve_points", [])

            if not curve or len(curve) < 5:
                R.warn(f"Fee sensitivity {seg}", "No fee curve data returned")
                continue

            fees  = [p["fee"]           for p in curve]
            probs = [p["win_probability"] for p in curve]

            low_prob  = probs[0]   # lowest fee  → should have HIGHEST win prob
            high_prob = probs[-1]  # highest fee → should have LOWEST win prob
            spread    = low_prob - high_prob

            # Compute slope: does win prob decrease as fee increases?
            # Allow some flatness in normal range but expect meaningful drop at extremes
            is_monotone_decreasing = high_prob <= low_prob  # at least non-increasing overall

            detail = (f"base_fee=${base_fee:,.0f}  seg_avg=${seg_avg:,.0f}  "
                      f"curve_spread={spread:.1f}pp  "
                      f"(low_fee={low_prob:.1f}% → high_fee={high_prob:.1f}%)")

            if not is_monotone_decreasing:
                R.warn(f"Fee sensitivity {seg}", f"Non-monotone curve — {detail}")
            elif spread < 5:
                R.warn(f"Fee sensitivity {seg}", f"Very flat curve ({spread:.1f}pp) — {detail}")
            elif spread < 15:
                R.ok(f"Fee sensitivity {seg}", f"Moderate slope ({spread:.1f}pp) — {detail}")
            else:
                R.ok(f"Fee sensitivity {seg}", f"Good slope ({spread:.1f}pp) — {detail}")

        except Exception as e:
            R.fail(f"Fee sensitivity {seg}", str(e))


# ============================================================================
# SECTION 6 — EV OPTIMIZATION
# ============================================================================

def check_ev_optimization(predictor):
    """
    The ev_optimal_fee should be different from predicted_fee (unless perfectly aligned),
    and EV at ev_optimal_fee should be >= EV at predicted_fee.
    """
    section("6. EV OPTIMIZATION")

    cases = [
        ("Financing", "Multifamily", "Illinois", 30),
        ("Consulting", "Commercial", "California", 45),
    ]

    for seg, ptype, state, ttime in cases:
        try:
            result = predictor.predict(
                business_segment=seg, property_type=ptype,
                property_state=state, target_time=ttime
            )
            predicted   = result["predicted_fee"]
            ev_optimal  = result.get("ev_optimal_fee", predicted)
            ev_score    = result["expected_value"]
            win_prob    = result["win_probability"]["probability"]
            curve       = result.get("fee_curve", {}).get("curve_points", [])

            diff_pct = abs(ev_optimal - predicted) / max(predicted, 1) * 100

            # If we have a curve, verify EV at ev_optimal_fee >= EV at predicted_fee
            ev_optimal_better = True
            if curve:
                ev_at_predicted = None
                ev_at_optimal   = None
                for pt in curve:
                    fee_pt = pt["fee"]
                    wp_pt  = pt["win_probability"] / 100
                    ev_pt  = fee_pt * wp_pt
                    if abs(fee_pt - predicted) < predicted * 0.05:
                        ev_at_predicted = ev_pt
                    if abs(fee_pt - ev_optimal) < ev_optimal * 0.05:
                        ev_at_optimal = ev_pt

                if ev_at_predicted is not None and ev_at_optimal is not None:
                    ev_optimal_better = ev_at_optimal >= ev_at_predicted - 1  # allow $1 rounding

            detail = (f"predicted=${predicted:,.0f}  ev_optimal=${ev_optimal:,.0f}  "
                      f"diff={diff_pct:.1f}%  EV=${ev_score:,.0f}  win={win_prob*100:.0f}%")

            if ev_optimal <= 0:
                R.fail(f"EV optimal {seg}", f"ev_optimal_fee is zero or missing — {detail}")
            elif not ev_optimal_better:
                R.warn(f"EV optimal {seg}", f"ev_optimal EV is not better than predicted_fee EV — {detail}")
            else:
                R.ok(f"EV optimal {seg}", detail)

        except Exception as e:
            R.fail(f"EV optimal {seg}", str(e))


# ============================================================================
# SECTION 7 — CONFIDENCE INTERVALS
# ============================================================================

def check_confidence_intervals(predictor):
    section("7. CONFIDENCE INTERVALS")

    cases = [
        ("Financing", "Multifamily", "Illinois", 30, "high-data segment"),
        ("Estate/Gift Tax", "Residential", "Alaska", 21, "low-data combo"),
        ("Consulting", "Commercial", "California", 45, "medium segment"),
    ]

    for seg, ptype, state, ttime, label in cases:
        try:
            result = predictor.predict(
                business_segment=seg, property_type=ptype,
                property_state=state, target_time=ttime
            )
            fee  = result["predicted_fee"]
            low  = result["confidence_interval"]["low"]
            high = result["confidence_interval"]["high"]
            conf = result["confidence_level"]

            fee_in_band = low <= fee <= high
            band_positive = low > 0 and high > 0
            band_ordered  = low < high
            band_width_pct = (high - low) / max(fee, 1) * 100

            detail = (f"CI=[${low:,.0f}–${high:,.0f}]  width={band_width_pct:.0f}%  "
                      f"conf={conf}  fee_in_band={'yes' if fee_in_band else 'NO'}")

            if not band_ordered:
                R.fail(f"CI ordered ({label})", f"low >= high — {detail}")
            elif not band_positive:
                R.fail(f"CI positive ({label})", f"negative bound — {detail}")
            elif not fee_in_band:
                R.warn(f"CI contains prediction ({label})", detail)
            elif band_width_pct > 300:
                R.warn(f"CI very wide ({label})", detail)
            else:
                R.ok(f"CI valid ({label})", detail)

        except Exception as e:
            R.fail(f"CI check ({label})", str(e))


# ============================================================================
# SECTION 8 — EDGE CASES
# ============================================================================

def check_edge_cases(predictor):
    section("8. EDGE CASES")

    # Very short turnaround
    try:
        r = predictor.predict(business_segment="Financing", property_type="Multifamily",
                              property_state="Illinois", target_time=1)
        fee = r["predicted_fee"]
        wp  = r["win_probability"]["probability_pct"]
        if fee > 100 and 5 <= wp <= 95:
            R.ok("1-day turnaround", f"fee=${fee:,.0f}  win={wp:.0f}%")
        else:
            R.warn("1-day turnaround", f"fee=${fee:,.0f}  win={wp:.0f}% — outside expected range")
    except Exception as e:
        R.fail("1-day turnaround", str(e))

    # Unknown/rare state
    try:
        r = predictor.predict(business_segment="Financing", property_type="Multifamily",
                              property_state="Wyoming", target_time=30)
        fee = r["predicted_fee"]
        wp  = r["win_probability"]["probability_pct"]
        if fee > 0 and 5 <= wp <= 95:
            R.ok("Rare state (Wyoming)", f"fee=${fee:,.0f}  win={wp:.0f}%")
        else:
            R.warn("Rare state (Wyoming)", f"fee=${fee:,.0f}  win={wp:.0f}%")
    except Exception as e:
        R.fail("Rare state (Wyoming)", str(e))

    # Office location auto-deriving region
    try:
        r1 = predictor.predict(business_segment="Financing", property_type="Multifamily",
                               property_state="Illinois", target_time=30,
                               office_location="Chicago")
        r2 = predictor.predict(business_segment="Financing", property_type="Multifamily",
                               property_state="Illinois", target_time=30,
                               office_location="Unknown")
        fee_diff = abs(r1["predicted_fee"] - r2["predicted_fee"])
        if fee_diff < r1["predicted_fee"] * 0.5:   # within 50%
            R.ok("Office location region derivation", f"Chicago vs Unknown fee diff={fee_diff:,.0f}")
        else:
            R.warn("Office location region derivation", f"Very large difference: {fee_diff:,.0f}")
    except Exception as e:
        R.fail("Office location region derivation", str(e))

    # open_date — past date should not cause error
    try:
        r = predictor.predict(business_segment="Financing", property_type="Multifamily",
                              property_state="Illinois", target_time=30,
                              open_date="2023-06-15")
        fee = r["predicted_fee"]
        R.ok("open_date past date", f"fee=${fee:,.0f}  date=2023-06-15")
    except Exception as e:
        R.fail("open_date past date", str(e))

    # Invalid open_date should fall back gracefully
    try:
        r = predictor.predict(business_segment="Financing", property_type="Multifamily",
                              property_state="Illinois", target_time=30,
                              open_date="not-a-date")
        R.ok("open_date invalid — falls back gracefully", f"fee=${r['predicted_fee']:,.0f}")
    except Exception as e:
        R.fail("open_date invalid", f"Should not throw: {e}")

    # Win probability must always be in [5, 95]
    probs_found = []
    for seg in ["Financing", "Consulting", "Estate/Gift Tax"]:
        for state in ["Illinois", "California", "Alaska"]:
            try:
                r = predictor.predict(business_segment=seg, property_type="Multifamily",
                                      property_state=state, target_time=30)
                probs_found.append(r["win_probability"]["probability_pct"])
            except Exception:
                pass

    if probs_found:
        out_of_range = [p for p in probs_found if not (5 <= p <= 95)]
        if out_of_range:
            R.fail("Win prob clipping [5,95]", f"Out-of-range values: {out_of_range}")
        else:
            R.ok("Win prob clipping [5,95]", f"All {len(probs_found)} values in range")


# ============================================================================
# SECTION 9 — SEGMENT VARIATION
# ============================================================================

def check_segment_variation(predictor):
    """Different segments should produce meaningfully different fee predictions."""
    section("9. SEGMENT VARIATION")

    segments = [
        "Financing", "Consulting", "Financial Reporting",
        "Estate/Gift Tax", "Litigation", "Transaction Services",
    ]

    fees = {}
    for seg in segments:
        try:
            r = predictor.predict(business_segment=seg, property_type="Multifamily",
                                  property_state="Illinois", target_time=30)
            fees[seg] = r["predicted_fee"]
        except Exception as e:
            R.warn(f"Segment {seg}", f"Prediction failed: {e}")

    if len(fees) < 3:
        R.fail("Segment variation", f"Only {len(fees)} segments returned successfully")
        return

    fee_values = list(fees.values())
    spread = max(fee_values) - min(fee_values)
    cv = np.std(fee_values) / max(np.mean(fee_values), 1) * 100  # coefficient of variation

    R.info(f"Fees by segment: " + "  ".join(f"{s}=${v:,.0f}" for s, v in fees.items()))

    if spread < 500:
        R.fail("Segment fee variation", f"All segments within ${spread:.0f} — may be defaulting to global avg")
    elif cv < 10:
        R.warn("Segment fee variation", f"Low CV={cv:.0f}%, spread=${spread:.0f}")
    else:
        R.ok("Segment fee variation", f"spread=${spread:,.0f}  CV={cv:.0f}%")


# ============================================================================
# SECTION 10 — LIVE API (optional)
# ============================================================================

def check_live_api(base_url: str):
    section(f"10. LIVE API ({base_url})")

    try:
        import requests
    except ImportError:
        R.warn("requests not installed", "Skipping live API checks (pip install requests)")
        return

    # Health check
    try:
        r = requests.get(f"{base_url}/api/health", timeout=10)
        data = r.json()
        if r.status_code == 200 and data.get("v2_model_loaded"):
            R.ok("/api/health", f"v2_model_loaded=True  status={data.get('status')}")
        elif r.status_code == 200:
            R.warn("/api/health", f"v2_model_loaded={data.get('v2_model_loaded')}  "
                   f"v2_error={data.get('v2_error')}")
        else:
            R.fail("/api/health", f"HTTP {r.status_code}")
    except Exception as e:
        R.fail("/api/health", str(e))
        return  # If health check fails, skip further API checks

    # Options
    try:
        r = requests.get(f"{base_url}/api/v2/options", timeout=10)
        data = r.json()
        if r.status_code == 200 and data.get("success"):
            opts = data.get("data", {})
            R.ok("/api/v2/options", f"segments={len(opts.get('segments',[]))}  "
                 f"states={len(opts.get('states',[]))}  "
                 f"subtypes={len(opts.get('sub_property_types',[]))}")
        else:
            R.fail("/api/v2/options", f"HTTP {r.status_code}: {data.get('error')}")
    except Exception as e:
        R.fail("/api/v2/options", str(e))

    # Basic prediction
    payload = {
        "business_segment": "Financing",
        "property_type": "Multifamily",
        "property_state": "Illinois",
        "target_time": 30,
        "delivery_days": 30,
    }
    try:
        t0 = time.time()
        r = requests.post(f"{base_url}/api/v2/predict", json=payload, timeout=30)
        elapsed_ms = (time.time() - t0) * 1000
        data = r.json()
        if r.status_code == 200 and data.get("success"):
            pred = data["prediction"]
            fee = pred["predicted_fee"]
            wp  = pred["win_probability"]["probability_pct"]
            ev  = pred["expected_value"]
            R.ok("/api/v2/predict basic", f"fee=${fee:,.0f}  win={wp:.0f}%  EV=${ev:,.0f}  [{elapsed_ms:.0f}ms]")
        else:
            R.fail("/api/v2/predict basic", f"HTTP {r.status_code}: {data.get('error')}")
    except Exception as e:
        R.fail("/api/v2/predict basic", str(e))

    # Win prob should vary across segments
    seg_probs = {}
    for seg in ["Financing", "Consulting", "Estate/Gift Tax"]:
        try:
            pl = {**payload, "business_segment": seg}
            r = requests.post(f"{base_url}/api/v2/predict", json=pl, timeout=15)
            if r.status_code == 200:
                seg_probs[seg] = r.json()["prediction"]["win_probability"]["probability_pct"]
        except Exception:
            pass

    if len(seg_probs) >= 2:
        spread = max(seg_probs.values()) - min(seg_probs.values())
        R.info(f"Win prob by segment (API): {seg_probs}")
        if spread < 5:
            R.fail("API win prob variation", f"All segments within {spread:.1f}pp — feature alignment bug?")
        else:
            R.ok("API win prob variation", f"{spread:.1f}pp spread across segments")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="End-to-end validation for BidRecommendation system")
    p.add_argument("--url",     default=None, help="Optional: live API base URL e.g. http://localhost:5001")
    p.add_argument("--verbose", action="store_true", help="Print prediction details for each check")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  BID RECOMMENDATION SYSTEM — END-TO-END VALIDATION")
    print(f"  {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. File checks (always run, no model needed)
    check_model_files()

    # 2-9. Direct model checks
    predictor = check_model_loading()
    if predictor is not None:
        results = check_basic_predictions(predictor, verbose=args.verbose)
        check_win_prob_distribution(predictor)
        check_fee_sensitivity(predictor)
        check_ev_optimization(predictor)
        check_confidence_intervals(predictor)
        check_edge_cases(predictor)
        check_segment_variation(predictor)
    else:
        R.fail("Skipping sections 3-9", "Predictor failed to load")

    # 10. Live API (optional)
    if args.url:
        check_live_api(args.url)

    # Final summary
    ok = R.summary()

    if ok:
        print("\n  All checks passed. System is healthy.")
    else:
        print(f"\n  {R.failed} check(s) FAILED. Review output above.")
        print("  Common causes:")
        print("    - Feature alignment: win prob all at 95% → add missing features to _generate_features()")
        print("    - Missing model files: run the training pipeline first")
        print("    - Fee curve flat: expected; win prob is driven by firm-market fit in normal range")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
