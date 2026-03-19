"""
Enhanced Prediction Service (v2)
===================================
Serves dual-model predictions using v2 models trained on JobsData + enriched BidData.

New capabilities vs v1:
  - SubType, Office_Region, CompanyLocation as features
  - Broader training data (2018+ instead of 2023+)
  - 7.3% MAPE bid fee, 0.948 AUC win probability

Usage:
    from api.enhanced_prediction_service import EnhancedBidPredictor

    predictor = EnhancedBidPredictor()
    result = predictor.predict(
        business_segment="Financing",
        property_type="Multifamily",
        property_state="Illinois",
        target_time=30,
        sub_property_type="Conventional",
        office_location="Chicago",
        delivery_days=45,
    )

Author: Ujjawal Dwivedi
Date: 2026-02-15
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np

from config.model_config import MODELS_DIR, PREDICTION_CONFIG, REPORTS_DIR

from api.empirical_bands import EmpiricalBandCalculator

# Location → Region mapping derived from training data (BidData_enriched_v2.csv).
# Keys are lowercase office location names for case-insensitive lookup.
# Used to auto-derive office_region from office_location at inference time.
LOCATION_TO_REGION = {
    'austin': 'Southwest',
    'birmingham, al': 'Southeast',
    'boise, id': 'West',
    'boston, ma': 'Northeast',
    'caribbean': 'Other',
    'charleston, sc': 'Southeast',
    'charlotte, nc': 'Southeast',
    'chicago': 'Central',
    'cincinnati, oh': 'Central',
    'cleveland': 'Central',
    'coastal new jersey': 'Northeast',
    'columbia, sc': 'Southeast',
    'columbus, oh': 'Central',
    'corporate office': 'Other',
    'denver, co': 'Southwest',
    'detroit, mi': 'Central',
    'dfw': 'Southwest',
    'florida property advisors': 'Southeast',
    'fort worth, tx': 'Southwest',
    'grand rapids, mi': 'Central',
    'greensboro, nc': 'Southeast',
    'hartford, ct': 'Northeast',
    'houston': 'Southwest',
    'houston annex': 'Southwest',
    'indianapolis, in': 'Central',
    'integra aim core': 'Other',
    'integra aim national accounts': 'Other',
    'jackson, ms': 'Southeast',
    'kansas city': 'Central',
    'knoxville, tn': 'Southeast',
    'la metro': 'West',
    'las vegas, nv': 'West',
    'little rock, ar': 'Southwest',
    'los angeles, ca': 'West',
    'louisville, ky': 'Central',
    'lubbock, tx': 'Southwest',
    'memphis, tn': 'Southeast',
    'miami, fl': 'Southeast',
    'minneapolis, mn': 'Central',
    'naples, fl': 'Southeast',
    'nashville, tn': 'Southeast',
    'new orleans, la': 'Southwest',
    'northern new jersey': 'Northeast',
    'oklahoma city, ok': 'Southwest',
    'orange county, ca': 'West',
    'orlando, fl': 'Southeast',
    'philadelphia': 'Northeast',
    'phoenix, az': 'Southwest',
    'pittsburgh, pa': 'Northeast',
    'portland, or': 'West',
    'providence, ri': 'Northeast',
    'puerto rico': 'Other',
    'raleigh, nc': 'Southeast',
    'richmond, va': 'Northeast',
    'rio grande valley, tx': 'Southwest',
    'sacramento': 'West',
    'salt lake city, ut': 'West',
    'san antonio, tx': 'Southwest',
    'san diego, ca': 'West',
    'san francisco': 'West',
    'seattle, wa': 'West',
    'st. louis, mo': 'Central',
    'syracuse, ny': 'Northeast',
    'tampa, fl': 'Southeast',
    'tellatin senior housing': 'Central',
    'washington baltimore metro': 'Northeast',
    'washington dc': 'Northeast',
}


class EnhancedBidPredictor:
    """v2 prediction service using JobsData-trained models."""

    def __init__(self):
        self.model = None
        self.model_features = None
        self.win_prob_model = None
        self.win_prob_features = None
        self.win_prob_calibrator = None
        self.stats = {}
        self.feature_defaults = {}
        self.zip_lookup = {}
        self.band_calculator = EmpiricalBandCalculator()

        self._load_models()
        self._load_stats()
        self._load_feature_defaults()
        self._load_empirical_bands()
        self._load_zip_lookup()

    def _load_models(self):
        """Load v2 LightGBM models."""
        # Phase 1A v2
        bidfee_path = MODELS_DIR / "lightgbm_bidfee_v2_model.txt"
        if bidfee_path.exists():
            self.model = lgb.Booster(model_file=str(bidfee_path))
            self.model_features = self.model.feature_name()
            print(f"[EnhancedPredictor] Bid fee v2 loaded: {len(self.model_features)} features")
        else:
            raise FileNotFoundError(f"v2 model not found: {bidfee_path}")

        # Phase 1B v2
        winprob_path = MODELS_DIR / "lightgbm_win_probability_v2.txt"
        if winprob_path.exists():
            self.win_prob_model = lgb.Booster(model_file=str(winprob_path))
            self.win_prob_features = self.win_prob_model.feature_name()

            meta_path = MODELS_DIR / "lightgbm_win_probability_v2_metadata.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                auc = meta["metrics"]["test"]["auc_roc"]
                print(f"[EnhancedPredictor] Win prob v2 loaded: AUC={auc:.4f}")

            # Load calibrator (isotonic regression)
            calibrator_path = MODELS_DIR / "win_probability_v2_calibrator.pkl"
            if calibrator_path.exists():
                with open(calibrator_path, "rb") as f:
                    self.win_prob_calibrator = pickle.load(f)
                print("[EnhancedPredictor] Win prob calibrator loaded")
        else:
            print("[EnhancedPredictor] Win prob v2 not found, using fallback")

    def _load_stats(self):
        """Load precomputed statistics."""
        stats_path = REPORTS_DIR / "api_precomputed_stats_v2.json"
        if stats_path.exists():
            with open(stats_path, "r") as f:
                self.stats = json.load(f)
            print(f"[EnhancedPredictor] Stats loaded: {len(self.stats['segments'])} segments")
        else:
            raise FileNotFoundError(f"v2 stats not found: {stats_path}")

    def _load_feature_defaults(self):
        """Load feature defaults."""
        path = REPORTS_DIR / "feature_defaults_v2.json"
        if path.exists():
            with open(path, "r") as f:
                self.feature_defaults = json.load(f)
            print(f"[EnhancedPredictor] Feature defaults loaded: {len(self.feature_defaults)} features")

    def _load_empirical_bands(self):
        """Load empirical confidence bands."""
        path = REPORTS_DIR / "empirical_bands.json"
        if path.exists():
            self.band_calculator.load_bands(path)

    def _load_zip_lookup(self):
        """Load zip code demographics lookup."""
        path = REPORTS_DIR / "zip_demographics_lookup.json"
        if path.exists():
            with open(path, "r") as f:
                self.zip_lookup = json.load(f)
            print(f"[EnhancedPredictor] Zip lookup loaded: {len(self.zip_lookup):,} zip codes")
        else:
            print("[EnhancedPredictor] Zip lookup not found, using defaults")

    def _generate_features(
        self,
        business_segment: str,
        property_type: str,
        property_state: str,
        target_time: int = 30,
        sub_property_type: str = "Unknown",
        office_location: str = "Unknown",
        office_region: str = "Unknown",
        delivery_days: Optional[int] = None,
        company_type: str = "Unknown",
        contact_type: str = "Unknown",
        ref_date: Optional[datetime] = None,
        zip_code: Optional[str] = None,
    ) -> Dict[str, float]:
        """Generate feature vector for v2 model from user inputs."""
        features = {}
        s = self.stats
        zip_data = self.zip_lookup.get(zip_code, {}) if zip_code else {}

        # Segment features
        features["segment_avg_fee"] = s["segment_avg_fee"].get(
            business_segment, s["global_avg_fee"]
        )
        features["segment_std_fee"] = s.get("segment_std_fee", {}).get(
            business_segment, s["global_std_fee"]
        )
        features["segment_frequency"] = s["segment_frequency"].get(business_segment, 100)

        # State features
        features["state_avg_fee"] = s["state_avg_fee"].get(
            property_state, s["global_avg_fee"]
        )
        features["state_frequency"] = s.get("state_frequency", {}).get(property_state, 0.02)
        features["state_count"] = s.get("state_count", {}).get(property_state, 100)

        # PropertyType features
        features["propertytype_avg_fee"] = s["propertytype_avg_fee"].get(
            property_type, s["global_avg_fee"]
        )
        features["propertytype_frequency"] = s["propertytype_frequency"].get(property_type, 1000)

        # SubType features (NEW in v2)
        features["subtype_avg_fee"] = s.get("subtype_avg_fee", {}).get(
            sub_property_type, features["propertytype_avg_fee"]
        )
        features["subtype_frequency"] = s.get("subtype_frequency", {}).get(sub_property_type, 100)

        # Office Region features (NEW in v2)
        # Fall back to segment_avg_fee (not global_avg_fee) when region is unknown.
        # global_avg_fee is inflated by high-fee segments and causes win prob model
        # to predict ~98% for any unspecified office location.
        region_avg_fallback = features["segment_avg_fee"]
        features["office_region_avg_fee"] = s.get("office_region_avg_fee", {}).get(
            office_region, region_avg_fallback
        )
        features["office_region_frequency"] = s.get("office_region_frequency", {}).get(
            office_region, features["segment_frequency"]
        )

        # CompanyLocation features (NEW in v2)
        features["company_location_frequency"] = s.get("company_location_frequency", {}).get(
            office_location, features["segment_frequency"]
        )

        # CompanyType/ContactType (Phase 1A)
        features["companytype_avg_fee"] = s.get("companytype_avg_fee", {}).get(
            company_type, s["global_avg_fee"]
        )
        features["companytype_frequency"] = s.get("companytype_frequency", {}).get(company_type, 100)
        features["contacttype_avg_fee"] = s.get("contacttype_avg_fee", {}).get(
            contact_type, s["global_avg_fee"]
        )

        # Rolling features (use segment averages as proxy)
        features["rolling_avg_fee_segment"] = features["segment_avg_fee"]
        features["rolling_std_fee_segment"] = features["segment_std_fee"]
        features["rolling_avg_fee_state"] = features["state_avg_fee"]

        # Interaction features
        features["segment_x_state_fee"] = features["segment_avg_fee"] * features["state_avg_fee"]

        # Temporal features — use ref_date (job open date) if provided, else today
        now = ref_date if ref_date else datetime.now()
        features["Year"] = now.year
        features["Month"] = now.month
        features["Quarter"] = (now.month - 1) // 3 + 1
        features["DayOfWeek"] = now.weekday()
        features["WeekOfYear"] = now.isocalendar()[1]
        features["Month_sin"] = np.sin(2 * np.pi * now.month / 12)
        features["Month_cos"] = np.cos(2 * np.pi * now.month / 12)
        features["DayOfWeek_sin"] = np.sin(2 * np.pi * now.weekday() / 7)
        features["DayOfWeek_cos"] = np.cos(2 * np.pi * now.weekday() / 7)

        # Geographic — prefer zip-level coordinates when available
        if zip_data.get("latitude"):
            features["RooftopLatitude"] = zip_data["latitude"]
            features["RooftopLongitude"] = zip_data.get("longitude", -95.0)
        else:
            features["RooftopLatitude"] = s.get("state_latitude", {}).get(property_state, 35.0)
            features["RooftopLongitude"] = s.get("state_longitude", {}).get(property_state, -95.0)

        # JobLength features (NEW in v2)
        jl = delivery_days if delivery_days and delivery_days > 0 else 30
        features["JobLength_Days"] = jl
        features["joblength_log"] = np.log1p(jl)
        features["joblength_bucket"] = min(int(jl / 30), 4)
        features["joblength_x_segment_fee"] = jl * features["segment_avg_fee"]

        # Building area features (defaults — user doesn't provide these)
        features["GrossBuildingSF"] = 0
        features["GLARentableSF"] = 0
        features["GrossLandAreaAcres"] = 0
        features["GrossLandAreaSF"] = 0
        features["building_sf_log"] = 0
        features["land_acres_log"] = 0
        features["YearBuilt"] = 2000

        # Zip features — use real demographics when zip_code is provided
        zip_defaults = {
            "Zip_Population": 3362, "Zip_PopDensity": 7282,
            "Zip_HouseholdsPerZip": 11901, "Zip_GrowthRank": 231200,
            "Zip_AverageHouseValue": 17042, "Zip_IncomePerHousehold": 2403,
            "Zip_MedianAge": 15, "Zip_MedianIncome": 616,
            "Zip_NumberOfBusinesses": 811, "Zip_NumberOfEmployees": 11722,
            "Zip_LandArea": 19.9, "Zip_PopulationEstimate": 29976,
            "Zip_PopCount": 1830, "Zip_DeliveryTotal": 14582,
            "Zip_WorkersOutZip": 1671683,
        }
        for k, v in zip_defaults.items():
            features[k] = zip_data.get(k, v)

        features["income_x_segment_fee"] = features["Zip_IncomePerHousehold"] * features["segment_avg_fee"]
        features["pop_density_log"] = np.log1p(features["Zip_PopDensity"])

        # Temporal (days since last in segment — use default)
        features["days_since_last_segment"] = 1

        # --- Win probability model feature alignment ---
        # The v2 win prob model was trained on column names that differ from the bid fee
        # model's feature names. Without these, 24 of 44 win prob features silently
        # default to 0 at inference, pushing raw model output to ~0.97 for every input.

        # Raw user inputs available at inference time
        features["TargetTime"] = target_time
        features["targettime_log"] = np.log1p(target_time)

        # Distance: prefer zip-level, then state-level, then 25-mile fallback
        if zip_data.get("distance_miles"):
            dist_miles = zip_data["distance_miles"]
        else:
            dist_miles = s.get("state_distance_miles", {}).get(property_state, 25.0)
        features["DistanceInMiles"] = dist_miles
        features["distance_log"] = np.log1p(dist_miles)

        # Demographic name aliases: win prob model was trained with non-prefixed column names
        # but the bid fee model uses Zip_* names. Alias them so both models get values.
        features["PopulationEstimate"] = features["Zip_PopulationEstimate"]
        features["AverageHouseValue"]  = features["Zip_AverageHouseValue"]
        features["IncomePerHousehold"] = features["Zip_IncomePerHousehold"]
        features["MedianAge"]          = features["Zip_MedianAge"]
        features["DeliveryTotal"]      = features["Zip_DeliveryTotal"]
        features["NumberofBusinesses"] = features["Zip_NumberOfBusinesses"]
        features["NumberofEmployees"]  = features["Zip_NumberOfEmployees"]
        features["ZipPopulation"]      = features["Zip_Population"]

        # Office-level average: proxy with region average (no per-office stats at inference)
        features["office_avg_fee"] = features["office_region_avg_fee"]

        # Client history: no history available at inference — proxy with segment averages.
        # Assumption: treat every API call as if it's from a client whose historical
        # behaviour matches the market average for this segment.
        features["client_avg_fee"]             = features["segment_avg_fee"]
        features["client_std_fee"]             = features["segment_std_fee"]
        features["lag1_bidfee_client"]         = features["segment_avg_fee"]
        features["cumulative_bids_client"]     = 0    # unknown → assume first bid
        features["days_since_last_bid_client"] = 365  # unknown → assume long gap

        return features

    def predict(
        self,
        business_segment: str,
        property_type: str,
        property_state: str,
        target_time: int = 30,
        sub_property_type: str = "Unknown",
        office_location: str = "Unknown",
        delivery_days: Optional[int] = None,
        company_type: str = "Unknown",
        contact_type: str = "Unknown",
        open_date: Optional[str] = None,
        zip_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Predict bid fee, win probability, and expected value."""

        # Derive office_region from office_location — no longer a user-supplied input
        office_region = LOCATION_TO_REGION.get(office_location.lower(), "Unknown")

        # Parse open_date for temporal features; fall back to today if absent/invalid
        ref_date: Optional[datetime] = None
        if open_date:
            try:
                ref_date = datetime.strptime(open_date, "%Y-%m-%d")
            except ValueError:
                ref_date = None

        # Clean and validate zip_code
        zip_found = False
        zip_derived_state = None
        zip_state_mismatch = False
        if zip_code:
            zip_code = str(zip_code).strip()
            zip_data = self.zip_lookup.get(zip_code, {})
            zip_found = bool(zip_data)
            if zip_found and zip_data.get("state"):
                zip_derived_state = zip_data["state"]
                # Auto-derive state if not provided or if user wants zip to drive it
                if property_state and zip_derived_state != property_state:
                    zip_state_mismatch = True

        features = self._generate_features(
            business_segment=business_segment,
            property_type=property_type,
            property_state=property_state,
            target_time=target_time,
            sub_property_type=sub_property_type,
            office_location=office_location,
            office_region=office_region,
            delivery_days=delivery_days,
            company_type=company_type,
            contact_type=contact_type,
            ref_date=ref_date,
            zip_code=zip_code,
        )

        # Build feature vector in model order
        feature_vector = []
        for feat_name in self.model_features:
            if feat_name in features:
                feature_vector.append(features[feat_name])
            elif feat_name in self.feature_defaults:
                defaults = self.feature_defaults[feat_name]
                seg_key = f"segment_{business_segment}_median"
                feature_vector.append(defaults.get(seg_key, defaults.get("global_median", 0)))
            else:
                feature_vector.append(0)

        # Predict fee
        X = np.array([feature_vector])
        prediction = self.model.predict(X)[0]
        prediction = float(np.expm1(prediction))

        # Benchmark-aware floor
        segment_avg = self.stats["segment_avg_fee"].get(
            business_segment, self.stats["global_avg_fee"]
        )
        segment_floor = 0.3 * segment_avg
        prediction = max(prediction, PREDICTION_CONFIG["min_fee"], segment_floor)

        # Blend with segment average for rare combos
        seg_count = self.stats.get("segment_count", {}).get(business_segment, 0)
        state_count = self.stats.get("state_count", {}).get(property_state, 0)
        min_samples = min(seg_count, state_count)
        if min_samples < 100 and segment_avg > 0:
            # Gradually increase blending: fewer samples = more reliance on segment avg
            blend_weight = max(0.2, min_samples / 100)  # model weight: 0.2 to 1.0
            prediction = blend_weight * prediction + (1 - blend_weight) * segment_avg

        # Confidence interval
        low, high, band_meta = self.band_calculator.get_confidence_interval(
            predicted_fee=prediction,
            segment=business_segment,
            state=property_state,
            confidence_level=0.80,
        )

        # Confidence level (seg_count, state_count computed above for blending)
        cfg = PREDICTION_CONFIG
        if seg_count > cfg["confidence_segment_high"] and state_count > cfg["confidence_state_high"]:
            data_conf = "high"
        elif seg_count > cfg["confidence_segment_medium"] and state_count > cfg["confidence_state_medium"]:
            data_conf = "medium"
        else:
            data_conf = "low"

        band_ratio = (high - low) / max(prediction, 1)
        if band_ratio < cfg["band_ratio_high"]:
            band_conf = "high"
        elif band_ratio < cfg["band_ratio_medium"]:
            band_conf = "medium"
        else:
            band_conf = "low"

        conf_rank = {"low": 0, "medium": 1, "high": 2}
        rank_label = {0: "low", 1: "medium", 2: "high"}
        confidence = rank_label[min(conf_rank[data_conf], conf_rank[band_conf])]

        # Win probability
        features["BidFee"] = prediction
        win_prob_result = self._predict_win_probability(features, prediction, segment_avg)

        # Expected value
        expected_value = win_prob_result["probability"] * prediction

        # Fee sensitivity curve
        curve_data = self.get_fee_sensitivity_curve(
            features=features,
            recommended_fee=prediction,
            segment_avg=segment_avg,
        )

        # Blended benchmark
        blended = (
            0.4 * features["segment_avg_fee"]
            + 0.3 * features["state_avg_fee"]
            + 0.3 * features["propertytype_avg_fee"]
        )

        diff_pct = ((prediction - blended) / blended) * 100
        win_pct = win_prob_result["probability_pct"]
        win_prob = win_prob_result["probability"]

        # EV-optimal fee from curve data
        ev_optimal_fee = None
        ev_optimal_diff_pct = 0
        if curve_data and curve_data.get("curve_points"):
            best_ev_point = max(
                curve_data["curve_points"],
                key=lambda p: (p["win_probability"] / 100) * p["fee"],
            )
            ev_optimal_fee = best_ev_point["fee"]
            ev_optimal_diff_pct = ((ev_optimal_fee - prediction) / prediction) * 100

        # Solid-win ceiling: highest fee where P(win) >= 30%
        P_SOLID = 0.30
        solid_win_ceiling = None
        if curve_data and curve_data.get("curve_points"):
            solid_candidates = [
                p for p in curve_data["curve_points"]
                if p["win_probability"] / 100 >= P_SOLID
            ]
            if solid_candidates:
                solid_win_ceiling = max(p["fee"] for p in solid_candidates)

        # Constrain: ev_optimal_fee must never exceed the displayed ceiling.
        # The displayed ceiling is solid_win_ceiling when P(win)>=30% was found on
        # the curve, otherwise it falls back to the CI upper bound (high).
        # Either way, the Optimal Bid shown in the UI cannot sit above what we call
        # the "Ceiling" — that would be a contradictory display.
        displayed_ceiling = solid_win_ceiling if solid_win_ceiling is not None else high
        ev_capped_at_ceiling = False
        if ev_optimal_fee and ev_optimal_fee > displayed_ceiling:
            ev_optimal_fee = displayed_ceiling
            ev_optimal_diff_pct = ((ev_optimal_fee - prediction) / prediction) * 100
            ev_capped_at_ceiling = True

        rec = self._build_recommendation(
            diff_pct=diff_pct,
            win_pct=win_pct,
            win_prob=win_prob,
            expected_value=expected_value,
            prediction=prediction,
            blended=blended,
            confidence=confidence,
            ev_optimal_fee=ev_optimal_fee,
            ev_optimal_diff_pct=ev_optimal_diff_pct,
            segment=business_segment,
            seg_count=seg_count,
            state_count=state_count,
        )

        # Sample-size warnings for rare combos
        warnings = []
        if seg_count < 100:
            warnings.append(
                f"Low data: only {seg_count} training samples for {business_segment} segment. "
                f"Prediction may be less reliable."
            )
        if state_count < 100:
            warnings.append(
                f"Low data: only {state_count} training samples for {property_state}. "
                f"Prediction may be less reliable."
            )
        if seg_count < 100 and state_count < 100:
            warnings.append(
                "Consider using segment benchmark as a safer reference point."
            )

        return {
            "predicted_fee": round(prediction, 2),
            "confidence_interval": {"low": round(low, 2), "high": round(high, 2)},
            "confidence_level": confidence,
            "win_probability": win_prob_result,
            "expected_value": round(expected_value, 2),
            "segment_benchmark": round(segment_avg, 2),
            "state_benchmark": round(
                self.stats["state_avg_fee"].get(property_state, segment_avg), 2
            ),
            "recommendation": rec,
            "warnings": warnings,
            "factors": {
                "segment_effect": round(features["segment_avg_fee"], 2),
                "state_effect": round(features["state_avg_fee"], 2),
                "subtype_effect": round(features["subtype_avg_fee"], 2),
                "office_region_effect": round(features["office_region_avg_fee"], 2),
                "delivery_days": delivery_days,
            },
            "ev_optimal_fee": round(ev_optimal_fee, 2) if ev_optimal_fee else round(prediction, 2),
            "solid_win_ceiling": round(displayed_ceiling, 2),
            "ev_capped_at_ceiling": ev_capped_at_ceiling,
            "fee_curve": curve_data,
            "metadata": {
                "model_version": "2.0",
                "prediction_date": datetime.now().isoformat(),
                "data_coverage": {
                    "segment_samples": seg_count,
                    "state_samples": state_count,
                    "total_training_samples": sum(self.stats.get("segment_count", {}).values()),
                },
                "zip_info": {
                    "zip_code": zip_code,
                    "zip_found": zip_found,
                    "zip_derived_state": zip_derived_state,
                    "zip_state_mismatch": zip_state_mismatch,
                } if zip_code else None,
            },
        }

    def _predict_win_probability(
        self, features: Dict, predicted_fee: float, segment_benchmark: float
    ) -> Dict[str, Any]:
        """Predict win probability using v2 classification model."""
        if self.win_prob_model is None:
            return self._fallback_win_probability(predicted_fee, segment_benchmark)

        # Compute fee-relative features here — BidFee is known at this point
        # but was not available when _generate_features() ran.
        # Defaulting these to 0 tells the model "fee is at zero relative to market"
        # which pushes win probability to ~98% for every prediction.
        seg_avg  = features.get("segment_avg_fee", segment_benchmark) or segment_benchmark
        state_avg = features.get("state_avg_fee", segment_benchmark) or segment_benchmark
        fee = features.get("BidFee", predicted_fee) or predicted_fee
        features["fee_vs_segment_ratio"]  = fee / seg_avg if seg_avg > 0 else 1.0
        features["fee_diff_from_segment"] = fee - seg_avg
        features["bid_vs_state_ratio"]    = fee / state_avg if state_avg > 0 else 1.0
        # fee_percentile_segment: approximate as ratio clamped 0-1
        features["fee_percentile_segment"] = min(1.0, max(0.0, fee / (2 * seg_avg)))
        # bid_vs_client_ratio: no client history available at inference → use segment ratio
        features["bid_vs_client_ratio"] = features["fee_vs_segment_ratio"]

        feature_vector = []
        for feat_name in self.win_prob_features:
            if feat_name in features:
                feature_vector.append(features[feat_name])
            elif feat_name in self.feature_defaults:
                feature_vector.append(self.feature_defaults[feat_name].get("global_median", 0))
            else:
                feature_vector.append(0)

        X = np.array([feature_vector])
        raw_probability = float(self.win_prob_model.predict(X)[0])

        # Apply isotonic calibration if available.
        # Wrapped in try/except: calibrator was saved under sklearn 1.3.0 but may be
        # running on a newer version, which can cause silent errors on .predict().
        if self.win_prob_calibrator is not None:
            try:
                probability = float(self.win_prob_calibrator.predict([raw_probability])[0])
            except Exception:
                probability = raw_probability
        else:
            probability = raw_probability

        probability = max(PREDICTION_CONFIG["win_prob_min"],
                         min(PREDICTION_CONFIG["win_prob_max"], probability))

        dist = abs(probability - 0.5)
        if dist > 0.3:
            win_conf = "high"
        elif dist > 0.15:
            win_conf = "medium"
        else:
            win_conf = "low"

        return {
            "probability": round(probability, 4),
            "probability_pct": round(probability * 100, 1),
            "confidence": win_conf,
            "model_used": "LightGBM Classifier v2 (AUC: 0.948)",
        }

    def _fallback_win_probability(self, predicted_fee: float, segment_benchmark: float) -> Dict:
        """Heuristic fallback when win prob model isn't available."""
        ratio = predicted_fee / max(segment_benchmark, 1)
        raw = 1 / (1 + np.exp(5 * (ratio - 1)))
        probability = 0.20 + raw * 0.55
        probability = max(0.05, min(0.95, probability))
        return {
            "probability": round(probability, 4),
            "probability_pct": round(probability * 100, 1),
            "confidence": "low",
            "model_used": "Heuristic fallback (no v2 classification model)",
        }

    def get_fee_sensitivity_curve(
        self,
        features: Dict[str, float],
        recommended_fee: float,
        segment_avg: float,
        num_points: int = 20,
    ) -> Dict[str, Any]:
        """Generate P(Win) across a range of fee levels."""
        fee_low = max(500, recommended_fee * 0.40)
        fee_high = recommended_fee * 2.0

        fee_points = np.geomspace(fee_low, fee_high, num=num_points)
        # Ensure recommended fee is always a data point
        fee_points = np.sort(np.unique(np.append(fee_points, recommended_fee)))

        curve_points = []
        for fee in fee_points:
            features_copy = features.copy()
            features_copy["BidFee"] = float(fee)

            wp_result = self._predict_win_probability(
                features_copy, float(fee), segment_avg
            )

            curve_points.append({
                "fee": round(float(fee), 0),
                "win_probability": round(wp_result["probability"] * 100, 1),
            })

        return {
            "curve_points": curve_points,
            "recommended_fee": round(recommended_fee, 0),
        }

    def _build_recommendation(
        self,
        diff_pct: float,
        win_pct: float,
        win_prob: float,
        expected_value: float,
        prediction: float,
        blended: float,
        confidence: str,
        ev_optimal_fee: float,
        ev_optimal_diff_pct: float,
        segment: str,
        seg_count: int,
        state_count: int,
    ) -> Dict[str, str]:
        """Build a plain-language recommendation based on fee position and win odds."""
        blended_str = f"${blended:,.0f}"
        abs_diff = abs(diff_pct)

        # Fee positioning tier
        if diff_pct < -20:
            fee_pos = "aggressive"
        elif diff_pct < -5:
            fee_pos = "competitive"
        elif diff_pct <= 10:
            fee_pos = "aligned"
        elif diff_pct <= 25:
            fee_pos = "premium"
        else:
            fee_pos = "stretch"

        # Win probability tier
        if win_pct >= 70:
            win_tier = "strong"
        elif win_pct >= 40:
            win_tier = "moderate"
        else:
            win_tier = "low"

        # Plain-English headlines — no jargon, no formula references
        headlines = {
            ("aggressive", "strong"):   "You're priced below market — there's room to charge more",
            ("aggressive", "moderate"): "Below-market bid with decent chances",
            ("aggressive", "low"):      "Low fee, but this market is hard to break into",
            ("competitive", "strong"):  "Strong position — well-priced and likely to win",
            ("competitive", "moderate"): "Solid bid in a competitive market",
            ("competitive", "low"):     "Fair price, but competition is stiff here",
            ("aligned", "strong"):      "Right at the market rate with strong win odds",
            ("aligned", "moderate"):    "Market-rate pricing",
            ("aligned", "low"):         "Market price, but this is a heavily contested segment",
            ("premium", "strong"):      "Above market and still likely to win",
            ("premium", "moderate"):    "Higher than average — good odds, but review before submitting",
            ("premium", "low"):         "Fee is above what the market typically sees here",
            ("stretch", "strong"):      "Aggressive fee, but conditions are in your favor",
            ("stretch", "moderate"):    "Fee is on the high end — worth a second look",
            ("stretch", "low"):         "Fee is too high relative to what this market supports",
        }
        headline = headlines.get((fee_pos, win_tier), "Market-rate pricing")

        # Build detail — reference market benchmark, no raw model fee or formulas
        if diff_pct < -5:
            market_context = (
                f"The average {segment} fee in this area runs around {blended_str}. "
                f"Your recommended bid comes in {abs_diff:.0f}% below that — "
                f"you're priced aggressively."
            )
        elif diff_pct > 10:
            market_context = (
                f"The average {segment} fee in this area runs around {blended_str}. "
                f"Your recommended bid is {abs_diff:.0f}% above that — "
                f"you're pricing at a premium."
            )
        else:
            market_context = (
                f"Your recommended bid is right in line with the average {segment} "
                f"fee in this area ({blended_str})."
            )

        if win_tier == "strong":
            odds_context = "Your win odds are strong at this price."
        elif win_tier == "moderate":
            odds_context = "Your win odds are reasonable, though not guaranteed."
        else:
            odds_context = (
                f"Win odds are modest at {win_pct:.0f}% — this is a competitive assignment."
            )

        detail = f"{market_context} {odds_context}"

        # Strategy tips — plain English, no slider, no EV formula
        strategy_tip = None
        if fee_pos == "aggressive" and win_tier == "strong":
            strategy_tip = (
                f"You'd likely win even at a higher fee. "
                f"Consider bidding closer to the market rate ({blended_str}) "
                f"to capture more revenue per job."
            )
        elif fee_pos == "aggressive" and win_tier in ("moderate", "low"):
            strategy_tip = (
                "Even at a below-market price, win odds are modest — "
                "this segment is likely very competitive. "
                "Your track record and client relationships may matter more than price here."
            )
        elif fee_pos in ("premium", "stretch") and win_tier == "low":
            strategy_tip = (
                f"Lowering your fee closer to {blended_str} could meaningfully "
                f"improve your chances. At this price point, winning will be difficult."
            )
        elif fee_pos in ("premium", "stretch") and win_tier == "moderate":
            strategy_tip = (
                "You're pricing above average, which is fine if you have a strong "
                "relationship with this client or a competitive edge on deliverables."
            )
        elif fee_pos == "aligned" and win_tier == "low":
            strategy_tip = (
                f"The {segment} segment here is highly competitive. "
                "Non-price factors — your experience, turnaround time, and client "
                "relationships — will likely matter as much as the fee itself."
            )

        # Low confidence context — plain language, no "limited data" alarm
        if confidence == "low":
            cfg = PREDICTION_CONFIG
            is_data_limited = not (
                seg_count > cfg["confidence_segment_medium"]
                and state_count > cfg["confidence_state_medium"]
            )
            if is_data_limited:
                headline = "Estimate based on limited data — " + headline.lower()
                detail = (
                    f"There aren't many historical bids for this exact combination, "
                    f"so treat this as a directional guide rather than a firm number. "
                    + detail
                )
            else:
                headline = "Fees vary widely here — " + headline.lower()
                detail = (
                    f"Fees for {segment} work in this state vary a lot, "
                    f"so the range is broader than usual. "
                    + detail
                )

        # Signal for frontend color coding
        if win_tier == "strong" and fee_pos in ("competitive", "aligned", "aggressive"):
            signal = "positive"
        elif win_tier == "low" and fee_pos in ("premium", "stretch"):
            signal = "caution"
        else:
            signal = "neutral"

        rec = {
            "headline": headline,
            "detail": detail,
            "signal": signal,
        }
        if strategy_tip:
            rec["strategy_tip"] = strategy_tip

        return rec

    def get_dropdown_options(self) -> Dict[str, List]:
        """Return options for UI dropdowns including new v2 fields."""
        return {
            "segments": self.stats.get("segments", []),
            "property_types": self.stats.get("property_types", []),
            "states": self.stats.get("states", []),
            "sub_property_types": self.stats.get("subtypes", []),
            "office_regions": self.stats.get("office_regions", []),
            "office_locations": self.stats.get("company_locations", []),
            "subtypes_by_property_type": self.stats.get("subtypes_by_property_type", {}),
        }

    def get_segment_stats(self, segment: str) -> Dict:
        """Get statistics for a specific segment."""
        return {
            "avg_fee": self.stats["segment_avg_fee"].get(segment, 0),
            "std_fee": self.stats.get("segment_std_fee", {}).get(segment, 0),
            "count": self.stats.get("segment_count", {}).get(segment, 0),
        }
