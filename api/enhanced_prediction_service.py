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
        self.band_calculator = EmpiricalBandCalculator()

        self._load_models()
        self._load_stats()
        self._load_feature_defaults()
        self._load_empirical_bands()

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
    ) -> Dict[str, float]:
        """Generate feature vector for v2 model from user inputs."""
        features = {}
        s = self.stats

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
        features["office_region_avg_fee"] = s.get("office_region_avg_fee", {}).get(
            office_region, s["global_avg_fee"]
        )
        features["office_region_frequency"] = s.get("office_region_frequency", {}).get(
            office_region, 1000
        )

        # CompanyLocation features (NEW in v2)
        features["company_location_frequency"] = s.get("company_location_frequency", {}).get(
            office_location, 500
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

        # Temporal features
        now = datetime.now()
        features["Year"] = now.year
        features["Month"] = now.month
        features["Quarter"] = (now.month - 1) // 3 + 1
        features["DayOfWeek"] = now.weekday()
        features["WeekOfYear"] = now.isocalendar()[1]
        features["Month_sin"] = np.sin(2 * np.pi * now.month / 12)
        features["Month_cos"] = np.cos(2 * np.pi * now.month / 12)
        features["DayOfWeek_sin"] = np.sin(2 * np.pi * now.weekday() / 7)
        features["DayOfWeek_cos"] = np.cos(2 * np.pi * now.weekday() / 7)

        # Geographic
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

        # Zip features (use medians)
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
            features[k] = v

        features["income_x_segment_fee"] = features["Zip_IncomePerHousehold"] * features["segment_avg_fee"]
        features["pop_density_log"] = np.log1p(features["Zip_PopDensity"])

        # Temporal (days since last in segment — use default)
        features["days_since_last_segment"] = 1

        return features

    def predict(
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
    ) -> Dict[str, Any]:
        """Predict bid fee, win probability, and expected value."""

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
            "fee_curve": curve_data,
            "metadata": {
                "model_version": "2.0",
                "prediction_date": datetime.now().isoformat(),
                "data_coverage": {
                    "segment_samples": seg_count,
                    "state_samples": state_count,
                },
            },
        }

    def _predict_win_probability(
        self, features: Dict, predicted_fee: float, segment_benchmark: float
    ) -> Dict[str, Any]:
        """Predict win probability using v2 classification model."""
        if self.win_prob_model is None:
            return self._fallback_win_probability(predicted_fee, segment_benchmark)

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

        # Apply isotonic calibration if available
        if self.win_prob_calibrator is not None:
            probability = float(self.win_prob_calibrator.predict([raw_probability])[0])
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
        """Build a structured recommendation based on multiple signals."""
        # Fee positioning (5 tiers)
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

        # Build headline
        headlines = {
            ("aggressive", "strong"): "Underpriced — room to increase",
            ("aggressive", "moderate"): "Below market with moderate odds",
            ("aggressive", "low"): "Low fee but still competitive",
            ("competitive", "strong"): "Strong competitive position",
            ("competitive", "moderate"): "Competitive pricing, fair odds",
            ("competitive", "low"): "Competitive fee, tough market",
            ("aligned", "strong"): "Well-positioned to win",
            ("aligned", "moderate"): "Market-rate pricing",
            ("aligned", "low"): "Fair price, contested segment",
            ("premium", "strong"): "Premium fee with strong odds",
            ("premium", "moderate"): "Above market — weigh the tradeoff",
            ("premium", "low"): "Premium pricing risk",
            ("stretch", "strong"): "High fee but favorable conditions",
            ("stretch", "moderate"): "Stretch pricing — proceed carefully",
            ("stretch", "low"): "Significant pricing risk",
        }
        headline = headlines.get((fee_pos, win_tier), "Market-rate pricing")

        # Build detail text
        fee_str = f"${prediction:,.0f}"
        blended_str = f"${blended:,.0f}"
        abs_diff = abs(diff_pct)

        if diff_pct < -5:
            position_text = f"At {fee_str}, your bid is {abs_diff:.0f}% below the market benchmark ({blended_str})."
        elif diff_pct > 10:
            position_text = f"At {fee_str}, your bid is {abs_diff:.0f}% above the market benchmark ({blended_str})."
        else:
            position_text = f"At {fee_str}, your bid aligns with the market benchmark ({blended_str})."

        detail = (
            f"{position_text} "
            f"Win probability is {win_pct}% with an expected value of ${expected_value:,.0f}."
        )

        # Build strategy tip
        strategy_tip = None
        if fee_pos == "aggressive" and win_tier == "strong":
            strategy_tip = (
                f"You'd likely win at a higher fee. "
                f"Consider increasing toward {blended_str} to capture more revenue."
            )
        elif fee_pos == "aggressive" and win_tier in ("moderate", "low"):
            strategy_tip = (
                "Fee is already below market. If win odds are still modest, "
                "the segment may be highly competitive — focus on non-price differentiators."
            )
        elif fee_pos in ("premium", "stretch") and win_tier == "low":
            strategy_tip = (
                f"Reducing fee toward {blended_str} could significantly improve win probability. "
                f"Check the sensitivity chart for the optimal tradeoff."
            )
        elif fee_pos in ("premium", "stretch") and win_tier == "moderate":
            strategy_tip = (
                "Fee is above market but odds are reasonable. "
                "Consider whether the higher revenue per win justifies the lower win rate."
            )
        elif fee_pos == "aligned" and win_tier == "low":
            strategy_tip = (
                f"This {segment} segment is highly contested. "
                "A modest fee reduction may improve odds without sacrificing much revenue."
            )

        # EV-optimal insight
        if ev_optimal_fee is not None and abs(ev_optimal_diff_pct) > 10:
            ev_direction = "higher" if ev_optimal_diff_pct > 0 else "lower"
            strategy_tip = (
                (strategy_tip + " " if strategy_tip else "")
                + f"The EV-optimal fee is ${ev_optimal_fee:,.0f} ({ev_direction} than recommended), "
                f"which may yield better risk-adjusted returns."
            )

        # Low confidence hedging
        if confidence == "low":
            headline = "Limited data — " + headline.lower()
            detail = "Note: prediction is based on limited training data. " + detail

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
