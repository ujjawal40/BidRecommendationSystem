"""
Flask API for Bid Recommendation System
========================================
REST API endpoints for Global Stat Solutions bid prediction service.

Endpoints:
- GET  /api/health          - Health check
- GET  /api/options         - Get dropdown options (segments, states, etc.)
- POST /api/predict         - Predict bid fee
- GET  /api/segment/<name>  - Get segment statistics

Usage:
    python api/app.py

    Or with gunicorn:
    gunicorn -w 4 -b 0.0.0.0:5000 api.app:app

Author: Global Stat Solutions
Date: 2026-01-29
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

from api.prediction_service import get_predictor, BidPredictor

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize predictor on startup
predictor: BidPredictor = None
init_error: str = None


@app.route('/api/debug', methods=['GET'])
def debug_info():
    """Debug endpoint - no predictor needed."""
    import os
    from pathlib import Path

    cwd = os.getcwd()
    base = Path(__file__).parent.parent

    # Check critical files
    files_to_check = [
        base / "outputs/models/lightgbm_bidfee_model.txt",
        base / "outputs/models/lightgbm_win_probability.txt",
        base / "outputs/reports/api_precomputed_stats.json",
        base / "outputs/reports/empirical_bands.json",
        base / "outputs/reports/feature_defaults.json",
        base / "config/model_config.py",
    ]

    file_status = {}
    for f in files_to_check:
        file_status[str(f.relative_to(base))] = f.exists()

    return jsonify({
        "cwd": cwd,
        "base_dir": str(base),
        "files": file_status,
        "init_error": init_error,
        "predictor_loaded": predictor is not None,
    })


@app.before_request
def initialize_predictor():
    """Lazy-load the predictor on first request."""
    global predictor, init_error

    # Skip initialization for debug endpoint
    if request.endpoint == 'debug_info':
        return

    if predictor is None and init_error is None:
        try:
            print("[API] Initializing predictor...")
            predictor = get_predictor()
            print("[API] Predictor ready!")
        except Exception as e:
            init_error = f"{type(e).__name__}: {e}"
            print(f"[API] FATAL ERROR initializing predictor: {init_error}")
            import traceback
            traceback.print_exc()
            raise


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Global Stat Solutions - Bid Recommendation API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": predictor is not None,
    })


@app.route('/api/options', methods=['GET'])
def get_options():
    """Get dropdown options for the UI."""
    try:
        options = predictor.get_dropdown_options()
        return jsonify({
            "success": True,
            "data": options,
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict_bid_fee():
    """
    Predict bid fee for a new opportunity.

    Request body (JSON):
    {
        "business_segment": "Financing",
        "property_type": "Multifamily",
        "property_state": "Illinois",
        "target_time": 30,
        "office_id": 123,          // optional
        "distance_km": 50,         // optional
        "on_due_date": 0,          // optional
        "client_history": {        // optional
            "avg_fee": 4500,
            "total_bids": 10,
            "total_wins": 5,
            "last_bid_fee": 4200
        }
    }

    Response:
    {
        "success": true,
        "prediction": {
            "predicted_fee": 3456.78,
            "confidence_interval": {"low": 2800, "high": 4100},
            "confidence_level": "high",
            "segment_benchmark": 3200,
            "recommendation": "...",
            ...
        }
    }
    """
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['business_segment', 'property_type', 'property_state', 'target_time']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": f"Missing required field: {field}",
                }), 400

        # Make prediction
        result = predictor.predict(
            business_segment=data['business_segment'],
            property_type=data['property_type'],
            property_state=data['property_state'],
            target_time=int(data['target_time']),
            office_id=data.get('office_id'),
            distance_km=float(data.get('distance_km', 0)),
            on_due_date=int(data.get('on_due_date', 0)),
            client_history=data.get('client_history'),
        )

        return jsonify({
            "success": True,
            "prediction": result,
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


@app.route('/api/segment/<segment_name>', methods=['GET'])
def get_segment_stats(segment_name: str):
    """Get statistics for a specific business segment."""
    try:
        stats = predictor.get_segment_stats(segment_name)
        return jsonify({
            "success": True,
            "segment": segment_name,
            "statistics": stats,
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction for multiple bids.

    Request body (JSON):
    {
        "bids": [
            {"business_segment": "Financing", ...},
            {"business_segment": "Consulting", ...}
        ]
    }
    """
    try:
        data = request.get_json()
        bids = data.get('bids', [])

        if not bids:
            return jsonify({
                "success": False,
                "error": "No bids provided",
            }), 400

        results = []
        for bid in bids:
            try:
                result = predictor.predict(
                    business_segment=bid['business_segment'],
                    property_type=bid['property_type'],
                    property_state=bid['property_state'],
                    target_time=int(bid['target_time']),
                    office_id=bid.get('office_id'),
                    distance_km=float(bid.get('distance_km', 0)),
                    on_due_date=int(bid.get('on_due_date', 0)),
                    client_history=bid.get('client_history'),
                )
                results.append({"success": True, "prediction": result})
            except Exception as e:
                results.append({"success": False, "error": str(e)})

        return jsonify({
            "success": True,
            "results": results,
            "total": len(results),
            "successful": sum(1 for r in results if r['success']),
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
    }), 404


@app.errorhandler(500)
def server_error(e):
    import traceback
    error_details = str(e)
    print(f"[API] 500 Error: {error_details}")
    traceback.print_exc()
    return jsonify({
        "success": False,
        "error": "Internal server error",
        "details": error_details,
    }), 500


if __name__ == '__main__':
    print("=" * 60)
    print("GLOBAL STAT SOLUTIONS - BID RECOMMENDATION API")
    print("=" * 60)
    print()

    # Initialize predictor before starting
    predictor = get_predictor()

    print()
    print("Starting Flask server...")
    print("API available at: http://localhost:5001")
    print()
    print("Endpoints:")
    print("  GET  /api/health          - Health check")
    print("  GET  /api/options         - Get dropdown options")
    print("  POST /api/predict         - Predict bid fee")
    print("  GET  /api/segment/<name>  - Get segment stats")
    print("  POST /api/batch-predict   - Batch predictions")
    print()

    app.run(host='0.0.0.0', port=5001, debug=True)
