# Bid Recommendation API

REST API for the Global Stat Solutions Bid Recommendation System.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server
python app.py
```

The API will be available at `http://localhost:5000`

## Endpoints

### Health Check
```
GET /api/health
```

### Get Options (Dropdowns)
```
GET /api/options
```

Returns available segments, property types, and states for the UI.

### Predict Bid Fee
```
POST /api/predict
Content-Type: application/json

{
  "business_segment": "Financing",
  "property_type": "Multifamily",
  "property_state": "Illinois",
  "target_time": 30,
  "distance_km": 50,
  "on_due_date": 0
}
```

Response:
```json
{
  "success": true,
  "prediction": {
    "predicted_fee": 3456.78,
    "confidence_interval": {
      "low": 2890.00,
      "high": 4120.00
    },
    "confidence_level": "high",
    "segment_benchmark": 3200.00,
    "recommendation": "..."
  }
}
```

### Get Segment Statistics
```
GET /api/segment/{segment_name}
```

### Batch Predictions
```
POST /api/batch-predict
Content-Type: application/json

{
  "bids": [
    {"business_segment": "Financing", ...},
    {"business_segment": "Consulting", ...}
  ]
}
```

## Production Deployment

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Notes

- Win probability is marked as **experimental** pending validation
- Confidence intervals use empirical quantile bands (stratified by fee bucket)
- Model trained on 2023+ data only
