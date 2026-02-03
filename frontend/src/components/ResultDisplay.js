import React from 'react';
import './ResultDisplay.css';

function ResultDisplay({ prediction, formData }) {
  const {
    predicted_fee,
    confidence_interval,
    confidence_level,
    segment_benchmark,
    state_benchmark,
    recommendation,
    factors,
    win_probability,
    expected_value,
  } = prediction;

  // Calculate how prediction compares to benchmarks
  const vsSegment = ((predicted_fee - segment_benchmark) / segment_benchmark * 100).toFixed(1);
  const vsState = ((predicted_fee - state_benchmark) / state_benchmark * 100).toFixed(1);

  // Use win probability from ML model (or fallback if not available)
  const winProbability = win_probability?.probability_pct || calculateFallbackWinProb(predicted_fee, segment_benchmark);
  const winProbConfidence = win_probability?.confidence || 'low';
  const modelUsed = win_probability?.model_used || 'Fallback Heuristic';

  return (
    <div className="result-display">
      {/* Main Prediction Card */}
      <div className="card result-main">
        <div className="result-header">
          <span className="result-label">Recommended Bid Fee</span>
          <span className={`confidence-badge confidence-${confidence_level}`}>
            {confidence_level} confidence
          </span>
        </div>

        <div className="predicted-fee">
          <span className="currency">$</span>
          <span className="amount">{predicted_fee.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}</span>
        </div>

        {/* Confidence Interval */}
        <div className="confidence-interval">
          <div className="interval-bar">
            <div
              className="interval-range"
              style={{
                left: `${Math.max(0, (confidence_interval.low / confidence_interval.high) * 40)}%`,
                right: '10%',
              }}
            />
            <div
              className="interval-marker"
              style={{ left: `${((predicted_fee - confidence_interval.low) / (confidence_interval.high - confidence_interval.low)) * 80 + 10}%` }}
            />
          </div>
          <div className="interval-labels">
            <span>${confidence_interval.low.toLocaleString()}</span>
            <span className="interval-label-center">80% Confidence Band</span>
            <span>${confidence_interval.high.toLocaleString()}</span>
          </div>
        </div>
      </div>

      {/* Win Probability Card */}
      <div className="card result-winprob">
        <div className="winprob-header">
          <h4>Win Probability</h4>
          <span className={`confidence-tag confidence-${winProbConfidence}`}>
            {winProbConfidence} confidence
          </span>
        </div>

        <div className="winprob-display">
          <div className="winprob-circle">
            <svg viewBox="0 0 36 36" className="winprob-chart">
              <path
                d="M18 2.0845
                  a 15.9155 15.9155 0 0 1 0 31.831
                  a 15.9155 15.9155 0 0 1 0 -31.831"
                fill="none"
                stroke="#e0e0e0"
                strokeWidth="3"
              />
              <path
                d="M18 2.0845
                  a 15.9155 15.9155 0 0 1 0 31.831
                  a 15.9155 15.9155 0 0 1 0 -31.831"
                fill="none"
                stroke="#38a169"
                strokeWidth="3"
                strokeDasharray={`${winProbability}, 100`}
              />
            </svg>
            <span className="winprob-value">{Math.round(winProbability)}%</span>
          </div>
          <p className="winprob-note">
            {modelUsed}
          </p>
          {expected_value && (
            <p className="expected-value">
              Expected Value: <strong>${expected_value.toLocaleString()}</strong>
            </p>
          )}
        </div>
      </div>

      {/* Benchmarks Comparison */}
      <div className="card result-benchmarks">
        <h4>Market Comparison</h4>

        <div className="benchmark-item">
          <div className="benchmark-info">
            <span className="benchmark-label">vs. Segment Average</span>
            <span className="benchmark-name">{formData.business_segment}</span>
          </div>
          <div className="benchmark-values">
            <span className="benchmark-base">${segment_benchmark.toLocaleString()}</span>
            <span className={`benchmark-diff ${parseFloat(vsSegment) >= 0 ? 'positive' : 'negative'}`}>
              {vsSegment > 0 ? '+' : ''}{vsSegment}%
            </span>
          </div>
        </div>

        <div className="benchmark-item">
          <div className="benchmark-info">
            <span className="benchmark-label">vs. State Average</span>
            <span className="benchmark-name">{formData.property_state}</span>
          </div>
          <div className="benchmark-values">
            <span className="benchmark-base">${state_benchmark.toLocaleString()}</span>
            <span className={`benchmark-diff ${parseFloat(vsState) >= 0 ? 'positive' : 'negative'}`}>
              {vsState > 0 ? '+' : ''}{vsState}%
            </span>
          </div>
        </div>
      </div>

      {/* Recommendation */}
      <div className="card result-recommendation">
        <h4>Recommendation</h4>
        <p>{recommendation}</p>
      </div>

      {/* Factors Breakdown */}
      <div className="card result-factors">
        <h4>Key Factors</h4>
        <div className="factors-grid">
          <div className="factor-item">
            <span className="factor-label">Segment Effect</span>
            <span className="factor-value">${factors.segment_effect?.toLocaleString() || '-'}</span>
          </div>
          <div className="factor-item">
            <span className="factor-label">State Effect</span>
            <span className="factor-value">${factors.state_effect?.toLocaleString() || '-'}</span>
          </div>
          <div className="factor-item">
            <span className="factor-label">Office Effect</span>
            <span className="factor-value">${factors.office_effect?.toLocaleString() || '-'}</span>
          </div>
          <div className="factor-item">
            <span className="factor-label">Time Factor</span>
            <span className="factor-value">{factors.time_factor} days</span>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Fallback win probability calculation when API doesn't return model prediction.
 * Uses smooth sigmoid curve instead of hard buckets.
 */
function calculateFallbackWinProb(predicted, benchmark) {
  const ratio = predicted / benchmark;

  // Smooth sigmoid-like function
  // Lower ratio (more competitive) = higher win probability
  const k = 5; // Steepness
  const probability = 1 / (1 + Math.exp(k * (ratio - 1)));

  // Scale to realistic range (20% - 75%)
  return 20 + (probability * 55);
}

export default ResultDisplay;
