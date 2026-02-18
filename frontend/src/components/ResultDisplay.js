import React from 'react';
import FeeSensitivityCharts from './FeeSensitivityCharts';
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
    fee_curve,
    warnings,
  } = prediction;

  // Calculate how prediction compares to benchmarks
  const vsSegment = ((predicted_fee - segment_benchmark) / segment_benchmark * 100).toFixed(1);
  const vsState = ((predicted_fee - state_benchmark) / state_benchmark * 100).toFixed(1);

  // Use win probability from ML model (or fallback if not available)
  const winProbability = win_probability?.probability_pct || calculateFallbackWinProb(predicted_fee, segment_benchmark);
  const winProbConfidence = win_probability?.confidence || 'low';

  // Win probability color tier
  const winTierClass = winProbability >= 70 ? 'winprob-high' : winProbability >= 40 ? 'winprob-moderate' : 'winprob-low-tier';
  const winStrokeColor = winProbability >= 70 ? '#10b981' : winProbability >= 40 ? '#f59e0b' : '#ef4444';

  // Structured vs legacy recommendation
  const isStructured = recommendation && typeof recommendation === 'object' && recommendation.headline;

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

      {/* Low Data Warnings */}
      {warnings && warnings.length > 0 && (
        <div className="card result-warnings">
          {warnings.map((w, i) => (
            <p key={i} className="warning-text">{w}</p>
          ))}
        </div>
      )}

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
                stroke="#e5e7eb"
                strokeWidth="3"
              />
              <path
                className="winprob-progress"
                d="M18 2.0845
                  a 15.9155 15.9155 0 0 1 0 31.831
                  a 15.9155 15.9155 0 0 1 0 -31.831"
                fill="none"
                stroke={winStrokeColor}
                strokeWidth="3"
                strokeDasharray={`${winProbability}, 100`}
                strokeLinecap="round"
              />
            </svg>
            <span className={`winprob-value ${winTierClass}`}>{Math.round(winProbability)}%</span>
          </div>
          <p className="winprob-note">
            Probability of winning this bid at the recommended fee
          </p>
        </div>
      </div>

      {/* Expected Value */}
      {expected_value && (
        <div className="card result-ev">
          <h4>Expected Value</h4>
          <div className="ev-display">
            <span className="ev-amount">
              ${Math.round(expected_value).toLocaleString()}
            </span>
            <span className="ev-formula">
              EV = {Math.round(winProbability)}% win prob x ${predicted_fee.toLocaleString('en-US', { maximumFractionDigits: 0 })} fee
            </span>
          </div>
        </div>
      )}

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
      {isStructured ? (
        <div className={`card result-recommendation signal-${recommendation.signal || 'neutral'}`}>
          <h4 className="rec-headline">{recommendation.headline}</h4>
          <p className="rec-detail">{recommendation.detail}</p>
          {recommendation.strategy_tip && (
            <p className="rec-strategy">{recommendation.strategy_tip}</p>
          )}
        </div>
      ) : (
        <div className="card result-recommendation signal-neutral">
          <h4>Recommendation</h4>
          <p className="legacy-rec">{typeof recommendation === 'string' ? recommendation : ''}</p>
        </div>
      )}

      {/* Fee Sensitivity Charts */}
      {fee_curve && (
        <FeeSensitivityCharts curveData={fee_curve} />
      )}

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
            <span className="factor-label">SubType Effect</span>
            <span className="factor-value">${factors.subtype_effect?.toLocaleString() || factors.office_effect?.toLocaleString() || '-'}</span>
          </div>
          <div className="factor-item">
            <span className="factor-label">Office Region</span>
            <span className="factor-value">${factors.office_region_effect?.toLocaleString() || factors.time_factor || '-'}</span>
          </div>
          {factors.delivery_days && (
            <div className="factor-item">
              <span className="factor-label">Delivery Days</span>
              <span className="factor-value">{factors.delivery_days} days</span>
            </div>
          )}
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
