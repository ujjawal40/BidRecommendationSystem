import { useState } from 'react';
import './ResultDisplay.css';

// ── Helpers ─────────────────────────────────────────────────────────────────

function fmt(n) {
  return Math.round(n).toLocaleString('en-US');
}

function interpolateWinProb(fee, curvePoints) {
  if (!curvePoints || curvePoints.length === 0) return null;
  const pts = [...curvePoints].sort((a, b) => a.fee - b.fee);
  if (fee <= pts[0].fee) return pts[0].win_probability;
  if (fee >= pts[pts.length - 1].fee) return pts[pts.length - 1].win_probability;
  for (let i = 0; i < pts.length - 1; i++) {
    if (fee >= pts[i].fee && fee <= pts[i + 1].fee) {
      const t = (fee - pts[i].fee) / (pts[i + 1].fee - pts[i].fee);
      return pts[i].win_probability + t * (pts[i + 1].win_probability - pts[i].win_probability);
    }
  }
  return pts[pts.length - 1].win_probability;
}

function winLabel(pct) {
  if (pct >= 65) return 'Strong chance of winning';
  if (pct >= 45) return 'Solid chance of winning';
  if (pct >= 30) return 'Moderate chance of winning';
  return 'Low chance of winning';
}

function winColor(pct) {
  if (pct >= 55) return '#10b981';
  if (pct >= 35) return '#f59e0b';
  return '#f43f5e';
}

// ── Main Component ────────────────────────────────────────────────────────────

function ResultDisplay({ prediction, formData }) {
  const {
    predicted_fee,
    ev_optimal_fee,
    solid_win_ceiling,
    ev_capped_at_ceiling,
    confidence_interval,
    confidence_level,
    segment_benchmark,
    state_benchmark,
    win_probability,
    fee_curve,
    warnings,
    factors,
    recommendation,
  } = prediction;

  const [showDetails, setShowDetails] = useState(false);

  const curvePoints = fee_curve?.curve_points || [];

  // Three anchor values
  const floorFee = confidence_interval.low;
  const recFee   = ev_optimal_fee    || predicted_fee;
  const maxFee   = solid_win_ceiling || confidence_interval.high;
  const evCapped = ev_capped_at_ceiling || false;

  // Win prob at recommended fee
  const rawWinProb = interpolateWinProb(recFee, curvePoints);
  const winProbPct = Math.round(rawWinProb ?? win_probability?.probability_pct ?? 50);
  const evAtRec    = Math.round((winProbPct / 100) * recFee);

  // Track position (2-98%)
  const recTrackPct = maxFee > floorFee
    ? Math.max(2, Math.min(98, ((recFee - floorFee) / (maxFee - floorFee)) * 100))
    : 50;

  const recEqualsMax = Math.abs(recFee - maxFee) < 50;

  // Flat curve detection
  const probValues  = curvePoints.map(p => p.win_probability);
  const probRange   = probValues.length > 1 ? Math.max(...probValues) - Math.min(...probValues) : 0;
  const isFlatCurve = probRange < 8;

  // Market context
  const vsMarket    = ((predicted_fee - segment_benchmark) / segment_benchmark * 100);
  const vsMarketStr = (vsMarket >= 0 ? '+' : '') + vsMarket.toFixed(0) + '%';

  const turnaroundDays = formData.turnaround_days || 30;
  const isRush         = turnaroundDays <= 21;
  const wpColor        = winColor(winProbPct);

  return (
    <div className="result-display">

      {/* ── Warnings ── */}
      {warnings?.length > 0 && (
        <div className="result-warning">
          <span className="warning-icon">⚠</span>
          <span>{warnings[0]}</span>
        </div>
      )}

      {/* ── Bid Range Panel ── */}
      <div className="card result-hero">

        <div className="bid-range-header">
          <span className="bid-range-label">Bid Range</span>
          <span className={`confidence-pill confidence-${confidence_level}`}>
            {confidence_level} confidence
          </span>
        </div>

        {/* Three anchors */}
        <div className="bid-anchors">

          <div className="anchor">
            <span className="anchor-tag">Floor</span>
            <span className="anchor-fee">${fmt(floorFee)}</span>
            <p className="anchor-legend">
              The lower boundary of what similar assignments in this market typically charge.
              Bidding below this is unusual and may signal underpricing.
            </p>
          </div>

          <div className="anchor anchor-center">
            <span className="anchor-tag anchor-tag-rec">Optimal Bid</span>
            <div className="anchor-rec-fee">
              <span className="anchor-rec-currency">$</span>
              <span className="anchor-rec-amount">{fmt(recFee)}</span>
            </div>
            <p className="anchor-legend">
              {evCapped
                ? <>The unconstrained optimal fee exceeds the Bid Ceiling. <strong>Capped here</strong> to keep win odds above 30%.</>
                : <>The fee most likely to maximize your earnings — it weighs both your chance of winning <em>and</em> the revenue when you do.</>
              }
            </p>
          </div>

          <div className="anchor anchor-right">
            <span className="anchor-tag">Bid Ceiling</span>
            <span className="anchor-fee">${fmt(maxFee)}</span>
            <p className="anchor-legend">
              The highest fee where you still have a reasonable shot at winning.
              Above this, your odds drop below our 30% business threshold — a long shot.
            </p>
          </div>
        </div>

        {/* Static scale */}
        <div className="range-scale-wrap">
          <div className="range-scale">
            <div className="range-scale-fill" style={{ width: `${recTrackPct}%` }} />
            <div className="scale-tick tick-floor" />
            <div className="scale-tick tick-rec" style={{ left: `${recTrackPct}%` }} />
            {!recEqualsMax && <div className="scale-tick tick-max" />}
            {recEqualsMax  && <div className="scale-tick tick-rec-max" />}
          </div>
          <div className="range-scale-labels">
            <span className="scale-label-left">Floor</span>
            <span className="scale-label-right">Ceiling</span>
          </div>
        </div>

        {/* Win probability */}
        <div className="win-section">
          <div className="win-row">
            <span className="win-pct" style={{ color: wpColor }}>
              {winProbPct}<span className="win-pct-sym">%</span>
            </span>
            <div className="win-meta">
              <span className="win-label" style={{ color: wpColor }}>{winLabel(winProbPct)}</span>
              <span className="win-ev">EV · ${fmt(evAtRec)} at optimal bid</span>
            </div>
          </div>
          <div className="win-bar-track">
            <div className="win-bar-fill" style={{ width: `${winProbPct}%`, background: wpColor }} />
          </div>
        </div>

      </div>

      {/* ── Flat curve note ── */}
      {isFlatCurve && !evCapped && (
        <div className="flat-curve-note">
          <span className="flat-curve-icon">ℹ</span>
          <span>
            For this type of work, your win odds stay roughly the same no matter where you price
            within the range — meaning charging more doesn't hurt your chances. Bidding near the
            Ceiling earns you more per job won without sacrificing much.
          </span>
        </div>
      )}

      {/* ── API recommendation card (from backend) ── */}
      {recommendation && (
        <div className={`card result-recommendation signal-${recommendation.signal}`}>
          <p className="rec-headline">{recommendation.headline}</p>
          <p className="rec-detail">{recommendation.detail}</p>
          {recommendation.strategy_tip && (
            <p className="rec-tip">{recommendation.strategy_tip}</p>
          )}
        </div>
      )}

      {/* ── Market context ── */}
      <div className="card result-context">
        <div className="context-row">
          <span className="context-segment">
            {formData.business_segment} · {formData.property_state}
          </span>
          <div className="context-right">
            <span className="context-bench">
              {formData.business_segment} avg ${fmt(segment_benchmark)}
            </span>
            <span className={`context-diff ${vsMarket >= 0 ? 'above' : 'below'}`}>
              {vsMarketStr} vs segment
            </span>
          </div>
        </div>
      </div>

      {/* ── Rush callout ── */}
      {isRush && (
        <div className="card result-rush">
          <span className="rush-icon">⚡</span>
          <div>
            <p className="rush-title">Short turnaround premium included</p>
            <p className="rush-body">
              Your {turnaroundDays}-day timeline earns a fee premium over standard assignments.
            </p>
          </div>
        </div>
      )}

      {/* ── Details toggle ── */}
      <button className="details-toggle" onClick={() => setShowDetails(v => !v)}>
        {showDetails ? '▲ Hide details' : '▾ Benchmarks & model factors'}
      </button>

      {showDetails && (
        <div className="result-details">
          <div className="card detail-card">
            <h5>Market Benchmarks</h5>
            <div className="detail-row">
              <span>Segment ({formData.business_segment})</span>
              <span>${fmt(segment_benchmark)}</span>
            </div>
            {state_benchmark && (
              <div className="detail-row">
                <span>State ({formData.property_state})</span>
                <span>${fmt(state_benchmark)}</span>
              </div>
            )}
            {factors?.subtype_effect > 0 && (
              <div className="detail-row">
                <span>Sub-type effect</span>
                <span>${fmt(factors.subtype_effect)}</span>
              </div>
            )}
            {factors?.office_region_effect > 0 && (
              <div className="detail-row">
                <span>Office region effect</span>
                <span>${fmt(factors.office_region_effect)}</span>
              </div>
            )}
            <div className="detail-row">
              <span>Market-typical fee (model)</span>
              <span>${fmt(predicted_fee)}</span>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

export default ResultDisplay;
