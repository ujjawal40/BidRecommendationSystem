import { useState } from 'react';
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis,
  ReferenceLine, ReferenceArea, Tooltip as RechartsTooltip,
  CartesianGrid,
} from 'recharts';
import {
  AlertTriangle, Zap, Info, HelpCircle,
  ChevronUp, ChevronDown, Check,
} from 'lucide-react';
import * as Tip from '@radix-ui/react-tooltip';
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
  if (pct >= 55) return 'Strong chance of winning';
  if (pct >= 35) return 'Moderate chance of winning';
  return 'Low chance of winning';
}

function winColor(pct) {
  if (pct >= 55) return '#1A7A4C';
  if (pct >= 35) return '#A16207';
  return '#B91C1C';
}

// ── Client-side contextual message ──────────────────────────────────────────

function buildContextMessage({ recFee, maxFee, floorFee, winProbPct, confidence, segment, evCapped, isFlatCurve, segBenchmark }) {
  const aboveBench = recFee > segBenchmark;
  const diffPct    = Math.abs(((recFee - segBenchmark) / segBenchmark) * 100).toFixed(0);

  // Case 1: Recommended was capped at ceiling — the model wanted to go higher
  if (evCapped) {
    return {
      headline: 'Max Recommended — the EV-optimal fee exceeds the 30% win threshold',
      body: `The fee that would mathematically maximize your expected earnings sits above the Bid Ceiling, ` +
            `where win odds fall below 30% — our minimum threshold for a viable bid. ` +
            `$${fmt(recFee)} is the highest fee where you still have a reasonable shot at winning.`,
      tip: `If you choose to bid above $${fmt(maxFee)}, win odds will drop below 30%. ` +
           `At that point, your track record, client relationships, and quality of work ` +
           `will matter more than price.`,
      signal: winProbPct >= 40 ? 'neutral' : 'caution',
    };
  }

  // Case 2: Flat curve — price barely matters
  if (isFlatCurve) {
    return {
      headline: 'Price has little effect on your win odds here',
      body: `For ${segment} assignments in this market, win probability stays roughly the same ` +
            `across the entire fee range — meaning raising your price won't meaningfully hurt your chances.`,
      tip: `Since the odds don't change much with fee, bidding near the Bid Ceiling ($${fmt(maxFee)}) ` +
           `gives you the same shot at winning while maximizing what you earn per job. ` +
           `Other factors — your firm's reputation, speed, and relationships — are what will actually decide this.`,
      signal: 'neutral',
    };
  }

  // Case 3: Low win probability (below 30%)
  if (winProbPct < 30) {
    const dataNote = confidence === 'low'
      ? ' The model also has limited data for this exact combination, so treat this as a rough estimate.'
      : '';
    return {
      headline: 'Tough market — win odds are modest even at the recommended fee',
      body: `Estimated win probability is ${winProbPct}%, which is below our 30% viability threshold. ` +
            `This is a highly competitive assignment.` + dataNote,
      tip: `In segments this competitive, price is rarely the deciding factor. ` +
           `Your track record for this type of work, your client relationships, and what you offer ` +
           `beyond the fee itself will likely matter more than where you price.`,
      signal: 'caution',
    };
  }

  // Case 4: Low confidence (wide fee spread or sparse data)
  if (confidence === 'low') {
    return {
      headline: 'Estimate is directional — fees vary widely in this market',
      body: `${segment} fees in this state have a wide spread, so the confidence interval is broader than usual. ` +
            `Your recommended bid of $${fmt(recFee)} is ${aboveBench ? `${diffPct}% above` : `${diffPct}% below`} ` +
            `the segment average ($${fmt(segBenchmark)}).`,
      tip: `Because fees vary a lot here, check what you've charged for similar jobs before committing to this bid. ` +
           `The floor and ceiling give you the safe range — anywhere in that band is defensible.`,
      signal: 'neutral',
    };
  }

  // Case 5: Strong odds — well positioned
  if (winProbPct >= 55) {
    return {
      headline: aboveBench
        ? `Above market with strong odds — good position`
        : `Well priced with strong odds`,
      body: `Your recommended bid of $${fmt(recFee)} gives you a ${winProbPct}% estimated chance of winning. ` +
            (aboveBench
              ? `Even at ${diffPct}% above the ${segment} average ($${fmt(segBenchmark)}), your odds are strong — the market supports this price.`
              : `At ${diffPct}% below the ${segment} average ($${fmt(segBenchmark)}), you're priced competitively.`),
      tip: aboveBench && diffPct > 20
        ? `You could potentially push toward the Bid Ceiling ($${fmt(maxFee)}) — win odds would still be reasonable and you'd capture more revenue per job.`
        : null,
      signal: 'positive',
    };
  }

  // Case 6: Default — moderate odds, normal range
  return {
    headline: aboveBench
      ? `Above market with moderate odds`
      : `Competitive bid with reasonable odds`,
    body: `Your recommended bid of $${fmt(recFee)} gives you a ${winProbPct}% estimated chance of winning. ` +
          `The ${segment} market average in this area is around $${fmt(segBenchmark)}, ` +
          `putting you ${aboveBench ? `${diffPct}% above` : `${diffPct}% below`} that benchmark.`,
    tip: winProbPct < 40
      ? `Win odds are modest — if this client is price-sensitive, consider whether sliding closer to the ` +
        `floor ($${fmt(recFee > floorFee ? floorFee : recFee)}) would meaningfully improve your chances.`
      : null,
    signal: 'neutral',
  };
}

// ── Radix Tooltip wrapper ────────────────────────────────────────────────────

function InfoTip({ children, content, side = 'top' }) {
  return (
    <Tip.Provider delayDuration={200}>
      <Tip.Root>
        <Tip.Trigger asChild>{children}</Tip.Trigger>
        <Tip.Portal>
          <Tip.Content className="tooltip-content" side={side} sideOffset={6}>
            {content}
            <Tip.Arrow className="tooltip-arrow" />
          </Tip.Content>
        </Tip.Portal>
      </Tip.Root>
    </Tip.Provider>
  );
}

// ── Custom Recharts tooltip ──────────────────────────────────────────────────

function ChartTooltipContent({ active, payload }) {
  if (!active || !payload || payload.length === 0) return null;
  const data = payload[0].payload;
  const wp = Math.round(data.win_probability);
  const ev = Math.round((wp / 100) * data.fee);
  const color = winColor(wp);
  return (
    <div className="recharts-custom-tooltip">
      <span className="rct-fee">${fmt(data.fee)}</span>
      <span className="rct-prob" style={{ color }}>{wp}% win</span>
      <span className="rct-ev">EV ${fmt(ev)}</span>
    </div>
  );
}

// ── SVG Fee Chart ─────────────────────────────────────────────────────────────

function FeeChart({ curvePoints, recFee, maxFee, floorFee, evCapped }) {
  const pts = [...curvePoints].sort((a, b) => a.fee - b.fee);
  if (pts.length < 2) return null;

  return (
    <div className="fee-chart">
      <div className="fee-chart-title">Fee vs. Win Probability</div>
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={pts} margin={{ top: 10, right: 16, bottom: 0, left: -12 }}>
          <defs>
            <linearGradient id="winProbGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#2B5A83" stopOpacity={0.12} />
              <stop offset="100%" stopColor="#2B5A83" stopOpacity={0.01} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.05)" vertical={false} />
          <XAxis
            dataKey="fee"
            tickFormatter={v => `$${fmt(v)}`}
            tick={{ fontSize: 10, fill: '#8C8C88' }}
            axisLine={{ stroke: 'rgba(0,0,0,0.08)' }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 100]}
            tickFormatter={v => `${v}%`}
            tick={{ fontSize: 10, fill: '#8C8C88' }}
            axisLine={false}
            tickLine={false}
          />
          <ReferenceArea
            x1={Math.max(floorFee, pts[0].fee)}
            x2={Math.min(maxFee, pts[pts.length - 1].fee)}
            fill="rgba(43,90,131,0.04)"
            ifOverflow="hidden"
          />
          <ReferenceLine
            y={30}
            stroke="rgba(185,28,28,0.25)"
            strokeDasharray="4 3"
            label={{ value: '30% min', position: 'insideTopLeft', fontSize: 9, fill: 'rgba(185,28,28,0.5)' }}
          />
          <ReferenceLine
            x={floorFee}
            stroke="rgba(0,0,0,0.15)"
            strokeDasharray="3 3"
            label={{ value: 'Floor', position: 'insideTopLeft', fontSize: 9, fill: '#8C8C88' }}
          />
          <ReferenceLine
            x={maxFee}
            stroke="rgba(43,90,131,0.3)"
            strokeDasharray="3 3"
            label={{ value: 'Ceiling', position: 'insideTopRight', fontSize: 9, fill: '#8C8C88' }}
          />
          <ReferenceLine
            x={recFee}
            stroke="#2B5A83"
            strokeWidth={1.5}
            label={{ value: evCapped ? 'Max Rec.' : 'Optimal', position: 'insideTopRight', fontSize: 9, fill: '#2B5A83', fontWeight: 600 }}
          />
          <Area
            type="monotone"
            dataKey="win_probability"
            stroke="#2B5A83"
            strokeWidth={2}
            fill="url(#winProbGrad)"
            dot={false}
            activeDot={{ r: 5, fill: '#2B5A83', stroke: '#fff', strokeWidth: 2 }}
          />
          <RechartsTooltip
            content={<ChartTooltipContent />}
            cursor={{ stroke: 'rgba(0,0,0,0.08)', strokeWidth: 1 }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
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
    metadata,
  } = prediction;

  const [showDetails, setShowDetails] = useState(false);
  const [showChart,   setShowChart]   = useState(true);
  const [whatIfFee,   setWhatIfFee]   = useState('');
  const [copied,      setCopied]      = useState(false);

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

  // Market context — use recFee (what the user is told to bid) not predicted_fee
  const vsMarket    = ((recFee - segment_benchmark) / segment_benchmark * 100);
  const vsMarketStr = (vsMarket >= 0 ? '+' : '') + vsMarket.toFixed(0) + '%';

  const turnaroundDays = formData.turnaround_days || 30;
  const isRush         = turnaroundDays <= 21;
  const wpColor        = winColor(winProbPct);

  // Build contextual message client-side (always fresh, no stale API text)
  const ctxMsg = buildContextMessage({
    recFee,
    maxFee,
    floorFee,
    winProbPct,
    confidence: confidence_level,
    segment:    formData.business_segment,
    evCapped,
    isFlatCurve,
    segBenchmark: segment_benchmark,
  });

  // Bid quality summary
  const bidQuality = (() => {
    if (winProbPct >= 55 && confidence_level !== 'low') return { label: 'Good bid', cls: 'quality-good' };
    if (winProbPct >= 35 || (winProbPct >= 30 && confidence_level === 'high')) return { label: 'Fair bid', cls: 'quality-fair' };
    return { label: 'Risky bid', cls: 'quality-risky' };
  })();

  return (
    <div className="result-display">

      {/* ── Warnings ── */}
      {warnings?.length > 0 && (
        <div className="result-warning">
          <AlertTriangle size={16} className="warning-icon" />
          <span>{warnings[0]}</span>
        </div>
      )}

      {/* ── Bid Range Panel ── */}
      <div className="card result-hero">

        <div className="bid-range-header">
          <div className="bid-range-left">
            <span className="bid-range-label">Bid Range</span>
            <span className={`quality-badge ${bidQuality.cls}`}>{bidQuality.label}</span>
          </div>
          <InfoTip content={
            confidence_level === 'high'
              ? `Based on ${fmt(metadata?.data_coverage?.segment_samples || 0)} similar bids — estimate is narrow and reliable`
              : confidence_level === 'medium'
              ? `Moderate data for this combination — estimate is directional but range is wider`
              : `Sparse data for this exact combination — treat as a rough guide, cross-reference with experience`
          }>
            <span className={`confidence-pill confidence-${confidence_level}`}>
              {confidence_level} confidence
              {metadata?.data_coverage?.segment_samples && (
                <span className="coverage-count"> · {fmt(metadata.data_coverage.segment_samples)} similar bids</span>
              )}
            </span>
          </InfoTip>
        </div>

        {/* Bid anchors — 2 columns when capped, 3 otherwise */}
        <div className={recEqualsMax ? 'bid-anchors-two' : 'bid-anchors'}>

          <div className="anchor">
            <span className="anchor-tag">Floor</span>
            <span className="anchor-fee">${fmt(floorFee)}</span>
          </div>

          {recEqualsMax ? (
            <div className="anchor anchor-right">
              <span
                className={`anchor-tag ${evCapped ? 'anchor-tag-capped' : 'anchor-tag-rec'}`}
              >
                Recommended Bid
              </span>
              <div className="anchor-rec-fee" style={{ justifyContent: 'flex-end', cursor: 'pointer' }}
                title="Click to copy"
                onClick={() => { navigator.clipboard.writeText(Math.round(recFee).toString()); setCopied(true); setTimeout(() => setCopied(false), 1500); }}>
                <span className="anchor-rec-currency">$</span>
                <span className="anchor-rec-amount">{fmt(recFee)}</span>
                {copied && <span className="copied-badge">Copied</span>}
              </div>
              {evCapped && (
                <span className="capped-badge">⚠ capped at 30% win floor</span>
              )}
              <span className="ceiling-subtitle">(at bid ceiling)</span>
            </div>
          ) : (
            <>
              <div className="anchor anchor-center">
                <span
                  className={`anchor-tag ${evCapped ? 'anchor-tag-capped' : 'anchor-tag-rec'}`}
                >
                  Optimal Bid
                </span>
                <div className="anchor-rec-fee"
                  title="Click to copy"
                  style={{ cursor: 'pointer' }}
                  onClick={() => { navigator.clipboard.writeText(Math.round(recFee).toString()); setCopied(true); setTimeout(() => setCopied(false), 1500); }}>
                  <span className="anchor-rec-currency">$</span>
                  <span className="anchor-rec-amount">{fmt(recFee)}</span>
                  {copied && <span className="copied-badge">Copied</span>}
                </div>
                {evCapped && (
                  <span className="capped-badge">⚠ capped at 30% win floor</span>
                )}
              </div>

              <div className="anchor anchor-right">
                <span className="anchor-tag">Bid Ceiling</span>
                <span className="anchor-fee">${fmt(maxFee)}</span>
              </div>
            </>
          )}
        </div>

        {/* Static scale */}
        <div className="range-scale-wrap">
          <div className="range-scale">
            <div className="range-scale-fill" style={{ width: `${recTrackPct}%`, opacity: evCapped ? 0.6 : 0.45, background: evCapped ? 'var(--warning)' : 'var(--accent)' }} />
            <div className="scale-tick tick-floor" />
            <div className={`scale-tick tick-rec${evCapped ? ' tick-rec-capped' : ''}`} style={{ left: `${recTrackPct}%` }} />
            {!recEqualsMax && <div className="scale-tick tick-max" />}
            {recEqualsMax  && <div className="scale-tick tick-rec-max" />}
          </div>
          <div className="range-scale-labels">
            <span className="scale-label-left">Floor</span>
            <span className="scale-label-right">Ceiling</span>
          </div>
          <p className="band-note">The defensible pricing range based on historical outcomes in this segment</p>
        </div>

        {/* Win probability */}
        <div className="win-section">
          <span className="win-section-label">Win Probability</span>
          <div className="win-row">
            <span className="win-pct" style={{ color: wpColor }}>
              {winProbPct}<span className="win-pct-sym">%</span>
            </span>
            <div className="win-meta">
              <span className="win-label" style={{ color: wpColor }}>{winLabel(winProbPct)}</span>
              <span className="win-ev" title="Expected Value = P(Win) × Fee — your average earnings per bid at this price">
                EV · ${fmt(evAtRec)} at {evCapped ? 'max recommended' : 'optimal bid'}
                <span className="ev-help">?</span>
              </span>
            </div>
          </div>
          <div className="win-bar-track">
            <div className="win-bar-fill" style={{ width: `${winProbPct}%`, background: wpColor }} />
          </div>
          <p className="win-interpretation">
            {winProbPct >= 55
              ? `If you bid $${fmt(recFee)} on 10 similar jobs, you'd expect to win about ${Math.round(winProbPct / 10)} of them.`
              : winProbPct >= 35
              ? `About ${Math.round(winProbPct / 10)} in 10 similar bids would be won at this price — non-price factors will tip the balance.`
              : `At this price you'd win roughly ${Math.round(winProbPct / 10)} in 10 similar bids — relationships and expertise will matter more than price.`}
          </p>
          {win_probability?.model_used && win_probability.model_used.includes('Heuristic') && (
            <span className="model-source-badge" title="The ML model is being supplemented by a calibrated heuristic for fee-sensitivity">
              Heuristic estimate
            </span>
          )}
        </div>

      </div>

      {/* ── What-if quick input ── */}
      {curvePoints.length > 1 && (
        <div className="card what-if-card">
          <div className="what-if-row">
            <label className="what-if-label" htmlFor="what-if-fee">What if I bid</label>
            <div className="what-if-input-wrap">
              <span className="what-if-dollar">$</span>
              <input
                id="what-if-fee"
                type="number"
                className="what-if-input"
                placeholder={fmt(recFee)}
                value={whatIfFee}
                onChange={e => setWhatIfFee(e.target.value)}
                min={0}
                step={50}
              />
            </div>
            {whatIfFee && Number(whatIfFee) > 0 && (() => {
              const wf = Number(whatIfFee);
              const wp = Math.round(interpolateWinProb(wf, curvePoints) ?? 50);
              const ev = Math.round((wp / 100) * wf);
              const evDelta = ev - evAtRec;
              return (
                <div className="what-if-result">
                  <span className="what-if-prob" style={{ color: winColor(wp) }}>{wp}%</span>
                  <span className="what-if-detail">
                    win · EV ${fmt(ev)}
                    <span className={`what-if-delta ${evDelta >= 0 ? 'positive' : 'negative'}`}>
                      {evDelta >= 0 ? '+' : ''}{fmt(evDelta)} EV
                    </span>
                  </span>
                </div>
              );
            })()}
          </div>
          {whatIfFee && Number(whatIfFee) > 0 && Number(whatIfFee) > maxFee && (
            <p className="what-if-warn">Above bid ceiling — win odds drop below 30%</p>
          )}
        </div>
      )}

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

      {/* ── Contextual insight card (client-side, always fresh) ── */}
      <div className={`card result-insight signal-${ctxMsg.signal}`}>
        <p className="insight-headline">{ctxMsg.headline}</p>
        <p className="insight-body">{ctxMsg.body}</p>
        {ctxMsg.tip && (
          <p className="insight-tip">{ctxMsg.tip}</p>
        )}
      </div>

      {/* ── Fee analysis chart (expandable) ── */}
      {curvePoints.length > 1 && (
        <div className="chart-section">
          <button className="chart-toggle" onClick={() => setShowChart(v => !v)}>
            {showChart ? '▲ Hide fee analysis' : '▾ Show fee vs. win probability chart'}
          </button>
          {showChart && (
            <div className="card chart-card">
              <FeeChart
                curvePoints={curvePoints}
                recFee={recFee}
                maxFee={maxFee}
                floorFee={floorFee}
                evCapped={evCapped}
              />
            </div>
          )}
        </div>
      )}

      {/* ── Market context ── */}
      <div className="card result-context">
        <div className="context-top">
          <span className="context-segment">
            {formData.business_segment} · {formData.property_state}
            {formData.zip_code && formData.zip_code.length === 5 && (
              <span className="context-zip"> · ZIP {formData.zip_code}</span>
            )}
          </span>
        </div>
        <div className="context-metrics">
          <div className="context-metric">
            <span className="context-metric-label">Segment avg</span>
            <span className="context-metric-value">${fmt(segment_benchmark)}</span>
          </div>
          <div className="context-metric">
            <span className="context-metric-label">Your bid vs segment</span>
            <span className={`context-metric-value context-diff ${vsMarket >= 0 ? 'above' : 'below'}`}>
              {vsMarketStr}
            </span>
          </div>
          <div className="context-metric">
            <span className="context-metric-label">Floor–Ceiling spread</span>
            <span className="context-metric-value">${fmt(maxFee - floorFee)}</span>
          </div>
          {state_benchmark && (
            <div className="context-metric">
              <span className="context-metric-label">State avg</span>
              <span className="context-metric-value">${fmt(state_benchmark)}</span>
            </div>
          )}
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
            <p className="detail-explainer">Average historical fees by category — these are reference points, not model inputs.</p>
            <div className="detail-row">
              <span>Avg fee in {formData.business_segment}</span>
              <span>${fmt(segment_benchmark)}</span>
            </div>
            {state_benchmark && (
              <div className="detail-row">
                <span>Avg fee in {formData.property_state}</span>
                <span>${fmt(state_benchmark)}</span>
              </div>
            )}
            {factors?.subtype_effect > 0 && (
              <div className="detail-row">
                <span>Avg fee for sub-type</span>
                <span>${fmt(factors.subtype_effect)}</span>
              </div>
            )}
            {factors?.office_region_effect > 0 && (
              <div className="detail-row">
                <span>Avg fee in office region</span>
                <span>${fmt(factors.office_region_effect)}</span>
              </div>
            )}
            <div className="detail-row detail-row-highlight">
              <span>Model prediction (market-typical fee)</span>
              <span>${fmt(predicted_fee)}</span>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

export default ResultDisplay;
