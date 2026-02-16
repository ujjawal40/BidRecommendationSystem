import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer,
} from 'recharts';
import './FeeSensitivityCharts.css';

function FeeSensitivityCharts({ curveData }) {
  if (!curveData || !curveData.curve_points || curveData.curve_points.length === 0) {
    return null;
  }

  const { curve_points, ev_optimal_fee, ev_optimal_ev, recommended_fee } = curveData;

  const fmtDollar = (val) => `$${Number(val).toLocaleString()}`;
  const fmtPct = (val) => `${val}%`;

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    const d = payload[0].payload;
    return (
      <div className="chart-tooltip">
        <p className="tooltip-fee">{fmtDollar(d.fee)}</p>
        <p>Win Probability: <strong>{d.win_probability}%</strong></p>
        <p>Expected Value: <strong>{fmtDollar(d.expected_value)}</strong></p>
      </div>
    );
  };

  const showEvDiffers = Math.abs(ev_optimal_fee - recommended_fee) / recommended_fee > 0.10;

  return (
    <div className="fee-sensitivity-charts">
      <div className="card chart-card">
        <h4>Win Probability vs. Fee</h4>
        <p className="chart-subtitle">
          How your chance of winning changes at different fee levels
        </p>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={curve_points} margin={{ top: 10, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis
              dataKey="fee"
              tickFormatter={fmtDollar}
            />
            <YAxis
              tickFormatter={fmtPct}
              domain={[0, 100]}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine
              x={recommended_fee}
              stroke="#2563eb"
              strokeDasharray="5 5"
              label={{ value: 'Recommended', position: 'top', fill: '#2563eb', fontSize: 11 }}
            />
            <Line
              type="monotone"
              dataKey="win_probability"
              stroke="#38a169"
              strokeWidth={2}
              dot={{ r: 3 }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="card chart-card">
        <h4>Expected Value vs. Fee</h4>
        <p className="chart-subtitle">
          Revenue-weighted outcome: EV = Win Probability &times; Fee
        </p>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={curve_points} margin={{ top: 10, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis
              dataKey="fee"
              tickFormatter={fmtDollar}
            />
            <YAxis
              tickFormatter={fmtDollar}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine
              x={recommended_fee}
              stroke="#2563eb"
              strokeDasharray="5 5"
              label={{ value: 'Recommended', position: 'top', fill: '#2563eb', fontSize: 11 }}
            />
            <ReferenceLine
              x={ev_optimal_fee}
              stroke="#d97706"
              strokeDasharray="3 3"
              label={{ value: 'Peak EV', position: 'top', fill: '#d97706', fontSize: 11 }}
            />
            <Line
              type="monotone"
              dataKey="expected_value"
              stroke="#d97706"
              strokeWidth={2}
              dot={{ r: 3 }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>

        {showEvDiffers && (
          <div className="ev-insight">
            <strong>Insight:</strong> The EV-maximizing fee ({fmtDollar(ev_optimal_fee)})
            differs from the model's recommended fee ({fmtDollar(recommended_fee)}).
            Peak expected revenue is {fmtDollar(ev_optimal_ev)}.
          </div>
        )}
      </div>
    </div>
  );
}

export default FeeSensitivityCharts;
