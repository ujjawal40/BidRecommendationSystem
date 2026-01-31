import React, { useState, useEffect } from 'react';
import './StateBenchmarks.css';

// Hardcoded state benchmarks (would come from API in production)
const STATE_DATA = {
  'Illinois': { avg_fee: 3245, volume: 26748, win_rate: 0.38, rank: 1 },
  'Florida': { avg_fee: 3512, volume: 15254, win_rate: 0.41, rank: 2 },
  'Texas': { avg_fee: 3678, volume: 8321, win_rate: 0.39, rank: 3 },
  'Missouri': { avg_fee: 2987, volume: 5697, win_rate: 0.36, rank: 4 },
  'Ohio': { avg_fee: 3124, volume: 5060, win_rate: 0.37, rank: 5 },
  'California': { avg_fee: 4521, volume: 3066, win_rate: 0.42, rank: 6 },
  'Indiana': { avg_fee: 2876, volume: 3299, win_rate: 0.35, rank: 7 },
  'Unknown': { avg_fee: 3345, volume: 32580, win_rate: 0.37, rank: 0 },
};

function StateBenchmarks({ state, states }) {
  const [selectedState, setSelectedState] = useState(state);
  const [stateStats, setStateStats] = useState(null);

  useEffect(() => {
    if (state) {
      setSelectedState(state);
      loadStateData(state);
    }
  }, [state]);

  const loadStateData = (stateName) => {
    // In production, this would be an API call
    const data = STATE_DATA[stateName] || {
      avg_fee: 3200,
      volume: 100,
      win_rate: 0.36,
      rank: '-',
    };
    setStateStats(data);
  };

  const handleStateChange = (e) => {
    const newState = e.target.value;
    setSelectedState(newState);
    loadStateData(newState);
  };

  // Get top states for comparison
  const topStates = Object.entries(STATE_DATA)
    .filter(([name]) => name !== 'Unknown')
    .sort((a, b) => b[1].volume - a[1].volume)
    .slice(0, 5);

  return (
    <div className="card state-benchmarks">
      <div className="benchmarks-header">
        <h3>State Benchmarks</h3>
        <select
          value={selectedState}
          onChange={handleStateChange}
          className="state-select"
        >
          {states.map(s => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>

      {stateStats && (
        <div className="selected-state-stats">
          <div className="state-stat-row">
            <span className="state-stat-label">Average Fee</span>
            <span className="state-stat-value">${stateStats.avg_fee.toLocaleString()}</span>
          </div>
          <div className="state-stat-row">
            <span className="state-stat-label">Bid Volume</span>
            <span className="state-stat-value">{stateStats.volume.toLocaleString()}</span>
          </div>
          <div className="state-stat-row">
            <span className="state-stat-label">Win Rate</span>
            <span className="state-stat-value">{(stateStats.win_rate * 100).toFixed(1)}%</span>
          </div>
        </div>
      )}

      <div className="top-states">
        <h4>Top States by Volume</h4>
        <div className="top-states-list">
          {topStates.map(([name, data], index) => (
            <div
              key={name}
              className={`top-state-item ${name === selectedState ? 'active' : ''}`}
            >
              <span className="state-rank">#{index + 1}</span>
              <span className="state-name">{name}</span>
              <div className="state-bar-container">
                <div
                  className="state-bar"
                  style={{ width: `${(data.volume / topStates[0][1].volume) * 100}%` }}
                />
              </div>
              <span className="state-volume">{(data.volume / 1000).toFixed(1)}k</span>
            </div>
          ))}
        </div>
      </div>

      <div className="illinois-note">
        <span className="note-icon">*</span>
        <span>Illinois accounts for ~23% of all bids - highest confidence predictions</span>
      </div>
    </div>
  );
}

export default StateBenchmarks;
