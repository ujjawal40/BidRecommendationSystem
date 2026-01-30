import React, { useState, useEffect } from 'react';
import { fetchSegmentStats } from '../services/api';
import './SegmentStats.css';

function SegmentStats({ segment, segments }) {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedSegment, setSelectedSegment] = useState(segment);

  useEffect(() => {
    if (segment) {
      setSelectedSegment(segment);
      loadStats(segment);
    }
  }, [segment]);

  const loadStats = async (seg) => {
    if (!seg) return;

    setLoading(true);
    try {
      const data = await fetchSegmentStats(seg);
      setStats(data);
    } catch (err) {
      console.error('Failed to load segment stats:', err);
      setStats(null);
    } finally {
      setLoading(false);
    }
  };

  const handleSegmentChange = (e) => {
    const newSegment = e.target.value;
    setSelectedSegment(newSegment);
    loadStats(newSegment);
  };

  return (
    <div className="card segment-stats">
      <div className="stats-header">
        <h3>Segment Benchmarks</h3>
        <select
          value={selectedSegment}
          onChange={handleSegmentChange}
          className="segment-select"
        >
          {segments.map(seg => (
            <option key={seg} value={seg}>{seg}</option>
          ))}
        </select>
      </div>

      {loading ? (
        <div className="stats-loading">
          <div className="loading-spinner small"></div>
          <span>Loading...</span>
        </div>
      ) : stats ? (
        <div className="stats-grid">
          <div className="stat-item">
            <span className="stat-value">${stats.avg_fee?.toLocaleString() || '-'}</span>
            <span className="stat-label">Avg Fee</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">${stats.std_fee?.toLocaleString() || '-'}</span>
            <span className="stat-label">Std Dev</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{stats.win_rate ? `${(stats.win_rate * 100).toFixed(1)}%` : '-'}</span>
            <span className="stat-label">Win Rate</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{stats.count?.toLocaleString() || '-'}</span>
            <span className="stat-label">Sample Size</span>
          </div>
        </div>
      ) : (
        <p className="no-data">Select a segment to view statistics</p>
      )}

      <div className="stats-footer">
        <span className="data-note">Based on 2023+ training data</span>
      </div>
    </div>
  );
}

export default SegmentStats;
