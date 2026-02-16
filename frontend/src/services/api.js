/**
 * API Service for Bid Recommendation System
 * Handles all communication with the Flask backend
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';

/**
 * Fetch dropdown options (segments, property types, states)
 */
export async function fetchOptions() {
  const response = await fetch(`${API_BASE_URL}/options`);

  if (!response.ok) {
    throw new Error('Failed to fetch options');
  }

  const data = await response.json();

  if (!data.success) {
    throw new Error(data.error || 'Failed to fetch options');
  }

  return data.data;
}

/**
 * Get bid fee prediction
 */
export async function predictBidFee(formData) {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(formData),
  });

  if (!response.ok) {
    throw new Error('Failed to get prediction');
  }

  const data = await response.json();

  if (!data.success) {
    throw new Error(data.error || 'Failed to get prediction');
  }

  return data.prediction;
}

/**
 * Get segment statistics
 */
export async function fetchSegmentStats(segmentName) {
  const response = await fetch(`${API_BASE_URL}/segment/${encodeURIComponent(segmentName)}`);

  if (!response.ok) {
    throw new Error('Failed to fetch segment stats');
  }

  const data = await response.json();

  if (!data.success) {
    throw new Error(data.error || 'Failed to fetch segment stats');
  }

  return data.statistics;
}

/**
 * Batch prediction for multiple bids
 */
export async function batchPredict(bids) {
  const response = await fetch(`${API_BASE_URL}/batch-predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ bids }),
  });

  if (!response.ok) {
    throw new Error('Failed to get batch predictions');
  }

  const data = await response.json();

  if (!data.success) {
    throw new Error(data.error || 'Failed to get batch predictions');
  }

  return data;
}

/**
 * Fetch v2 dropdown options (includes subtypes, office regions, office locations)
 */
export async function fetchV2Options() {
  const response = await fetch(`${API_BASE_URL}/v2/options`);

  if (!response.ok) {
    throw new Error('Failed to fetch v2 options');
  }

  const data = await response.json();

  if (!data.success) {
    throw new Error(data.error || 'Failed to fetch v2 options');
  }

  return data.data;
}

/**
 * Get v2 bid fee prediction (enhanced with subtypes, office region, delivery days)
 */
export async function predictV2BidFee(formData) {
  const response = await fetch(`${API_BASE_URL}/v2/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(formData),
  });

  if (!response.ok) {
    throw new Error('Failed to get v2 prediction');
  }

  const data = await response.json();

  if (!data.success) {
    throw new Error(data.error || 'Failed to get v2 prediction');
  }

  return data.prediction;
}

/**
 * Health check
 */
export async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    const data = await response.json();
    return data.status === 'healthy';
  } catch {
    return false;
  }
}
