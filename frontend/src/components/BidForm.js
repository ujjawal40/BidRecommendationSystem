import React from 'react';
import './BidForm.css';

function BidForm({ formData, options, onChange, onSubmit, onReset, loading }) {
  return (
    <form className="bid-form" onSubmit={onSubmit}>
      <div className="form-grid">
        {/* Business Segment */}
        <div className="form-group">
          <label htmlFor="business_segment">Business Segment</label>
          <select
            id="business_segment"
            name="business_segment"
            value={formData.business_segment}
            onChange={onChange}
            required
          >
            <option value="">Select segment...</option>
            {options.segments.map(segment => (
              <option key={segment} value={segment}>{segment}</option>
            ))}
          </select>
        </div>

        {/* Property Type */}
        <div className="form-group">
          <label htmlFor="property_type">Property Type</label>
          <select
            id="property_type"
            name="property_type"
            value={formData.property_type}
            onChange={onChange}
            required
          >
            <option value="">Select type...</option>
            {options.property_types.map(type => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>
        </div>

        {/* Property State */}
        <div className="form-group">
          <label htmlFor="property_state">Property State</label>
          <select
            id="property_state"
            name="property_state"
            value={formData.property_state}
            onChange={onChange}
            required
          >
            <option value="">Select state...</option>
            {options.states.map(state => (
              <option key={state} value={state}>{state}</option>
            ))}
          </select>
        </div>

        {/* Target Time */}
        <div className="form-group">
          <label htmlFor="target_time">
            Target Time (days)
            <span className="field-hint">Time to complete appraisal</span>
          </label>
          <input
            type="number"
            id="target_time"
            name="target_time"
            value={formData.target_time}
            onChange={onChange}
            min="1"
            max="365"
            required
          />
        </div>

        {/* Distance */}
        <div className="form-group">
          <label htmlFor="distance_km">
            Distance (km)
            <span className="field-hint">Distance to property</span>
          </label>
          <input
            type="number"
            id="distance_km"
            name="distance_km"
            value={formData.distance_km}
            onChange={onChange}
            min="0"
            step="0.1"
          />
        </div>

        {/* On Due Date */}
        <div className="form-group">
          <label htmlFor="on_due_date">Delivery</label>
          <select
            id="on_due_date"
            name="on_due_date"
            value={formData.on_due_date}
            onChange={onChange}
          >
            <option value={0}>Before due date</option>
            <option value={1}>On due date</option>
          </select>
        </div>
      </div>

      <div className="form-actions">
        <button
          type="submit"
          className="btn-primary btn-predict"
          disabled={loading}
        >
          {loading ? (
            <>
              <span className="btn-spinner"></span>
              Predicting...
            </>
          ) : (
            'Get Prediction'
          )}
        </button>

        <button
          type="button"
          className="btn-secondary"
          onClick={onReset}
          disabled={loading}
        >
          Reset
        </button>
      </div>
    </form>
  );
}

export default BidForm;
