import React from 'react';
import './BidForm.css';

function BidForm({ formData, options, onChange, onSubmit, onReset, loading }) {
  // Get subtypes filtered by selected property type
  const subtypesForPropertyType = formData.property_type
    ? (options.subtypes_by_property_type || {})[formData.property_type] || []
    : options.sub_property_types || [];

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

        {/* Sub-Property Type (cascading from Property Type) */}
        <div className="form-group">
          <label htmlFor="sub_property_type">
            Sub-Property Type
            <span className="field-hint">Filtered by property type</span>
          </label>
          <select
            id="sub_property_type"
            name="sub_property_type"
            value={formData.sub_property_type}
            onChange={onChange}
          >
            <option value="">Any / Unknown</option>
            {subtypesForPropertyType.map(st => (
              <option key={st} value={st}>{st}</option>
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

        {/* Office Region */}
        <div className="form-group">
          <label htmlFor="office_region">
            Office Region
            <span className="field-hint">Originating office region</span>
          </label>
          <select
            id="office_region"
            name="office_region"
            value={formData.office_region}
            onChange={onChange}
          >
            <option value="">Any / Unknown</option>
            {(options.office_regions || []).map(region => (
              <option key={region} value={region}>{region}</option>
            ))}
          </select>
        </div>

        {/* Delivery Days */}
        <div className="form-group">
          <label htmlFor="delivery_days">
            Delivery Days
            <span className="field-hint">Expected job duration</span>
          </label>
          <input
            type="number"
            id="delivery_days"
            name="delivery_days"
            value={formData.delivery_days}
            onChange={onChange}
            min="1"
            max="365"
            placeholder="Optional"
          />
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
