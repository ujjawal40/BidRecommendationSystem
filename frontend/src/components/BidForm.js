import React from 'react';
import './BidForm.css';

function BidForm({ formData, options, onChange, onSubmit, onReset, loading }) {
  const subtypesForPropertyType = formData.property_type
    ? (options.subtypes_by_property_type || {})[formData.property_type] || []
    : options.sub_property_types || [];

  return (
    <form className="bid-form" onSubmit={onSubmit}>
      <div className="form-fields">

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
            <option value="">Select segment…</option>
            {options.segments.map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>

        {/* Property Type + Sub-type */}
        <div className="form-row">
          <div className="form-group">
            <label htmlFor="property_type">Property Type</label>
            <select
              id="property_type"
              name="property_type"
              value={formData.property_type}
              onChange={onChange}
              required
            >
              <option value="">Select type…</option>
              {options.property_types.map(t => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="sub_property_type">Sub-type</label>
            <select
              id="sub_property_type"
              name="sub_property_type"
              value={formData.sub_property_type}
              onChange={onChange}
            >
              <option value="">Any</option>
              {subtypesForPropertyType.map(st => (
                <option key={st} value={st}>{st}</option>
              ))}
            </select>
          </div>
        </div>

        {/* State */}
        <div className="form-group">
          <label htmlFor="property_state">State</label>
          <select
            id="property_state"
            name="property_state"
            value={formData.property_state}
            onChange={onChange}
            required
          >
            <option value="">Select state…</option>
            {options.states.map(st => (
              <option key={st} value={st}>{st}</option>
            ))}
          </select>
        </div>

        {/* Days to Complete */}
        <div className="form-group">
          <label htmlFor="turnaround_days">
            Days to Complete
            <span className="field-hint">How long will this assignment take?</span>
          </label>
          <div className="input-with-unit">
            <input
              type="number"
              id="turnaround_days"
              name="turnaround_days"
              value={formData.turnaround_days}
              onChange={onChange}
              min="1"
              max="365"
              required
            />
            <span className="input-unit">days</span>
          </div>
          <div className="turnaround-presets">
            {[14, 21, 30, 45, 60].map(d => (
              <button
                key={d}
                type="button"
                className={`preset-chip ${formData.turnaround_days === d ? 'active' : ''}`}
                onClick={() => onChange({ target: { name: 'turnaround_days', value: d, type: 'number' } })}
              >
                {d}d
              </button>
            ))}
          </div>
        </div>

        {/* Office Region + Office Name */}
        <div className="form-row">
          <div className="form-group">
            <label htmlFor="office_region">Office Region</label>
            <select
              id="office_region"
              name="office_region"
              value={formData.office_region}
              onChange={onChange}
            >
              <option value="">Any</option>
              {(options.office_regions || []).map(r => (
                <option key={r} value={r}>{r}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="office_location">Office Name</label>
            <select
              id="office_location"
              name="office_location"
              value={formData.office_location}
              onChange={onChange}
            >
              <option value="">Any</option>
              {(options.office_locations || []).map(loc => (
                <option key={loc} value={loc}>{loc}</option>
              ))}
            </select>
          </div>
        </div>

      </div>

      {/* Actions */}
      <div className="form-actions">
        <button
          type="submit"
          className="btn-primary btn-predict"
          disabled={loading}
        >
          {loading ? (
            <><span className="btn-spinner" /> Analyzing…</>
          ) : (
            'Get Recommendation'
          )}
        </button>
        <button
          type="button"
          className="btn-ghost btn-reset"
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
