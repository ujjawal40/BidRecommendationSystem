import { useState, useEffect } from 'react';
import { BarChart3 } from 'lucide-react';
import Header from './components/Header';
import BidForm from './components/BidForm';
import ResultDisplay from './components/ResultDisplay';
import { fetchV2Options, predictV2BidFee, lookupZip } from './services/api';
import './App.css';

function App() {
  const [options, setOptions] = useState({
    segments: [],
    property_types: [],
    states: [],
    sub_property_types: [],
    office_regions: [],
    office_locations: [],
    subtypes_by_property_type: {},
  });
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  // Single turnaround_days replaces separate target_time + delivery_days
  const [formData, setFormData] = useState({
    business_segment: '',
    property_type: '',
    property_state: '',
    turnaround_days: 30,
    sub_property_type: '',
    office_location: '',
    open_date: '',
    zip_code: '',
  });
  const [zipStatus, setZipStatus] = useState(null); // { found, state, loading }

  useEffect(() => { loadOptions(); }, []);

  const loadOptions = async () => {
    try {
      setLoading(true);
      const data = await fetchV2Options();
      setOptions(data);

      if (data.segments.length > 0) {
        const defaultPropType = data.property_types.includes('Multifamily')
          ? 'Multifamily' : data.property_types[0];
        const subtypesForProp = (data.subtypes_by_property_type || {})[defaultPropType] || [];
        setFormData(prev => ({
          ...prev,
          business_segment: data.segments.includes('Financing') ? 'Financing' : data.segments[0],
          property_type: defaultPropType,
          property_state: data.states.includes('Illinois') ? 'Illinois' : data.states[0],
          sub_property_type: subtypesForProp[0] || '',
          open_date: '',
        }));
      }
    } catch (err) {
      setError('Failed to load options. Is the API server running?');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value, type } = e.target;
    const parsed = type === 'number' ? (value === '' ? '' : Number(value)) : value;
    const newData = { ...formData, [name]: parsed };

    // When property type changes, reset sub_property_type
    if (name === 'property_type') {
      const subtypesForProp = (options.subtypes_by_property_type || {})[value] || [];
      newData.sub_property_type = subtypesForProp[0] || '';
    }

    // When zip code reaches 5 digits, auto-lookup state
    if (name === 'zip_code') {
      const cleaned = value.replace(/\D/g, '').slice(0, 5);
      newData.zip_code = cleaned;
      if (cleaned.length === 5) {
        setZipStatus({ loading: true });
        lookupZip(cleaned).then(data => {
          setZipStatus({ found: data.found, state: data.state, loading: false });
          if (data.found && data.state) {
            setFormData(prev => ({ ...prev, property_state: data.state }));
          }
        }).catch(() => setZipStatus({ found: false, loading: false }));
      } else {
        setZipStatus(null);
      }
    }

    setFormData(newData);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setPredicting(true);

    try {
      // turnaround_days feeds both API parameters — one field, one timeline
      const days = Number(formData.turnaround_days) || 30;
      const payload = {
        business_segment: formData.business_segment,
        property_type: formData.property_type,
        property_state: formData.property_state,
        target_time: days,
        delivery_days: days,
      };

      if (formData.sub_property_type) payload.sub_property_type = formData.sub_property_type;
      if (formData.office_location)    payload.office_location = formData.office_location;
      if (formData.open_date)          payload.open_date = formData.open_date;
      if (formData.zip_code && formData.zip_code.length === 5) payload.zip_code = formData.zip_code;

      const result = await predictV2BidFee(payload);
      setPrediction(result);
    } catch (err) {
      setError('Failed to get prediction. Please try again.');
      console.error(err);
    } finally {
      setPredicting(false);
    }
  };

  const handleReset = () => {
    setPrediction(null);
    setError(null);
  };

  if (loading) {
    return (
      <div className="app">
        <Header />
        <main className="main-content">
          <div className="loading-container">
            <div className="loading-spinner" />
            <p>Loading…</p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="app">
      <Header />

      <main className="main-content">
        <div className="content-grid">
          {/* Left: Form */}
          <div className="form-section">
            <div className="card">
              <h2>New Bid</h2>
              <p className="subtitle">Get a recommended fee with win probability</p>

              <BidForm
                formData={formData}
                options={options}
                onChange={handleInputChange}
                onSubmit={handleSubmit}
                onReset={handleReset}
                loading={predicting}
                zipStatus={zipStatus}
              />

              {error && <div className="error-message">{error}</div>}
            </div>
          </div>

          {/* Right: Results */}
          <div className="results-section">
            {predicting ? (
              <div className="skeleton-wrap">
                <div className="card skeleton-card skeleton-hero">
                  <div className="skeleton-line skeleton-short" />
                  <div className="skeleton-anchors">
                    <div className="skeleton-block" />
                    <div className="skeleton-block skeleton-block-lg" />
                    <div className="skeleton-block" />
                  </div>
                  <div className="skeleton-bar" />
                  <div className="skeleton-line skeleton-wide" />
                </div>
                <div className="card skeleton-card">
                  <div className="skeleton-line skeleton-mid" />
                  <div className="skeleton-line skeleton-wide" />
                </div>
              </div>
            ) : prediction ? (
              <ResultDisplay prediction={prediction} formData={formData} />
            ) : (
              <div className="card placeholder-card">
                <div className="placeholder-icon">
                  <BarChart3 size={48} strokeWidth={1.5} />
                </div>
                <h3>Ready to Analyze</h3>
                <p>Fill out the form and click "Get Recommendation" to see your optimal bid with live win probability.</p>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="footer">
        <p>&copy; 2026 Global Stat Solutions</p>
      </footer>
    </div>
  );
}

export default App;
