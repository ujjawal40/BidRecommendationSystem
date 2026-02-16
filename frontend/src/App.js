import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import BidForm from './components/BidForm';
import ResultDisplay from './components/ResultDisplay';
import SegmentStats from './components/SegmentStats';
import StateBenchmarks from './components/StateBenchmarks';
import { fetchV2Options, predictV2BidFee } from './services/api';
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

  // Form state
  const [formData, setFormData] = useState({
    business_segment: '',
    property_type: '',
    property_state: '',
    target_time: 30,
    sub_property_type: '',
    office_region: '',
    delivery_days: '',
  });

  // Load options on mount
  useEffect(() => {
    loadOptions();
  }, []);

  const loadOptions = async () => {
    try {
      setLoading(true);
      const data = await fetchV2Options();
      setOptions(data);

      // Set defaults
      if (data.segments.length > 0) {
        const defaultPropType = data.property_types.includes('Multifamily') ? 'Multifamily' : data.property_types[0];
        const subtypesForProp = (data.subtypes_by_property_type || {})[defaultPropType] || [];
        setFormData(prev => ({
          ...prev,
          business_segment: data.segments.includes('Financing') ? 'Financing' : data.segments[0],
          property_type: defaultPropType,
          property_state: data.states.includes('Illinois') ? 'Illinois' : data.states[0],
          sub_property_type: subtypesForProp[0] || '',
          office_region: (data.office_regions || [])[0] || '',
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
    const newData = {
      ...formData,
      [name]: type === 'number' ? (value === '' ? '' : Number(value)) : value,
    };

    // When property type changes, reset sub_property_type to first available
    if (name === 'property_type') {
      const subtypesForProp = (options.subtypes_by_property_type || {})[value] || [];
      newData.sub_property_type = subtypesForProp[0] || '';
    }

    setFormData(newData);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setPredicting(true);

    try {
      // Build payload — omit empty optional fields
      const payload = {
        business_segment: formData.business_segment,
        property_type: formData.property_type,
        property_state: formData.property_state,
        target_time: formData.target_time,
      };
      if (formData.sub_property_type) payload.sub_property_type = formData.sub_property_type;
      if (formData.office_region) payload.office_region = formData.office_region;
      if (formData.delivery_days !== '' && formData.delivery_days != null) {
        payload.delivery_days = Number(formData.delivery_days);
      }

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
            <div className="loading-spinner"></div>
            <p>Loading Bid Recommendation System...</p>
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
          {/* Left column: Form */}
          <div className="form-section">
            <div className="card">
              <h2>New Bid Prediction</h2>
              <p className="subtitle">Enter bid details to get a recommended fee with confidence intervals</p>

              <BidForm
                formData={formData}
                options={options}
                onChange={handleInputChange}
                onSubmit={handleSubmit}
                onReset={handleReset}
                loading={predicting}
              />

              {error && (
                <div className="error-message">
                  {error}
                </div>
              )}
            </div>
          </div>

          {/* Right column: Results */}
          <div className="results-section">
            {prediction ? (
              <ResultDisplay prediction={prediction} formData={formData} />
            ) : (
              <div className="card placeholder-card">
                <div className="placeholder-icon">
                  <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                  </svg>
                </div>
                <h3>Ready to Predict</h3>
                <p>Fill out the form and click "Get Prediction" to see the recommended bid fee with confidence intervals.</p>
              </div>
            )}
          </div>
        </div>

        {/* Bottom section: Benchmarks */}
        <div className="benchmarks-section">
          <div className="benchmarks-grid">
            <SegmentStats
              segment={formData.business_segment}
              segments={options.segments}
            />
            <StateBenchmarks
              state={formData.property_state}
              states={options.states}
            />
          </div>
        </div>
      </main>

      <footer className="footer">
        <p>&copy; 2026 Global Stat Solutions. Bid Recommendation System v2.0</p>
        <p className="disclaimer">
          Enhanced model — MAPE 7.3%, AUC 0.948
        </p>
      </footer>
    </div>
  );
}

export default App;
