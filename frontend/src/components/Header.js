import React from 'react';
import './Header.css';

function Header() {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo-section">
          <img src="/logo.svg" alt="Global Stat Solutions" className="logo" />
          <div className="brand-text">
            <h1>Bid Recommendation System</h1>
            <p className="tagline">Intelligent Pricing for Commercial Real Estate Appraisals</p>
          </div>
        </div>

        <nav className="header-nav">
          <span className="version-badge">v2.0</span>
        </nav>
      </div>
    </header>
  );
}

export default Header;
