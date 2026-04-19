import React from 'react';
import './Header.css';

function Header() {
  return (
    <header className="header">
      <div className="header-inner">
        <div className="header-brand">
          <span className="brand-name">Global Stat Solutions</span>
          <span className="brand-sep">·</span>
          <span className="brand-product">Bid Intelligence</span>
        </div>
        <span className="header-badge">v2.0</span>
      </div>
    </header>
  );
}

export default Header;
