# Bid Recommendation System - Frontend

React-based UI for Global Stat Solutions Bid Recommendation System.

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm start
```

The app will be available at `http://localhost:3000`

## Features

- **Bid Prediction Form**: Enter bid details to get recommended fee
- **Confidence Intervals**: 80% empirical confidence bands
- **Win Probability**: Experimental probability estimate
- **Segment Benchmarks**: Compare against segment averages
- **State Benchmarks**: View state-level statistics

## Design System

- **Colors**: Off-white background (#fafafa), dark text (#1a1a1a)
- **Typography**: Inter font family (sans-serif)
- **Style**: Clean, calm, professional

## Components

- `Header` - Logo and branding
- `BidForm` - Input form for bid details
- `ResultDisplay` - Prediction output with confidence bands
- `SegmentStats` - Segment benchmark comparison
- `StateBenchmarks` - State-level statistics

## API Configuration

The frontend proxies API requests to `http://localhost:5000` in development.

For production, set the `REACT_APP_API_URL` environment variable.

## Build

```bash
npm run build
```

Creates optimized production build in `build/` directory.
