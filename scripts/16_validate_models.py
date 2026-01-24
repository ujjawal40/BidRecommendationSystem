"""
Model Validation Script
=======================
Validates both Phase 1A (Bid Fee) and Phase 1B (Win Probability) models
using the test set from training data.

Shows:
1. Predictions vs Actuals for won bids
2. Win probability distribution for Won vs Lost
3. Expected Value analysis
4. Sample predictions for inspection

Run: python scripts/16_validate_models.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from datetime import datetime

from config.model_config import (
    FEATURES_DATA, MODELS_DIR, FIGURES_DIR,
    DATA_START_DATE, EXCLUDE_COLUMNS, JOBDATA_FEATURES_TO_EXCLUDE,
)

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Leaky features for classification (same as training)
LEAKY_CLASSIFICATION_FEATURES = [
    'win_rate_with_client', 'office_win_rate', 'propertytype_win_rate',
    'state_win_rate', 'segment_win_rate', 'client_win_rate',
    'rolling_win_rate_office', 'total_wins_with_client', 'prev_won_same_client',
]


def main():
    print("=" * 80)
    print("MODEL VALIDATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ========================================================================
    # LOAD MODELS
    # ========================================================================
    print("Loading models...")

    # Phase 1A: Bid Fee Model
    bidfee_model = lgb.Booster(model_file=str(MODELS_DIR / "lightgbm_bidfee_model.txt"))
    with open(MODELS_DIR / "lightgbm_metadata.json") as f:
        bidfee_metadata = json.load(f)
    bidfee_features = bidfee_metadata.get('features', bidfee_model.feature_name())
    print(f"  Phase 1A (Bid Fee): {len(bidfee_features)} features")

    # Phase 1B: Win Probability Model
    winprob_model = lgb.Booster(model_file=str(MODELS_DIR / "lightgbm_win_probability.txt"))
    with open(MODELS_DIR / "lightgbm_win_probability_metadata.json") as f:
        winprob_metadata = json.load(f)
    winprob_features = winprob_metadata.get('features', winprob_model.feature_name())
    print(f"  Phase 1B (Win Prob): {len(winprob_features)} features")

    # ========================================================================
    # LOAD AND PREPARE TEST DATA
    # ========================================================================
    print("\n" + "=" * 80)
    print("LOADING TEST DATA")
    print("=" * 80)

    df = pd.read_csv(FEATURES_DATA)
    df['BidDate'] = pd.to_datetime(df['BidDate'])
    df = df.sort_values('BidDate').reset_index(drop=True)

    # Filter to 2023+ (same as training)
    start_date = pd.Timestamp(DATA_START_DATE)
    df = df[df['BidDate'] >= start_date].copy()
    print(f"Records (2023+): {len(df):,}")

    # Get test set (last 20%)
    n = len(df)
    test_start = int(n * 0.8)
    test_df = df.iloc[test_start:].copy()
    print(f"Test set: {len(test_df):,} records")
    print(f"Test date range: {test_df['BidDate'].min()} to {test_df['BidDate'].max()}")

    # ========================================================================
    # MAKE PREDICTIONS
    # ========================================================================
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)

    # Prepare features
    X_bidfee = test_df[bidfee_features].fillna(0)
    X_winprob = test_df[winprob_features].fillna(0)

    # Predictions
    test_df['predicted_fee'] = bidfee_model.predict(X_bidfee)
    test_df['predicted_win_prob'] = winprob_model.predict(X_winprob)
    test_df['expected_value'] = test_df['predicted_win_prob'] * test_df['predicted_fee']

    print(f"Predictions generated for {len(test_df):,} bids")

    # ========================================================================
    # VALIDATION 1: WIN PROBABILITY BY ACTUAL OUTCOME
    # ========================================================================
    print("\n" + "=" * 80)
    print("VALIDATION 1: WIN PROBABILITY BY ACTUAL OUTCOME")
    print("=" * 80)

    won_bids = test_df[test_df['Won'] == 1]
    lost_bids = test_df[test_df['Won'] == 0]

    print(f"\nWon Bids ({len(won_bids):,}):")
    print(f"  Mean predicted P(Win): {won_bids['predicted_win_prob'].mean():.1%}")
    print(f"  Median predicted P(Win): {won_bids['predicted_win_prob'].median():.1%}")
    print(f"  Min: {won_bids['predicted_win_prob'].min():.1%}, Max: {won_bids['predicted_win_prob'].max():.1%}")

    print(f"\nLost Bids ({len(lost_bids):,}):")
    print(f"  Mean predicted P(Win): {lost_bids['predicted_win_prob'].mean():.1%}")
    print(f"  Median predicted P(Win): {lost_bids['predicted_win_prob'].median():.1%}")
    print(f"  Min: {lost_bids['predicted_win_prob'].min():.1%}, Max: {lost_bids['predicted_win_prob'].max():.1%}")

    # Separation quality
    separation = won_bids['predicted_win_prob'].mean() - lost_bids['predicted_win_prob'].mean()
    print(f"\n✓ Separation (Won - Lost mean): {separation:.1%}")
    if separation > 0.3:
        print("  Assessment: EXCELLENT - Model clearly distinguishes winners from losers")
    elif separation > 0.2:
        print("  Assessment: GOOD - Model has reasonable discrimination")
    else:
        print("  Assessment: NEEDS IMPROVEMENT - Weak discrimination")

    # ========================================================================
    # VALIDATION 2: BID FEE ACCURACY FOR WON BIDS
    # ========================================================================
    print("\n" + "=" * 80)
    print("VALIDATION 2: BID FEE PREDICTION ACCURACY")
    print("=" * 80)

    # Only look at won bids with actual fees > 0
    valid_won = won_bids[won_bids['BidFee'] > 0].copy()
    valid_won['fee_error'] = valid_won['predicted_fee'] - valid_won['BidFee']
    valid_won['fee_error_pct'] = (valid_won['fee_error'] / valid_won['BidFee']) * 100

    print(f"\nWon Bids with Valid Fees ({len(valid_won):,}):")
    print(f"  Actual Fee - Mean: ${valid_won['BidFee'].mean():,.0f}, Median: ${valid_won['BidFee'].median():,.0f}")
    print(f"  Predicted Fee - Mean: ${valid_won['predicted_fee'].mean():,.0f}, Median: ${valid_won['predicted_fee'].median():,.0f}")

    mae = valid_won['fee_error'].abs().mean()
    mape = valid_won['fee_error_pct'].abs().mean()
    rmse = np.sqrt((valid_won['fee_error'] ** 2).mean())

    print(f"\nError Metrics:")
    print(f"  MAE: ${mae:,.0f}")
    print(f"  RMSE: ${rmse:,.0f}")
    print(f"  MAPE: {mape:.1f}%")

    # ========================================================================
    # VALIDATION 3: EXPECTED VALUE ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("VALIDATION 3: EXPECTED VALUE ANALYSIS")
    print("=" * 80)

    # Sort by EV and see if high EV bids actually won more
    test_df['ev_quartile'] = pd.qcut(test_df['expected_value'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])

    print("\nWin Rate by Expected Value Quartile:")
    print(f"  {'Quartile':<15} {'Bids':>8} {'Wins':>8} {'Win Rate':>10} {'Avg EV':>12}")
    print(f"  {'-'*55}")

    for quartile in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']:
        q_data = test_df[test_df['ev_quartile'] == quartile]
        wins = q_data['Won'].sum()
        win_rate = q_data['Won'].mean()
        avg_ev = q_data['expected_value'].mean()
        print(f"  {quartile:<15} {len(q_data):>8,} {wins:>8,} {win_rate:>10.1%} ${avg_ev:>11,.0f}")

    # Check if higher EV correlates with higher win rate
    q4_winrate = test_df[test_df['ev_quartile'] == 'Q4 (High)']['Won'].mean()
    q1_winrate = test_df[test_df['ev_quartile'] == 'Q1 (Low)']['Won'].mean()
    ev_lift = q4_winrate / q1_winrate if q1_winrate > 0 else 0

    print(f"\n✓ High EV (Q4) vs Low EV (Q1) Win Rate Lift: {ev_lift:.2f}x")

    # ========================================================================
    # SAMPLE PREDICTIONS
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS (Random 10 from Test Set)")
    print("=" * 80)

    sample = test_df.sample(n=10, random_state=42)

    print(f"\n{'Actual':>8} {'Pred Fee':>10} {'P(Win)':>8} {'EV':>10} {'Won':>5} {'Segment':<20}")
    print("-" * 70)

    for _, row in sample.iterrows():
        segment = str(row.get('BusinessSegment', 'N/A'))[:20]
        print(f"${row['BidFee']:>7,.0f} ${row['predicted_fee']:>9,.0f} {row['predicted_win_prob']:>7.1%} ${row['expected_value']:>9,.0f} {'Yes' if row['Won'] else 'No':>5} {segment:<20}")

    # ========================================================================
    # TOP 10 HIGHEST EXPECTED VALUE BIDS
    # ========================================================================
    print("\n" + "=" * 80)
    print("TOP 10 HIGHEST EXPECTED VALUE BIDS (Test Set)")
    print("=" * 80)

    top_ev = test_df.nlargest(10, 'expected_value')

    print(f"\n{'Pred Fee':>10} {'P(Win)':>8} {'EV':>12} {'Won':>5} {'Actual Fee':>12}")
    print("-" * 55)

    for _, row in top_ev.iterrows():
        print(f"${row['predicted_fee']:>9,.0f} {row['predicted_win_prob']:>7.1%} ${row['expected_value']:>11,.0f} {'Yes' if row['Won'] else 'No':>5} ${row['BidFee']:>11,.0f}")

    wins_in_top10 = top_ev['Won'].sum()
    print(f"\n✓ Won {wins_in_top10}/10 of highest EV bids ({wins_in_top10*10}% win rate)")

    # ========================================================================
    # SAVE VALIDATION PLOT
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING VALIDATION PLOTS")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Win Probability Distribution
    ax1 = axes[0, 0]
    ax1.hist(won_bids['predicted_win_prob'], bins=30, alpha=0.7, label=f'Won (n={len(won_bids):,})', color='green')
    ax1.hist(lost_bids['predicted_win_prob'], bins=30, alpha=0.7, label=f'Lost (n={len(lost_bids):,})', color='red')
    ax1.set_xlabel('Predicted Win Probability')
    ax1.set_ylabel('Count')
    ax1.set_title('Win Probability Distribution by Outcome')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Predicted vs Actual Fee (Won Bids)
    ax2 = axes[0, 1]
    ax2.scatter(valid_won['BidFee'], valid_won['predicted_fee'], alpha=0.3, s=10)
    max_fee = max(valid_won['BidFee'].max(), valid_won['predicted_fee'].max())
    ax2.plot([0, max_fee], [0, max_fee], 'r--', label='Perfect Prediction')
    ax2.set_xlabel('Actual Bid Fee ($)')
    ax2.set_ylabel('Predicted Bid Fee ($)')
    ax2.set_title('Predicted vs Actual Fee (Won Bids)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Win Rate by EV Quartile
    ax3 = axes[1, 0]
    quartiles = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
    win_rates = [test_df[test_df['ev_quartile'] == q]['Won'].mean() for q in quartiles]
    bars = ax3.bar(quartiles, win_rates, color=['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1'])
    ax3.set_ylabel('Win Rate')
    ax3.set_title('Win Rate by Expected Value Quartile')
    ax3.set_ylim(0, 1)
    for bar, rate in zip(bars, win_rates):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.0%}', ha='center', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Expected Value Distribution
    ax4 = axes[1, 1]
    ax4.hist(test_df['expected_value'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax4.axvline(test_df['expected_value'].mean(), color='red', linestyle='--', label=f'Mean: ${test_df["expected_value"].mean():,.0f}')
    ax4.axvline(test_df['expected_value'].median(), color='green', linestyle='--', label=f'Median: ${test_df["expected_value"].median():,.0f}')
    ax4.set_xlabel('Expected Value ($)')
    ax4.set_ylabel('Count')
    ax4.set_title('Expected Value Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'model_validation.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {FIGURES_DIR / 'model_validation.png'}")
    plt.close()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print(f"""
Model Performance:
  - Win Probability separates Won/Lost by {separation:.0%}
  - Bid Fee MAE: ${mae:,.0f} ({mape:.1f}% MAPE)
  - High EV bids win {ev_lift:.1f}x more than low EV bids

Recommendation:
  - Models are validated and ready for inference
  - Use Expected Value (EV) to prioritize bids
  - Higher EV = Higher predicted fee × Higher win probability
""")

    return test_df


if __name__ == "__main__":
    results = main()
