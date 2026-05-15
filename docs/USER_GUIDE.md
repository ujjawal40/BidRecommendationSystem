# Bid Intelligence Tool — User Guide

## What This Tool Does

You are deciding what fee to bid on a commercial real estate appraisal engagement. Bid too high and you lose the job. Bid too low and you win but leave money on the table. This tool analyzes over 215,000 historical bids to recommend a fee that balances both — maximizing the revenue you can realistically expect to earn.

It answers two questions at once:

1. **What should we charge?** A regression model predicts the market-typical fee for this combination of segment, property type, state, and turnaround time.
2. **Will we win at this price?** A classification model estimates the probability of winning at the recommended fee, based on how that price compares to what the market typically pays.

The tool combines both into an **Expected Value** score: `EV = P(Win) x Fee`. A $5,200 bid with a 25% win chance (EV = $1,300) is worse than a $3,500 bid with a 65% win chance (EV = $2,275). The system recommends the fee that maximizes EV — not just the highest fee, and not just the most winnable fee, but the best trade-off between the two.

---

## Filling Out the Form

| Field | What to enter | Why it matters |
|-------|--------------|----------------|
| **Business Segment** | The service line (Financing, Litigation, etc.) | The single most important factor — segment alone explains ~64% of fee variation |
| **Property Type / Sub-type** | The asset class (Multifamily, Office, Retail, etc.) | Different property types command different fees in different markets |
| **Zip Code** | The property's zip code (optional) | Auto-fills the state and loads real demographic data for that area, improving accuracy |
| **Property State** | Where the property is located | State-level market dynamics significantly affect pricing |
| **Days to Complete** | How long this assignment will take | Shorter turnarounds (under 21 days) typically command a premium |
| **Job Open Date** | When this opportunity was posted | Captures seasonal pricing patterns. Leave blank to use today's date |
| **Office Location** | Which office is handling this bid | Regional pricing patterns vary across offices |

Click **Get Recommendation** to run the analysis.

---

## Reading the Results

### Bid Quality Badge

Next to the "Bid Range" header, a colored badge gives you an at-a-glance assessment:

- **Good bid** (green) — Win probability is 55%+ and confidence is not low. You're well-positioned.
- **Fair bid** (amber) — Win probability is 35%+ or 30%+ with high confidence. Competitive but not a lock.
- **Risky bid** (red) — Win probability below those thresholds. Proceed with caution.

### The Bid Range

The large panel at the top shows three price anchors:

**Floor** — The low end of the defensible range. 80% of historical bids for similar work fall between the Floor and the Ceiling. Bidding below the floor is unusually aggressive — you are almost certainly underpricing.

**Recommended Bid (or Optimal Bid)** — The fee the system recommends. This is the price point where your **expected revenue is maximized**, considering both the fee amount and the probability of winning. This is not simply the average market price — it is the fee where the math says you earn the most over many similar bids. You can click the recommended fee to copy it to your clipboard.

**Bid Ceiling** — The high end of the range. The highest fee where you still have at least a 30% chance of winning. Bidding above the ceiling is possible but your odds drop sharply and you are relying on non-price factors (reputation, relationships, speed) to win.

> **The scale bar** between Floor and Ceiling shows where the Recommended Bid sits within the range. If the dot is near the left, you are priced conservatively. Near the right, you are pricing aggressively within the viable range.

### Capped at 30% Win Floor

Sometimes the label says **"Capped at 30% win floor"** with a warning badge. This means the mathematically optimal fee (the one that maximizes EV) would actually sit *above* the Ceiling, where win odds drop below 30%. The system caps the recommendation at the Ceiling instead of showing you a fee with poor odds. In this scenario, the Recommended Bid equals the Ceiling — shown in a two-column layout instead of three.

**What to do when you see the cap:** The model is telling you this is a situation where the market can bear a high price, but competition or market dynamics limit how far you can push. Bid at or near the recommended amount. Going higher is a gamble — your track record with the client and the quality of your proposal will matter more than price.

### Confidence Badge

The pill in the top-right corner of the Bid Range panel tells you how much data backs this estimate:

- **High confidence** — Large sample of similar historical bids (typically 1,000+). The range is narrow and reliable.
- **Medium confidence** — Moderate data. The estimate is directional but the range is wider.
- **Low confidence** — Sparse data for this exact combination. Treat as a rough guide, not a precise number. Cross-reference with your own experience.

The number next to it (e.g., "150,062 similar bids") shows how many historical bids in this segment informed the prediction.

### Win Probability

Below the bid range, a large percentage and colored bar show your estimated chance of winning at the recommended fee:

- **Green (55%+):** Strong chance of winning. You are well-priced for this market and likely to win.
- **Amber (35–54%):** Moderate chance. Competitive assignment — price is one factor among several.
- **Red (below 35%):** Low chance. Even at the recommended fee, odds are modest. Non-price factors (experience, client relationship, turnaround speed) will likely decide the outcome.

Below the percentage, a plain-language line puts the odds in context — for example, "If you bid $2,714 on 10 similar jobs, you'd expect to win about 7 of them."

**EV (Expected Value)** is shown next to the win probability. This is `P(Win) x Fee` — the average revenue you would earn if you bid this amount repeatedly on similar assignments. Use this to compare different fee options: a lower fee with higher win odds often has a better EV than a premium fee with poor odds. Hover over the EV label for a tooltip explanation.

### "What If I Bid $X?"

Directly below the bid range panel, a quick input lets you type any dollar amount and instantly see:

- The estimated **win probability** at that fee
- The **EV** at that fee
- The **EV delta** compared to the recommended bid (green if better, red if worse)

If you enter a fee above the Bid Ceiling, a warning tells you win odds drop below 30%. This is faster than dragging the chart slider and lets you test exact dollar amounts.

### Market Context Bar

The bar at the bottom of the results shows four metrics in a grid:

- **Segment avg** — The historical average fee for this business segment
- **Your bid vs segment** — Whether your recommended bid is above or below the segment average
- **Floor–Ceiling spread** — How wide the defensible range is (wider = more uncertainty)
- **State avg** — The historical average fee for the selected state

Being below the segment average is not automatically bad — it may reflect state-level pricing or property type dynamics. Being above average with good win odds means the market supports a premium for this type of work.

---

## The Fee vs. Win Probability Chart

The chart is shown by default below the results. It plots win probability (vertical axis) against bid fee (horizontal axis). A slider lets you explore different fee points and see the estimated win probability and EV at each. The **shaded blue region** between Floor and Ceiling marks the defensible pricing zone.

Key markers on the chart:
- **Floor** — Left dashed line
- **Optimal / Recommended** — Blue label
- **Ceiling** — Right dashed line
- **30% threshold** — Red dashed horizontal line. Below this, the bid is considered non-viable.

**How to use it:** If you are considering bidding above or below the recommendation, use the slider to see what happens to your odds. The chart helps you make an informed trade-off rather than guessing.

---

## When to Bid and When to Walk Away

### Good signals to bid:
- Win probability is 45% or above
- The recommended fee aligns with or exceeds the segment average
- Confidence is medium or high
- The assignment matches your firm's core expertise

### Caution signals:
- Win probability below 30% — even at the optimized fee, this is a long shot
- Low confidence with a wide Floor-to-Ceiling spread — the model is uncertain
- The recommended fee is significantly below your internal cost to deliver
- The "Capped at 30% win floor" badge appears with win odds under 35%

### Walking away:
This tool tells you what fee maximizes your expected revenue — it does not tell you whether the job is worth pursuing. If the recommended fee is below your cost basis, or if win odds are low and the client is unfamiliar, the best bid may be no bid. The EV helps here: if EV is lower than the opportunity cost of your team's time, redirect effort to higher-value pursuits.

---

## What This Tool Cannot Do

- **It cannot guarantee a win.** A 65% win probability means you lose 35% of the time. Over many bids, the recommendations optimize your revenue, but any single bid may go either way.
- **It does not know your client relationship.** If you have a long-standing relationship with the client, your real odds may be higher than shown. If you are a new entrant, they may be lower.
- **It does not capture deal-specific factors.** Property complexity, special requirements, competing firms on this specific deal, and the client's budget are not inputs the model has access to.
- **It does not replace judgment.** The recommendation is a data-informed starting point. Your experience with the client, the competitive landscape, and your firm's strategic priorities should all factor into the final number.

---

## Quick Reference

| Term | Meaning |
|------|---------|
| **Floor** | Lowest defensible bid — 10th percentile of similar historical bids |
| **Recommended Bid** | Fee that maximizes expected revenue (P(Win) x Fee) |
| **Ceiling** | Highest fee with >= 30% win probability |
| **Win Probability** | Estimated chance of winning at the recommended fee |
| **EV (Expected Value)** | P(Win) x Fee — your average earnings per bid at this price |
| **Segment Average** | Historical mean fee for this business segment |
| **Confidence** | How much historical data supports this estimate (high/medium/low) |
| **Capped at 30% Win Floor** | The EV-optimal fee exceeds the ceiling; recommendation is capped |
| **Bid Quality Badge** | At-a-glance assessment: Good (green), Fair (amber), Risky (red) |
| **What If I Bid $X?** | Quick input to test any fee against the model instantly |
| **Heuristic Estimate** | Win probability is using a calibrated fallback instead of the ML model |
