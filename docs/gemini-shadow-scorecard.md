# Gemini desk: evidence-based shadow scorecard

The independent shadow collector records simulated Mark-to-Mark returns *after*
a conservative, per-side stress deduction. These are neither Bybit venue fills
nor account cashflows. Missing quotes and unfillable entries are **not**
counted as successful trades.

The learning API publishes `shadow_scorecard`. It is not an Invo score or
copy-trader ranking. Its documented evidence gates are:

- At least 30 **settled** simulated trades spanning at least 3 distinct UTC
  days, with 75% or higher net win rate.
- Gross net profits / absolute gross net losses (profit factor) at least 1.3.
  A sample containing **no losing trades** has unknown PF and cannot qualify.
- Positive arithmetic net expectancy **after** modeled costs.
- Modeled equal-notional compounded peak-to-trough drawdown no more than 15%.

The scorecard also reports a descriptive Wilson 95%-interval lower win-rate
bound to communicate small-sample uncertainty. It does not guarantee the
lower-bound performance, nor is it a replacement for walk-forward holdouts.

**Critical limitations**

- Equal-notional per-trade compounded drawdown is *not* actual account
  drawdown. Actual Bybit Demo equity, executions, funding, margin, stop fills
  and liquidation risk require separate independent reconciliation.
- A qualifying shadow scorecard is necessary but insufficient for profitable
  trading. Gemini's bounded opinion, the existing chronological lower-bound
  learner, exchange-side protections, quote freshness, account risk guard,
  and an explicitly authorized Demo strategy start still apply.
- The program cannot auto-promote model weights or increase leverage based on
  this scorecard, and it never connects to or copies Invo users or accounts.
- Change thresholds only under a versioned protocol followed by independent
  out-of-sample testing. Do not optimize on the evaluation window.
