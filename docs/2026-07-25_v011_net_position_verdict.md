# V011 (fine-tuned net position) — NOT PROMOTED

**Date:** 2026-07-25
**Verdict:** V011 fine-tune **does not beat** V010 zero-shot. Keeping V010 in the
scheduled slot. Registry: V011 → `rejected`, V010 → `ready`.

## Result (canonical / optimistic — see caveat)

Mean MAE over the 12 held-out backtest weeks (W01–W12), MW, lower is better:

| Country | V010 (zero-shot) | V011 (fine-tuned) | Δ | V011 wins |
|---|---|---|---|---|
| BE | 875.0 | 994.6 | **+13.7% worse** | 4/12 |
| FR | 1578.2 | 1682.5 | **+6.6% worse** | 7/12 |
| NL | 1589.9 | 1918.7 | **+20.7% worse** | 1/12 |
| AT | 877.1 | 902.1 | **+2.9% worse** | 5/12 |
| **Pooled** | **1230.0** | **1374.5** | **+11.7% worse** | **17/48 (35%)** |

V011 is worse in **every** country tested and wins only 35% of country-weeks.
DE produced no comparable rows in this run (excluded, not counted either way).
The direction is consistent — this is not a marginal or noise-level difference.

Fine-tuning itself ran correctly: 5000 steps, 1h42m, train_loss 0.2927,
eval_loss 0.2971, 478 MB checkpoint. The model trained; it just generalizes worse
than the pretrained zero-shot model on this target.

Plausible reading: 18 series × ~1 year is a small corpus for fine-tuning a
foundation model, and net position is noisy/regime-driven — fine-tuning appears to
overfit training-period regimes while zero-shot Chronos-2 retains broader priors.
Not investigated further (out of scope).

## Promotion-gate findings

1. **Incumbent:** correct slot = net_position × D+2 × acceptance-scheduled →
   **V010 zero-shot** (`able-net-position-forecast`, 08:00). No prod net-position
   serving exists. Compared against the true incumbent. ✓
2. **Look-ahead / lag-parity: ⚠️ ISSUE FOUND (pre-existing, affects BOTH models).**
   `build_for_country` pulls past covariates through **D+1 23:00** for a D+2 target,
   but `crossborder_flows` is consistently **~24 h stale** (identical `max(timestamp)`
   across all 12+ countries checked; part sync cadence, part ENTSO-E publication).
   In a historical backtest those hours exist in the DB; at real inference time they
   do not. **All numbers above are therefore `canonical` (optimistic), not
   serve-faithful.** Because the flaw is in the shared input builder, it hits V010 and
   V011 equally, so the *relative* V010-vs-V011 comparison stands; the *absolute*
   skill of either model is unproven.
3. **Prior art:** no recorded verdict rejecting Chronos-2 fine-tuning in this repo
   (V003 marked `completed`, no win/loss recorded). Not re-deriving a known NO-GO. ✓
4. **Break-test:** unnecessary to defend a win — there is no win. The result was
   adversarial to the *candidate* by construction (V011 lost).
5. **Comparability:** same metric (MAE), same 12 backtest weeks, same countries, both
   models on the same (new) aggregate cross-border covariates. Labelled canonical. ✓
6. **Claim:** V011 is **not better**. No promotion. ✓

## Follow-up (not done here)

- **Lag-realistic re-run** to establish serve-faithful skill for V010: delay
  `crossborder_flows` to its true availability (≥24 h) in the input builder and
  re-measure. Until then V010's production skill is unknown — it is *running*, but
  its accuracy is not validated.
- The scheduled job stays on **V010**; `run-net-position.ps1` unchanged.
