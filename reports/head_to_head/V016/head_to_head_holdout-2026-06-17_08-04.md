# Head-to-head: `chronos-2-V016` vs `chronos-2-V010`

**Generated:** 2026-08-08 05:59 UTC · **Window:** 2026-06-17 .. 2026-08-04T23:00
**Paired rows:** 22,344 over 49 vintages · rows only in `chronos-2-V010`: 0 · only in `chronos-2-V016`: 0

Scored only on `(country, target hour, vintage)` rows where **both** models produced a value and an actual exists. Per-model reports are *not* comparable to each other — they cover different vintage sets.

## Pooled

`chronos-2-V010` **775.2 MW** MAE · `chronos-2-V016` **786.1 MW** MAE · challenger is **1.4% worse** (+11.0 MW)

Materially better (>= 0.5%) in **1/19** countries; identical (pass-through) in **3**.

Pooled MAE mixes countries of very different size — the per-country table is the one that gates.

## Per country

| country | n | chronos-2-V010 MAE | chronos-2-V016 MAE | Δ MW | Δ % | verdict |
|---|---:|---:|---:|---:|---:|---|
| AT | 1,176 | 801.5 | 800.6 | -0.9 | -0.1% | tie |
| BE | 1,176 | 933.8 | 954.5 | +20.7 | +2.2% | worse |
| BG | 1,176 | 367.3 | 367.3 | +0.0 | +0.0% | identical |
| CZ | 1,176 | 649.9 | 661.5 | +11.6 | +1.8% | worse |
| DE | 1,176 | 2,996.0 | 3,014.1 | +18.1 | +0.6% | worse |
| EE | 1,176 | 85.2 | 88.0 | +2.7 | +3.2% | worse |
| ES | 1,176 | 1,129.8 | 1,240.3 | +110.5 | +9.8% | worse |
| FI | 1,176 | 742.3 | 780.1 | +37.8 | +5.1% | worse |
| FR | 1,176 | 1,955.7 | 1,915.6 | -40.1 | -2.1% | better |
| HR | 1,176 | 301.1 | 310.5 | +9.4 | +3.1% | worse |
| HU | 1,176 | 441.9 | 454.0 | +12.1 | +2.7% | worse |
| LT | 1,176 | 252.7 | 252.7 | +0.0 | +0.0% | identical |
| LV | 1,176 | 132.1 | 131.9 | -0.2 | -0.2% | tie |
| NL | 1,176 | 1,730.7 | 1,742.2 | +11.5 | +0.7% | worse |
| PL | 1,176 | 882.0 | 878.3 | -3.7 | -0.4% | tie |
| PT | 1,176 | 591.8 | 607.9 | +16.1 | +2.7% | worse |
| RO | 1,176 | 444.5 | 444.5 | +0.0 | +0.0% | identical |
| SI | 1,176 | 107.1 | 107.4 | +0.3 | +0.3% | tie |
| SK | 1,176 | 182.8 | 185.3 | +2.6 | +1.4% | worse |

`identical` means the challenger passed that country through uncorrected — it *is* the champion there, by design, not by tie.
