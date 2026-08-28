# ABL-606 — the 8 standing container failures: diagnosis and disposition

**Verdict: disposition 1.** `chronos-bolt-small` and `tso-correction` are
**workstation-only by design**. They are not meant to serve, they have not
served since 2026-03-03, and the dashboard already refuses to register them.
The defect is that they were registered in the production matrix at all.

**Pre-existing, not from ABL-597.** The two interpreter paths have been in
`config.py` since the initial commit (996c45a, 2026-03-05) and have never been
edited. ABL-597 touched five files, none of them `config.py` or
`forecast_daily.py`.

**After the fix the expected floor is `Total: 432, Failed: 0`** — that is the
number the next deploy diffs against (ABL-603 hazard 1).

---

## 1. The measurement, reproduced without a container

ABL-601, inside the rebuilt container:

```
Total: 440, Success: 154, Empty: 0, Unreported: 0, Skipped: 278, Failed: 8
```

`scripts/abl606_runner_matrix_probe.py` re-derives both numbers from `config`
alone, on `origin/main` at e0ec351, no container and no database:

```
countries=24 types=9 horizons=2
builtin cells            : 432
external runners, pre-fix selection (type=external and enabled):
  chronos-bolt-small   production=False cells=2   exe=C:\Users\guill\.openclaw\...\chronos-venv\Scripts\python.exe
  tso-correction       production=False cells=6   exe=C:\Users\guill\miniconda3\python.exe
external cells pre-fix   : 8
TOTAL pre-fix            : 440  (ABL-601 measured 440)
```

The 8 are exactly, and only, the two runners:

| runner | matrix | cells |
|---|---|---|
| `chronos-bolt-small` | BE × price × {D+1, D+2} | 2 |
| `tso-correction` | BE × {solar, wind_onshore, wind_offshore} × {D+1, D+2} | 6 |

`154 + 278 = 432` closes the other side: every builtin cell is accounted for as
success or skip, and all 8 failures are external. Nothing else in ABL-601's run
was failing.

## 2. Why it happens

`MODEL_RUNNERS` carries a `"production"` key on **every** entry, each with a
hand-written comment explaining the choice. Nothing read it. Selection was:

```python
external_runners = [r for r in config.MODEL_RUNNERS
                    if r.get('type') == 'external' and r.get('enabled', False)]
```

Both offending entries are `"enabled": True, "production": False`. The flag
that says "not the scheduled matrix" was inert, so the container launched them
and hit `FileNotFoundError` on a `C:\Users\guill\...` path 8 times per run.

This is not a container-specific defect. The same paths fail on the
workstation; CLAUDE.md has recorded that `chronos-bolt-small` is "genuinely
unrunnable on this box (venv missing)" since ABL-370.

## 3. Which disposition, on three independent readings

**(a) Neither model name has written a forecast row since 2026-03-03.** Replica
census, `country_code='BE'`, the four registered types:

| type | model_name | n | first generated_at | last generated_at |
|---|---|---:|---|---|
| price | `xgboost` | 36 072 | 2025-12-26 | **2026-08-27T19:00:05** |
| price | `chronos-bolt-small` | 504 | 2026-02-22 | **2026-03-03T16:34:18** |
| solar | `catboost` | 35 040 | 2026-02-02 | **2026-08-27T19:00:06** |
| solar | `tso_corrected` | 506 | 2026-02-25 | **2026-03-03T15:34:39** |
| solar | `tso_raw` | 506 | 2026-02-25 | **2026-03-03T15:34:39** |
| wind_onshore | `catboost` | 35 040 | 2026-02-02 | **2026-08-27T19:00:06** |
| wind_onshore | `tso_corrected` / `tso_raw` | 506 each | 2026-02-25 | **2026-03-03T15:34:39** |
| wind_offshore | `xgboost` | 36 000 | 2025-12-26 | **2026-08-27T19:00:06** |
| wind_offshore | `tso_corrected` / `tso_raw` | 506 each | 2026-02-25 | **2026-03-03T15:34:39** |

The production names write to the hour. These four stop dead on 2026-03-03 —
**two days before this repo's initial commit**. They have never produced a row
from `energy-forecast` as it exists today, let alone from a container.

> Caveat on this table: the replica was mid-refresh (a 600 MB rollback journal,
> an exclusive writer lock), so it was read with `nolock=1` and could in
> principle be a torn read. It is corroborated independently in (b), which
> names the same date from a different repository, and it is a coarse census —
> do not reuse these counts as a metric.

**(b) The dashboard already decided this, and holds it with a test.**
`energy-dashboard-frontend/server/src/config/forecastModels.ts:48`:

> *"Stale models are deliberately absent: chronos-bolt-small (price), lightgbm
> (solar) and tso_raw / tso_corrected all last wrote in Feb–Mar 2026."*

`forecastModels.test.ts:61` asserts all four stay unregistered, and
`client/src/components/dashboard/forecastVintage.ts:86` states
`chronos-bolt-small` "is deliberately unregistered and last wrote 2026-03-03" —
the same date the census returns, arrived at independently.

**(c) This repo's own tests already treat those paths as box-specific.**
`tests/test_script_imports.py::test_model_runner_launches`: *"The runner's
configured `python_executable` is deliberately not used: it is an absolute path
to one box's venv."* And the `tso-correction` conda pin is **deliberate**, not
stale — CLAUDE.md, "The interpreter is part of the configuration": its
artifacts are LightGBM, which round-trips as text under conda.

So the flag was right and the selection was wrong. The fix makes the flag
load-bearing; it does not move, rewrite, or containerise any path.

## 4. Pre-existing or new with ABL-597 — settled

ABL-597 (`ee9deb2`, merged `689c716`) changed five files:

```
reports/abl_597_artifact_load_path.json | 666 +
reports/abl_597_artifact_load_path.md   | 167 +
requirements.txt                        |  64 +-
scripts/abl597_artifact_load_path.py    | 261 +
tests/test_abl597_requirement_pins.py   | 178 +
```

Neither `config.py` nor `scripts/forecast_daily.py`. And
`git log -S` on both interpreter strings in `config.py` returns exactly one
commit: `996c45a`, 2026-03-05, the initial commit. **The 8 have been failing
since day one of this repo.**

### Why ABL-585's baseline said `Failed: 4`

Because before ABL-370 (`cbe2a46`, 2026-08-13) an `Executable not found` was
counted as a *skip*, not a failure. `src/runner_report.is_skip`'s docstring
records it verbatim:

> *"it decided by looking for `not found` in the error text … A runner whose
> configured `python_executable` does not exist fails with `Executable not
> found: [WinError 2]` — a runner that cannot run at all, counted as benign."*

ABL-585's `Total: 392, Success: 148, Failed: 4` is consistent with that
reading: 384 builtin cells = 148 success + 4 failed (the CZ/RO solar wall) +
232 skipped, plus 8 external absorbed into skip = 240 skipped, 392 total. The
384 implies the stale image carried one fewer forecast type than
`origin/main`'s 9, which I cannot check from outside the image — so treat the
decomposition as a *consistent reconstruction*, not a measurement. What does
not depend on it: 4 < 8, and ABL-601 attributes those 4 to CZ/RO solar, so the
8 cannot have been inside ABL-585's `Failed` column either way.

**ABL-370 did not create these failures. It made them sayable.** That is the
same defect species one layer up: the count was always 8; for five months it
was filed under a heading that read as benign.

## 5. The change

`select_external_runners()` in `scripts/forecast_daily.py` separates three
questions that had been collapsed into two flags:

| key | question |
|---|---|
| `type` | subprocess, or the in-process builtin? |
| `enabled` | wired up at all, or parked (`chronos-2`, awaiting fine-tuning)? |
| `production` | belongs in the **scheduled** matrix? |

A scheduled run takes `production` runners only. `--include-non-production` is
the opt-in that brings the workstation experiments back on a box that has their
interpreters — so nothing is deleted, and the scheduled job cannot re-acquire
them by accident. Both entries keep `enabled: True`, which also keeps them
inside the ABL-354 and ABL-364 coverage sweeps (`test_script_imports`,
`test_help_text_encoding`), neither of which filters on these flags.

The run log now names the selection, so a moved `Total:` is attributable
without re-reading config:

```
External runners: none (non-production, not run: ['chronos-bolt-small', 'tso-correction'])
```

`tests/test_abl606_production_runner_matrix.py` (10 tests) holds it, including
the rule rather than the two instances: **a runner whose `python_executable` is
an absolute path outside the repo `.venv` must be `production: False`**, which
is what a future entry would trip over. Mutation check — reverting the filter
turns 4 of the 10 red, on `assert 8 == 0`.

## 6. What the next deploy should see

| | before | after |
|---|---:|---:|
| `Total` | 440 | **432** |
| `Failed` (expected floor) | 8 | **0** |
| `Success` + `Skipped` | 432 | 432 |

`Failed: 0` is now the floor, so ABL-603's hazard-1 guard — diff the
`Total/Success/Failed` counts across a deploy — works again: **any** non-zero
`Failed` is a real regression, and a `Total` other than 432 means the matrix
itself moved.

One consequence worth stating for the deployer: `forecast_daily` returns exit 1
and writes `status='failed'` on the `forecast_runs` row whenever `failed > 0`
(`scripts/forecast_daily.py`, the `complete_forecast_run` call). On the reading
above, every scheduled production run since ABL-370 shipped has been recording
itself as a failed run for these 8 cells alone. I could not confirm that
against `forecast_runs` — the replica held an exclusive writer lock for the
duration of this work — so it is stated as a read of the code, not a
measurement. It is cheap to check once the refresh clears.

## 7. Scope and boundaries

- No serving output changes. The `builtin` runner (`production: True`) produces
  every row the dashboard reads; it is untouched. The two removed runners
  produce nothing served, on the evidence in §3.
- Contamination: none of ABL-71 / ABL-67 / ABL-111 / ABL-109 bears on this. The
  census in §3 is a `generated_at` recency read on the `forecasts` table, not
  an accuracy measurement against actuals.
- Landing this in production is a container rebuild, which is the Deployment
  Engineer's call, not mine. This is an evidence pack and a diff.
- Not touched, deliberately: ABL-599 (rail vs container xgboost majors) and
  ABL-600 (unpinned transitive deps). Neither explains an `Executable not
  found`, and neither is fixed by this.
