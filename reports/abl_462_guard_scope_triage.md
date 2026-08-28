# ABL-462 — the TSO plausibility sweep only walked `src/`: scope fix and triage

**Owner:** Forecasting Scientist · **Date measured:** 2026-08-28
**Tree:** branch `ABL-462-sweep-scope`, branched from `origin/main` = `e0ec351`
**Interpreter:** `C:\Code\able\energy-forecast\.venv\Scripts\python.exe` (the rail)
**Data:** live replica `C:\Code\able\data\energy_dashboard.db` (10,718,515,200 bytes),
opened read-only (`file:...?mode=ro`, `uri=True`). Nothing was written to it.
**Serving:** unchanged. No ingest, dashboard or registry change in this work.

---

## 1. The defect, reproduced at today's head

ABL-462 was filed against `origin/main` = `e6aa104` (2026-08-14). Re-measured at
`e0ec351`, unchanged. Control pair, byte-identical file, only the directory differs:

| location of an unguarded `forecast_vintage_archive` reader | `tests/test_tso_plausibility.py` |
|---|---|
| `src/_zz_abl462_control.py` | **1 failed**, 34 passed — the sweep names the file |
| `scripts/_zz_abl462_control.py` | **35 passed** — not detected |

`test_no_unguarded_module_reads_a_tso_forecast_table` walked
`(REPO_ROOT / "src").rglob("*.py")`, so `scripts/` and `experiments/` were out of
scope. Both probe files were temporary and removed; the worktree was verified
clean afterwards.

**Why it mattered for ABL-247.** `abl247-prereg` requires every archive read to go
through `guard_tso_series`. ABL-247's gated backtest lands under `scripts/`. So
the pre-registered requirement was unenforceable in the one directory the work
occupies — a check whose scope silently excluded the code it existed to cover.

---

## 2. The fix: a denylist of non-source directories, not an allowlist of roots

The sweep now walks **every `*.py` under the repo root**, skipping only tooling
and cache directories plus `tests/`. An allowlist of roots acquires a new blind
spot every time someone adds a directory, and nothing notices; a denylist of
non-source directories cannot.

`tests/` is the single excluded source directory, by name and with a reason: the
fixtures in `tests/test_tso_plausibility.py` **create** all three tables and write
a 140,996 MW row on purpose, so sweeping them would make the guard's own negative
controls unwritable. Pinned by
`test_the_suite_is_out_of_scope_deliberately_not_incidentally`.

**Positive controls.** A passing sweep proved nothing before — that was the
defect. `SWEPT_DIRECTORIES` is now derived from the tree rather than listed
(today: `.`, `experiments`, `scripts`, `src`), and each one gets a parameterized
control that plants an unguarded archive read in a synthetic tree and asserts the
sweep names it. A new source directory arrives already controlled.
`test_the_directories_abl247_will_write_in_are_swept` names `scripts` and
`experiments` explicitly, because that is the ABL-247 requirement.

Suite: **45 passed** (was 35), rail interpreter.

---

## 3. Triage of the four `scripts/` files the widened sweep sees

The issue was explicit that this is not a one-line widening: some of these read
raw values *deliberately*, and blanket-exempting `scripts/` would restore the gap
under a different name. Each file was measured, not assumed.

| file | disposition | measured basis |
|---|---|---|
| `scripts/abl247_vintage_availability_probe.py` | already guarded (PR #76) | calls `guard_tso_series` |
| `scripts/abl430_ro_country_diagnosis.py` — A2 wind read | **now guarded** | guard nulls **96 of 408,159** rows |
| `scripts/abl430_ro_country_diagnosis.py` — load coverage census | **deliberately raw**, documented in file | guard would null **0 of 872,355** rows |
| `scripts/abl439_reporting_basis_probe.py` | **now guarded** | guard nulls **0 of 393,044** rows |
| `scripts/attest_net_position_serve_faithfulness.py` | **exempt: mention-only** | no `FROM`/`JOIN` against either table |

### 3.1 `abl430` A2 wind-vs-TSO alignment — guarded

Read: `energy_generation_forecast.wind_onshore_mw`, `forecast_type='day_ahead'`,
`target_timestamp_utc` in `[2026-01-01, 2026-08-11)`, 24 countries, 408,159 rows.

The guard nulls **96 rows, all HU, all 2026-02-03/04**, largest 140,996.2 MW =
498x the 283.2 MW p99.5 reference. The other 23 countries: 0 rows. These are the
exact rows ABL-431 was written for. Guarding runs per country and **before**
`to_hourly`, per CLAUDE.md — a refused value averaged into its neighbour first is
not refused.

### 3.2 `abl430` load coverage census — deliberately raw

Read: `energy_load_forecast.forecast_value_mw`, `[2023-01-01, 2026-03-01)`, 19
gate countries, 872,355 rows.

This counts hours the ingest actually holds. The guard nulls values, so guarding a
**presence census** would report the pipeline as having fetched fewer hours than
it did — it would measure the guard, not the coverage. The semantic argument
stands on its own; the measurement corroborates it, at **0 rows** flagged over
this window, so nothing published rests on the distinction today. Documented in
the function's own docstring so the next reader does not "fix" it.

*Sweep limitation, stated rather than hidden:* the sweep is file-level, so a file
on the guarded list is not re-checked read-by-read. `abl430` now holds one guarded
read and one deliberately raw one. That is pre-existing sweep behaviour, not
introduced here, and the raw read is annotated at the site.

### 3.3 `abl439` NL/DE reporting-basis probe — guarded

Read: `energy_generation_forecast.wind_onshore_mw`, NL and DE,
`[2021-01-01, 2026-08-11)`, 393,044 rows. Guard nulls **0 rows**.

The guard is wired into `_hourly` on a `TSO_FORECAST_SOURCES` membership test,
whose keys are exactly the forecast `(table, column)` pairs — so the two actuals
tables the same helper serves are untouched by construction rather than by a
hand-maintained list. Verified by running both arms against the same replica at
the same instant (guarded, and with the membership map emptied so the guard never
fires): **the two arms are exactly equal** across NL and DE monthly ratios
2024-2026 and yearly ratios 2021-2026. Every number this script published is
unchanged; the guard is there so the next contaminated pair is caught rather than
averaged in.

### 3.4 `attest_net_position_serve_faithfulness.py` — exempt, mention-only

It names both live tables only inside the `source_cutoffs` block of the
attestation manifest it emits — a description of where V014's features come from.
The read itself is `src/challengers/v014_features.py`, which is on the guarded
list. Verified: no `FROM`/`JOIN`/`INTO`/`UPDATE` against either table anywhere in
the file.

This exemption is the only one that is a claim about file *contents* rather than
about intent, so it is pinned:
`test_mention_only_exemptions_execute_no_query_against_a_tso_table` fails if a
real query is ever added, rather than silently covering it.

---

## 4. Consequence: the published ABL-430 A2 table has a superseded HU row

The scope gap was not hypothetical. `reports/abl_430_ro_diagnosis.json` published
its A2 wind-vs-TSO table from the unguarded read. HU was the one country in the
fleet that looked broken:

| | published (unguarded) | corrected (guarded) |
|---|---|---|
| `n` | 5,328 | 5,304 |
| `corr` | **0.0237** | **0.8805** |
| `argmax_lag_h` | **3** | **0** |
| `slope_tso_on_actual` | 2.0992 | 0.812 |
| `wape_pct` | **597.0** | **34.13** |

Read at face value, the published row says HU's TSO wind series is uncorrelated
with our actuals and peaks three hours off — the signature of a zone swap or a
clock error, which is exactly what A2 exists to detect. It is neither. It is 96
rows of 140,996 MW against a 283 MW fleet. Corrected, HU is an unremarkable member
of the fleet, comparable to BG (0.858 / 42.9%) and EE (0.865 / 32.5%).

**The guard term and the data term are separated, not conflated.** Re-running today
mixes two changes: the guard, and 14 days of actuals revision inside the window
(`energy_generation` has a ~28-day revision horizon). Both arms were run against
the same replica at the same instant, differing only in whether the guard fires:

| comparison | countries that move (of 21) | which |
|---|---|---|
| published (2026-08-14, unguarded) vs today unguarded — **data-revision term** | 1 | PT: `wape_pct` 14.44 -> 14.45, `actual_mean_mw` 1481.3 -> 1481.5 |
| today unguarded vs today guarded — **guard term** | 1 | HU, the five fields above |
| published vs today guarded — both | 2 | HU and PT |

The two terms are disjoint. The entire HU flip is attributable to the guard; the
PT move is rounding-level revision and touches nothing this report claims.

`reports/abl_430_ro_diagnosis.json` is **not edited** — it is the record of what
that run produced. The superseding value lives here and is pointed at from the
`check_wind_against_tso` docstring. The ABL-430 markdown report does not quote the
A2 HU cell, so no narrative text is contradicted.

---

## 5. Caveats

- **Sample and window are stated per measurement above**; every figure is a full
  population count over the named window, not a sample estimate. Nothing here is
  a model score, so in-sample / out-of-sample does not apply.
- **Known contamination (ABL-71 / ABL-67 / ABL-111 / ABL-109) touching these
  windows.** A2 compares `energy_generation` against
  `energy_generation_forecast`; ABL-67 (fabricated `net_position` rows) does not
  reach either. ABL-111/109 (zero-as-missing rows in actual load) reach
  `energy_load`, which enters this work only as the p99.5 reference denominator
  for the `energy_load_forecast` census check — a spurious **zero** cannot move a
  p99.5 upper reference, and that check measured 0 flagged rows regardless.
  ABL-71 (prod ingest stale, fixes undeployed) bounds how fresh the tail of any
  window is; it is the mechanism behind the PT revision term isolated in §4, and
  it is why the guard and data terms were measured separately rather than by
  re-running once.
- **The 140,996 MW HU rows are still in the replica.** The guard is a read-time
  refusal; it never touches stored rows. Any other consumer reading that pair
  unguarded still gets them — which is the whole point of widening the sweep.
- **This changes no promotion, no registry entry and no served model.** Two
  diagnostic scripts changed their read, one published diagnostic cell is
  superseded here, and the sweep got wider.

---

## 6. What this unblocks

ABL-247's `abl247-prereg` precondition "clear ABL-462" is discharged: an
unguarded archive read under `scripts/` — the directory ABL-247's backtest will
occupy — now fails `tests/test_tso_plausibility.py`, and that claim is carried by
a positive control per directory rather than by an assertion that has never been
seen to fire.
