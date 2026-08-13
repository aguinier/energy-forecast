"""ABL-385: run the registered variance sweep. Derives every invocation from the
frozen registration, so the run cannot drift from what was pre-registered.

This is a driver, not a harness. It shells out to
`scripts/abl338_solar_holdout.py` - the harness ABL-338 wrote, ABL-375 added
`--seeds` to, and this issue added `--type` to - once per
(forecast type, algorithm, window), with every country of that type in the same
invocation so the read and the feature build are paid once rather than per seed.

Two properties worth knowing:

**It is resumable.** Each invocation writes its own JSON, and an invocation whose
output already exists is skipped unless ``--force``. A sweep that is interrupted
after 40 of 88 invocations can be restarted and will not refit the 40. That
matters because the alternative - one long process holding every result in
memory - loses everything to a single failure, and this sweep is ~90 minutes.

**It does not decide anything.** It fits and writes. Every number that reaches a
verdict is computed by `scripts/abl385_read_margin.py`, which never fits, so the
analysis can be re-run and reviewed without touching the fits it reads.

Usage
-----
    .venv\\Scripts\\python.exe scripts/abl385_run_sweep.py --out reports/abl_385_sweep
    .venv\\Scripts\\python.exe scripts/abl385_run_sweep.py --dry-run     # print the plan
    .venv\\Scripts\\python.exe scripts/abl385_run_sweep.py --only primary

`ENERGY_DB_PATH` must be passed explicitly from a worktree - `.env` is gitignored
and `config.DATABASE_PATH` otherwise degrades to a bare `\\data\\energy_dashboard.db`.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402

REGISTRATION = REPO_ROOT / "experiments" / "ABL385" / "config.json"
HARNESS = REPO_ROOT / "scripts" / "abl338_solar_holdout.py"

#: The interpreter that trains, serves and evaluates (ABL-69). Hardcoded to the
#: rail rather than `sys.executable`, because a driver launched by accident under
#: the conda Python would otherwise spawn 88 subprocesses that all silently reset
#: a fitted xgboost intercept to 0.5 and produce a series with shape and no
#: level. `sys.executable` is used when it *is* the rail, so a worktree venv or a
#: future relocation still works.
RAIL_PYTHON = Path(r"C:\Code\able\energy-forecast\.venv\Scripts\python.exe")


def _interpreter() -> str:
    here = Path(sys.executable).resolve()
    if here.name.lower() == "python.exe" and "miniconda" not in str(here).lower():
        return str(here)
    if RAIL_PYTHON.exists():
        return str(RAIL_PYTHON)
    raise SystemExit(
        f"refusing to run under {sys.executable}: an xgboost 3.3.0 artifact loaded "
        f"under 2.1.4 silently resets its fitted intercept (ABL-69). Use "
        f"{RAIL_PYTHON}."
    )


def _arms_for(reg: dict, forecast_type: str) -> list:
    """The registered arms for a type.

    Solar carries two (`control` and `geometry`) because both exist for it and
    only for it, and fitting both at matched seeds is what lets the seed-pairing
    question be answered. Every other type carries `control` alone - the geometry
    pair is appended by `create_all_features` for solar only, so a `geometry` arm
    elsewhere would be byte-identical to `control`.
    """
    if forecast_type == "solar":
        return list(reg["scope"]["solar_second_arm"]["arms"])
    return ["control"]


def build_plan(reg: dict, out_dir: Path, only: str) -> list:
    """Every registered invocation, as (label, argv, expected_output_path)."""
    seeds = ",".join(str(s) for s in reg["scope"]["seeds"])
    algorithms = reg["scope"]["algorithms"]

    groups = {}  # forecast_type -> [country, ...], in registration order
    selected = []
    if only in ("primary", "all"):
        selected += reg["scope"]["pairs_primary"]
    if only in ("secondary", "all"):
        selected += reg["scope"]["pairs_secondary"]
    for pair in selected:
        groups.setdefault(pair["type"], []).append(pair["country"])

    plan = []
    for window in reg["scope"]["windows"]:
        for forecast_type, countries in groups.items():
            for algorithm in algorithms:
                plan.append(_invocation(
                    out_dir, forecast_type, countries, algorithm, seeds,
                    _arms_for(reg, forecast_type),
                    window["start"], window["end"], window["window"],
                    start_date="2023-01-01",
                ))

    # The season-controlled fit-length ablation, W6 only. `only=secondary` is a
    # partial rerun of the non-gated pairs and does not include it.
    if only in ("primary", "all"):
        ablation = reg["scope"]["fit_length_ablation"]
        w6 = reg["scope"]["windows"][-1]
        countries = [p["country"] for p in reg["scope"]["pairs_primary"]
                     if f"{p['country']}/{p['type']}" in ablation["pairs"]]
        for fit_start in ablation["fit_starts"]:
            for algorithm in algorithms:
                plan.append(_invocation(
                    out_dir, "solar", countries, algorithm, seeds,
                    _arms_for(reg, "solar"), w6["start"], w6["end"],
                    f"abl-{fit_start}", start_date=fit_start,
                ))
    return plan


def _invocation(out_dir, forecast_type, countries, algorithm, seeds, arms,
                holdout_start, holdout_end, tag, start_date):
    argv = [
        str(HARNESS),
        "--type", forecast_type,
        "--countries", ",".join(countries),
        "--start", start_date,
        "--holdout", f"{holdout_start}:{holdout_end}",
        "--arms", ",".join(arms),
        "--seeds", seeds,
        "--force-algorithm", algorithm,
        "--out", str(out_dir),
        "--tag", tag,
    ]
    # Solar-only, and the harness refuses it elsewhere rather than ignoring it.
    name = f"holdout_{tag}"
    if forecast_type != "solar":
        name += f"_{forecast_type}"
    name += f"_{algorithm}"
    if forecast_type == "solar":
        argv.append("--drop-impossible-night")
        name += "_cleaned"
    label = f"{forecast_type:13s} {algorithm:9s} {tag:16s} {','.join(countries)}"
    return label, argv, out_dir / f"{name}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/abl_385_sweep")
    parser.add_argument("--only", default="all", choices=["all", "primary", "secondary"],
                        help="Which registered pair set to run (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and the commands; fit nothing")
    parser.add_argument("--force", action="store_true",
                        help="Refit invocations whose output already exists")
    args = parser.parse_args()

    reg = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(reg, out_dir, args.only)
    python = _interpreter()

    print(f"registration : {REGISTRATION}")
    print(f"interpreter  : {python}")
    print(f"replica      : {config.DATABASE_PATH}")
    print(f"seeds        : {reg['scope']['n_seeds']} -> {reg['scope']['seeds']}")
    print(f"invocations  : {len(plan)}")
    print()

    if args.dry_run:
        for label, argv, expected in plan:
            state = "SKIP (exists)" if expected.exists() and not args.force else "run"
            print(f"  [{state:13s}] {label}")
            print(f"      {python} {' '.join(argv)}")
        return 0

    started = time.time()
    failures = []
    for index, (label, argv, expected) in enumerate(plan, start=1):
        if expected.exists() and not args.force:
            print(f"[{index:3d}/{len(plan)}] SKIP {label}  ({expected.name} exists)",
                  flush=True)
            continue
        step = time.time()
        print(f"[{index:3d}/{len(plan)}] {label} ...", end="", flush=True)
        completed = subprocess.run([python] + argv, capture_output=True, text=True)
        if completed.returncode != 0:
            # A failed invocation must not look like a skipped one on the next
            # pass: no output file is written, so a rerun retries it.
            print(f" FAILED rc={completed.returncode}", flush=True)
            print(completed.stderr[-1500:], flush=True)
            failures.append(label)
            continue
        if not expected.exists():
            print(f" rc=0 but {expected.name} is missing", flush=True)
            failures.append(f"{label} (no output)")
            continue
        print(f" ok  {time.time() - step:5.1f}s", flush=True)

    elapsed = time.time() - started
    print(f"\n{len(plan) - len(failures)}/{len(plan)} invocations complete "
          f"in {elapsed / 60:.1f} min")
    if failures:
        print("FAILED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
