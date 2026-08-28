#!/usr/bin/env python
"""Is CH solar's readmission premise still true in the source tree? By value.

ABL-583 section 1. The readmission rests on a claim about four constants, and
this script checks it the only way that is not vacuous: it resolves what each
name actually *holds* at each revision, rather than comparing how it is spelled.

WHY AN AST-DUMP COMPARISON IS THE WRONG INSTRUMENT HERE
-------------------------------------------------------
The obvious check is to hash `ast.dump` of each constant's value node across the
revisions -- immune to the comment-only hunks the read commit carried, and it is
what this pack did first. On this particular table it is **two-thirds vacuous**,
because two of the three names are derived expressions rather than literals:

    LEGACY_FEATURE_COLUMNS = tuple(c for c in FEATURE_COLUMNS
                                   if c not in SOLAR_GEOMETRY_FEATURES)
    DEFAULT_SCOPE_FEATURES = FEATURE_COLUMNS

`DEFAULT_SCOPE_FEATURES`'s value node is a bare `Name`. Its dump is
`Name(id='FEATURE_COLUMNS', ctx=Load())` at every revision **whatever
FEATURE_COLUMNS holds** -- which is exactly the move ABL-395 made when it went
from 25 names to 27, and exactly the move that withdrew CH solar from the ship
set. An AST hash on that row reads "identical" across the very change it exists
to detect. `LEGACY_FEATURE_COLUMNS` is a `GeneratorExp` over the same name and
has the same blind spot.

`FEATURE_COLUMNS` itself is only *partly* a literal -- it ends `*SOLAR_GEOMETRY_FEATURES`,
splatting a tuple from a third module -- so it carries the same blind spot one
level up, and that one is live on today's tree. `--demonstrate-blind-spot`
proves it rather than arguing it: it re-resolves the chain **in memory** with one
extra name appended to `SOLAR_GEOMETRY_FEATURES` and reports both instruments
side by side. The result is that `FEATURE_COLUMNS` goes 27 -> 28 names -- the
precise scenario this pack says must fail the suite rather than silently re-base
the artifact -- while its AST hash and `DEFAULT_SCOPE_FEATURES`'s do not move at
all. Nothing is written to the tree; the mutation is a string substitution on a
blob already in memory.

So the constants are lifted out by AST and then **evaluated**, in dependency
order, against the revision's own source. What is compared is the tuple, not the
spelling.

THE FOUR NAMES DO NOT LIVE IN ONE MODULE
-----------------------------------------
A second reason to resolve rather than to read: the chain crosses three files,
and a report that says "the constants in solar_retrain" is already wrong.

    src/solar_features.py            SOLAR_GEOMETRY_FEATURES   (an AnnAssign)
    src/evaluation/solar_retrain.py  FEATURE_COLUMNS           (splats the above)
    scripts/evaluate_solar_retrain.py
                                     LEGACY_FEATURE_COLUMNS
                                     DEFAULT_SCOPE_FEATURES
                                     SCOPE_FEATURES

WHAT IT DOES NOT DO
-------------------
It does not import the modules. Importing gives one revision -- the checked-out
one -- and the claim is about four. It reads each revision's blob with
`git show`, which is also why it needs no database and touches no replica.

It grades nothing and reads no gate. The disposition is the CEO's.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

#: The constant chain, in dependency order. `(path, name)` -- resolving in this
#: order is what lets a derived expression be evaluated against the revision's
#: own upstream value instead of against the checked-out tree's.
CONSTANT_CHAIN = (
    ("src/solar_features.py", "SOLAR_GEOMETRY_FEATURES"),
    ("src/evaluation/solar_retrain.py", "FEATURE_COLUMNS"),
    ("scripts/evaluate_solar_retrain.py", "LEGACY_FEATURE_COLUMNS"),
    ("scripts/evaluate_solar_retrain.py", "DEFAULT_SCOPE_FEATURES"),
    ("scripts/evaluate_solar_retrain.py", "SCOPE_FEATURES"),
)

#: The revisions the readmission premise spans: where the new scope was
#: registered, where it was read, what merged, and what this branch would ship.
DEFAULT_REVISIONS = ("82e3108", "49ab9e9", "origin/main", "HEAD")

#: The scope ABL-581 registered and read CH solar under.
NEW_SCOPE = "abl581-ch-solar-f27"

#: The tranche whose pin withdrew CH solar, and which must stay pinned.
LEGACY_PINNED_SCOPE = "abl316-t1b"


def blob(rev: str, path: str, repo: Path) -> str:
    """The file's contents at `rev`, without checking anything out."""
    return subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        capture_output=True, text=True, check=True, cwd=repo,
    ).stdout


def assigned_expressions(source: str, names: set[str]) -> dict[str, str]:
    """`name -> unparsed value expression` for module-level assignments.

    Handles `AnnAssign` as well as `Assign`: `SOLAR_GEOMETRY_FEATURES` carries a
    `Tuple[str, ...]` annotation, and a walker that only knows `Assign` misses it
    and fails with a `KeyError` that reads like a missing constant.
    """
    found: dict[str, str] = {}
    for node in ast.parse(source).body:
        target = None
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            target = node.targets[0].id
        elif (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.value is not None):
            target = node.target.id
        if target in names:
            found[target] = ast.unparse(node.value)
    return found


def resolve(rev: str, repo: Path) -> dict[str, object]:
    """Every constant in the chain, as the value it holds at `rev`."""
    namespace: dict[str, object] = {}
    cache: dict[str, str] = {}
    for path, name in CONSTANT_CHAIN:
        if path not in cache:
            cache[path] = blob(rev, path, repo)
        expressions = assigned_expressions(cache[path], {name})
        if name not in expressions:
            raise SystemExit(f"{rev}: {name} not assigned at module level in {path}")
        # `eval` against the namespace built so far, so a derived expression sees
        # this revision's upstream value and not the checked-out tree's.
        namespace[name] = eval(expressions[name], dict(namespace))  # noqa: S307
    return namespace


def value_hash(value: object) -> str:
    """Canonical digest of a resolved constant. Tuples and dicts both land here,
    so `default=list` normalises the tuple/list distinction that JSON erases
    anyway -- the claim is about membership and order, not about the container.
    """
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=list).encode()
    ).hexdigest()[:16]


def ast_hash(rev: str, path: str, name: str, repo: Path) -> str:
    """The instrument this script replaces, kept so the record shows the contrast."""
    node = ast.parse(blob(rev, path, repo))
    for item in node.body:
        targets = (item.targets if isinstance(item, ast.Assign)
                   else [item.target] if isinstance(item, ast.AnnAssign) else [])
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name and item.value:
                return hashlib.sha256(ast.dump(item.value).encode()).hexdigest()[:16]
    raise SystemExit(f"{rev}: {name} not found in {path}")


#: The upstream name the blind-spot demonstration appends. Any name absent from
#: the real tuple works; a plausible one is used so the printed contrast reads as
#: the change it is standing in for rather than as a typo.
BLIND_SPOT_PROBE_NAME = "clearsky_index"


def blind_spot_demonstration(rev: str, repo: Path) -> dict[str, object]:
    """Re-resolve the chain with one extra upstream geometry name, in memory.

    Shows the two instruments disagreeing on a change that matters: appending a
    name to `SOLAR_GEOMETRY_FEATURES` moves `FEATURE_COLUMNS` from 27 to 28 while
    leaving its `ast.dump` untouched, because the splat is the same expression
    either way. Touches no file -- the substitution is on the expression string.
    """
    sources = {path: blob(rev, path, repo) for path, _ in CONSTANT_CHAIN}
    expressions = {
        name: assigned_expressions(sources[path], {name})[name]
        for path, name in CONSTANT_CHAIN
    }

    def resolve_with(geometry_expression: str) -> dict[str, object]:
        namespace: dict[str, object] = {}
        for _, name in CONSTANT_CHAIN:
            source = (geometry_expression if name == "SOLAR_GEOMETRY_FEATURES"
                      else expressions[name])
            namespace[name] = eval(source, dict(namespace))  # noqa: S307
        return namespace

    actual = resolve_with(expressions["SOLAR_GEOMETRY_FEATURES"])
    probed_expression = repr(
        tuple(actual["SOLAR_GEOMETRY_FEATURES"]) + (BLIND_SPOT_PROBE_NAME,))
    probed = resolve_with(probed_expression)

    rows = []
    for _, name in CONSTANT_CHAIN:
        expression_actual = expressions[name]
        expression_probed = (probed_expression if name == "SOLAR_GEOMETRY_FEATURES"
                             else expression_actual)
        ast_actual = hashlib.sha256(
            ast.dump(ast.parse(expression_actual, mode="eval").body).encode()
        ).hexdigest()[:16]
        ast_probed = hashlib.sha256(
            ast.dump(ast.parse(expression_probed, mode="eval").body).encode()
        ).hexdigest()[:16]
        value_moved = value_hash(actual[name]) != value_hash(probed[name])
        rows.append({
            "constant": name,
            "n_actual": len(actual[name]),
            "n_probed": len(probed[name]),
            "ast_hash_actual": ast_actual,
            "ast_hash_probed": ast_probed,
            "ast_detects_the_change": ast_actual != ast_probed,
            "value_hash_actual": value_hash(actual[name]),
            "value_hash_probed": value_hash(probed[name]),
            "value_detects_the_change": value_moved,
            # The row that makes the case: the value moved and the AST did not.
            "ast_is_blind_to_this_change": value_moved and ast_actual == ast_probed,
        })

    return {
        "revision": rev,
        "mutation": (f"append {BLIND_SPOT_PROBE_NAME!r} to SOLAR_GEOMETRY_FEATURES, "
                     "in memory only -- no file is written"),
        "writes_to_the_tree": False,
        "constants": rows,
        "constants_the_ast_check_would_miss": [
            row["constant"] for row in rows if row["ast_is_blind_to_this_change"]
        ],
        "reading": (
            "FEATURE_COLUMNS goes 27 -> 28 names, which is the scenario this pack "
            "says must fail the suite rather than silently re-base the artifact, "
            "and the ast.dump hash of FEATURE_COLUMNS and DEFAULT_SCOPE_FEATURES "
            "does not move. An AST comparison would have reported 'identical' "
            "across exactly the change it was there to detect."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Resolve the solar feature-list constant chain by VALUE "
                     "across revisions (ABL-583 section 1)."))
    parser.add_argument(
        "--demonstrate-blind-spot", action="store_true",
        help=("Also re-resolve the chain in memory with one extra upstream "
              "geometry name, to show the AST instrument missing a real change."))
    parser.add_argument("--repo", default=str(Path(__file__).parent.parent),
                        help="Repository to read blobs from.")
    parser.add_argument("--revisions", default=",".join(DEFAULT_REVISIONS),
                        help="Comma-separated revisions to compare.")
    parser.add_argument("--reference", default="origin/main",
                        help="Revision the scope-table assertions are read on.")
    parser.add_argument("--json-out", default="reports/abl_583_scope_value_check.json")
    args = parser.parse_args()

    repo = Path(args.repo)
    revisions = [r.strip() for r in args.revisions.split(",") if r.strip()]
    resolved = {rev: resolve(rev, repo) for rev in revisions}

    constants = []
    for path, name in CONSTANT_CHAIN:
        by_rev = {rev: value_hash(resolved[rev][name]) for rev in revisions}
        constants.append({
            "constant": name,
            "defined_in": path,
            "value_hash_by_revision": by_rev,
            "identical_across_revisions": len(set(by_rev.values())) == 1,
            "ast_hash_by_revision": {
                rev: ast_hash(rev, path, name, repo) for rev in revisions
            },
            "ast_hash_is_vacuous_for_this_constant": name in {
                "LEGACY_FEATURE_COLUMNS", "DEFAULT_SCOPE_FEATURES",
            },
            "value_expression_on_reference": assigned_expressions(
                blob(args.reference, path, repo), {name})[name],
        })

    ref = resolved[args.reference]
    scope_features = ref["SCOPE_FEATURES"]
    legacy = tuple(ref["LEGACY_FEATURE_COLUMNS"])
    current = tuple(ref["FEATURE_COLUMNS"])

    assertions = {
        "n_current_feature_columns": len(current),
        "n_legacy_feature_columns": len(legacy),
        "geometry_names_added": list(ref["SOLAR_GEOMETRY_FEATURES"]),
        "default_scope_features_is_the_current_list":
            tuple(ref["DEFAULT_SCOPE_FEATURES"]) == current,
        "scope_features_keys": sorted(scope_features),
        "new_scope_is_absent_from_scope_features": NEW_SCOPE not in scope_features,
        "legacy_tranche_is_still_pinned":
            tuple(scope_features[LEGACY_PINNED_SCOPE]) == legacy,
        "every_registered_pin_is_the_legacy_list":
            all(tuple(v) == legacy for v in scope_features.values()),
    }

    payload = {
        "issue": "ABL-583",
        "check": "solar feature-list constant chain, resolved by value",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "revisions": revisions,
        "reference_revision": args.reference,
        "reads_the_database": False,
        "why_not_ast": (
            "DEFAULT_SCOPE_FEATURES is a bare Name node and LEGACY_FEATURE_COLUMNS "
            "a GeneratorExp over it, so an ast.dump hash on either is identical at "
            "every revision whatever FEATURE_COLUMNS holds -- including across "
            "ABL-395's 25->27 move, the move that withdrew CH solar. The value "
            "hashes below are the load-bearing check; the ast hashes are carried "
            "only to show the contrast."),
        "constants": constants,
        "reference_assertions": assertions,
        "all_constants_identical_across_revisions":
            all(c["identical_across_revisions"] for c in constants),
        "all_reference_assertions_hold": all(
            v is True for k, v in assertions.items()
            if isinstance(v, bool)),
    }

    if args.demonstrate_blind_spot:
        payload["blind_spot_demonstration"] = blind_spot_demonstration(
            args.reference, repo)

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for entry in constants:
        flag = "vacuous-by-AST" if entry["ast_hash_is_vacuous_for_this_constant"] else ""
        print(f"  {entry['constant']:24s} "
              f"{entry['value_hash_by_revision'][args.reference]}  "
              f"identical={entry['identical_across_revisions']}  {flag}")
    print(f"\n  {len(current)} current / {len(legacy)} legacy names; "
          f"{NEW_SCOPE} absent={assertions['new_scope_is_absent_from_scope_features']}; "
          f"{LEGACY_PINNED_SCOPE} still pinned="
          f"{assertions['legacy_tranche_is_still_pinned']}")

    demonstration = payload.get("blind_spot_demonstration")
    if demonstration:
        print(f"\n  blind-spot demonstration ({demonstration['mutation']}):")
        print(f"    {'constant':24s} {'n':>7s}  {'ast':^21s}  {'value':^21s}")
        for row in demonstration["constants"]:
            print(f"    {row['constant']:24s} "
                  f"{row['n_actual']:2d}->{row['n_probed']:2d}  "
                  f"{row['ast_hash_actual']} "
                  f"{'MOVED' if row['ast_detects_the_change'] else ' same'}  "
                  f"{row['value_hash_actual']} "
                  f"{'MOVED' if row['value_detects_the_change'] else ' same'}")
        print("    AST would have missed: "
              f"{', '.join(demonstration['constants_the_ast_check_would_miss'])}")

    print(f"Wrote {out}")
    return 0 if payload["all_constants_identical_across_revisions"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
