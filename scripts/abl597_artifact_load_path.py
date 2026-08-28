"""ABL-597: derive which libraries the serving artifacts' load path actually needs.

Read-only. Opens no database and writes nothing except its own --json-out.

Why a script and not a number in a comment: `models/` is gitignored and is
retrained out of band, so any count written down here rots. This re-derives it.

Method
------
1. Replay the pickle opcodes of every `models/<CC>/<type>/model.joblib` with
   `pickletools` and resolve each GLOBAL / STACK_GLOBAL to a (module, qualname)
   pair. Those names are literally in the artifact bytes: they are what the
   unpickler will import, so they are the load-path requirement, not a guess.

2. Decode the WRITER library version out of the stored model blob. xgboost
   (>=1.6) serialises UBJSON carrying 'version' -> [major, minor, patch].

   The trap this avoids: unpickling the estimator and asking it for its version
   reports the *current* library, because `save_raw`/`save_config` re-serialise
   with whatever is installed now. That reads as "artifact matches environment"
   no matter how far the two have drifted. Only the untouched bytes are evidence.

3. With --check-intercept, compare `base_score` as the writer stored it against
   what the booster reports once loaded. CLAUDE.md's failure mode is an xgboost
   pickle that keeps its trees across a version move and silently resets the
   fitted intercept -- shape intact, level gone, which reads as a bad model
   rather than a bad load. This is the witness for that.

Usage
-----
  .venv\\Scripts\\python.exe scripts/abl597_artifact_load_path.py
  .venv\\Scripts\\python.exe scripts/abl597_artifact_load_path.py --check-intercept
"""

import argparse
import collections
import io
import json
import pickletools
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config

# UBJSON as xgboost writes it: 'version' then array-of-3 typed ints, and
# 'base_score' as a length-prefixed string.
_XGB_VERSION = re.compile(rb"version\[#L.{8}", re.S)
_XGB_BASE_SCORE = re.compile(rb"base_scoreSL.{8}", re.S)
_UBJ_INT_WIDTH = {b"i": 1, b"U": 1, b"I": 2}


def artifact_symbols(raw):
    """Every (module, qualname) the unpickler would import for this artifact.

    STACK_GLOBAL takes its two operands off the stack, and the pickler always
    emits them immediately before it -- either as literal string pushes or, for
    a module name it has already seen, as a BINGET into the memo. So this keeps
    a shadow stack of the last few pushed values and resolves BINGET through the
    memo.

    Two traps, both of which produce a plausible-looking wrong answer rather than
    an error, and both of which this has been caught by:

    - Peeking at "the last two strings seen" without popping pairs the operands
      of one STACK_GLOBAL with the next string pushed, inventing a symbol that
      is not in the file at all.
    - MEMOIZE numbers *every* memoised object, not just the strings. Advancing
      the memo index only on strings drifts it out of step with the stream, and
      a later BINGET then resolves to the wrong module name. Every push
      therefore advances the shadow stack, non-strings included, as None.
    """
    symbols = set()
    recent = []
    memo = {}
    memo_index = 0
    string_ops = ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8")
    for op, arg, _pos in pickletools.genops(io.BytesIO(raw)):
        name = op.name
        if name == "GLOBAL":
            module, _, qualname = arg.partition(" ")
            symbols.add((module, qualname))
            recent.append(None)
        elif name in string_ops:
            recent.append(arg)
        elif name in ("BINGET", "LONG_BINGET", "GET"):
            recent.append(memo.get(arg))
        elif name == "MEMOIZE":
            memo[memo_index] = recent[-1] if recent else None
            memo_index += 1
        elif name in ("BINPUT", "LONG_BINPUT", "PUT"):
            memo[arg] = recent[-1] if recent else None
        elif name == "STACK_GLOBAL":
            qualname = recent.pop() if recent else None
            module = recent.pop() if recent else None
            if isinstance(module, str) and isinstance(qualname, str):
                symbols.add((module, qualname))
            recent.append(None)
        else:
            recent.append(None)
        del recent[:-4]
    return symbols


def xgboost_writer_version(raw):
    """The xgboost version that wrote this artifact, from the stored blob."""
    match = _XGB_VERSION.search(raw)
    if match is None:
        return None
    count = int.from_bytes(raw[match.end() - 8:match.end()], "big")
    offset = match.end()
    parts = []
    for _ in range(min(count, 4)):
        marker = raw[offset:offset + 1]
        offset += 1
        width = _UBJ_INT_WIDTH.get(marker)
        if width is None:
            return None
        parts.append(int.from_bytes(raw[offset:offset + width], "big", signed=marker == b"i"))
        offset += width
    return ".".join(str(part) for part in parts)


def stored_base_score(raw):
    """base_score as the writing library stored it, before any re-serialisation."""
    match = _XGB_BASE_SCORE.search(raw)
    if match is None:
        return None
    length = int.from_bytes(raw[match.end() - 8:match.end()], "big")
    return raw[match.end():match.end() + length].decode("ascii", "replace")


def _as_float(value):
    """xgboost 3.x reports base_score as a vector string ('[6.69E3]'); 2.x as a scalar."""
    return float(str(value).strip().lstrip("[").rstrip("]").split(",")[0])


def loaded_base_score(path):
    """base_score after joblib.load under the *current* interpreter, or None."""
    import joblib

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        payload = joblib.load(path)
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    if model is None or type(model).__module__.split(".")[0] != "xgboost":
        return None
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    config_json = json.loads(booster.save_config())
    return config_json["learner"]["learner_model_param"]["base_score"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", default=None,
                        help="artifact root (default: config.MODELS_DIR)")
    parser.add_argument("--check-intercept", action="store_true",
                        help="load each xgboost artifact and compare base_score "
                             "against the value its writer stored (slow)")
    parser.add_argument("--json-out", default=None, help="write the machine record here")
    args = parser.parse_args()

    root = Path(args.models_dir) if args.models_dir else config.MODELS_DIR
    artifacts = sorted(root.glob("*/*/model.joblib"))
    if not artifacts:
        print(f"No artifacts under {root}")
        return 1

    module_counts = collections.Counter()
    symbol_counts = collections.Counter()
    writer_counts = collections.Counter()
    per_artifact = {}
    parse_errors = []
    intercept_moved = []
    intercept_checked = 0

    for path in artifacts:
        raw = path.read_bytes()
        key = str(path.relative_to(root)).replace("\\", "/")
        try:
            symbols = artifact_symbols(raw)
        except Exception as exc:  # a truncated artifact is a finding, not a crash
            parse_errors.append((key, repr(exc)))
            symbols = set()
        for module, qualname in symbols:
            module_counts[module.split(".")[0]] += 1
            symbol_counts[f"{module}.{qualname}"] += 1

        writer = xgboost_writer_version(raw) if b"xgboost" in raw else None
        if b"xgboost" in raw:
            writer_counts[writer] += 1
        per_artifact[key] = {
            "symbols": sorted(f"{m}.{q}" for m, q in symbols),
            "xgboost_writer_version": writer,
            "stored_base_score": stored_base_score(raw),
        }

        if args.check_intercept and per_artifact[key]["stored_base_score"] is not None:
            stored = per_artifact[key]["stored_base_score"]
            try:
                loaded = loaded_base_score(path)
                same = abs(_as_float(stored) - _as_float(loaded)) <= 1e-9 * max(
                    1.0, abs(_as_float(stored)))
            except Exception as exc:
                loaded, same = f"ERROR {exc!r}", False
            per_artifact[key]["loaded_base_score"] = str(loaded)
            intercept_checked += 1
            if not same:
                intercept_moved.append((key, stored, str(loaded)))

    print(f"artifacts: {len(artifacts)}   parse errors: {len(parse_errors)}")
    for key, err in parse_errors:
        print(f"  PARSE ERROR {key}: {err}")

    print("\n-- top-level module -> artifacts naming it in their own bytes --")
    for module, count in module_counts.most_common():
        print(f"   {module:<16} {count}")

    print("\n-- distinct symbols the unpickler will import --")
    for symbol, count in sorted(symbol_counts.items()):
        print(f"   {symbol:<52} {count}")

    print("\n-- xgboost WRITER version -> artifact count --")
    for version, count in writer_counts.most_common():
        print(f"   xgboost {str(version):<10} {count}")

    if args.check_intercept:
        print(f"\n-- intercept witness: {intercept_checked} xgboost artifact(s) checked --")
        if intercept_moved:
            print(f"   *** base_score MOVED on {len(intercept_moved)} artifact(s) ***")
            for key, stored, loaded in intercept_moved:
                print(f"     {key}: stored={stored} loaded={loaded}")
        else:
            print("   base_score identical on every artifact (rel err <= 1e-9)")

    if args.json_out:
        record = {
            "models_dir": str(root),
            "artifact_count": len(artifacts),
            "module_counts": dict(module_counts),
            "symbol_counts": dict(symbol_counts),
            "xgboost_writer_versions": {str(k): v for k, v in writer_counts.items()},
            "parse_errors": parse_errors,
            "intercept_checked": intercept_checked,
            "intercept_moved": intercept_moved,
            "per_artifact": per_artifact,
        }
        Path(args.json_out).write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(f"\nmachine record -> {args.json_out}")

    return 1 if parse_errors or intercept_moved else 0


if __name__ == "__main__":
    sys.exit(main())
