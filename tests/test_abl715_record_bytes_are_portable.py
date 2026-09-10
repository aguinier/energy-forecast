"""A digest in a record must name bytes every checkout can reproduce (ABL-715).

Nine records in this repo stored SHA-256 digests taken over CRLF working-tree
bytes, because `core.autocrlf=true` on the workstation they were recorded on
turns the LF blobs git stores into CRLF on checkout. Those digests matched on
exactly one machine. CI was the first reader outside it, and four tests --
ABL-438's, ABL-443's and two of ABL-444's -- failed there for that reason
alone, while the same tests passed locally.

The failure is quiet in the worst way: a provenance record whose whole job is
to let a later reader confirm "these grades were computed from these bytes"
answers "no" everywhere except the box that wrote it, and answers it the same
way whether the bytes really changed or merely got checked out.

Two halves have to hold, and this file holds both because either alone rots:

1. **Storage.** `*.json` is pinned `eol=lf` in `.gitattributes` and every
   tracked JSON blob is LF, so what checkout writes is what git stores, on
   Windows and Linux alike.
2. **Writing.** The generators that produce these records write LF explicitly.
   Without that, Python's text mode translates to CRLF on Windows, git
   normalises it back to LF on commit, and the next re-record hashes bytes that
   differ from the ones anybody else gets -- reopening the hole from the other
   side.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Records that pin the files they graded by SHA-256, and the key naming the
#: file each digest is OF. Listed rather than sniffed: the repo holds ~143 other
#: `*_sha256` values that digest model artifacts and training payloads, which
#: are not files in this tree and must not be swept into a file-digest check.
DIGEST_KEYS = {
    "sha256": ("path",),
    "results_sha256": ("results_path", "results"),
    "night_screen_sha256": ("night_screen",),
    "record_sha256": ("record",),
    "source_record_sha256": ("source_record",),
    "source_reread_sha256": ("source_reread",),
    "source_offshore_reread_sha256": ("source_offshore_reread",),
}
PINNING_RECORDS = [
    "experiments/ABL443/config.json",
    "reports/abl_418_retro_grade.json",
    "reports/abl_419_tranche2c_tables.json",
    "reports/abl_421_tranche2d_tables.json",
    "reports/abl_426_source_arm_delta.json",
    "reports/abl_437_causal_levelling_reread.json",
    "reports/abl_438_retro_grade.json",
    "reports/abl_443_offshore_trailing_reread.json",
    "reports/abl_444_g23_floor_reread.json",
]


def git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"`git {' '.join(args)}` failed:\n{proc.stderr}"
    return proc.stdout


def _stored_digests(rel: str):
    """Yield (digest, target path) for every file-digest in one record."""
    def walk(node):
        if isinstance(node, dict):
            for key, val in node.items():
                if key in DIGEST_KEYS and isinstance(val, str):
                    target = next((node[c] for c in DIGEST_KEYS[key]
                                   if isinstance(node.get(c), str)), None)
                    if target is not None:
                        yield val, target
                yield from walk(val)
        elif isinstance(node, list):
            for item in node:
                yield from walk(item)
    yield from walk(json.loads((REPO_ROOT / rel).read_text(encoding="utf-8")))


@pytest.mark.parametrize("record", PINNING_RECORDS)
def test_every_digest_in_a_record_matches_the_file_it_names(record):
    """The whole point of the digest, asserted for every record that stores one.

    Four of these were already checked by their own issue's tests, which is how
    CI found the problem; five were not, and held digests that matched nothing
    on any machine. A per-issue test only covers the record whose issue thought
    to write one, so the rule is asserted here over the set instead.
    """
    checked, wrong = 0, []
    for digest, target in _stored_digests(record):
        path = REPO_ROOT / target
        if not path.is_file():
            wrong.append(f"{target}: named by a digest but not in the tree")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        # Records store either the full digest or a documented prefix.
        if actual[:len(digest)] != digest:
            wrong.append(f"{target}: recorded {digest[:16]}..., file is {actual[:16]}...")
        checked += 1
    assert not wrong, (
        f"{record} names bytes that are not the bytes on disk:\n  "
        + "\n  ".join(wrong)
        + "\n\nIf the file legitimately changed, re-record the digest in the "
        "same commit. If it did not, this is the ABL-715 line-ending failure: "
        "the digest was taken over a CRLF checkout."
    )
    assert checked, f"{record} stores no file digest — has its shape changed?"


def test_every_tracked_json_is_stored_and_checked_out_as_lf():
    """`git ls-files --eol` reads the index and the attributes, not this
    working tree, so the answer does not depend on which box runs it."""
    offenders, examined = [], 0
    for line in git("ls-files", "--eol", "--", "*.json").splitlines():
        # "i/lf    w/lf    attr/text eol=lf\tpath/to/file.json"
        fields, _, path = line.partition("\t")
        index_eol, _, rest = fields.strip().partition(" ")
        examined += 1
        if index_eol != "i/lf" or "eol=lf" not in rest:
            offenders.append(f"{path.strip()}: {fields.strip()}")
    # Without this the check passes on an empty list, which is what it would
    # get the day the pathspec stops matching.
    assert examined > 100, f"only {examined} JSON file(s) examined — pathspec broken?"
    assert not offenders, (
        "JSON that is not stored-and-pinned as LF. A record digest over these "
        "bytes is a digest of whatever the recording box happened to check "
        "out:\n  " + "\n  ".join(offenders)
        + "\n\nPin it in .gitattributes (`*.json text eol=lf`) and re-checkout "
        "(`git checkout -- '*.json'`) before recording any digest over it."
    )


def test_the_gitattributes_pin_is_still_there():
    """The check above passes vacuously on a tree with no JSON, and would go
    quiet the day someone trims `.gitattributes`. Assert the rule itself."""
    text = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    rules = [line.split() for line in text.splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    assert any(rule[0] == "*.json" and "eol=lf" in rule for rule in rules), (
        ".gitattributes no longer pins *.json to eol=lf. Every SHA-256 in "
        "reports/ and experiments/ is taken over these bytes (ABL-715)."
    )


@pytest.mark.parametrize("script", [
    "scripts/abl444_g23_floor_reread.py",
])
def test_a_regenerated_record_is_written_with_lf(script, tmp_path):
    """The write half, proven by writing rather than by reading the source.

    ABL-444's reread is the one that regenerates from files alone -- no
    database, no artifacts -- so it is the one that can be re-run here. It is
    also the record four ship-set documents pin by blob id, which is exactly
    the pin that breaks if a re-run writes different bytes on a different box.
    """
    out = tmp_path / "record.json"
    proc = subprocess.run(
        [sys.executable, script, "--json-out", str(out),
         "--report-out", str(tmp_path / "record.md")],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, f"{script} failed:\n{proc.stderr[-2000:]}"
    written = out.read_bytes()
    assert written, f"{script} wrote nothing"
    assert b"\r\n" not in written, (
        f"{script} wrote CRLF. git will normalise that to LF on commit, so the "
        "digests this run recorded are over bytes no other checkout has "
        "(ABL-715). Pass `newline=\"\\n\"` to write_text."
    )
