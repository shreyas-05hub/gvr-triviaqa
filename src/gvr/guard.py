"""Test-split guard: any validation/test access raises unless freeze_manifest says test_unlocked: true."""
import json
from .paths import FREEZE_MANIFEST

LOCKED_SPLITS = {"validation", "test"}


class TestSplitLockedError(PermissionError):
    __test__ = False
    pass


def test_unlocked(manifest_path=FREEZE_MANIFEST) -> bool:
    with open(manifest_path) as f:
        return json.load(f).get("test_unlocked", False) is True


def check_split(split: str, manifest_path=FREEZE_MANIFEST):
    s = split.lower()
    if any(s == x or s.startswith(x) for x in LOCKED_SPLITS) and not test_unlocked(manifest_path):
        raise TestSplitLockedError(f"access to split '{split}' is locked (test_unlocked is not true)")


def check_file(filename: str, manifest_path=FREEZE_MANIFEST):
    base = filename.rsplit("/", 1)[-1].lower()
    if any(base.startswith(x) for x in LOCKED_SPLITS) and not test_unlocked(manifest_path):
        raise TestSplitLockedError(f"access to file '{filename}' is locked (test_unlocked is not true)")
