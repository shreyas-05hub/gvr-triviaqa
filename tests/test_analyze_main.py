"""Statistics used by scripts/analyze_main.py (exact McNemar, Holm, random-gate expectation)."""
import sys
import math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_main as A  # noqa: E402
from gvr.random_gate import random_gate


def test_mcnemar_exact():
    a = np.array([True] * 7 + [False] * 1 + [True] * 10)
    b = np.array([False] * 7 + [True] * 1 + [True] * 10)
    n10, n01, p = A.mcnemar_exact(a, b)
    assert (n10, n01) == (7, 1)
    assert abs(p - 2 * (math.comb(8, 0) + math.comb(8, 1)) / 2 ** 8) < 1e-12      # 0.0703125
    assert A.mcnemar_exact(np.array([True, False]), np.array([True, False]))[2] == 1.0
    assert A.mcnemar_exact(np.array([True]), np.array([False]))[2] == 1.0          # n=1 -> p capped at 1


def test_holm():
    adj = A.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    assert adj["a"] == 0.04 and adj["c"] == 0.09 and adj["b"] == 0.09 and adj["d"] == 0.5


def test_random_gate_expectation_matches_simulation():
    rng = np.random.default_rng(1)
    ids = [f"i{j}" for j in range(200)]
    inv = set(ids[:5])
    e0 = [dict(item_id=i, invalid=i in inv, u=0.0) for i in ids]
    c0 = {i: bool(rng.random() < 0.5) for i in ids}; c1 = {i: bool(rng.random() < 0.7) for i in ids}
    k = 60
    accs, _ = random_gate(e0, c0, c1, k, range(1000))
    valid = np.array([i not in inv for i in ids]); pm = np.where(valid, k / valid.sum(), 1.0)
    expect = float(np.mean(pm * np.array([c1[i] for i in ids]) + (1 - pm) * np.array([c0[i] for i in ids])))
    assert abs(accs.mean() - expect) < 0.003


def test_analysis_constants_frozen():
    assert (A.BOOT_B, A.BOOT_SEED, A.AUDIT_SEED, A.AUDIT_N, len(A.RG_SEEDS)) == (10_000, 20261006, 20261005, 50, 1000)
