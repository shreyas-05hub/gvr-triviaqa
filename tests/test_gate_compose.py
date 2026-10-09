import json
import numpy as np
import pytest
from gvr.gate import fit_gate, fit_tau, apply_gate, rates, save_gate
from gvr.compose import compose
from gvr.random_gate import random_gate, summarize

GC = {"quantile": 0.60, "quantile_method": "higher", "comparison": "strict_gt"}


def mk(n=20, invalid=(3, 7), seed=0):
    rng = np.random.default_rng(seed)
    return [dict(item_id=f"q{i}", invalid=i in invalid, u=None if i in invalid else float(rng.random()),
                 answer=f"a{i}", arm="E0", cost=dict(n_calls=1, input_tokens=10, output_tokens=2, wall_s=0.1))
            for i in range(n)]


def test_tau_higher_and_strict():
    u = [0.1, 0.2, 0.3, 0.4, 0.5]
    tau = fit_tau(u, 0.6, "higher")
    assert tau == np.quantile(u, 0.6, method="higher") == 0.4
    d = apply_gate([dict(item_id=i, invalid=False, u=x) for i, x in enumerate(u)], tau)
    assert [d[i]["flag"] for i in range(5)] == [False, False, False, False, True]  # 0.4 not flagged


def test_ties_no_tiebreak():
    items = [dict(item_id=i, invalid=False, u=0.5) for i in range(5)]
    tau, dec, r = fit_gate(items, GC)
    assert r["k_flagged_valid"] == 0  # all tied at tau -> none flagged (strict), no tie-breaking


def test_invalid_forced_and_rates():
    items = mk()
    tau, dec, r = fit_gate(items, GC)
    assert dec["q3"] == dict(flag=False, forced=True, mitigate=True)
    assert r["n_valid"] == 18 and r["n_invalid"] == 2
    assert r["nominal_rate"] == r["k_flagged_valid"] / 18
    assert r["overall_mitigation_rate"] == (r["k_flagged_valid"] + 2) / 20


def test_tau_reproducible_from_saved(tmp_path):
    items = mk(seed=5)
    tau, dec, r = fit_gate(items, GC)
    save_gate(tmp_path / "g.json", tau, r, GC, {"u_values": {i["item_id"]: i["u"] for i in items}}, True)
    g = json.load(open(tmp_path / "g.json"))
    re_items = [dict(item_id=k, invalid=v is None, u=v) for k, v in g["u_values"].items()]
    tau2, dec2, r2 = fit_gate(re_items, {"quantile": g["quantile"], "quantile_method": g["quantile_method"]})
    assert tau2 == g["tau"] == tau and r2 == g["realized"] == r and dec2 == dec
    assert g["provisional"] is True and "DISCARDED" in g["status"]


def test_compose_unflagged_equal_e0():
    e0 = {x["item_id"]: dict(x, final=True) for x in mk()}
    e1 = {k: dict(v, arm="E1", answer="rev_" + k, cost=dict(n_calls=5, input_tokens=100, output_tokens=20, wall_s=1.0))
          for k, v in e0.items()}
    tau, dec, r = fit_gate(list(e0.values()), GC)
    e2 = compose(e0, e1, dec, "E2")
    for k, d in dec.items():
        if d["mitigate"]:
            assert e2[k]["answer"] == "rev_" + k and e2[k]["cost"]["n_calls"] == 6
        else:
            assert e2[k]["answer"] == e0[k]["answer"] and e2[k]["cost"]["n_calls"] == 1
    assert e2["q3"]["mitigated"] and e2["q3"]["forced"]


def test_compose_assert_catches_mismatch():
    e0 = {"a": dict(item_id="a", invalid=False, u=0.1, answer="x", arm="E0", cost={})}
    dec = {"a": dict(flag=False, forced=False, mitigate=False)}
    out = compose(e0, {"a": dict(e0["a"], arm="E1")}, dec, "E2")
    assert out["a"]["answer"] == "x"


def test_random_gate_rate_and_determinism():
    items = mk(n=60, invalid=(1, 2, 3))
    tau, dec, r = fit_gate(items, GC)
    k = r["k_flagged_valid"]
    c0 = {i["item_id"]: False for i in items}
    c1 = {i["item_id"]: True for i in items}
    accs, rr = random_gate(items, c0, c1, k, range(1000))
    assert all(x == r["nominal_rate"] for x in rr)
    assert np.allclose(accs, (k + 3) / 60)  # every mitigated item correct
    accs2, _ = random_gate(items, c0, {i: (int(i[1:]) % 2 == 0) for i in c1}, k, range(1000))
    accs3, _ = random_gate(items, c0, {i: (int(i[1:]) % 2 == 0) for i in c1}, k, range(1000))
    assert np.array_equal(accs2, accs3) and accs2.std() > 0
    s = summarize(accs2)
    assert s["n_seeds"] == 1000 and s["p2_5"] <= s["mean"] <= s["p97_5"]
