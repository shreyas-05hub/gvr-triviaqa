"""Uncertainty gate. tau = quantile(u[valid], q, method) ; flag = u > tau (strict). No tie-breaking."""
import json
import numpy as np


def fit_tau(u_valid, q, method):
    return float(np.quantile(np.asarray(u_valid, dtype=np.float64), q, method=method))


def apply_gate(items, tau):
    """items: [{item_id, invalid, u}] -> {item_id: dict(flag, forced, mitigate)}"""
    out = {}
    for it in items:
        if it["invalid"]:
            out[it["item_id"]] = dict(flag=False, forced=True, mitigate=True)
        else:
            f = bool(it["u"] > tau)
            out[it["item_id"]] = dict(flag=f, forced=False, mitigate=f)
    return out


def rates(decisions):
    n = len(decisions)
    nv = sum(not d["forced"] for d in decisions.values())
    fv = sum(d["flag"] for d in decisions.values())
    inv = n - nv
    return dict(N=n, n_valid=nv, n_invalid=inv, k_flagged_valid=fv,
                nominal_rate=fv / nv if nv else None, overall_mitigation_rate=(fv + inv) / n if n else None)


def fit_gate(e0_items, gcfg):
    valid = [it for it in e0_items if not it["invalid"]]
    for it in valid:
        assert it["u"] is not None, f"valid item without u: {it['item_id']}"
    tau = fit_tau([it["u"] for it in valid], gcfg["quantile"], gcfg["quantile_method"])
    dec = apply_gate(e0_items, tau)
    return tau, dec, rates(dec)


def save_gate(path, tau, r, gcfg, provenance, provisional):
    rec = dict(tau=tau, quantile=gcfg["quantile"], quantile_method=gcfg["quantile_method"],
               comparison=gcfg["comparison"], realized=r, provisional=provisional,
               status="PROVISIONAL - fitted on pilot E0 for code exercise; DISCARDED" if provisional
               else "final", **provenance)
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    return rec
