"""E2/E4 composition from logged outputs (no generation)."""


def compose(e0, mit, decisions, mit_arm_name):
    """e0, mit: {item_id: item dict incl. labels+cost}. Returns {item_id: composed item}."""
    out = {}
    for iid, base in e0.items():
        d = decisions[iid]
        if d["mitigate"]:
            src = mit[iid]
            cost = add_costs(base["cost"], src["cost"])
        else:
            src = base
            cost = dict(base["cost"])
        it = {k: v for k, v in src.items() if k not in ("arm", "cost")}
        it.update(arm=mit_arm_name, source_arm=src["arm"], mitigated=d["mitigate"],
                  forced=d["forced"], cost=cost)
        out[iid] = it
    for iid, d in decisions.items():
        if not d["mitigate"]:
            assert out[iid]["answer"] == e0[iid]["answer"], "unflagged item must equal E0 exactly"
    return out


def add_costs(a, b):
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}
