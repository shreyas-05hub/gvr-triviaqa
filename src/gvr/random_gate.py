"""Random-gate control: same k valid items mitigated, chosen uniformly; invalid always mitigated."""
import numpy as np


def random_gate(e0_items, correct_e0, correct_mit, k, seeds):
    """e0_items: [{item_id, invalid}], correct_*: {item_id: bool final}. Returns per-seed accuracies."""
    ids = [it["item_id"] for it in e0_items]
    valid = sorted(it["item_id"] for it in e0_items if not it["invalid"])
    invalid = {it["item_id"] for it in e0_items if it["invalid"]}
    accs, rates = [], []
    for s in seeds:
        rng = np.random.Generator(np.random.PCG64(s))
        chosen = set(rng.choice(valid, size=k, replace=False).tolist()) if k else set()
        mit = chosen | invalid
        accs.append(float(np.mean([correct_mit[i] if i in mit else correct_e0[i] for i in ids])))
        rates.append(len(chosen) / len(valid) if valid else None)
    return np.array(accs), rates


def summarize(accs):
    return dict(n_seeds=len(accs), mean=float(np.mean(accs)), sd=float(np.std(accs, ddof=1)),
                p2_5=float(np.percentile(accs, 2.5)), p97_5=float(np.percentile(accs, 97.5)))
