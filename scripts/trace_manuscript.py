"""READ-ONLY numerical trace of a manuscript text export against the frozen/hashed artifacts.

Usage: scripts/trace_manuscript.py --manuscript reports/manuscript_v2.md [--out reports/verification/trace_manuscript_v2]

For every number in the manuscript: candidate artifact values (all numeric leaves of S1, S2, S4, gate.json,
freeze_manifest, config, index_record, run_meta, judge-decision files), matched on the value as printed:
  exact               artifact value (x1 or x100) equals the printed value
  rounding-consistent |artifact*scale - printed| <= half a unit in the printed last place
  NO SOURCE           no artifact leaf (or registered derived value) matches
Explicit checks (CHECKS) compare named quantities against artifact-derived expectations and can yield MISMATCH.
Derived ratios are computed from artifact values, never from manuscript prose."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "S1": "runs/analysis_main/results.json",
    "S2": "runs/post_freeze/audit_adjusted_v2.json",
    "S4": "runs/post_hoc/audit_adjusted_corrected_labels.json",
    "gate": "configs/gate.json",
    "manifest": "configs/freeze_manifest.json",
    "index_record": "configs/index_record.json",
    "judge_decision_v1": "runs/pilot-01/judge_decision.json",
    "judge_decision_v2": "runs/judgedev/judge_decision_v2.json",
    "calib_summary": "runs/calib-01/calibration_summary.json",
}
JSONL = {f"run_meta_{c}": f"runs/test-{c}/run_meta.jsonl" for c in ("c1", "c2", "c3")}


def sha(p):
    return hashlib.sha256(open(ROOT / p, "rb").read()).hexdigest()


def leaves(obj, ptr=""):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield ptr or "/", float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{ptr}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{ptr}/{i}")


def load_all():
    data, idx = {}, []
    for key, p in SOURCES.items():
        if (ROOT / p).exists():
            data[key] = json.load(open(ROOT / p))
    for key, p in JSONL.items():
        if (ROOT / p).exists():
            data[key] = [json.loads(l) for l in open(ROOT / p) if l.strip()]
    import yaml
    data["config"] = yaml.safe_load(open(ROOT / "configs/config.yaml"))
    paths = dict(SOURCES, **JSONL, config="configs/config.yaml")
    for key, obj in data.items():
        for ptr, v in leaves(obj):
            idx.append(dict(source=key, path=paths[key], pointer=ptr, value=v))
    return data, idx, paths


def derived(d):
    """Named derived quantities computed ONLY from artifact values."""
    S1 = d["S1"]; em = S1["descriptives"]["em_accuracy"]; cp = S1["cost_pipeline"]; b = S1["bootstrap"]["em"]
    t = lambda a, m: cp[a]["total"][m]
    out = {
        "E4_gain_share_of_E3_gain (E4-E0)/(E3-E0)": (em["E4"] - em["E0"]) / (em["E3"] - em["E0"]),
        "E4_gain_share_bootstrap_diffs b[E4-E0]/b[E3-E0]": b["E4-E0"]["diff"] / b["E3-E0"]["diff"],
        "E3_minus_E4_pts": 100 * (em["E3"] - em["E4"]),
        "E1_minus_E2_pts": 100 * (em["E1"] - em["E2"]),
        "E2_over_E4_gpu_s": t("E2", "gpu_s") / t("E4", "gpu_s"),
        "E2_over_E4_gen_tokens": t("E2", "gen_tokens") / t("E4", "gen_tokens"),
        "E2_over_E4_total_tokens": (t("E2", "gen_tokens") + t("E2", "in_tokens")) / (t("E4", "gen_tokens") + t("E4", "in_tokens")),
        "E2_over_E4_input_tokens": t("E2", "in_tokens") / t("E4", "in_tokens"),
        "E2_over_E4_calls": t("E2", "calls") / t("E4", "calls"),
        "E1pipeline_over_E0_gpu_s": t("E1", "gpu_s") / t("E0", "gpu_s"),
        "E1armonly_over_E0_gpu_s": S1["cost_arm_only"]["E1"]["gpu_s"] / t("E0", "gpu_s"),
        "overall_mitigation_rate": S1["descriptives"]["gate"]["overall_mitigation_rate"],
        "nominal_flag_rate": S1["descriptives"]["gate"]["nominal_rate"],
    }
    for a in cp:
        for m in ("gpu_s", "gen_tokens", "in_tokens", "calls", "passages"):
            out[f"per_item_{m}_{a}"] = cp[a]["per_item"][m]
        out[f"retrieval_calls_{a}"] = cp[a]["total"]["passages"] / 5
    return out


NUM = re.compile(r"(?<![\w.])([−\-–]?\d[\d,]*(?:\.\d+)?(?:\s?[×x]\s?10\^?[−\-]?\d+|e[−\-+]?\d+)?)(\s?%|\s?×|\s?pts?|\s?points)?", re.I)


def parse_num(s):
    s0 = s.replace("−", "-").replace("–", "-").replace(",", "").replace(" ", "")
    m = re.match(r"^(-?\d+(?:\.\d+)?)(?:[×x]10\^?(-?\d+)|e([-+]?\d+))?$", s0, re.I)
    if not m:
        return None, None
    base = m.group(1); exp = m.group(2) or m.group(3)
    val = float(base) * (10 ** int(exp) if exp else 1)
    dec = len(base.split(".")[1]) if "." in base else 0
    half = 0.5 * 10 ** (-dec) * (10 ** int(exp) if exp else 1)
    return val, half


def match(val, half, unit, idx, der):
    hits = []
    scales = (100.0,) if unit and unit.strip() in ("%",) else (1.0, 100.0)
    for e in idx:
        for sc in scales:
            v = e["value"] * sc
            if v == val:
                hits.append(dict(e, scale=sc, kind="exact")); break
            if abs(v - val) <= half + 1e-12 * max(1, abs(v)):
                hits.append(dict(e, scale=sc, kind="rounding-consistent")); break
    for k, v in der.items():
        for sc in ((100.0,) if unit and unit.strip() == "%" else (1.0, 100.0)):
            if abs(v * sc - val) <= half + 1e-12:
                hits.append(dict(source="derived", path="(computed from artifacts)", pointer=k, value=v, scale=sc, kind="rounding-consistent (derived)"))
                break
    return hits


CHECKS = [  # (label, regex near-context, expected-from-artifacts lambda, tolerance-kind)
    ("primary E2-E4 diff", r"E2\s*[−\-–]\s*E4", lambda d: 100 * d["S1"]["primary"]["diff"]),
    ("primary CI low", r"E2\s*[−\-–]\s*E4", lambda d: 100 * d["S1"]["bootstrap"]["em"]["E2-E4"]["ci95"][0]),
    ("primary CI high", r"E2\s*[−\-–]\s*E4", lambda d: 100 * d["S1"]["bootstrap"]["em"]["E2-E4"]["ci95"][1]),
]


def explicit_checks(text, d, der):
    S1 = d["S1"]; sec = S1["secondary_holm"]; g = S1["descriptives"]["gate"]
    want = {
        "N 3,000": (3000, g["N"]), "valid 2,998": (2998, g["n_valid"]), "k 1,180": (1180, g["k_flagged_valid"]),
        "audit 50/arm": (50, min(v["n"] for v in d["S2"]["audit_fn"].values())),
        "93% = (E4-E0)/(E3-E0)": (93, 100 * der["E4_gain_share_of_E3_gain (E4-E0)/(E3-E0)"]),
        "1.16 pts": (1.16, der["E3_minus_E4_pts"]),
        "2.7x time (E2/E4 GPU-s)": (2.7, der["E2_over_E4_gpu_s"]),
        "5.6x tokens (E2/E4 generated)": (5.6, der["E2_over_E4_gen_tokens"]),
        "2.1x calls (E2/E4)": (2.1, der["E2_over_E4_calls"]),
        "~14x (E1 vs E0 time, pipeline)": (14, der["E1pipeline_over_E0_gpu_s"]),
        "39.4% (overall mitigation)": (39.4, 100 * der["overall_mitigation_rate"]),
        "primary p 1.24e-76": (1.24e-76, S1["primary"]["p"]),
        "primary n10 55 / n01 444": ((55, 444), (S1["primary"]["n10"], S1["primary"]["n01"])),
        "E2-E4 CI negative": ("negative", "negative" if S1["bootstrap"]["em"]["E2-E4"]["ci95"][1] < 0 else "NOT negative"),
    }
    for k in sec:
        want[f"{k} p"] = (None, sec[k]["p"]); want[f"{k} p_holm"] = (None, sec[k]["p_holm"])
    rows = []
    for label, (claimed, art) in want.items():
        if isinstance(claimed, (int, float)) and isinstance(art, (int, float)) and claimed is not None:
            dec = len(str(claimed).split(".")[1]) if "." in str(claimed) else 0
            if art == claimed:
                status = "exact"
            else:
                ok = (f"{art:.3g}" == f"{claimed:.3g}") if claimed < 1e-6 else abs(art - claimed) <= 0.5 * 10 ** (-dec) + 1e-12
                status = "rounding-consistent" if ok else "MISMATCH"
        elif claimed is None:
            status = "artifact value (for reference)"
        else:
            status = "exact" if claimed == art else "MISMATCH"
        present = None
        if isinstance(claimed, (int, float)):
            pat = {93: r"\b93\s?%", 1.16: r"\b1\.16\b", 2.7: r"\b2\.7\s?[×x]", 5.6: r"\b5\.6\s?[×x]", 2.1: r"\b2\.1\s?[×x]", 14: r"~?\s?14\s?[×x]", 39.4: r"\b39\.4\s?%",
                   3000: r"\b3,?000\b", 2998: r"\b2,?998\b", 1180: r"\b1,?180\b", 50: r"\b50\b"}.get(claimed)
            present = bool(re.search(pat, text)) if pat else None
        rows.append(dict(check=label, claimed=claimed, artifact_value=art, status=status, found_in_manuscript=present))
    # CI sign convention near "E2 − E4"
    for m in re.finditer(r"E2\s*[−\-–]\s*E4[^\n]{0,120}", text):
        seg = re.split(r"\bp\s*[=<]|McNemar|;", m.group(0)[len("E2-E4"):])[0]           # stop before p-values / next clause
        nums = [parse_num(x)[0] for x, u in NUM.findall(seg) if not (u and u.strip() == "%")]  # drop "95%"-style numbers
        nums = [x for x in nums if x is not None and x not in (2.0, 4.0)]
        neg = [x < 0 for x in nums]
        rows.append(dict(check="CI sign near 'E2−E4'", claimed=m.group(0)[:100], artifact_value="all negative",
                         status="exact" if nums and all(neg) else ("MISMATCH" if nums else "NO NUMBERS"), found_in_manuscript=True))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    text = open(ROOT / a.manuscript, encoding="utf-8").read()
    d, idx, paths = load_all()
    der = derived(d)
    rows = []
    for ln, line in enumerate(text.splitlines(), 1):
        for m in NUM.finditer(line):
            val, half = parse_num(m.group(1))
            if val is None:
                continue
            hits = match(val, half, m.group(2), idx, der)
            kinds = {h["kind"] for h in hits}
            status = "exact" if "exact" in kinds else ("rounding-consistent" if hits else "NO SOURCE")
            ctx = line[max(0, m.start() - 40): m.end() + 40]
            rows.append(dict(line=ln, text=ctx.strip(), value=m.group(0).strip(), status=status, n_candidates=len(hits),
                             candidates=[dict(source=h["source"], path=h["path"], pointer=h["pointer"], artifact_value=h["value"], scale=h["scale"], kind=h["kind"]) for h in hits[:5]]))
    checks = explicit_checks(text, d, der)
    out = Path(a.out or f"reports/verification/trace_{Path(a.manuscript).stem}")
    (ROOT / out).parent.mkdir(parents=True, exist_ok=True)
    res = dict(manuscript=a.manuscript, manuscript_sha256=sha(a.manuscript), sources={k: dict(path=p, sha256=sha(p)) for k, p in paths.items() if (ROOT / p).exists()},
               derived=der, explicit_checks=checks, numbers=rows,
               summary={s: sum(r["status"] == s for r in rows) for s in ("exact", "rounding-consistent", "NO SOURCE")},
               note=("candidate matching is value-based: a small integer can match many unrelated leaves; review candidates for "
                     "semantic correctness. MISMATCH is reported only by explicit checks (named quantities)."))
    json.dump(res, open(ROOT / f"{out}.json", "w"), indent=2, ensure_ascii=False)
    L = [f"# Numerical trace: {a.manuscript} (sha {res['manuscript_sha256'][:12]}…)", "", "## Explicit checks", "",
         "| check | claimed | artifact value | status | in manuscript |", "|---|---|---|---|---|"]
    fmt = lambda v: (f"{v:.6g}" if isinstance(v, float) else str(v))
    for c in checks:
        L.append(f"| {c['check']} | {fmt(c['claimed'])} | {fmt(c['artifact_value'])} | {c['status']} | {c['found_in_manuscript']} |")
    L += ["", "## Every number", "", "| line | manuscript text | value | match | source path + pointer (first candidate) | artifact value |", "|---|---|---|---|---|---|"]
    for r in rows:
        c0 = r["candidates"][0] if r["candidates"] else None
        src = f"`{c0['path']}` `{c0['pointer']}` (x{c0['scale']:g}; {r['n_candidates']} cand.)" if c0 else "—"
        L.append(f"| {r['line']} | {r['text'].replace('|', '/')} | {r['value']} | {r['status']} | {src} | {fmt(c0['artifact_value']) if c0 else '—'} |")
    (ROOT / f"{out}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps(dict(summary=res["summary"], explicit=[(c["check"], c["status"]) for c in checks]), indent=1))


if __name__ == "__main__":
    main()
