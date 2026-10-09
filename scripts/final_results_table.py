"""Final results table (ONE table, rendered as panels A/B) built ONLY from hash-verified artifacts; every cell records
its source path, sha256 and JSON pointer. No new statistics.
S1 = runs/analysis_main/results.json (frozen PRIMARY analysis)
S2 = runs/post_freeze/audit_adjusted_v2.json (post-freeze addition v2; SECONDARY SENSITIVITY ONLY)
S3 = runs/test-c{1,2,3}/run_meta.jsonl (session wall time)
S4 = runs/post_hoc/audit_adjusted_corrected_labels.json (POST-HOC sensitivity; corrected labels; not of record)"""
import hashlib
import json
from _common import ROOT, write_json

h = lambda p: hashlib.sha256(open(ROOT / p, "rb").read()).hexdigest()
man = json.load(open(ROOT / "configs/freeze_manifest.json"))
S1, S2 = "runs/analysis_main/results.json", "runs/post_freeze/audit_adjusted_v2.json"
assert h(S1) == man["analysis_run"]["outputs_sha256"][S1]
assert h(S2) == next(x for x in man["post_freeze_additions"] if x["name"] == "audit_adjusted_v2")["outputs_sha256"][S2]
J = {S1: json.load(open(ROOT / S1)), S2: json.load(open(ROOT / S2))}
S3 = [f"runs/test-{c}/run_meta.jsonl" for c in ("c1", "c2", "c3")]
S4 = "runs/post_hoc/audit_adjusted_corrected_labels.json"
assert h(S4) == next(x for x in man["post_freeze_additions"] if x["name"] == "audit_adjusted_corrected_labels_POST_HOC")["outputs_sha256"][S4]
J[S4] = json.load(open(ROOT / S4))
SHA = {p: h(p) for p in [S1, S2, S4] + S3}


def get(src, ptr):
    v = J[src]
    for k in ptr.strip("/").split("/"):
        v = v[int(k)] if isinstance(v, list) else v[k]
    return v


def cell(src, ptr, fmt):
    v = get(src, ptr)
    return dict(display=fmt(v), value=v, source=src, sha256=SHA[src], pointer=ptr)


pct = lambda v: f"{100*v:.2f}%"
pts = lambda v: f"{100*v:+.2f}"
ci = lambda v: f"[{100*v[0]:.2f}, {100*v[1]:.2f}]"
num = lambda v: f"{v:,.0f}"
sec = lambda v: f"{v:,.1f}"
g3 = lambda v: f"{v:.3g}"
ARMS = ("E0", "E1", "E3", "E2", "E4")
D = "/descriptives"
rows = {}
for a in ARMS:
    c = {}
    c["N"] = cell(S1, f"{D}/n_items", num)
    if a in ("E0", "E1", "E3"):
        c["invalid"] = cell(S1, f"{D}/invalid/{a}", num); c["abstain"] = cell(S1, f"{D}/abstain/{a}", num)
    c["EM"] = cell(S1, f"{D}/em_accuracy/{a}", pct)
    c["F1"] = cell(S1, f"{D}/f1_mean/{a}", lambda v: f"{v:.4f}")
    c["audit_adjusted_SECONDARY"] = cell(S2, f"/audit_adjusted/{a}/point", pct)
    c["audit_adjusted_ci95"] = cell(S2, f"/audit_adjusted/{a}/ci95", ci)
    if a in ("E1", "E3", "E2", "E4"):
        c[f"EM_diff_vs_E0"] = cell(S1, f"/bootstrap/em/{a}-E0/diff", pts)
        c[f"EM_diff_vs_E0_ci95"] = cell(S1, f"/bootstrap/em/{a}-E0/ci95", ci)
    if a in ("E2", "E4"):
        for f in ("n10", "n01", "p", "p_holm"):
            c[f"mcnemar_vs_E0_{f}"] = cell(S1, f"/secondary_holm/{a}_vs_E0/{f}", num if f in ("n10", "n01") else g3)
        c["gate_k_flagged_valid"] = cell(S1, f"{D}/gate/k_flagged_valid", num)
        c["gate_n_valid"] = cell(S1, f"{D}/gate/n_valid", num)
        c["gate_nominal_rate"] = cell(S1, f"{D}/gate/nominal_rate", pct)
        c["gate_overall_rate"] = cell(S1, f"{D}/gate/overall_mitigation_rate", pct)
        rg = f"/secondary_holm/{a}_vs_random_gate"
        c["rg_null_mean"] = cell(S1, f"{rg}/null_mean", pct); c["rg_null_sd"] = cell(S1, f"{rg}/null_sd", lambda v: f"{100*v:.2f}")
        c["rg_percentile_observed"] = cell(S1, f"{rg}/percentile_of_observed", lambda v: f"{v:.1f}")
        c["rg_p"] = cell(S1, f"{rg}/p", g3); c["rg_p_holm"] = cell(S1, f"{rg}/p_holm", g3)
        c["EM_diff_vs_RG"] = cell(S1, f"/bootstrap/em/{a}-RG/diff", pts); c["EM_diff_vs_RG_ci95"] = cell(S1, f"/bootstrap/em/{a}-RG/ci95", ci)
    if a == "E3":
        c["recall5"] = cell(S1, f"{D}/bm25_answer_string_recall5", pct)
    t = f"/cost_pipeline/{a}/total"
    c["calls"] = cell(S1, f"{t}/calls", num); c["gen_tokens"] = cell(S1, f"{t}/gen_tokens", num)
    c["input_tokens"] = cell(S1, f"{t}/in_tokens", num); c["passages"] = cell(S1, f"{t}/passages", num)
    c["retrieval_calls"] = dict(cell(S1, f"{t}/passages", lambda v: f"{v/5:,.0f}"), derivation="passages / k (k=5)")
    c["gpu_s"] = cell(S1, f"{t}/gpu_s", sec)
    rows[a] = c
primary = {k: cell(S1, f"/primary/{k}", (num if k in ("n10", "n01") else g3 if k == "p" else pts)) for k in ("n10", "n01", "p", "diff")}
primary["ci95"] = cell(S1, "/bootstrap/em/E2-E4/ci95", ci)
adj_diff = dict(point=cell(S2, "/audit_adjusted_E2_minus_E4/point", pts), ci95=cell(S2, "/audit_adjusted_E2_minus_E4/ci95", ci))
wall = [dict(display=sec(json.loads(open(ROOT / p).read().splitlines()[-1])["wall_s"]), source=p, sha256=SHA[p], pointer="last line /wall_s") for p in S3]
wall_total = sum(float(x["display"].replace(",", "")) for x in wall)

dsp = lambda a, k: rows[a][k]["display"] if k in rows[a] else "—"
L = ["# Final results — test (validation split), N = 3,000", "",
     "**PRIMARY = frozen EM result [S1].** Audit-adjusted accuracy = SECONDARY SENSITIVITY ANALYSIS only [S2].", "",
     f"Primary E2 vs E4 [S1 /primary, /bootstrap/em/E2-E4]: n10 = {primary['n10']['display']}, n01 = {primary['n01']['display']}, "
     f"exact McNemar p = {primary['p']['display']}; E2 − E4 = {primary['diff']['display']} pts, bootstrap 95% CI {primary['ci95']['display']}.", "",
     "### Panel A — accuracy and tests", "",
     "| Arm | N | Invalid / abstain | EM | F1 | Audit-adjusted (SECONDARY) [95% CI] | EM − E0 [95% CI] | McNemar vs E0: n10 / n01 / p / p Holm | Gate: k / valid; nominal; overall | Random gate: null mean (SD); obs. pct; p / p Holm; EM − RG [95% CI] | Recall@5 |",
     "|---|---|---|---|---|---|---|---|---|---|---|"]
for a in ARMS:
    inv = f"{dsp(a,'invalid')} / {dsp(a,'abstain')}" if "invalid" in rows[a] else "via E0/E1/E3"
    dif = f"{dsp(a,'EM_diff_vs_E0')} {dsp(a,'EM_diff_vs_E0_ci95')}" if "EM_diff_vs_E0" in rows[a] else "reference"
    mc = (f"{dsp(a,'mcnemar_vs_E0_n10')} / {dsp(a,'mcnemar_vs_E0_n01')} / {dsp(a,'mcnemar_vs_E0_p')} / {dsp(a,'mcnemar_vs_E0_p_holm')}"
          if a in ("E2", "E4") else ("not tested (descriptive)" if a != "E0" else "—"))
    gate = (f"{dsp(a,'gate_k_flagged_valid')} / {dsp(a,'gate_n_valid')}; {dsp(a,'gate_nominal_rate')}; {dsp(a,'gate_overall_rate')}" if a in ("E2", "E4") else "—")
    rg = (f"{dsp(a,'rg_null_mean')} ({dsp(a,'rg_null_sd')}); {dsp(a,'rg_percentile_observed')}; {dsp(a,'rg_p')} / {dsp(a,'rg_p_holm')}; "
          f"{dsp(a,'EM_diff_vs_RG')} {dsp(a,'EM_diff_vs_RG_ci95')}" if a in ("E2", "E4") else "—")
    rec = dsp(a, "recall5") if a == "E3" else ("not computed in frozen output" if a == "E4" else "—")
    L.append(f"| {a} | {dsp(a,'N')} | {inv} | {dsp(a,'EM')} | {dsp(a,'F1')} | {dsp(a,'audit_adjusted_SECONDARY')} {dsp(a,'audit_adjusted_ci95')} | {dif} | {mc} | {gate} | {rg} | {rec} |")
L += ["", "### Panel B — cost (full pipeline totals)", "",
      "| Arm | Model calls | Generated tokens | Input tokens | Retrieval calls | Passages | GPU-s (model-call wall time, 1×T4) |", "|---|---|---|---|---|---|---|"]
for a in ARMS:
    L.append(f"| {a} | {dsp(a,'calls')} | {dsp(a,'gen_tokens')} | {dsp(a,'input_tokens')} | {dsp(a,'retrieval_calls')} | {dsp(a,'passages')} | {dsp(a,'gpu_s')} |")
ph = {x: dict(point=cell(S4, f"/audit_adjusted/{x}/point", pct), ci95=cell(S4, f"/audit_adjusted/{x}/ci95", ci)) for x in ARMS}
ph["E2-E4"] = dict(point=cell(S4, "/audit_adjusted_E2_minus_E4/point", pts), ci95=cell(S4, "/audit_adjusted_E2_minus_E4/ci95", ci))
ph_fn = {x: cell(S4, f"/audit_fn/{x}/correct", num) for x in ("E0", "E1", "E3")}
L += ["", f"Audit-adjusted E2 − E4 (SECONDARY) [S2 /audit_adjusted_E2_minus_E4]: {adj_diff['point']['display']} pts, 95% CI {adj_diff['ci95']['display']}.",
      "Audit numbers above are the AUDIT OF RECORD: AI-assisted blind audit, labels sha 9078bfca… (hash recorded before key read).",
      ("POST-HOC SENSITIVITY ANALYSIS [S4] (NOT the audit of record): after unblinding, the PI changed a092 and a143 CORRECT→INCORRECT "
       f"(labels sha dac7400c…; provenance_corrections PC6). Audit CORRECT counts E0/E1/E3 = {ph_fn['E0']['display']}/{ph_fn['E1']['display']}/{ph_fn['E3']['display']} of 50. "
       "Audit-adjusted: " + "; ".join(f"{x} {ph[x]['point']['display']} {ph[x]['ci95']['display']}" for x in ARMS) +
       f"; E2 − E4 {ph['E2-E4']['point']['display']} pts {ph['E2-E4']['ci95']['display']}. Same bootstrap and seed as S2."),
      f"Session wall time [S3], all arms incl. BM25/model setup: c1 {wall[0]['display']} s, c2 {wall[1]['display']} s, c3 {wall[2]['display']} s; total {wall_total:,.1f} s. "
      "Not separable by arm; per-arm time is the GPU-s column.",
      "Pipelines: E1 = E0 + E1 (E1 consumes A1); E2 = E0 + E1 on flagged items; E4 = E0 + E3 on flagged items. Retrieval calls = passages / 5.",
      "Audit-adjusted CI: two-level bootstrap, B = 10,000, PCG64(20261008): per replicate, each arm's 50 audit labels resampled (FN rates for E0/E1/E3) "
      "and the 3,000 test items resampled with one shared index draw; percentile 95% intervals.",
      "Transferability assumption: FN rates were estimated on audited EM-failed outputs of each arm overall, while E2/E4 use E1/E3 only on the gated "
      "(flagged) subset; the sensitivity result depends on these rates transferring to that subset. EM false positives are not estimated.",
      "EM differences: paired bootstrap 95% CIs from the frozen analysis (B = 10,000, PCG64(20261006)). Holm family: E2 vs E0, E4 vs E0, E2 vs RG, E4 vs RG. "
      "Random-gate p = 0.002 is the minimum attainable with 1,000 seeds.", "",
      "**Sources** (every cell in reports/final_results_table.json records path + sha256 + JSON pointer):",
      f"- S1 `{S1}` sha256 `{SHA[S1]}` (frozen primary analysis)",
      f"- S2 `{S2}` sha256 `{SHA[S2]}` (post-freeze addition v2, secondary; audit of record)",
      f"- S4 `{S4}` sha256 `{SHA[S4]}` (POST-HOC sensitivity, corrected labels; not of record)"] + \
     [f"- S3 `{p}` sha256 `{SHA[p]}`" for p in S3]
(ROOT / "reports/final_results_table.md").write_text("\n".join(L) + "\n", encoding="utf-8")
write_json(ROOT / "reports/final_results_table.json", dict(primary=primary, rows=rows, audit_adjusted_E2_minus_E4=adj_diff,
                                                          post_hoc_corrected_labels=dict(audit_adjusted=ph, audit_correct_counts=ph_fn),
                                                          session_wall_s=wall, session_wall_s_total=wall_total, sources=SHA))
print("\n".join(L))
