"""Regenerates reports/pilot_report.md from logs/results. Missing measurements print as NOT MEASURED."""
import json
import numpy as np

HEADER = "Engineering pilot; descriptive only; no scientific conclusions."
NM = "NOT MEASURED"


def _f(x, pct=False):
    if x is None:
        return NM
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return f"{100 * x:.1f}%" if pct else f"{x:.4g}"
    return str(x)


def _hist(vals, bins=10):
    if not vals:
        return NM
    c, e = np.histogram(vals, bins=bins)
    w = max(c.max(), 1)
    return "\n".join(f"{e[i]:7.3f}–{e[i + 1]:7.3f} | {'#' * int(round(30 * n / w))} {n}" for i, n in enumerate(c))


def _status(ok):
    return {True: "PASS", False: "FAIL", None: "PENDING"}[ok]


def go_no_go(res, checks, thr):
    s = res["summary"]
    rows = []
    worst_inv = max(s[a]["invalid"] for a in ("E0", "E1", "E3"))
    rows.append(("invalid ≤ {:.0%} per arm".format(thr["max_invalid_rate_per_arm"]),
                 ", ".join(f"{a}={_f(s[a]['invalid'], True)}" for a in ("E0", "E1", "E3")),
                 worst_inv <= thr["max_invalid_rate_per_arm"]))
    rows.append(("unrecovered failures ≤ {}".format(thr["max_unrecovered_failures"]),
                 checks.get("unrecovered"), None if checks.get("unrecovered") is None
                 else checks["unrecovered"] <= thr["max_unrecovered_failures"]))
    rr = checks.get("retried_rate")
    rows.append(("retried ≤ {:.0%}".format(thr["max_retried_rate"]), _f(rr, True),
                 None if rr is None else rr <= thr["max_retried_rate"]))
    rows.append(("u present for all valid E0", checks.get("u_all_valid"), checks.get("u_all_valid")))
    for key, label in [("alignment_pass", "u alignment: logits length, greedy argmax, EOS exclusion (7B NF4)"),
                       ("same_path_repro_pass", "u same-path reproducibility |du|<1e-6, identical tokens (replaces re-score; u_decision_rule case 1)"),
                       ("bm25_ok", "BM25 5 hits/query, k1/b confirmed"),
                       ("tau_reproducible", "τ reproducible"),
                       ("determinism_ok", "determinism (see section)"),
                       ("no_oom_2048", "no OOM at 2048 input tokens")]:
        rows.append((label, checks.get(key), checks.get(key)))
    ju = checks.get("judge_unparsed")
    rows.append(("judge unparsed outputs (scored judge_correct=False; in all denominators)",
                 f"{ju} total; " + ", ".join(f"{a}={s[a].get('judge_unparsed', 0)}" for a in ("E0", "E1", "E3", "E2", "E4"))
                 if ju is not None else None, None if ju is None else ju == 0))
    ja = checks.get("judge_agreement")
    rows.append(("judge vs hand labels ≥ {:.0%}".format(thr["min_judge_agreement"]),
                 _f(ja, True) if ja is not None else "PENDING (user hand labels)",
                 None if ja is None else ja >= thr["min_judge_agreement"]))
    gh, cap = checks.get("gpu_hours_quota"), thr.get("gpu_hour_budget_pilot")
    rows.append((f"pilot GPU-hours ≤ {cap} (Kaggle quota delta, incl. setup and failed attempts)",
                 _f(gh), None if gh is None or cap is None else gh <= cap))
    return rows


def render(res, manifest, checks, thr, timing, projection):
    L = [f"# Pilot report\n\n**{HEADER}**\n",
         "Pilot accuracies below are descriptive engineering checks only and are not evidence that any method "
         "improves accuracy. The gate threshold τ here is PROVISIONAL (fit on pilot E0) and is discarded.\n",
         "## Accepted interpretations\n",
         "- Trailing punctuation = `string.punctuation` (plus whitespace) stripped from the end of the parsed answer.\n"
         "- CoVe plan parser strips leading list markers (`1.`, `1)`, `1:`, `-`, `*`, `•`).\n"
         "- Answer-string recall@5 is computed over passage text excluding the title line; it is not support.\n",
         "## Environment and manifests\n", "```json", json.dumps(manifest, indent=2), "```\n",
         "## Go / no-go\n", "| criterion | measured | status |", "|---|---|---|"]
    for c, m, ok in go_no_go(res, checks, thr):
        L.append(f"| {c} | {m if m is not None else NM} | {_status(ok)} |")
    L += ["\n## Accuracy per arm (descriptive)\n",
          "| arm | N | EM | final (EM∨judge) | F1 | invalid | abstain | judged | judge unparsed | hallucination proxy |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for a in ("E0", "E1", "E3", "E2", "E4"):
        s = res["summary"][a]
        L.append(f"| {a} | {s['N']} | {_f(s['em'], 1)} | {_f(s['final'], 1)} | {_f(s['f1'])} | "
                 f"{_f(s['invalid'], 1)} | {_f(s['abstain'], 1)} | {s['judged']} | {s.get('judge_unparsed', 0)} | "
                 f"{_f(s['hallucination_proxy'], 1)} |")
    L += ["\n## Flips vs E0\n", "| arm | right→wrong | wrong→right |", "|---|---|---|"]
    for a, f in res["flips"].items():
        L.append(f"| {a} | {f['right_to_wrong']} | {f['wrong_to_right']} |")
    g = res["gate_rates"]
    L += ["\n## Gate (PROVISIONAL, discarded)\n",
          f"τ = {_f(res['tau'])} (quantile 0.60, method=higher, strict >). Valid {g['n_valid']}/{g['N']}, "
          f"flagged valid k={g['k_flagged_valid']}, nominal rate {_f(g['nominal_rate'], 1)}, "
          f"overall mitigation rate {_f(g['overall_mitigation_rate'], 1)}.\n",
          "### Random gate (1000 seeds, same k)\n", "| arm | mean | SD | 2.5% | 97.5% |", "|---|---|---|---|---|"]
    for a, r in res["random_gate"].items():
        L.append(f"| {a} | {_f(r['mean'], 1)} | {_f(r['sd'])} | {_f(r['p2_5'], 1)} | {_f(r['p97_5'], 1)} |")
    L += ["\n## Retrieval\n", f"Answer-string recall@5 (not support): {_f(res['recall5'], 1)}. "
          f"E3 prompts truncated: {res['e3_truncated']}.\n",
          "## u histogram (E0 valid)\n", "```", _hist(res["u_valid"]), "```\n",
          "## CoVe diagnostics\n",
          f"Verification questions: {res['paraphrase']['n']}; paraphrase-flagged (Jaccard ≥ 0.7): "
          f"{res['paraphrase']['flagged']}; sv_plan_invalid items: {res['sv_plan_invalid']}. "
          "CoVe-style (factored); same-model verification is not independent evidence.\n",
          "## Cost\n", "| arm | model calls | retrievals | input tok | output tok | wall s |", "|---|---|---|---|---|---|"]
    for a in ("E0", "E1", "E3", "E2", "E4"):
        s = res["summary"][a]
        L.append(f"| {a} | {s['cost_calls']} | {s['cost_retrievals']} | {s['cost_in_tokens']} | "
                 f"{s['cost_out_tokens']} | {_f(s['cost_wall_s'])} |")
    L += ["\n## Cost decomposition (base / detector / mitigation)\n",
          "Detector = uncertainty u from the E0 logits already computed: 0 extra model calls (its compute is inside "
          "the E0 call time). Mitigation = E1/E3 calls on mitigated items (always-on for E1/E3).\n",
          "| arm | base calls | detector calls | mitigation calls | retrievals | input tok | output tok | wall s |",
          "|---|---|---|---|---|---|---|---|"]
    b = res["summary"]["E0"]
    for a_ in ("E0", "E1", "E3", "E2", "E4"):
        s_ = res["summary"][a_]
        if a_ == "E0":
            base, mit = s_["cost_calls"], 0
        elif a_ in ("E1", "E3"):
            base, mit = 0, s_["cost_calls"]
        else:
            base, mit = b["cost_calls"], s_["cost_calls"] - b["cost_calls"]
        L.append(f"| {a_} | {base} | 0 | {mit} | {s_['cost_retrievals']} | {s_['cost_in_tokens']} | "
                 f"{s_['cost_out_tokens']} | {_f(s_['cost_wall_s'])} |")
    L.append("\nE1/E3 rows are the arms on their own (always-on mitigation, consuming the logged E0 answer for E1); "
             "a full always-on pipeline additionally pays the E0 base row.\n")
    jd = checks.get("judge_decision")
    if jd:
        L += ["## Judge decision (pre-registered rule, sha256 " + jd["rule_sha256"][:12] + "…)\n",
              f"**Case {jd['case']}**: agreement {_f(jd['agreement'], True)} on {jd['n_compared']} labelled rows "
              f"(CORRECT {jd['label_counts']['CORRECT']}, INCORRECT {jd['label_counts']['INCORRECT']}, UNSURE "
              f"{jd['label_counts']['UNSURE']} excluded, blank/other {jd['label_counts']['blank_or_other']}). "
              f"Judge false accepts {jd['false_accept']}, false rejects {jd['false_reject']}. "
              f"EM-vs-labels agreement {_f(jd['em_vs_labels_agreement'], True)} (on judge-invoked rows only, where EM=0 "
              "by construction).\n",
              "| arm | n | judge agreement | false accept | false reject | EM agreement | labelled CORRECT |",
              "|---|---|---|---|---|---|---|"]
        for a_, v in jd["per_arm"].items():
            L.append(f"| {a_} | {v['n']} | {_f(v['agree'] / v['n'], True)} | {v['false_accept']} | {v['false_reject']} | "
                     f"{_f(v['em_agree'] / v['n'], True)} | {v['human_correct']} |")
        if jd["case"] == 2:
            L.append("\nJudge REJECTED as-is. The 'final (EM∨judge)' columns above use a rejected judge and must not be "
                     "read as accuracy; EM columns are unaffected. No prompt/model change made; awaiting user decision.\n")
    bd = checks.get("bm25_detail")
    if bd:
        L += ["## BM25 session checks (before any arm)\n", "```json", json.dumps(bd, indent=2), "```\n"]
    th = checks.get("test_history")
    if th:
        L += ["## Test-suite history\n", "| count | added | why |", "|---|---|---|"]
        for t in th:
            L.append(f"| {t['count']} | {'; '.join(t.get('added', [])) or t.get('note', '')} | {t.get('why', '')} |")
    L += ["\n## Timing and projection\n", "```json", json.dumps(timing, indent=2), "```",
          "Projection (formula, not a claim): " + projection.get("formula", NM), "```json",
          json.dumps(projection, indent=2), "```",
          "\n## Determinism\n", "```json", json.dumps(checks.get("determinism"), indent=2), "```"]
    return "\n".join(L) + "\n"
