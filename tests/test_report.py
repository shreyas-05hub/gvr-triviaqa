from gvr.report import render, HEADER


def test_render_minimal():
    s = dict(N=2, em=0.5, final=0.5, f1=0.5, invalid=0.0, abstain=0.0, judged=1, hallucination_proxy=0.5,
             failed=0, cost_calls=2, cost_retrievals=0, cost_in_tokens=10, cost_out_tokens=4, cost_wall_s=1.0)
    res = dict(summary={a: s for a in ("E0", "E1", "E3", "E2", "E4")},
               flips={a: dict(right_to_wrong=0, wrong_to_right=0) for a in ("E1", "E3", "E2", "E4")},
               tau=0.3, gate_rates=dict(N=2, n_valid=2, k_flagged_valid=0, nominal_rate=0.0,
                                         overall_mitigation_rate=0.0),
               random_gate={"E2": dict(mean=0.5, sd=0.0, p2_5=0.5, p97_5=0.5)}, recall5=None,
               e3_truncated=0, u_valid=[0.1, 0.2], paraphrase=dict(n=0, flagged=0), sv_plan_invalid=0)
    thr = dict(max_invalid_rate_per_arm=0.05, max_unrecovered_failures=0, max_retried_rate=0.02,
               min_judge_agreement=0.9)
    md = render(res, {"m": 1}, {}, thr, {}, {})
    assert HEADER in md and "PENDING" in md and "NOT MEASURED" in md
