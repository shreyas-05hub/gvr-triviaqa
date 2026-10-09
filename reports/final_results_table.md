# Final results — test (validation split), N = 3,000

**PRIMARY = frozen EM result [S1].** Audit-adjusted accuracy = SECONDARY SENSITIVITY ANALYSIS only [S2].

Primary E2 vs E4 [S1 /primary, /bootstrap/em/E2-E4]: n10 = 55, n01 = 444, exact McNemar p = 1.24e-76; E2 − E4 = -12.97 pts, bootstrap 95% CI [-14.33, -11.60].

### Panel A — accuracy and tests

| Arm | N | Invalid / abstain | EM | F1 | Audit-adjusted (SECONDARY) [95% CI] | EM − E0 [95% CI] | McNemar vs E0: n10 / n01 / p / p Holm | Gate: k / valid; nominal; overall | Random gate: null mean (SD); obs. pct; p / p Holm; EM − RG [95% CI] | Recall@5 |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | 3,000 | 2 / 1 | 43.53% | 0.4919 | 51.42% [46.11, 57.32] | reference | — | — | — | — |
| E1 | 3,000 | 0 / 1 | 45.37% | 0.5124 | 49.73% [45.67, 54.47] | +1.83 [1.10, 2.53] | not tested (descriptive) | — | — | — |
| E3 | 3,000 | 1 / 0 | 59.33% | 0.6672 | 69.90% [64.86, 75.05] | +15.80 [13.83, 17.73] | not tested (descriptive) | — | — | 67.20% |
| E2 | 3,000 | via E0/E1/E3 | 45.20% | 0.5104 | 50.92% [47.49, 54.68] | +1.67 [1.13, 2.20] | 58 / 8 / 1.8e-10 / 5.39e-10 | 1,180 / 2,998; 39.36%; 39.40% | 44.26% (0.18); 100.0; 0.002 / 0.004; +0.95 [0.56, 1.32] | — |
| E4 | 3,000 | via E0/E1/E3 | 58.17% | 0.6469 | 66.36% [62.86, 69.91] | +14.63 [13.23, 16.03] | 484 / 45 / 5.49e-94 / 2.2e-93 | 1,180 / 2,998; 39.36%; 39.40% | 49.75% (0.49); 100.0; 0.002 / 0.004; +8.41 [7.39, 9.42] | not computed in frozen output |

### Panel B — cost (full pipeline totals)

| Arm | Model calls | Generated tokens | Input tokens | Retrieval calls | Passages | GPU-s (model-call wall time, 1×T4) |
|---|---|---|---|---|---|---|
| E0 | 3,000 | 11,740 | 267,817 | 0 | 0 | 1,263.1 |
| E1 | 17,948 | 214,875 | 1,987,709 | 0 | 0 | 17,516.7 |
| E3 | 3,000 | 12,072 | 2,656,172 | 3,000 | 15,000 | 3,953.5 |
| E2 | 8,888 | 92,275 | 947,340 | 0 | 0 | 7,698.4 |
| E4 | 4,182 | 16,566 | 1,313,943 | 1,182 | 5,910 | 2,827.4 |

Audit-adjusted E2 − E4 (SECONDARY) [S2 /audit_adjusted_E2_minus_E4]: -15.45 pts, 95% CI [-18.99, -11.75].
Audit numbers above are the AUDIT OF RECORD: AI-assisted blind audit, labels sha 9078bfca… (hash recorded before key read).
POST-HOC SENSITIVITY ANALYSIS [S4] (NOT the audit of record): after unblinding, the PI changed a092 and a143 CORRECT→INCORRECT (labels sha dac7400c…; provenance_corrections PC6). Audit CORRECT counts E0/E1/E3 = 6/3/13 of 50. Audit-adjusted: E0 50.30% [45.30, 55.99]; E1 48.64% [45.03, 52.86]; E3 69.90% [64.86, 75.05]; E2 49.82% [46.62, 53.27]; E4 65.92% [62.46, 69.37]; E2 − E4 -16.10 pts [-19.49, -12.63]. Same bootstrap and seed as S2.
Session wall time [S3], all arms incl. BM25/model setup: c1 7,738.4 s, c2 8,277.6 s, c3 7,893.5 s; total 23,909.5 s. Not separable by arm; per-arm time is the GPU-s column.
Pipelines: E1 = E0 + E1 (E1 consumes A1); E2 = E0 + E1 on flagged items; E4 = E0 + E3 on flagged items. Retrieval calls = passages / 5.
Audit-adjusted CI: two-level bootstrap, B = 10,000, PCG64(20261008): per replicate, each arm's 50 audit labels resampled (FN rates for E0/E1/E3) and the 3,000 test items resampled with one shared index draw; percentile 95% intervals.
Transferability assumption: FN rates were estimated on audited EM-failed outputs of each arm overall, while E2/E4 use E1/E3 only on the gated (flagged) subset; the sensitivity result depends on these rates transferring to that subset. EM false positives are not estimated.
EM differences: paired bootstrap 95% CIs from the frozen analysis (B = 10,000, PCG64(20261006)). Holm family: E2 vs E0, E4 vs E0, E2 vs RG, E4 vs RG. Random-gate p = 0.002 is the minimum attainable with 1,000 seeds.

**Sources** (every cell in reports/final_results_table.json records path + sha256 + JSON pointer):
- S1 `runs/analysis_main/results.json` sha256 `432e206d7b4e59233e6ace02d756840fe4cd9b77d582fb33b200af94ab4bb5ea` (frozen primary analysis)
- S2 `runs/post_freeze/audit_adjusted_v2.json` sha256 `b05b35105c3af645a600c3a4fbef7217279dbdd8daebe3c63ae77c5549d1255e` (post-freeze addition v2, secondary; audit of record)
- S4 `runs/post_hoc/audit_adjusted_corrected_labels.json` sha256 `cc1b42090df483479f662f7650c65c7e215704b76cfeb1df08e5515263f0c245` (POST-HOC sensitivity, corrected labels; not of record)
- S3 `runs/test-c1/run_meta.jsonl` sha256 `6d4a6e4e19ec5fcf071105c882969057c7ca1e9e26fb9764295e0d071e366caa`
- S3 `runs/test-c2/run_meta.jsonl` sha256 `de43cc265da5615912124b81729063aee434fe2244bb6651d5ccfdfa050904fa`
- S3 `runs/test-c3/run_meta.jsonl` sha256 `b3d56a062c19b4f2eab59c32202ee4e624d21af360c84e94c1f3415ad54ec4b9`
