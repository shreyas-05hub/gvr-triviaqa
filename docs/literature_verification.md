# Literature verification record

**Verified by:** the PI, 2026-10-09, against primary pages (arXiv abstract/HTML pages, ACL Anthology, NeurIPS proceedings).
**Recorded by:** Claude, from the PI's report. Claude did not independently re-check these pages in this session.

## VERIFIED

| Reference | Identifier | Venue / details confirmed |
|---|---|---|
| Dhuliawala et al. 2023, Chain-of-Verification | arXiv:2309.11495 | — |
| Lewis et al. 2020, Retrieval-Augmented Generation | arXiv:2005.11401 | NeurIPS 2020, pp. 9459–9474 |
| Jiang et al. 2023, Active Retrieval Augmented Generation (FLARE) | arXiv:2305.06983 | EMNLP 2023, pp. 7969–7992 |
| Jeong et al. 2024, Adaptive-RAG | arXiv:2403.14403 | NAACL 2024, pp. 7036–7050 |
| Mallen et al. 2023, When Not to Trust Language Models | arXiv:2212.10511 | ACL 2023, pp. 9802–9822 |
| Lin et al. 2021, Pyserini | arXiv:2102.10073 | SIGIR 2021, pp. 2356–2362 |
| Karpukhin et al. 2020, Dense Passage Retrieval | arXiv:2004.04906 | EMNLP 2020, pp. 6769–6781 |
| Dettmers et al. 2023, QLoRA | arXiv:2305.14314 | NeurIPS 2023; introduces NF4 |
| Qwen Team 2024, Qwen2.5 Technical Report | arXiv:2412.15115 | — |
| Joshi et al. 2017, TriviaQA | arXiv:1705.03551 | ACL 2017, pp. 1601–1611 |
| Li et al. 2024, Confidence Matters | arXiv:2402.12563 | — |
| Jang et al. 2025, Verbalized Confidence Triggers Self-Verification | arXiv:2506.03723 | — |
| Chen et al. 2025, Query-Level Uncertainty in LLMs | arXiv:2506.09669 | — |
| Kamalloo et al. 2023, Evaluating Open-Domain QA in the Era of LLMs | arXiv:2305.06984 | ACL 2023, pp. 5591–5606 |
| McNemar 1947 | — | Psychometrika 12(2):153–157 |
| Kumaran et al. 2026, How LLMs Detect and Correct Their Own Errors | arXiv:2604.22271 | Gemma 3 27B + Qwen2.5-7B replication on TriviaQA; closed-book, no retrieval |
| Moskvoretskii et al. 2025, Adaptive Retrieval Without Self-Knowledge? | — | ACL 2025, pp. 6355–6384; 35 methods = 8 pipelines + 27 uncertainty-estimation techniques |
| Dey, Ounis, McDonald 2026, Interpretable Uncertainty for Adaptive Retrieval and Reasoning in QA | arXiv:2607.07380 | only the retrieval branch evaluated; Llama-2-7b-chat, Natural Questions |
| Sah, Zhang, Lian 2026, Beyond Self-Knowledge | arXiv:2607.25600 | verbalized confidence; closed API models; top-5 TF–IDF; six datasets incl. TriviaQA; route-count-matched random control; no self-verification arm |

## PARTIAL

| Reference | Identifier | Open item |
|---|---|---|
| Asai et al. 2023, Self-RAG | arXiv:2310.11511 | venue unconfirmed |
| Robertson & Zaragoza 2009, The Probabilistic Relevance Framework: BM25 and Beyond | — | publisher page failed to load; pages/DOI unconfirmed |
| Holm 1979, A Simple Sequentially Rejective Multiple Test Procedure | — | volume, pages, DOI unconfirmed |

## Relation to earlier notes

`docs/literature_notes.md` recorded 2607.25600 from an HTML summary, pending full-text verification. This record supersedes that pending status, per the PI's verification above.

## Provenance correction (appended 2026-10-09; supersedes the "Verified by" line above)

Primary-page verification was performed by an AI assistant (Claude, in the PI's chat session, via arXiv abs/HTML, ACL Anthology, NeurIPS proceedings and Springer pages) on 2026-10-09 and relayed by the PI; not independently re-checked by a human or by Claude Code.

## Status update (appended 2026-10-09): PARTIAL items moved to VERIFIED (relayed)

Provenance: relayed by the PI from a second AI assistant's check of JSTOR, now publishers and ICLR proceedings; not re-checked by Claude Code.

| Reference | Status | Details |
|---|---|---|
| Holm 1979, A Simple Sequentially Rejective Multiple Test Procedure | VERIFIED (relayed) | Scand. J. Statist. 6(2):65–70, doi:10.2307/4615733 |
| Robertson & Zaragoza 2009, The Probabilistic Relevance Framework: BM25 and Beyond | VERIFIED (relayed) | Foundations and Trends in Information Retrieval 3(4):333–389, doi:10.1561/1500000019. The DOI landing page shows conflicting volume/pagination (Crossref/doi.org record: vol. 4, issue 1–2, pp. 1–174); the original front matter was used. |
| Asai et al., Self-RAG | VERIFIED (relayed) | ICLR 2024; author name Zeqiu Wu |
| Chen et al., Query-Level Uncertainty in Large Language Models (arXiv:2506.09669) | VERIFIED (relayed) | ICLR 2026 |

The PARTIAL table above is retained for the record; this section supersedes it.
