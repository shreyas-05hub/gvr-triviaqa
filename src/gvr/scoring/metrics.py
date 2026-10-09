from .normalize import normalize_answer, f1_score, contains_token_seq


def exact_match(pred: str, gts: set) -> bool:
    return normalize_answer(pred) in gts


def max_f1(pred: str, gts: set) -> float:
    return max((f1_score(pred, g) for g in gts), default=0.0)


def is_abstain(pred: str, abstain_list) -> bool:
    return normalize_answer(pred) in {normalize_answer(a) for a in abstain_list}


def answer_string_recall(passage_texts, gts: set) -> bool:
    """Answer-string recall@k (NOT support): any GT as whole-token sequence in any passage."""
    return any(contains_token_seq(t, g) for t in passage_texts for g in gts)


def needs_judge(valid: bool, abstain: bool, em: bool) -> bool:
    return valid and not abstain and not em


def label_item(pred, valid, gts, abstain_list, judge_correct=None, passages=None, judge_enabled=True):
    """judge_enabled=False (main study after judge_rule_v2 fallback): final = EM; no judge verdict used or required."""
    abstain = valid and is_abstain(pred, abstain_list)
    em = bool(valid and exact_match(pred, gts))
    f1 = max_f1(pred, gts) if valid else 0.0
    if not judge_enabled:
        judge_correct = None
    elif needs_judge(valid, abstain, em):
        if judge_correct is None:
            raise ValueError("judge verdict required but missing")
    else:
        judge_correct = None
    final = em or bool(judge_correct)
    incorrect = not final
    out = dict(valid=valid, abstain=abstain, em=em, f1=f1, judged=judge_correct is not None,
               judge_correct=judge_correct, final=final, incorrect=incorrect,
               hallucination_proxy=incorrect and valid and not abstain)
    if passages is not None:
        out["answer_string_recall5"] = answer_string_recall(passages, gts)
        out["unsupported"] = not out["answer_string_recall5"]
    return out


def flips(base_final: bool, arm_final: bool):
    return dict(right_to_wrong=base_final and not arm_final,
                wrong_to_right=(not base_final) and arm_final)
