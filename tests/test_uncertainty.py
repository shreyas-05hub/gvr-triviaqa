import math
import torch
from gvr.uncertainty import token_logprobs, mean_nll


def test_logprobs_and_eos_exclusion():
    V = 151646
    logits = []
    ids = [5, 7, 151645]
    for t in ids:
        l = torch.zeros(1, V, dtype=torch.float16)
        l[0, t] = 4.0
        logits.append(l)
    lps = token_logprobs(tuple(logits), ids)
    expect = 4.0 - math.log(math.exp(4.0) + V - 1)
    # float32 log_softmax over a 151k vocab: ~1e-4 rounding vs exact float64 value
    assert all(abs(x - expect) < 5e-4 for x in lps)
    ref = [torch.log_softmax(l[0].float(), -1)[t].item() for l, t in zip(logits, ids)]
    assert lps == ref
    u = mean_nll(lps, ids, [151645, 151643])
    assert abs(u + (lps[0] + lps[1]) / 2) < 1e-9
    assert mean_nll([lps[2]], [151645], [151645, 151643]) is None


def test_length_mismatch_raises():
    import pytest
    with pytest.raises(AssertionError):
        token_logprobs((torch.zeros(1, 3),), [0, 1])
