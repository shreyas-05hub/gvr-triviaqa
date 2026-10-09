import torch


def token_logprobs(logits, gen_ids):
    """logits: tuple (one per generated position) of [1, V] tensors. log_softmax in float32."""
    assert len(logits) == len(gen_ids), (len(logits), len(gen_ids))
    return [torch.log_softmax(l[0].float(), dim=-1)[t].item() for l, t in zip(logits, gen_ids)]


def mean_nll(logprobs, gen_ids, excluded_ids):
    """-> u or None if no non-excluded tokens."""
    kept = [lp for lp, t in zip(logprobs, gen_ids) if t not in set(excluded_ids)]
    if not kept:
        return None
    return -sum(kept) / len(kept)
