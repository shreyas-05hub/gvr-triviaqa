"""Append-only calls.jsonl logger with one-retry policy."""
import json
import time
import traceback
import uuid
from pathlib import Path


class CallLogger:
    def __init__(self, run_dir, run_id, split, env_versions, fixture=False):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "calls.jsonl"
        self.run_id, self.split, self.env, self.fixture = run_id, split, env_versions, fixture

    def log(self, **rec):
        base = dict(call_id=uuid.uuid4().hex, run_id=self.run_id, split=self.split,
                    fixture=self.fixture, env_versions=self.env, retry_of=None, error=None,
                    truncated=None, retrieval={"called": False})
        base.update(rec)
        with open(self.path, "a") as f:
            f.write(json.dumps(base, ensure_ascii=False) + "\n")
        return base

    def run(self, fn, meta, max_retries=1):
        """fn() -> dict of call fields. Logs every attempt; returns (record, ok)."""
        retry_of = None
        for attempt in range(max_retries + 1):
            t0 = time.time()
            try:
                out = fn()
                return self.log(**meta, **out, retry_of=retry_of), True
            except Exception as e:  # logged, then retried at most once
                rec = self.log(**meta, t_start=t0, t_end=time.time(), retry_of=retry_of,
                               error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
                retry_of = rec["call_id"]
        return rec, False


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]
