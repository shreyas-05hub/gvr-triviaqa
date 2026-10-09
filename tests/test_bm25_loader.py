"""Shared guarded BM25 loader is the ONLY path to a searcher; smoke-regression comparison; env hygiene."""
import ast
import json
import os
import pathlib
import sys
import types
import pytest
from gvr import retrieval as R

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _names(path):
    tree = ast.parse(path.read_text())
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out.update(a.name for a in n.names)
    return out


def test_lucenesearcher_only_in_retrieval_module():
    files = list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py"))
    users = [f.relative_to(ROOT).as_posix() for f in files
             if "LuceneSearcher" in _names(f) or "FaissSearcher" in _names(f)]
    assert users == ["src/gvr/retrieval.py"], users


def test_bm25_constructed_only_inside_loader():
    for f in list((ROOT / "scripts").glob("*.py")) + [p for p in (ROOT / "src").rglob("*.py")
                                                         if p.name != "retrieval.py"]:
        tree = ast.parse(f.read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == "BM25":
                pytest.fail(f"direct BM25(...) construction in {f}")
    src = (ROOT / "src/gvr/retrieval.py").read_text()
    assert src.count("BM25(rcfg)") == 1  # single construction site, inside load_bm25_guarded


@pytest.mark.parametrize("script", ["run_arms.py", "stage4_bm25.py"])
def test_scripts_use_shared_loader(script):
    names = _names(ROOT / "scripts" / script)
    assert "load_bm25_guarded" in names and "smoke_regression" in names


def test_reference_file_consistent():
    ref = json.load(open(ROOT / "configs/bm25_smoke_reference.json"))
    pilot = json.load(open(ROOT / "data/pilot_ids.json"))
    assert ref["item_order"] == pilot[:10]
    assert all(len(ref["top5_titles"][i]) == 5 for i in ref["item_order"])
    assert ref["index_md5"] == "7b58c08da992b2ea7e96667f0b176651" and (ref["k1"], ref["b"], ref["k"]) == (0.9, 0.4, 5)


def test_compare_smoke():
    ref = dict(item_order=["a", "b"], top5_titles={"a": ["t1"] * 5, "b": ["t2"] * 5})
    ok = [dict(item_id="a", titles=["t1"] * 5), dict(item_id="b", titles=["t2"] * 5)]
    assert R.compare_smoke(ok, ref) == []
    bad = [dict(item_id="a", titles=["t1"] * 4 + ["x"]), dict(item_id="b", titles=["t2"] * 5)]
    assert len(R.compare_smoke(bad, ref)) == 1
    assert R.compare_smoke(list(reversed(ok)), ref)  # order matters


def test_placeholder_key_set_only_during_import(monkeypatch):
    seen = {}
    fake = types.ModuleType("pyserini.search.lucene")
    class LuceneSearcher:  # noqa: N801
        pass
    fake.LuceneSearcher = LuceneSearcher
    def spy(name, *a, **k):
        if name == "pyserini.search.lucene":
            seen["key"] = os.environ.get("OPENAI_API_KEY")
            return fake
        return real_import(name, *a, **k)
    import builtins
    real_import = builtins.__import__
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(builtins, "__import__", spy)
    R._import_pyserini_with_placeholder_key()
    monkeypatch.setattr(builtins, "__import__", real_import)
    assert seen["key"] == R.PLACEHOLDER_OPENAI_KEY
    assert "OPENAI_API_KEY" not in os.environ
