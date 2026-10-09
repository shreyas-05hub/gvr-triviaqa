"""The OpenAI guard makes any OpenAI client/encoder use fail (Pyserini imports one at module load)."""
import types
import pytest
from gvr.retrieval import install_openai_guard, OpenAICalledError


def fake_module():
    m = types.ModuleType("pyserini.encode._openai")
    emb = types.SimpleNamespace(create=lambda **k: "real call")
    comp = types.SimpleNamespace(create=lambda **k: "real call")
    m.client = types.SimpleNamespace(embeddings=emb, chat=types.SimpleNamespace(completions=comp))

    class OpenAiQueryEncoder:
        def __init__(self, *a, **k):
            self.ok = True
    m.OpenAiQueryEncoder = OpenAiQueryEncoder
    m.OpenAiDocumentEncoder = type("OpenAiDocumentEncoder", (), {"__init__": lambda self: None})
    return m


def test_guard_blocks_all_openai_use():
    m = fake_module()
    calls = install_openai_guard([m])
    with pytest.raises(OpenAICalledError):
        m.client.embeddings.create(input="x")
    with pytest.raises(OpenAICalledError):
        m.client.chat.completions.create(messages=[])
    with pytest.raises(OpenAICalledError):
        m.OpenAiQueryEncoder()
    with pytest.raises(OpenAICalledError):
        m.OpenAiDocumentEncoder()
    assert len(calls) == 4


def test_placeholder_key_only_in_bm25_command():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "src" / "runner.py").read_text()
    assert src.count("OPENAI_API_KEY") == src.count("OPENAI_API_KEY=unused-placeholder") >= 1
    for line in src.splitlines():
        if "OPENAI_API_KEY" in line:
            assert "stage4_bm25.py" in line and "os.environ" not in line
