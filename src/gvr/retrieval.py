"""BM25 via Pyserini LuceneSearcher on the prebuilt SPARSE index wikipedia-dpr-100w.
Never FaissSearcher / dense indexes. k1/b fixed by config; never tuned."""
import json
import re
import time


def index_metadata(name):
    """Pyserini's registry entry for a prebuilt index (no download)."""
    from pyserini import prebuilt_index_info as pii
    for attr in dir(pii):
        reg = getattr(pii, attr)
        if isinstance(reg, dict) and name in reg and attr.upper().startswith(("TF_", "LUCENE_")):
            return attr, reg[name]
    raise KeyError(f"{name} not found in Pyserini sparse registries")


def parse_dpr_doc(raw):
    """wikipedia-dpr-100w docs store contents as '"Title"\\ntext'."""
    d = json.loads(raw)
    contents = d.get("contents", "")
    title, _, text = contents.partition("\n")
    return title.strip().strip('"'), text.strip()


class BM25:
    def __init__(self, rcfg):
        from pyserini.search.lucene import LuceneSearcher
        assert rcfg["searcher"] == "LuceneSearcher"
        self.cfg = rcfg
        self.searcher = LuceneSearcher.from_prebuilt_index(rcfg["prebuilt_index"])
        self.searcher.set_bm25(k1=rcfg["k1"], b=rcfg["b"])

    def search(self, query):
        t0 = time.time()
        hits = self.searcher.search(query, k=self.cfg["k"])
        t1 = time.time()
        passages = []
        for h in hits:
            title, text = parse_dpr_doc(self.searcher.doc(h.docid).raw())
            passages.append(dict(id=h.docid, score=float(h.score), title=title, text=text))
        return passages, dict(called=True, k=self.cfg["k"], k1=self.cfg["k1"], b=self.cfg["b"],
                              passage_ids=[p["id"] for p in passages],
                              scores=[p["score"] for p in passages], latency_s=t1 - t0,
                              t_start=t0, t_end=t1)


class OpenAICalledError(RuntimeError):
    pass


def install_openai_guard(modules=None):
    """Pyserini 1.6.0 constructs an OpenAI client at import time (pyserini/encode/_openai.py).
    We never use OpenAI: make every client method and the OpenAI encoders raise if ever invoked."""
    import sys
    calls = []

    def _boom(name):
        def f(*a, **k):
            calls.append(name)
            raise OpenAICalledError(f"OpenAI used via {name}: forbidden in this protocol")
        return f

    mods = modules or [sys.modules[m] for m in ("pyserini.encode._openai",) if m in sys.modules]
    for mod in mods:
        client = getattr(mod, "client", None)
        if client is not None:
            for path in ("embeddings.create", "chat.completions.create", "completions.create", "responses.create"):
                obj = client
                parts = path.split(".")
                try:
                    for p in parts[:-1]:
                        obj = getattr(obj, p)
                    setattr(obj, parts[-1], _boom(f"client.{path}"))
                except AttributeError:
                    pass
        for cls_name in ("OpenAiQueryEncoder", "OpenAiDocumentEncoder"):
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                cls.__init__ = _boom(f"{cls_name}.__init__")
    return calls


PLACEHOLDER_OPENAI_KEY = "unused-placeholder"


def _import_pyserini_with_placeholder_key():
    """Pyserini 1.6.0 builds openai.OpenAI(api_key=...) at import. Set a placeholder key ONLY for that import,
    then restore the environment. Returns seconds spent (JVM start + import)."""
    import os
    import time
    prev = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = PLACEHOLDER_OPENAI_KEY
    t0 = time.time()
    try:
        from pyserini.search.lucene import LuceneSearcher  # noqa: F401  (starts the JVM)
    finally:
        if prev is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = prev
    return time.time() - t0


def _try_java(obj, names):
    for n in names:
        try:
            v = getattr(obj, n)
            v = v() if callable(v) else v
            if v is not None:
                return n, v
        except Exception:
            continue
    return None, None


def load_bm25_guarded(rcfg, timing_log=None):
    """The ONLY entry point to a BM25 searcher (stage4 and run_arms). Behavior identical to
    LuceneSearcher.from_prebuilt_index + set_bm25(k1, b); adds the OpenAI guard, md5-verified download via
    pyserini.util.download_url, and separate JVM / download / extract / load timings.
    Returns (BM25, record)."""
    import json
    import os
    import pathlib
    import re
    import tarfile
    import time
    dl = rcfg["index_download"]
    jvm_s = _import_pyserini_with_placeholder_key()
    openai_calls = install_openai_guard()
    from pyserini.util import download_url, get_cache_home
    reg, meta = index_metadata(rcfg["prebuilt_index"])
    assert meta["md5"] == dl["expected_md5"], (meta["md5"], dl["expected_md5"])
    url = meta["urls"][0]
    index_name = re.sub(r".tar.gz.*$", "", url.split("/")[-1])
    index_directory = os.path.join(get_cache_home(), "indexes")
    index_path = os.path.join(index_directory, f"{index_name}.{meta['md5']}")
    os.makedirs(index_directory, exist_ok=True)
    dl_s = ex_s = 0.0
    downloaded = False
    if not os.path.exists(index_path):   # mirrors pyserini.util.download_and_unpack_index(prebuilt=True)
        need = dl["preflight_min_free_factor"] * dl["compressed_bytes"]
        import shutil
        free = shutil.disk_usage(index_directory).free
        if free < need:
            raise SystemExit(f"STOP: {free / 1e9:.1f} GB free < {need / 1e9:.1f} GB needed; retriever unchanged")
        tarball = os.path.join(index_directory, f"{index_name}.tar.gz")
        if os.path.exists(tarball):
            os.remove(tarball)
        t0 = time.time()
        download_url(url, index_directory, verbose=False, md5=meta["md5"], expected_size=dl["compressed_bytes"])
        dl_s = time.time() - t0
        t0 = time.time()
        tb = tarfile.open(tarball)
        dirs = [m.name for m in tb if m.isdir()]
        tb.extractall(index_directory)
        tb.close()
        os.remove(tarball)
        name_in_tar = dirs[0] if dirs[0] != index_name else index_name
        os.rename(os.path.join(index_directory, name_in_tar), index_path)
        ex_s = time.time() - t0
        downloaded = True
    t0 = time.time()
    bm = BM25(rcfg)   # LuceneSearcher.from_prebuilt_index (finds index_path, validates stats) + set_bm25
    load_s = time.time() - t0
    s = bm.searcher
    from jnius import autoclass
    d = autoclass("org.apache.lucene.store.FSDirectory").open(autoclass("java.nio.file.Paths").get(s.index_dir))
    seg = autoclass("org.apache.lucene.index.SegmentInfos").readLatestCommit(d)
    reader = autoclass("org.apache.lucene.index.DirectoryReader").open(d)
    an_name, an = _try_java(s.object, ["get_analyzer", "getAnalyzer", "analyzer"])
    sim_name, sim = _try_java(s.object, ["get_similarity", "getSimilarity", "similarity"])
    rec = dict(name=rcfg["prebuilt_index"], registry=reg, index_dir=s.index_dir, registry_md5=meta["md5"],
               md5_verified_by="pyserini.util.download_url(md5=registry md5)" if downloaded else "already present",
               index_dir_has_md5_suffix=s.index_dir.endswith(dl["expected_md5"]),
               build_tag_in_path=dl["build_tag"] in s.index_dir,
               num_docs=int(reader.numDocs()), registry_documents=meta.get("documents"),
               lucene_commit_version=str(seg.getCommitLuceneVersion().toString()),
               lucene_min_segment_version=str(seg.getMinSegmentLuceneVersion().toString()),
               searcher_class=type(s).__name__,
               analyzer=f"{an.getClass().getName()} (via {an_name})" if an is not None else "UNRESOLVED",
               similarity_readback=f"{sim.toString()} (via {sim_name})" if sim is not None else "UNRESOLVED",
               k1=rcfg["k1"], b=rcfg["b"], k=rcfg["k"],
               archive_deleted=not os.path.exists(os.path.join(index_directory, f"{index_name}.tar.gz")),
               extracted_index_bytes=sum(f.stat().st_size for f in pathlib.Path(s.index_dir).rglob("*") if f.is_file()),
               timing_s=dict(jvm_start=jvm_s, download_md5=dl_s, extract=ex_s, load_validate=load_s),
               downloaded_this_session=downloaded, openai_guard_calls=openai_calls)
    reader.close()
    assert rec["num_docs"] == dl["documents"] == rec["registry_documents"], rec
    assert rec["index_dir_has_md5_suffix"] and rec["build_tag_in_path"] and rec["archive_deleted"], rec
    if timing_log:
        with open(timing_log, "a") as f:
            for step, sec in rec["timing_s"].items():
                f.write(json.dumps(dict(step=f"bm25_{step}", seconds=sec, t_end=time.time())) + "\n")
    return bm, rec


def readback_k1_b(bm):
    """Per-query BM25 parameters as reported by Lucene's similarity object."""
    _, sim = _try_java(bm.searcher.object, ["get_similarity", "getSimilarity", "similarity"])
    if sim is None:
        return None
    m = re.search(r"k1=([0-9.]+),\s*b=([0-9.]+)", sim.toString())
    return (float(m.group(1)), float(m.group(2))) if m else None


def compare_smoke(rows, reference):
    """rows: [{item_id, titles}] ; reference: configs/bm25_smoke_reference.json. Returns list of mismatches."""
    ref = reference["top5_titles"]
    bad = []
    for r in rows:
        exp = ref.get(r["item_id"])
        if exp is None or r["titles"] != exp:
            bad.append(dict(item_id=r["item_id"], expected=exp, got=r["titles"]))
    if [r["item_id"] for r in rows] != reference["item_order"]:
        bad.append(dict(item_order_expected=reference["item_order"], got=[r["item_id"] for r in rows]))
    return bad


def smoke_regression(bm, questions, reference, rcfg):
    """Re-runs the reference smoke queries; records passage IDs, titles, per-query k1/b readback, latency.
    Raises SystemExit (STOP) on any title mismatch, wrong hit count, k1/b readback mismatch or OpenAI use."""
    rows = []
    for iid in reference["item_order"]:
        ps, meta = bm.search(questions[iid])
        kb = readback_k1_b(bm)
        rows.append(dict(item_id=iid, n_hits=len(ps), passage_ids=[p["id"] for p in ps],
                         titles=[p["title"] for p in ps], scores=[p["score"] for p in ps],
                         k1_b_readback=kb, latency_s=meta["latency_s"]))
    problems = compare_smoke(rows, reference)
    for r in rows:
        if r["n_hits"] != rcfg["k"]:
            problems.append(dict(item_id=r["item_id"], n_hits=r["n_hits"]))
        if r["k1_b_readback"] != (rcfg["k1"], rcfg["b"]):
            problems.append(dict(item_id=r["item_id"], k1_b_readback=r["k1_b_readback"]))
    return rows, problems
