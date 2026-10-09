"""Stage 2: train-only data access, ID sampling, freeze hashes, schema report (metadata only)."""
import json
import sys
import pyarrow.parquet as pq
from huggingface_hub import HfApi

sys.path.insert(0, "src")
from gvr.config import load_config
from gvr.data import download_split_file, sample_train_ids, write_ids

cfg = load_config()
d = cfg["dataset"]
info = HfApi().dataset_info(d["repo_id"], revision=d["revision"])
assert info.sha == d["revision"], (info.sha, d["revision"])
card = info.card_data.to_dict() if info.card_data else {}
print("dataset revision pinned:", info.sha)
print("license field (card):", card.get("license"), "| tags:", [t for t in info.tags if t.startswith("license")])

feat = next(c for c in card["dataset_info"] if c["config_name"] == d["config"])
names = {f["name"]: f for f in feat["features"]}
ans = [x["name"] for x in names["answer"]["struct"]]
for need in ["question", "question_id"]:
    assert need in names, need
for need in ["value", "aliases", "normalized_aliases"]:
    assert need in ans, need
print("card features (all splits share one schema):", list(names), "answer.*:", ans)
print("split sizes (card):", {s["name"]: s["num_examples"] for s in feat["splits"]})

path = download_split_file(cfg, "train")
schema = pq.read_schema(path)          # parquet footer schema only
print("train parquet schema answer struct:", schema.field("answer").type)

pilot, calib = sample_train_ids(cfg)
paths = write_ids(pilot, calib)
assert set(pilot).isdisjoint(calib) and len(pilot) == cfg["sampling"]["n_pilot"] and len(calib) == cfg["sampling"]["n_calib"]
print("pilot", len(pilot), "calib", len(calib), "intersection", len(set(pilot) & set(calib)))
print(json.dumps(json.load(open("configs/freeze_manifest.json")), indent=2))
