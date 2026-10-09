"""Writes configs/env_manifest.json (this machine) and prompts/registry.json."""
import argparse
from _common import ROOT
from gvr import prompts as P
from gvr.manifests import write_env_manifest

ap = argparse.ArgumentParser()
ap.add_argument("--role", default="laptop-cpu")
a = ap.parse_args()
P.write_registry()
name = "env_manifest.json" if a.role == "laptop-cpu" else f"env_manifest_{a.role}.json"
rec = write_env_manifest(ROOT / "configs" / name, role=a.role)
print(rec)
