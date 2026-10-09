from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
PROMPTS = ROOT / "prompts"
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
CONFIG_YAML = CONFIGS / "config.yaml"
FREEZE_MANIFEST = CONFIGS / "freeze_manifest.json"
if not FREEZE_MANIFEST.exists():   # public release ships only the redacted copy
    FREEZE_MANIFEST = CONFIGS / "freeze_manifest.public.json"
