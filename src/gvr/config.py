import hashlib
import yaml
from .paths import CONFIG_YAML


def load_config(path=CONFIG_YAML):
    with open(path) as f:
        return yaml.safe_load(f)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
