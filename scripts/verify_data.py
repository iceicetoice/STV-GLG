import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stv_glg.config import read_config
from stv_glg.data import load_seasons


root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "data" / "manifest.json").read_text(encoding="utf-8"))
for item in manifest["files"]:
    actual = hashlib.sha256((root / "data" / item["file"]).read_bytes()).hexdigest()
    if actual != item["sha256"]:
        raise ValueError(f"Dataset checksum differs: {item['file']}")
seasons, audit = load_seasons(read_config(root / "configs" / "reproduce_2022_2024.json"))
for season in seasons.values():
    if season.env_mask.any() or season.cloud_mask.any():
        raise ValueError("The area-only profile must not include environmental or cloud values.")
print(json.dumps({"checksums_valid": True, "years": audit["years"]}, indent=2))
