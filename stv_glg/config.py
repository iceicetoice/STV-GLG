from __future__ import annotations

import json
from pathlib import Path

PAPER_VARIABLES = ["sst", "sal", "cur", "po4", "no3", "si", "rad", "pre", "ws", "wd"]


def validate_paper_structure(config: dict):
    model = config["model"]
    expected = {"d_model": 128, "heads": 8, "encoder_layers": 4, "query_layers": 2,
                "observation_quality_embedding": False, "cyclic_wind_embedding": False}
    if config["variables"] != PAPER_VARIABLES or (config["region_rows"], config["region_cols"]) != (4, 2):
        raise ValueError("Paper structure requires the ten ordered variables and eight regions.")
    if any(model.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Paper structure settings must match {expected}.")
    if config.get("area_input_transform", "linear") != "linear" or config.get("calendar_doy_offset") != 0:
        raise ValueError("Paper inputs require linear area scaling and the manuscript day-of-year coordinates.")
    if config["training"].get("env_dropout", 0) or config["training"].get("progressive_warmup_epochs", 0):
        raise ValueError("Paper training cannot silently add modality dropout or postpone the progressive loss.")
    if model["dropout"] != 0:
        raise ValueError("The equation-aligned profile uses dropout=0 for deterministic adjacent-prefix comparisons.")
    if config["min_prefix"] != 7:
        raise ValueError("Appendix A specifies a minimum observed prefix of seven days.")
    if config["max_days"] != 177 or config["operation"].get("require_post_peak", True):
        raise ValueError("Use the paper's 177-day maximum and unmodified three-low-day stopping rule.")

def read_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("area_file", "output_dir", "cloud_file", "alignment_file"):
        if config.get(key):
            value = Path(config[key])
            config[key] = str(value if value.is_absolute() else path.parent / value)
    config["env_files"] = [
        str(Path(p) if Path(p).is_absolute() else path.parent / p)
        for p in config.get("env_files", [])
    ]
    config["input_workbooks"] = [
        str(Path(p) if Path(p).is_absolute() else path.parent / p)
        for p in config.get("input_workbooks", [])
    ]
    partitions = [set(config[name]) for name in ("train_years", "val_years", "test_years")]
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Training, validation and test years must be disjoint.")
    if not partitions[0] or not partitions[2]:
        raise ValueError("Training and test year partitions must be nonempty.")
    if not partitions[1] and (not config["training"].get("allow_no_validation") or config.get("require_paper_data")):
        raise ValueError("Empty validation requires fixed-epoch training with allow_no_validation=true.")
    ordered = [years for years in partitions if years]
    if any(max(a) >= min(b) for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Use chronological train/validation/test partitions.")
    if not config.get("variables") or len(set(config["variables"])) != len(config["variables"]):
        raise ValueError("Environmental variable names must be nonempty and unique.")
    if config["model"]["d_model"] % config["model"]["heads"]:
        raise ValueError("d_model must be divisible by heads.")
    if config.get("max_days", 177) < 3:
        raise ValueError("max_days must be at least three.")
    if not 2 <= config["min_prefix"] < config["max_days"]:
        raise ValueError("min_prefix must be at least two and smaller than max_days.")
    if not config["eval_prefixes"] or any(n < config["min_prefix"] or n >= config["max_days"] for n in config["eval_prefixes"]):
        raise ValueError("Evaluation prefixes must be between min_prefix and max_days - 1.")
    if config["region_rows"] <= 0 or config["region_cols"] <= 0:
        raise ValueError("Region grid dimensions must be positive.")
    training = config["training"]
    if training["lambda_prog"] < 0 or training["gamma_km4"] < 0:
        raise ValueError("Progressive weight and tolerance must be non-negative.")
    if not 0 <= training.get("env_dropout", 0) <= 1:
        raise ValueError("env_dropout must be in [0,1].")
    if config.get("area_input_transform", "linear") not in ("linear", "log1p"):
        raise ValueError("area_input_transform must be linear or log1p.")
    if config.get("paper_structure"):
        validate_paper_structure(config)
    return config
