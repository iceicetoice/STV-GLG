from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import Normalizer, PrefixSampler, load_seasons, make_batch, preceding_batch, provenance
from .evaluation import evaluate, export_evaluation
from .loss import progressive_objective
from .model import STVGLGFormer


def set_seed(seed: int, cpu_threads: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def select_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu.")
    return torch.device(name)


def prepare(config: dict):
    seasons, audit = load_seasons(config)
    normalizer = Normalizer.fit(seasons, config["train_years"], config["variables"])
    normalizer.env_clip = None if config.get("paper_structure") else 8.0
    audit["normalization"] = normalizer.to_dict()
    audit["partitions"] = {name: config[name] for name in ("train_years", "val_years", "test_years")}
    audit["sources"] = provenance(config)
    audit["interpolation"] = "Linear interpolation between observations; historical interpolation uses only anchors at or before issue date. No labels beyond final observation."
    audit["warnings"] = []
    if config.get("alignment_file"):
        alignment = json.loads(Path(config["alignment_file"]).read_text(encoding="utf-8"))
        alignment["limitations"] = [x for x in alignment["limitations"]
            if not x.startswith("Pilot uses") and not x.startswith("Only one independent")]
        audit["paper_alignment"] = alignment
        audit["warnings"].extend(alignment["limitations"])
    audit["warnings"].extend(audit.get("dataset_limitations", []))
    audit["active_environment_variables"] = [v for j, v in enumerate(config["variables"]) if normalizer.env_active[:, j].any()]
    audit["inactive_environment_variables"] = [v for j, v in enumerate(config["variables"]) if not normalizer.env_active[:, j].any()]
    return seasons, normalizer, audit


def save_checkpoint(path: Path, model, optimizer, scheduler, normalizer, config, epoch, score):
    portable_config = dict(config)
    if config.get("input_workbooks"):
        portable_config["input_workbooks"] = [Path(p).name for p in config["input_workbooks"]]
        portable_config["area_file"] = None
        portable_config["alignment_file"] = None
        portable_config["output_dir"] = "."
    elif config.get("area_file") and not config.get("env_files") and not config.get("cloud_file"):
        portable_config["area_file"] = Path(config["area_file"]).name
        portable_config["alignment_file"] = None
        portable_config["output_dir"] = "."
    state = {"format_version": 2, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "normalizer": normalizer.to_dict(), "config": portable_config,
             "epoch": epoch, "validation_mse_normalized": score,
             "torch_version": str(torch.__version__)}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, device, data_dir=None):
    # Only load checkpoints produced by this project, never untrusted pickle files.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format_version") != 2:
        raise ValueError("Legacy checkpoints use a different pooling architecture. Retrain with the equation-aligned version.")
    if checkpoint["config"].get("input_workbooks"):
        supplied = checkpoint["config"]["input_workbooks"]
        root = Path(data_dir) if data_dir else Path(__file__).resolve().parents[1] / "data"
        checkpoint["config"]["input_workbooks"] = [str(root / Path(p).name) for p in supplied]
    elif checkpoint["config"].get("area_file"):
        area = Path(checkpoint["config"]["area_file"])
        if data_dir or not area.is_absolute():
            root = Path(data_dir) if data_dir else Path(__file__).resolve().parents[1] / "data"
            checkpoint["config"]["area_file"] = str(root / area.name)
    normalizer = Normalizer.from_dict(checkpoint["normalizer"])
    model = STVGLGFormer(checkpoint["config"], normalizer.glg_initial, normalizer.area_scale).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, normalizer, checkpoint["config"], checkpoint


def train(config: dict, device_name="auto"):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    output = Path(config["output_dir"])
    if (output / "best.pt").exists():
        raise FileExistsError(f"{output}/best.pt exists. Choose a new --run-dir to preserve the previous run.")
    output.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    set_seed(config["seed"], training["cpu_threads"])
    device = select_device(device_name)
    seasons, normalizer, audit = prepare(config)
    audit["device"] = str(device)
    audit["torch_version"] = str(torch.__version__)
    (output / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    if config.get("require_paper_data") and (
        not normalizer.cloud_active or not normalizer.env_active.all()
        or not audit.get("paper_alignment", {}).get("fully_aligned", False)
    ):
        raise ValueError("Required input coverage or product alignment failed. See data_audit.json.")
    model = STVGLGFormer(config, normalizer.glg_initial, normalizer.area_scale).to(device)
    sampler = PrefixSampler(seasons, config["train_years"], config, config["seed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=12, min_lr=1e-6)
    gamma = training["gamma_km4"] / normalizer.area_scale ** 2
    print(json.dumps({"event": "start", "device": str(device), "parameter_count": sum(p.numel() for p in model.parameters()),
                      "train_years": config["train_years"], "val_years": config["val_years"], "test_years": config["test_years"],
                      "target_column": config["target_column"], "warnings": audit["warnings"]}, ensure_ascii=False), flush=True)
    best_score, best_epoch, history = float("inf"), 0, []
    start = time.monotonic()
    warmup = min(training["progressive_warmup_epochs"], training["epochs"] - 1)
    for epoch in range(1, training["epochs"] + 1):
        model.train()
        records = []
        weight = training["lambda_prog"] if epoch > warmup else 0.0
        for _ in range(training["steps_per_epoch"]):
            sample, n = sampler.sample(training["batch_size"])
            batch = make_batch(sample, n, normalizer, device)
            drop_probability = training.get("env_dropout", 0)
            if drop_probability > 0:
                keep = torch.rand(len(sample), 1, 1, 1, device=device) >= drop_probability
                batch["env_mask"] = batch["env_mask"] & keep
            optimizer.zero_grad(set_to_none=True)
            previous = None
            if weight > 0:
                with torch.no_grad():
                    previous = model(preceding_batch(batch, config["min_prefix"]))
            current = model(batch)
            loss, record = progressive_objective(current, previous, batch, weight, gamma)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch={epoch}, prefix={n}.")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            records.append({**record, "grad_norm": float(norm)})
        val_mse, val_mae, score = None, None, None
        if config["val_years"]:
            _, validation, _ = evaluate(model, seasons, config["val_years"], normalizer, config, device)
            val_mse = float(validation.groupby("year").MSE_km4.mean().mean())
            val_mae = float(validation.MAE_km2.mean())
            score = val_mse / normalizer.area_scale ** 2
            scheduler.step(score)
        row = {"epoch": epoch, **pd.DataFrame(records).mean().to_dict(), "val_mse_km4": val_mse,
               "val_mae_km2": val_mae, "learning_rate": optimizer.param_groups[0]["lr"],
               "lambda_prog": weight, "elapsed_seconds": round(time.monotonic() - start, 2)}
        history.append(row)
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        # Select only by validation; never use test metrics for checkpoint selection.
        if config["val_years"] and score < best_score and (weight > 0 or training["lambda_prog"] == 0):
            best_score, best_epoch = score, epoch
            save_checkpoint(output / "best.pt", model, optimizer, scheduler, normalizer, config, epoch, score)
        save_checkpoint(output / "last.pt", model, optimizer, scheduler, normalizer, config, epoch, score)
        if not config["val_years"] and epoch == training["epochs"]:
            best_epoch = epoch
            save_checkpoint(output / "best.pt", model, optimizer, scheduler, normalizer, config, epoch, score)
        print(json.dumps(row), flush=True)
        if config["val_years"] and epoch - best_epoch >= training["patience"]:
            print(json.dumps({"event": "early_stopping", "best_epoch": best_epoch}), flush=True)
            break
    model, normalizer, _, _ = load_checkpoint(output / "best.pt", device)
    forecasts, by_prefix, updates = evaluate(model, seasons, config["test_years"], normalizer, config, device, include_updates=True)
    summary = export_evaluation(output / "test", forecasts, by_prefix, updates, seasons, config)
    (output / "run_summary.json").write_text(json.dumps({"best_epoch": best_epoch, "epochs_completed": epoch,
            "checkpoint_selection": "validation_mse" if config["val_years"] else "fixed_final_epoch_no_validation",
            "elapsed_seconds": round(time.monotonic() - start, 2), "test_metrics": json.loads(summary.to_json(orient="records"))},
            indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"Saved checkpoint and reports: {output.resolve()}", flush=True)
    return output
