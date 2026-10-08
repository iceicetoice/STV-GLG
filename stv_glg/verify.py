from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import make_batch, preceding_batch
from .evaluation import metrics
from .loss import progressive_objective
from .preprocess import file_record
from .training import load_checkpoint, prepare


def verify_run(run_dir: str | Path, data_dir=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    run = Path(run_dir).resolve()
    model, normalizer, config, checkpoint = load_checkpoint(run / "best.pt", torch.device("cpu"), data_dir)
    torch.set_num_threads(config["training"]["cpu_threads"])
    seasons, fitted, audit = prepare(config)
    assert normalizer.to_dict() == fitted.to_dict(), "Normalization must reproduce from training years only."
    year = config["test_years"][0]
    season = seasons[year]
    verified = []
    for n in config["eval_prefixes"]:
        batch = make_batch([season], n, normalizer, torch.device("cpu"))
        with torch.inference_mode():
            result = model(batch)
        prediction = result["prediction"][0].numpy() * normalizer.area_scale
        tau = result["tau"][0].numpy()
        assert prediction.shape == (config["max_days"] - n,)
        assert np.isfinite(prediction).all() and (prediction >= 0).all()
        assert (np.diff(tau) > 0).all() and np.isclose(tau[-1], 1)
        assert tau[0] > (n - 1) / (config["max_days"] - 1)
        difference = np.diff(prediction)
        signs = np.sign(difference[np.abs(difference) > max(prediction.max(), 1) * 1e-7])
        assert not (np.diff(signs) > 0).any(), "GLG must not develop a recession-to-growth reversal."
        assert torch.allclose(result["parameters"]["weights"].sum(1), torch.ones(1))
        assert torch.allclose(result["pool_weights"].sum(1), torch.ones(1, config["model"]["d_model"]), atol=1e-6)
        previous = preceding_batch(batch, config["min_prefix"])
        independent = make_batch([season], n - 1, normalizer, torch.device("cpu"))
        assert torch.allclose(previous["area"], independent["area"])
        assert torch.equal(previous["future_calendar"][:, 1:], batch["future_calendar"])
        for variable in ("rad", "pre"):
            assert not batch["env_mask"][..., config["variables"].index(variable)].any()
        changed = copy.deepcopy(season)
        changed.target[n:] += 10000
        changed.env[n:] += 10000
        changed.env_mask[n:] = True
        changed_batch = make_batch([changed], n, normalizer, torch.device("cpu"))
        with torch.inference_mode():
            changed_output = model(changed_batch)["prediction"]
        assert torch.equal(result["prediction"], changed_output), "Future measurements leaked into prediction."
        verified.append({"year": year, "prefix_days": n, "horizon": len(prediction),
                         "minimum_increment": float(np.diff(tau).min()), "nonnegative": True,
                         "unimodal": True, "future_input_invariance": True})

    batch = make_batch([season], 28, normalizer, torch.device("cpu"))
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        reference = model(preceding_batch(batch, config["min_prefix"]))
    current = model(batch)
    loss, _ = progressive_objective(current, reference, batch, config["training"]["lambda_prog"],
                                    config["training"]["gamma_km4"] / normalizer.area_scale ** 2)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    env_grad = float(model.env_embedding[0].weight.grad.abs().sum())
    assert env_grad > 0, "Available environmental measurements must participate in the network."
    max_device_difference = None
    if torch.cuda.is_available():
        gpu_model, _, _, _ = load_checkpoint(run / "best.pt", torch.device("cuda"), data_dir)
        gpu_batch = make_batch([season], 28, normalizer, torch.device("cuda"))
        with torch.inference_mode():
            gpu_prediction = gpu_model(gpu_batch)["prediction"].cpu().numpy()
        cpu_prediction = current["prediction"].detach().numpy()
        max_device_difference = float(np.abs(gpu_prediction - cpu_prediction).max() * normalizer.area_scale)
        assert np.allclose(cpu_prediction, gpu_prediction, rtol=1e-4, atol=1e-5)

    forecasts = pd.read_csv(run / "test" / "forecasts_long.csv")
    recomputed = metrics(forecasts.observed_area_km2, forecasts.predicted_area_km2)
    summary = pd.read_excel(run / "test" / "test_forecasts_and_metrics.xlsx", sheet_name="summary")
    stored = summary.loc[summary.aggregation.eq("pooled_year_prefix_date")].iloc[0]
    for key, value in recomputed.items():
        assert np.isclose(stored[key], value), f"Metric mismatch: {key}"
    alignment = audit.get("paper_alignment", {})
    originals = audit["sources"] + alignment.get("sources", [])
    if alignment.get("paper"):
        originals.append(alignment["paper"])
    unique = {item["path"]: item for item in originals}
    for path, record in unique.items():
        assert file_record(Path(path))["sha256"] == record["sha256"], f"Source changed: {path}"
    if config.get("input_workbooks"):
        manifest_path = Path(config["input_workbooks"][0]).parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest["years"]:
            path = manifest_path.parent / record["file"]
            assert file_record(path)["sha256"] == record["sha256"], f"Published dataset hash mismatch: {path.name}"
    history = pd.read_csv(run / "training_history.csv")
    if not config["val_years"]:
        assert checkpoint["epoch"] == config["training"]["epochs"]
        assert history.val_mse_km4.isna().all()
    # Refresh the audit with partition-independent source limitations; keep run device metadata.
    old_audit = json.loads((run / "data_audit.json").read_text(encoding="utf-8"))
    audit.update({key: old_audit[key] for key in ("device", "torch_version") if key in old_audit})
    (run / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    report = {"passed": True, "checkpoint_epoch": checkpoint["epoch"], "checks": verified,
              "environment_gradient_l1": env_grad, "maximum_cpu_cuda_difference_km2": max_device_difference,
              "metric_xlsx_matches_recalculation": True, "unchanged_source_files": len(unique),
              "epochs_with_active_progressive_penalty": int((history.progressive > 0).sum()),
              "active_variables": audit["active_environment_variables"],
              "inactive_variables": audit["inactive_environment_variables"], "pooled_metrics": recomputed}
    (run / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify a trusted local STV-GLGFormer training run")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir")
    args = parser.parse_args()
    verify_run(args.run_dir, args.data_dir)
