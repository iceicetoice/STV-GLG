from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import read_config
from .data import historical_area, load_seasons, make_batch
from .evaluation import disappearance_date, evaluate, export_evaluation, plot_prediction
from .training import load_checkpoint, prepare, select_device, train


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="STV-GLGFormer: audit, train, evaluate, predict")
    sub = parser.add_subparsers(dest="command", required=True)
    preprocessing = sub.add_parser("preprocess", help="Prepare verified daily fields from local raw NetCDF files")
    preprocessing.add_argument("--config", required=True, help="Configuration for your own raw NetCDF source directories")
    for name in ("audit", "train"):
        p = sub.add_parser(name)
        default = Path(__file__).resolve().parents[1] / "configs" / "reproduce_2022_2024.json"
        if not default.exists():
            default = Path(__file__).parent / "configs" / "pilot_two_years.json"
        p.add_argument("--config", default=str(default))
        p.add_argument("--run-dir")
        if name == "train":
            p.add_argument("--epochs", type=int)
            p.add_argument("--steps-per-epoch", type=int)
            p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    for name in ("evaluate", "predict"):
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        p.add_argument("--data-dir", help="Directory containing the published annual training XLSX files")
        if name == "predict":
            p.add_argument("--year", type=int, required=True)
            group = p.add_mutually_exclusive_group(required=True)
            group.add_argument("--prefix-days", type=int)
            group.add_argument("--as-of", help="YYYY-MM-DD; only observations through this date are inputs")
            p.add_argument("--area-file", help="Optional replacement workbook, with the saved column schema")
            p.add_argument("--env-file", action="append", help="Replacement environmental CSV; repeat for multiple files")
            p.add_argument("--cloud-file", help="Optional Time,confidence CSV")
    args = parser.parse_args()
    if args.command == "preprocess":
        from .preprocess import prepare_local
        prepare_local(args.config)
        return
    if args.command in ("audit", "train"):
        config = read_config(args.config)
        if args.run_dir:
            config["output_dir"] = str(Path(args.run_dir).resolve())
        if args.command == "audit":
            _, _, audit = prepare(config)
            output = Path(config["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            (output / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
            pd.DataFrame(audit["years"]).to_csv(output / "data_audit_years.csv", index=False, encoding="utf-8-sig")
            print(json.dumps(audit, indent=2, ensure_ascii=False))
        else:
            for arg, key in ((args.epochs, "epochs"), (args.steps_per_epoch, "steps_per_epoch")):
                if arg is not None:
                    if arg <= 0:
                        parser.error("Training counts must be positive.")
                    config["training"][key] = arg
            train(config, args.device)
        return
    device = select_device(args.device)
    model, normalizer, config, checkpoint = load_checkpoint(args.checkpoint, device, args.data_dir)
    torch.set_num_threads(config["training"]["cpu_threads"])
    if args.command == "evaluate":
        seasons, _ = load_seasons(config)
        forecasts, prefixes, updates = evaluate(model, seasons, config["test_years"], normalizer, config, device, include_updates=True)
        summary = export_evaluation(args.output, forecasts, prefixes, updates, seasons, config)
        print(summary.to_string(index=False))
        return
    if args.area_file:
        config["area_file"] = str(Path(args.area_file).resolve())
    if args.env_file:
        config["env_files"] = [str(Path(p).resolve()) for p in args.env_file]
    if args.cloud_file:
        config["cloud_file"] = str(Path(args.cloud_file).resolve())
    # A new inference year need not appear in the original training partitions.
    inference_config = {**config, "train_years": [], "val_years": [], "test_years": [args.year]}
    seasons, _ = load_seasons(inference_config)
    season = seasons[args.year]
    n = args.prefix_days
    if args.as_of:
        as_of = pd.Timestamp(args.as_of).normalize()
        n = (as_of - season.dates[0]).days + 1
    if not config["min_prefix"] <= n < config["max_days"]:
        parser.error(f"Prefix must be between {config['min_prefix']} and {config['max_days'] - 1} calendar days.")
    batch = make_batch([season], n, normalizer, device)
    with torch.inference_mode():
        result = model(batch)
    prediction = result["prediction"][0].cpu().numpy() * normalizer.area_scale
    dates = season.dates[n:]
    stop = disappearance_date(dates, prediction, float(batch["area"].max()) * normalizer.area_scale,
                              config["operation"]["area_threshold_km2"], config["operation"]["consecutive_days"],
                              config["operation"].get("require_post_peak", True))
    frame = pd.DataFrame({"date": dates, "lead_days": np.arange(1, len(dates) + 1),
                          "predicted_area_km2": prediction, "lifecycle_coordinate": result["tau"][0].cpu().numpy(),
                          "retained_operationally": dates <= stop if stop is not None else True})
    frame["reference_area_km2"] = np.where(season.target_mask[n:], season.target[n:], np.nan)
    frame["reference_is_original"] = season.observed_mask[n:]
    frame["date"] = frame.date.dt.strftime("%Y-%m-%d")
    parameters = {k: v[0].cpu().tolist() for k, v in result["parameters"].items()}
    parameters["amplitude_normalized"] = parameters.pop("amplitude")
    parameters["amplitude_km2"] = parameters["amplitude_normalized"] * normalizer.area_scale
    metadata = {"year": args.year, "prefix_days": n, "issue_date": str(season.dates[n - 1].date()),
                "checkpoint_epoch": checkpoint["epoch"], "candidate_horizon": len(dates),
                "disappearance_date": str(stop.date()) if stop is not None else None, "parameters": parameters}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "forecast.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(output / "forecast.xlsx", engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="complete_forecast", index=False)
        frame.loc[frame.retained_operationally].to_excel(writer, sheet_name="operational_forecast", index=False)
        pd.DataFrame({"date": season.dates[:n].strftime("%Y-%m-%d"), "input_area_km2": historical_area(season, n),
                      "original_observation": season.observed_mask[:n]}).to_excel(writer, sheet_name="historical_input", index=False)
    (output / "forecast_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_prediction(output / "forecast.png", season, n, prediction, stop)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
