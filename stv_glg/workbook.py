from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Season, calendar_features

SCHEMA_VERSION = "stv-glg-daily-v1"


def load_training_workbooks(config: dict) -> tuple[dict[int, Season], dict]:
    requested = set(config["train_years"] + config["val_years"] + config["test_years"])
    seasons, reports, sources, limitations = {}, [], [], set()
    maximum = config["max_days"]
    regions = config["region_rows"] * config["region_cols"]
    variables = config["variables"]
    for path in config["input_workbooks"]:
        with pd.ExcelFile(path, engine="openpyxl") as book:
            metadata = dict(book.parse("metadata", keep_default_na=False)[["key", "value"]].itertuples(index=False, name=None))
            if metadata["schema_version"] != SCHEMA_VERSION:
                raise ValueError(f"{path}: unsupported workbook schema.")
            year = int(metadata["year"])
            if year not in requested:
                continue
            if year in seasons:
                raise ValueError(f"Duplicate workbook for year {year}.")
            if int(metadata["max_days"]) != maximum or json.loads(metadata["variables"]) != variables:
                raise ValueError("Workbook dimensions or variable ordering differ from the model configuration.")
            if json.loads(metadata["domain"]) != config["domain"]:
                raise ValueError("Workbook spatial domain differs from the configured domain.")
            daily = book.parse("area_daily")
            regional = book.parse("environment_daily")
            original = book.parse("area_observations")
            boundaries = book.parse("regions")
            expected_ids = np.arange(1, regions + 1)
            if not np.array_equal(boundaries.region_id, expected_ids):
                raise ValueError("Region identifiers must be consecutive and north-to-south, west-to-east.")
            if int(metadata["region_rows"]) != config["region_rows"] or int(metadata["region_cols"]) != config["region_cols"]:
                raise ValueError("Workbook region partition differs from the configured partition.")
            domain = config["domain"]
            dy = (domain["lat_max"] - domain["lat_min"]) / config["region_rows"]
            dx = (domain["lon_max"] - domain["lon_min"]) / config["region_cols"]
            for row in boundaries.itertuples(index=False):
                r, c = divmod(row.region_id - 1, config["region_cols"])
                expected = [domain["lat_max"] - (r + 1) * dy, domain["lat_max"] - r * dy,
                            domain["lon_min"] + c * dx, domain["lon_min"] + (c + 1) * dx]
                if not np.allclose([row.lat_min, row.lat_max, row.lon_min, row.lon_max], expected):
                    raise ValueError("Workbook region boundaries differ from the equal rectangular partition.")
            limitations.update(json.loads(metadata["limitations"]))
        daily["date"] = pd.to_datetime(daily.date, errors="raise").dt.normalize()
        calendar = pd.DatetimeIndex(daily.date)
        if len(calendar) != maximum or not calendar.equals(pd.date_range(calendar[0], periods=maximum)):
            raise ValueError(f"{year}: area_daily must contain exactly max_days consecutive dates.")
        original["date"] = pd.to_datetime(original.date, errors="raise").dt.normalize()
        values = pd.to_numeric(original.unmixed_area_km2, errors="raise")
        if original.date.duplicated().any() or not original.date.isin(calendar).all():
            raise ValueError("Area observations must have unique dates inside the seasonal interval.")
        if not (original.date.dt.year == year).all() or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Area observations must be finite, non-negative and belong to the workbook year.")
        labels = pd.Series(values.to_numpy(), index=original.date).reindex(calendar)
        if not original.date.is_monotonic_increasing or original.date.min() != calendar[0] or not labels.notna().any():
            raise ValueError("The seasonal interval must begin at the first supplied area observation.")
        observed = labels.notna().to_numpy()
        filled = labels.interpolate(method="linear", limit_area="inside")
        target_mask = filled.notna().to_numpy()
        target = filled.fillna(0).to_numpy(np.float32)
        for column, expected in [("observed_area_km2", labels), ("target_area_km2", filled)]:
            stored = pd.to_numeric(daily[column], errors="raise").to_numpy(float)
            if not np.allclose(stored, expected.to_numpy(float), equal_nan=True, rtol=1e-7, atol=1e-6):
                raise ValueError(f"{year}: {column} differs from the original observations and specified interpolation.")
        for column, expected in [("is_observed", observed), ("target_valid", target_mask)]:
            if not np.array_equal(_binary(daily[column], column), expected):
                raise ValueError(f"{year}: invalid {column} mask.")
        cloud = pd.to_numeric(daily.cloud_confidence, errors="raise").to_numpy(float)
        cloud_mask = np.isfinite(cloud)
        if not np.array_equal(_binary(daily.cloud_valid, "cloud_valid"), cloud_mask):
            raise ValueError("Cloud confidence mask does not match the available values.")
        if ((cloud[cloud_mask] < 0) | (cloud[cloud_mask] > 1)).any():
            raise ValueError("Cloud confidence must lie within [0,1].")
        regional["date"] = pd.to_datetime(regional.date, errors="raise").dt.normalize()
        index = pd.MultiIndex.from_product([calendar, expected_ids], names=["date", "region_id"])
        if regional.duplicated(["date", "region_id"]).any() or len(regional) != len(index):
            raise ValueError("Environmental dates and region identifiers must form a complete unique grid.")
        regional = regional.set_index(["date", "region_id"])
        if not index.isin(regional.index).all():
            raise ValueError("Environmental date-region keys do not match area_daily.")
        regional = regional.reindex(index)
        env = regional[variables].apply(pd.to_numeric, errors="raise").to_numpy(float)
        if np.isinf(env).any():
            raise ValueError("Infinite environmental values are not allowed.")
        env_mask = np.isfinite(env)
        stored_masks = np.column_stack([_binary(regional[f"{v}_valid"], f"{v}_valid") for v in variables])
        if not np.array_equal(stored_masks, env_mask):
            raise ValueError("Environmental masks do not match the available values.")
        shape = (maximum, regions, len(variables))
        env = np.where(env_mask, env, 0).astype(np.float32).reshape(shape)
        env_mask = env_mask.reshape(shape)
        last_seen = pd.Series(np.where(observed, np.arange(maximum), np.nan)).ffill()
        age = (np.arange(maximum) - last_seen.to_numpy()).astype(np.float32)
        seasons[year] = Season(year, calendar, target, target_mask, observed,
                               labels.ffill().fillna(0).to_numpy(np.float32), age,
                               np.where(cloud_mask, cloud, 0).astype(np.float32), cloud_mask,
                               env, env_mask, calendar_features(calendar, maximum, config.get("calendar_doy_offset", 0)),
                               original)
        reports.append({"year": year, "start": str(calendar[0].date()),
                        "last_observation": str(original.date.max().date()), "source_rows": len(original),
                        "blank_area_rows": 0, "same_date_extra_rows_averaged": 0,
                        "observed_area_dates": int(observed.sum()), "calendar_span_days": int(target_mask.sum()),
                        "interpolated_area_dates": int(target_mask.sum() - observed.sum()),
                        "area_max_km2": float(values.max()), "cloud_observations": int(cloud_mask.sum()),
                        "environmental_values_available": int(env_mask.sum())})
        sources.append({"file": Path(path).name, "format": SCHEMA_VERSION, "explicit_sea_mask": True})
    if requested != set(seasons):
        raise ValueError(f"No training workbook supplied for years {sorted(requested - set(seasons))}.")
    return seasons, {"environment_sources": sources, "years": sorted(reports, key=lambda x: x["year"]),
                     "target_column": "unmixed_area_km2", "warnings": [],
                     "dataset_limitations": sorted(limitations)}


def _binary(series: pd.Series, name: str) -> np.ndarray:
    if not series.isin([0, 1, False, True]).all():
        raise ValueError(f"{name} must contain only boolean or 0/1 values.")
    return series.to_numpy(bool)
