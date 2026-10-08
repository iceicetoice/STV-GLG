from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import torch


@dataclass
class Season:
    year: int
    dates: pd.DatetimeIndex
    target: np.ndarray
    target_mask: np.ndarray
    observed_mask: np.ndarray
    area: np.ndarray
    area_age: np.ndarray
    cloud: np.ndarray
    cloud_mask: np.ndarray
    env: np.ndarray
    env_mask: np.ndarray
    calendar: np.ndarray
    original: pd.DataFrame


def calendar_features(dates: pd.DatetimeIndex, max_days: int, doy_offset: int = 1) -> np.ndarray:
    period = np.where(dates.is_leap_year, 366.0, 365.0)
    angle = 2 * np.pi * (dates.dayofyear.to_numpy() - doy_offset) / period
    stage = np.arange(len(dates), dtype=np.float32) / (max_days - 1)
    return np.column_stack((np.sin(angle), np.cos(angle), stage)).astype(np.float32)


def assign_regions(lat: np.ndarray, lon: np.ndarray, config: dict) -> np.ndarray:
    domain = config["domain"]
    rows, cols = config["region_rows"], config["region_cols"]
    dy = (domain["lat_max"] - domain["lat_min"]) / rows
    dx = (domain["lon_max"] - domain["lon_min"]) / cols
    valid = (lat >= domain["lat_min"]) & (lat <= domain["lat_max"])
    valid &= (lon >= domain["lon_min"]) & (lon <= domain["lon_max"])
    south_row = np.floor((lat - domain["lat_min"]) / dy).astype(int).clip(0, rows - 1)
    column = np.floor((lon - domain["lon_min"]) / dx).astype(int).clip(0, cols - 1)
    return np.where(valid, (rows - 1 - south_row) * cols + column, -1)


def load_environment(config: dict) -> tuple[pd.DataFrame, list[dict]]:
    variables = config["variables"]
    frames, reports = [], []
    for path in config.get("env_files", []):
        frame = pd.read_csv(path)
        required = {"Time", "Latitude", "Longitude"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path}: required columns are {sorted(required)}")
        frame["Time"] = pd.to_datetime(frame["Time"], errors="raise").dt.normalize()
        for col in ["Latitude", "Longitude"] + [v for v in variables if v in frame]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["Latitude", "Longitude"])
        depth_range = None
        if "depth" in frame:
            frame["depth"] = pd.to_numeric(frame["depth"], errors="raise")
            depth_range = [float(frame.depth.min()), float(frame.depth.max())]
        if "sample_depth_m" in frame:
            frame["sample_depth_m"] = pd.to_numeric(frame["sample_depth_m"], errors="raise")
            frame = frame.sort_values("sample_depth_m").drop_duplicates(["Time", "Latitude", "Longitude"])
        if "is_ocean" in frame:
            ocean = pd.to_numeric(frame["is_ocean"], errors="raise")
            if not ocean.isin([0, 1]).all():
                raise ValueError("is_ocean must contain only 0/1 values.")
            frame = frame.loc[ocean.eq(1)].copy()
        frame["region"] = assign_regions(frame.Latitude.to_numpy(), frame.Longitude.to_numpy(), config)
        frame = frame.loc[frame.region.ge(0)].copy()
        if frame.empty:
            raise ValueError(f"{path}: no ocean samples inside the configured domain.")
        available = [v for v in variables if v in frame]
        if not available:
            raise ValueError(f"{path}: none of {variables} found.")
        reports.append({
            "file": str(Path(path).resolve()), "rows_in_domain": len(frame),
            "first_date": str(frame.Time.min().date()), "last_date": str(frame.Time.max().date()),
            "variables": available, "regions_present": sorted((frame.region.unique() + 1).tolist()),
            "depth_range_m_as_supplied": depth_range,
            "depth_interpretation": "Unspecified depth column; not assumed to be water sampling depth." if depth_range else None,
            "sample_depth_m": float(frame.sample_depth_m.min()) if "sample_depth_m" in frame else None,
            "explicit_sea_mask": "is_ocean" in frame,
        })
        frames.append(frame[["Time", "Latitude", "Longitude", "region"] + available])
    if not frames:
        return pd.DataFrame(columns=["Time", "region"] + variables), reports
    merged = pd.concat(frames, ignore_index=True)
    for variable in variables:
        if variable not in merged:
            merged[variable] = np.nan
    # Complementary CSVs can share grid points, but conflicting duplicates are rejected.
    keys = ["Time", "Latitude", "Longitude", "region"]
    if merged.duplicated(keys).any():
        spread = merged.groupby(keys)[variables].agg(lambda x: x.max() - x.min())
        if (spread.fillna(0) > 1e-5).any().any():
            raise ValueError("Conflicting environmental records at the same date/grid point.")
        merged = merged.groupby(keys, as_index=False)[variables].mean()
    weights = np.cos(np.deg2rad(merged.Latitude.to_numpy()))
    aggregates = merged[["Time", "region"]].drop_duplicates().set_index(["Time", "region"])
    for variable in variables:
        valid = np.isfinite(merged[variable].to_numpy())
        temporary = merged[["Time", "region"]].copy()
        temporary["w"] = weights * valid
        if variable == "wd":
            # Meteorological directions must be averaged on the circle.
            radians = np.deg2rad(merged[variable].fillna(0).to_numpy())
            temporary["s"] = np.sin(radians) * temporary.w
            temporary["c"] = np.cos(radians) * temporary.w
            sums = temporary.groupby(["Time", "region"])[["w", "s", "c"]].sum()
            value = np.rad2deg(np.arctan2(sums.s, sums.c)) % 360
            value = value.where((sums.w > 0) & (np.hypot(sums.s, sums.c) > 1e-8))
        else:
            temporary["v"] = merged[variable].fillna(0).to_numpy() * temporary.w
            sums = temporary.groupby(["Time", "region"])[["w", "v"]].sum()
            value = sums.v / sums.w.replace(0, np.nan)
        aggregates[variable] = value
    return aggregates.reset_index(), reports


def load_seasons(config: dict) -> tuple[dict[int, Season], dict]:
    if config.get("input_workbooks"):
        from .workbook import load_training_workbooks
        return load_training_workbooks(config)
    maximum = config["max_days"]
    k, f = config["region_rows"] * config["region_cols"], len(config["variables"])
    environment, env_reports = load_environment(config)
    cloud = None
    if config.get("cloud_file"):
        cloud = pd.read_csv(config["cloud_file"])
        cloud["Time"] = pd.to_datetime(cloud["Time"], errors="raise").dt.normalize()
        cloud["confidence"] = pd.to_numeric(cloud["confidence"], errors="raise")
        if not cloud.confidence.between(0, 1).all() or cloud.Time.duplicated().any():
            raise ValueError("Cloud CSV requires unique Time and confidence in [0, 1].")
        cloud = cloud.set_index("Time").confidence
    book = openpyxl.load_workbook(config["area_file"], data_only=True, read_only=True)
    requested = sorted(set(config["train_years"] + config["val_years"] + config["test_years"]))
    seasons, annual_reports = {}, []
    try:
        for year in requested:
            sheet = book[str(year)]
            rows = list(sheet.values)
            original = pd.DataFrame(rows[1:], columns=rows[0]).dropna(how="all")
            if config["target_column"] not in original:
                raise ValueError(f"Missing target column {config['target_column']}; found {list(original)}")
            dates = pd.to_datetime(original[config["date_column"]], errors="raise").dt.normalize()
            values = pd.to_numeric(original[config["target_column"]], errors="raise")
            if not (dates.dt.year == year).all():
                raise ValueError(f"{year}: dates must belong to the sheet year.")
            present = values.notna()
            if not present.any() or not np.isfinite(values[present]).all() or (values[present] < 0).any():
                raise ValueError(f"{year}: observed areas must be finite and non-negative, with at least one observation.")
            series = pd.Series(values[present].to_numpy(), index=pd.DatetimeIndex(dates[present])).groupby(level=0).mean().sort_index()
            if (series.index[-1] - series.index[0]).days >= maximum:
                raise ValueError(f"{year}: observations exceed max_days={maximum}.")
            calendar = pd.date_range(series.index[0], periods=maximum, freq="D")
            labels = series.reindex(calendar)
            observed_mask = labels.notna().to_numpy()
            filled = labels.interpolate(method="linear", limit_area="inside")
            target_mask = filled.notna().to_numpy()
            target = filled.fillna(0).to_numpy(dtype=np.float32)
            area = labels.ffill().fillna(0).to_numpy(dtype=np.float32)
            last_seen = pd.Series(np.where(observed_mask, np.arange(maximum), np.nan)).ffill()
            age = (np.arange(maximum) - last_seen.to_numpy()).astype(np.float32)
            c = cloud.reindex(calendar) if cloud is not None else pd.Series(np.nan, index=calendar)
            c_mask = c.notna().to_numpy()
            env = np.full((maximum, k, f), np.nan, dtype=np.float32)
            for region in range(k):
                regional = environment.loc[environment.region.eq(region)]
                if not regional.empty:
                    aligned = regional.set_index("Time")[config["variables"]].reindex(calendar)
                    aligned = aligned.ffill(limit=config["env_max_carry_days"] or None) if config["env_max_carry_days"] else aligned
                    env[:, region] = aligned.to_numpy(dtype=np.float32)
            env_mask = np.isfinite(env)
            seasons[year] = Season(
                year, calendar, target, target_mask, observed_mask, area, age,
                c.fillna(0).to_numpy(dtype=np.float32), c_mask,
                np.where(env_mask, env, 0), env_mask,
                calendar_features(calendar, maximum, config.get("calendar_doy_offset", 1)), original,
            )
            annual_reports.append({
                "year": year, "start": str(calendar[0].date()), "last_observation": str(series.index[-1].date()),
                "source_rows": len(original), "blank_area_rows": int((~present).sum()),
                "same_date_extra_rows_averaged": int(present.sum() - len(series)),
                "observed_area_dates": int(observed_mask.sum()), "calendar_span_days": int((series.index[-1] - calendar[0]).days + 1),
                "interpolated_area_dates": int(target_mask.sum() - observed_mask.sum()),
                "area_max_km2": float(values.max()), "cloud_observations": int(c_mask.sum()),
                "environmental_values_available": int(env_mask.sum()),
            })
    finally:
        book.close()
    return seasons, {"environment_sources": env_reports, "years": annual_reports,
                     "target_column": config["target_column"], "warnings": []}


@dataclass
class Normalizer:
    area_scale: float
    env_mean: np.ndarray
    env_std: np.ndarray
    env_active: np.ndarray
    cloud_active: bool
    glg_initial: list[float]
    env_clip: float | None = None

    @classmethod
    def fit(cls, seasons: dict[int, Season], years: list[int], variables: list[str]):
        training = [seasons[y] for y in years]
        areas = np.concatenate([s.target[s.target_mask] for s in training])
        scale = max(float(np.quantile(areas, 0.95)), 1.0)
        values = np.concatenate([s.env for s in training], axis=0)
        mask = np.concatenate([s.env_mask for s in training], axis=0)
        count = mask.sum((0, 1))
        mean = (values * mask).sum((0, 1)) / count.clip(1)
        variance = (((values - mean) ** 2) * mask).sum((0, 1)) / count.clip(1)
        std = np.sqrt(variance).clip(1e-5)
        for j, variable in enumerate(variables):
            if variable == "wd":
                mean[j], std[j] = 180, 180
        active = mask.sum(0) >= 3
        peaks, positions, widths = [], [], []
        for s in training:
            peaks.append(float(s.target.max()) / scale)
            positions.append(float(s.target.argmax()) / (len(s.target) - 1))
            high = np.flatnonzero(s.target_mask & (s.target >= s.target.max() / 2))
            widths.append(max((high[-1] - high[0]) / (2.355 * (len(s.target) - 1)), 0.025))
        sigma = float(np.median(widths))
        initial = [float(np.median(peaks)), float(np.median(positions)), sigma, sigma / 1.7, 1.0 / sigma]
        return cls(scale, mean.astype(np.float32), std.astype(np.float32), active,
                   any(s.cloud_mask.any() for s in training), initial)

    def to_dict(self) -> dict:
        return {"area_scale": self.area_scale, "env_mean": self.env_mean.tolist(),
                "env_std": self.env_std.tolist(), "env_active": self.env_active.tolist(),
                "cloud_active": self.cloud_active, "glg_initial": self.glg_initial, "env_clip": self.env_clip}

    @classmethod
    def from_dict(cls, state: dict):
        state = dict(state)
        for key, dtype in [("env_mean", np.float32), ("env_std", np.float32), ("env_active", bool)]:
            state[key] = np.asarray(state[key], dtype=dtype)
        return cls(**state)


def make_batch(seasons: list[Season], n: int, normalizer: Normalizer, device: torch.device) -> dict:
    if n < 1 or n >= len(seasons[0].dates):
        raise ValueError("Prefix length must be in [1, max_days - 1].")
    arrays = {
        "area": np.stack([historical_area(s, n) / normalizer.area_scale for s in seasons]),
        "area_mask": np.stack([s.observed_mask[:n] for s in seasons]),
        "area_age": np.stack([s.area_age[:n].clip(0, 30) / 30 for s in seasons]),
        "cloud": np.stack([s.cloud[:n] for s in seasons]) * normalizer.cloud_active,
        "cloud_mask": np.stack([s.cloud_mask[:n] for s in seasons]) & normalizer.cloud_active,
        "env": np.stack([(s.env[:n] - normalizer.env_mean) / normalizer.env_std for s in seasons]),
        "env_mask": np.stack([s.env_mask[:n] & normalizer.env_active for s in seasons]),
        "calendar": np.stack([s.calendar[:n] for s in seasons]),
        "future_calendar": np.stack([s.calendar[n:] for s in seasons]),
        "target": np.stack([s.target[n:] / normalizer.area_scale for s in seasons]),
        "target_mask": np.stack([s.target_mask[n:] for s in seasons]),
    }
    arrays["env"] = np.where(arrays["env_mask"], arrays["env"], 0)
    if normalizer.env_clip is not None:
        arrays["env"] = arrays["env"].clip(-normalizer.env_clip, normalizer.env_clip)
    return {key: torch.as_tensor(value, dtype=torch.bool if value.dtype == bool else torch.float32, device=device)
            for key, value in arrays.items()}


def historical_area(season: Season, n: int) -> np.ndarray:
    values = pd.Series(np.where(season.observed_mask[:n], season.target[:n], np.nan))
    return values.interpolate(method="linear", limit_area="inside").ffill().fillna(0).to_numpy(dtype=np.float32)


def preceding_batch(batch: dict, min_prefix: int = 1) -> dict:
    if batch["area"].shape[1] <= min_prefix:
        raise ValueError("Both adjacent prefixes must meet the minimum observed length.")
    previous = {k: v[:, :-1] for k, v in batch.items() if k not in ("future_calendar", "target", "target_mask")}
    area, mask = previous["area"], previous["area_mask"]
    positions = torch.arange(area.shape[1], device=area.device).expand_as(area)
    left = torch.where(mask, positions, 0).cummax(dim=1).values
    right = torch.where(mask, positions, area.shape[1]).flip(1).cummin(dim=1).values.flip(1)
    bounded_right = right.clamp_max(area.shape[1] - 1)
    left_value, right_value = area.gather(1, left), area.gather(1, bounded_right)
    fraction = (positions - left).float() / (right - left).clamp_min(1).float()
    interpolated = left_value + fraction * (right_value - left_value)
    previous["area"] = torch.where(right < area.shape[1], interpolated, left_value)
    previous["future_calendar"] = torch.cat((batch["calendar"][:, -1:], batch["future_calendar"]), dim=1)
    return previous


class PrefixSampler:
    def __init__(self, seasons: dict[int, Season], years: list[int], config: dict, seed: int):
        self.seasons = [seasons[y] for y in years]
        self.rng = np.random.default_rng(seed)
        self.pools = {}
        start = config["min_prefix"] + int(config["training"].get("lambda_prog", 0) > 0)
        for n in range(start, config["max_days"]):
            eligible = [s for s in self.seasons if s.target_mask[n:].sum() >= config["training"]["min_future_labels"]]
            if eligible:
                self.pools[n] = eligible
        if not self.pools:
            raise ValueError("No valid training prefixes with future labels.")

    def sample(self, size: int) -> tuple[list[Season], int]:
        n = int(self.rng.choice(list(self.pools)))
        pool = self.pools[n]
        indices = self.rng.choice(len(pool), size=size, replace=True)
        return [pool[i] for i in indices], n


def provenance(config: dict) -> list[dict]:
    paths = config.get("input_workbooks") or [config["area_file"]] + config.get("env_files", [])
    if config.get("cloud_file"):
        paths.append(config["cloud_file"])
    result = []
    for path in paths:
        p = Path(path)
        digest = hashlib.sha256()
        with p.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result.append({"path": str(p.resolve()), "bytes": p.stat().st_size, "sha256": digest.hexdigest()})
    return result
