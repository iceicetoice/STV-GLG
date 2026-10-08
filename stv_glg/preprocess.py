from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import PAPER_VARIABLES
from .data import assign_regions


def read_netcdf(path: Path) -> xr.Dataset:
    # Passing a file handle also supports Chinese paths on Windows.
    with path.open("rb") as stream:
        magic = stream.read(8)
        stream.seek(0)
        engine = "h5netcdf" if magic.startswith(b"\x89HDF") else "scipy"
        with xr.open_dataset(stream, engine=engine) as dataset:
            return dataset.load()


def file_record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def surface_field(dataset: xr.Dataset, variable: str, year: int) -> xr.DataArray:
    field = dataset[variable]
    if "depth" in field.dims:
        depth = np.asarray(field.depth.values, dtype=float)
        if not np.isfinite(depth).all() or (depth < 0).any():
            raise ValueError("Invalid water-layer depth coordinate.")
        field = field.isel(depth=int(depth.argmin()), drop=True)
    if "valid_time" in field.dims:
        field = field.rename({"valid_time": "time"})
    field = field.sortby("time").sortby("latitude").sortby("longitude")
    times = pd.DatetimeIndex(field.time.values)
    if times.duplicated().any() or not (times.year == year).all():
        raise ValueError(f"{variable}: duplicate timestamps or data outside requested year {year}.")
    return field.assign_coords(latitude=np.round(field.latitude.values, 4),
                               longitude=np.round(field.longitude.values, 4))


def require_daily_mean(field: xr.DataArray):
    times = pd.DatetimeIndex(field.time.values)
    if times.normalize().duplicated().any() or not (times == times.normalize()).all():
        raise ValueError("Daily ocean fields require one midnight timestamp per day.")


def daily_wind(u: xr.DataArray, v: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    u, v = xr.align(u, v, join="exact")
    times = pd.DatetimeIndex(u.time.values)
    if times.duplicated().any() or not (times == times.floor("h")).all():
        raise ValueError("Wind timestamps must be unique whole hours.")
    joint = np.isfinite(u) & np.isfinite(v)
    complete = joint.resample(time="1D").sum() == 24
    speed = np.hypot(u, v).resample(time="1D").mean().where(complete)
    um, vm = u.where(joint).resample(time="1D").mean(), v.where(joint).resample(time="1D").mean()
    direction = (180 + np.rad2deg(np.arctan2(um, vm))) % 360
    direction = direction.where(complete & (np.hypot(um, vm) > 1e-6))
    return speed, direction


def daily_accumulation(field: xr.DataArray, variable: str) -> xr.DataArray:
    """ERA5 hourly accumulations ending at valid time, grouped by their start date."""
    if variable not in ("ssrd", "tp"):
        raise ValueError("Use ssrd for downward solar radiation; net radiation is not interchangeable.")
    if field.attrs.get("GRIB_stepType") != "accum":
        raise ValueError("Expected ERA5 hourly accumulation metadata.")
    times = pd.DatetimeIndex(field.time.values)
    if times.duplicated().any() or not (times == times.floor("h")).all():
        raise ValueError("Accumulation timestamps must be unique whole hours.")
    unit = str(field.attrs.get("units", ""))
    if (variable == "tp" and unit != "m") or (variable == "ssrd" and unit not in ("J m**-2", "J m-2")):
        raise ValueError(f"Unexpected {variable} units: {unit}")
    shifted = field.assign_coords(time=times - pd.Timedelta(hours=1))
    complete = shifted.notnull().resample(time="1D").sum() == 24
    total = shifted.resample(time="1D").sum().where(complete)
    return total / 86400 if variable == "ssrd" else total * 1000


def remap(field: xr.DataArray, latitude: np.ndarray, longitude: np.ndarray) -> xr.DataArray:
    result = field.reindex(latitude=latitude, longitude=longitude, method="nearest", tolerance=0.06)
    inside = (result.latitude >= float(field.latitude.min()) - 1e-4)
    inside &= result.latitude <= float(field.latitude.max()) + 1e-4
    inside_lon = (result.longitude >= float(field.longitude.min()) - 1e-4)
    inside_lon &= result.longitude <= float(field.longitude.max()) + 1e-4
    return result.where(inside & inside_lon)


def prepare_local(config_path: str | Path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("phy_root", "bio_root", "wind_root", "output_dir", "paper_file"):
        p = Path(config[key])
        config[key] = p if p.is_absolute() else (config_path.parent / p).resolve()
    output = config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "paper_alignment.json").exists():
        raise FileExistsError(f"{output} already contains a prepared archive; choose a new output directory.")
    domain = config["domain"]
    lat = np.arange(domain["lat_min"], domain["lat_max"] + 0.001, 0.25)
    lon = np.arange(domain["lon_min"], domain["lon_max"] + 0.001, 0.25)
    sources, checks, coverage, csv_checks = [], [], [], []
    products = {}

    def read(path, variable, year, register=True):
        dataset = read_netcdf(path)
        field = surface_field(dataset, variable, year)
        attrs = dataset[variable].attrs
        depth = float(dataset.depth.min()) if "depth" in dataset else None
        sources.append({**file_record(path), "year": year, "variable": variable,
                        "units": attrs.get("units"), "long_name": attrs.get("long_name"),
                        "product": dataset.attrs.get("subset:productId", "ERA5"),
                        "dataset": dataset.attrs.get("subset:datasetId"), "sample_depth_m": depth,
                        "first_time": str(field.time.values.min()), "last_time": str(field.time.values.max()),
                        "latitude_min": float(field.latitude.min()), "latitude_max": float(field.latitude.max()),
                        "longitude_min": float(field.longitude.min()), "longitude_max": float(field.longitude.max())})
        if register:
            products[year, variable] = sources[-1]["product"]
        return field

    for year in config["years"]:
        phy = config["phy_root"] / str(year)
        bio = config["bio_root"] / str(year)
        sst = read(phy / "Shoal_tem_0.083deg.nc", "thetao", year)
        sal = read(phy / "Shoal_sal_0.083deg.nc", "so", year)
        u = read(phy / "Shoal_u_0.083deg.nc", "uo", year)
        v = read(phy / "Shoal_v_0.083deg.nc", "vo", year)
        for field in (sst, sal, u, v):
            require_daily_mean(field)
        for field, accepted in [(sst, {"degrees_C"}), (sal, {"1e-3"}), (u, {"m s-1"}), (v, {"m s-1"})]:
            if field.attrs.get("units") not in accepted:
                raise ValueError(f"Unexpected ocean units: {field.attrs}")
        u, v = xr.align(u, v, join="exact")
        fields = {"sst": sst, "sal": sal, "cur": np.hypot(u, v)}
        original_csv = pd.read_csv(bio / f"{year}_shoal_bio_data_ori.csv")
        original_csv.Time = pd.to_datetime(original_csv.Time)
        for variable, filename in [("po4", "phosphate"), ("no3", "nitrate"), ("si", "silicate")]:
            field = read(bio / f"Shoal_{filename}_0.25deg.nc", variable, year)
            require_daily_mean(field)
            if field.attrs.get("units") != "mmol m-3":
                raise ValueError(f"Unexpected nutrient units: {field.attrs}")
            fields[variable] = field
            raw = field.to_dataframe(name="raw").reset_index().rename(
                columns={"time": "Time", "latitude": "Latitude", "longitude": "Longitude"})
            compared = original_csv.merge(raw, on=["Time", "Latitude", "Longitude"], how="left")
            valid = compared[variable].notna()
            differences = np.abs(compared.loc[valid, variable] - compared.loc[valid, "raw"])
            if compared.loc[valid, "raw"].isna().any() or (differences > 1e-5).any():
                raise ValueError(f"{year} {variable}: raw NetCDF does not reproduce the supplied CSV.")
            csv_checks.append({"year": year, "variable": variable, "compared_rows": int(valid.sum()),
                               "max_absolute_difference": float(differences.max())})
        wind_path = config["wind_root"] / f"{year}_wind_era.nc"
        wu, wv = read(wind_path, "u10", year), read(wind_path, "v10", year)
        if wu.attrs.get("units") != "m s**-1" or wv.attrs.get("units") != "m s**-1":
            raise ValueError("Unexpected ERA5 wind units.")
        fields["ws"], fields["wd"] = daily_wind(wu, wv)
        mask_path = config["wind_root"] / f"surfacecurrent_{year}_CMEMS.nc"
        mask_field = read(mask_path, "uo", year, register=False)
        ocean = remap(mask_field.notnull().any("time").astype(float), lat, lon).fillna(0) > 0.5
        remapped = {key: remap(value, lat, lon).where(ocean) for key, value in fields.items()}
        dataset = xr.Dataset(remapped)
        # Empty slots retain the manuscript's input schema without invented observations.
        for variable in ("rad", "pre"):
            dataset[variable] = xr.full_like(dataset.sst, np.nan)
        dataset["is_ocean"] = ocean.astype(int)
        frame = dataset.to_dataframe().reset_index().rename(
            columns={"time": "Time", "latitude": "Latitude", "longitude": "Longitude"})
        frame = frame.loc[frame.is_ocean.eq(1)].copy()
        frame["region"] = assign_regions(frame.Latitude.to_numpy(), frame.Longitude.to_numpy(), config)
        for variable in PAPER_VARIABLES:
            valid = frame[variable].notna()
            expected = "GLOBAL_MULTIYEAR_PHY_001_030" if variable in ("sst", "sal", "cur") else (
                "GLOBAL_MULTIYEAR_BGC_001_029" if variable in ("po4", "no3", "si") else "ERA5")
            raw_name = {"sst": "thetao", "sal": "so", "cur": "uo", "ws": "u10", "wd": "v10"}.get(variable, variable)
            actual_product = products.get((year, raw_name)) if valid.any() else None
            units = {"sst": "degC", "sal": "1e-3", "cur": "m/s", "po4": "mmol/m3", "no3": "mmol/m3",
                     "si": "mmol/m3", "rad": "W/m2", "pre": "mm/day", "ws": "m/s", "wd": "degree"}[variable]
            checks.append({"year": year, "variable": variable, "units": units, "used": bool(valid.any()),
                           "expected_product": expected, "actual_product": actual_product,
                           "product_matches": bool(actual_product == expected),
                           "regions_available": ",".join(str(r + 1) for r in sorted(frame.loc[valid, "region"].unique())),
                           "status": "missing; masked" if not valid.any() else (
                               "definition and product match; coverage incomplete" if actual_product == expected
                               else "definition matches; product and coverage differ")})
            for region in range(8):
                sub = frame.loc[valid & frame.region.eq(region)]
                coverage.append({"year": year, "region": region + 1, "variable": variable,
                                 "valid_grid_days": len(sub), "valid_days": sub.Time.nunique(),
                                 "grid_cells": len(sub[["Latitude", "Longitude"]].drop_duplicates()),
                                 "first_date": str(sub.Time.min().date()) if len(sub) else None,
                                 "last_date": str(sub.Time.max().date()) if len(sub) else None})
        frame["Time"] = frame.Time.dt.strftime("%Y-%m-%d")
        frame[["Time", "Latitude", "Longitude", "is_ocean"] + PAPER_VARIABLES].to_csv(
            output / f"environment_{year}.csv", index=False, encoding="utf-8-sig")

    excluded = []
    for p in config["phy_root"].rglob("data_stream-oper_stepType-*.nc"):
        if "instant" in p.name:
            continue
        dataset = read_netcdf(p)
        for variable in dataset.data_vars:
            field = dataset[variable]
            t = pd.DatetimeIndex(dataset.valid_time.values if "valid_time" in dataset else dataset.time.values)
            counts = pd.Series(1, index=t).groupby(t.normalize()).sum()
            excluded.append({**file_record(p), "variable": variable, "long_name": field.attrs.get("long_name"),
                             "units": field.attrs.get("units"), "hours_per_day_min": int(counts.min()),
                             "hours_per_day_max": int(counts.max()),
                             "reason": "Net shortwave radiation is not ssrd." if variable == "avg_snswrf"
                             else "One hourly accumulation per day cannot provide daily total precipitation."})
    limitations = [
        "Archive covers 2022-2024 only, not all years required by the manuscript.",
        "RAD is unavailable: avg_snswrf is net shortwave flux, not downward ssrd; RAD is masked.",
        "PRE is unavailable: local tp files contain only 00:00 hourly accumulations; PRE is masked.",
        "Daily MODIS cloud confidence is unavailable for these years and is masked.",
        "Physical fields cover only 33-35 N / 120-122 E; 2024 uses ANALYSISFORECAST_PHY_001_024, not MULTIYEAR_PHY_001_030.",
        "Nutrients use ANALYSISFORECAST_BGC_001_028 at 0.494025 m, not MULTIYEAR_BGC_001_029 at approximately 0.506 m.",
        "Original nutrient CSV depth is bathymetry, not water sampling depth; raw nutrient values match the CSV.",
        "Hourly ERA5 wind covers all eight regions only from June through July; missing dates remain masked.",
    ]
    audit = {"fully_aligned": False, "paper": file_record(config["paper_file"]),
             "variables": PAPER_VARIABLES, "years": config["years"], "limitations": limitations,
             "variable_checks": checks, "sources": sources, "excluded_sources": excluded,
             "biochemical_csv_checks": csv_checks,
             "spatial_processing": "Nearest-neighbour remapping to 0.25 degree; no spatial extrapolation; valid ocean-current mask before cosine-weighted regional averaging.",
             "time_processing": "UTC days; only complete 24-hour wind days; WS=mean(hypot(u10,v10)); WD from mean u10/v10."}
    (output / "paper_alignment.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    with pd.ExcelWriter(output / "paper_alignment.xlsx", engine="openpyxl") as writer:
        pd.DataFrame(checks).to_excel(writer, sheet_name="variable_alignment", index=False)
        pd.DataFrame(coverage).to_excel(writer, sheet_name="year_region_coverage", index=False)
        pd.DataFrame(sources).to_excel(writer, sheet_name="raw_sources", index=False)
        pd.DataFrame(excluded).to_excel(writer, sheet_name="excluded_sources", index=False)
        pd.DataFrame(csv_checks).to_excel(writer, sheet_name="bio_csv_consistency", index=False)
        pd.DataFrame({"limitations": limitations}).to_excel(writer, sheet_name="limitations", index=False)
    print(json.dumps({"output_dir": str(output), "active_variables": [v for v in PAPER_VARIABLES if v not in ("rad", "pre")],
                      "fully_aligned": False, "limitations": limitations}, ensure_ascii=False, indent=2))
