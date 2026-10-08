import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from stv_glg.config import PAPER_VARIABLES, read_config, validate_paper_structure
from stv_glg.evaluation import disappearance_date
from stv_glg.model import STVGLGFormer
from stv_glg.preprocess import daily_accumulation, daily_wind, remap, surface_field


def config_path(local, release):
    root = Path(__file__).parents[1]
    path = root / "configs" / local
    return path if path.exists() else root / "configs" / release


def series(values, start="2024-06-01", freq="h", attrs=None):
    return xr.DataArray(values, dims="time", coords={"time": pd.date_range(start, periods=len(values), freq=freq)},
                        attrs=attrs)


def test_hourly_wind_daily_definitions_and_incomplete_days():
    u = series(np.r_[np.ones(24) * 3, np.ones(23) * 6])
    v = series(np.r_[np.ones(24) * 4, np.ones(23) * 8])
    speed, direction = daily_wind(u, v)
    assert speed.values[0] == pytest.approx(5)
    assert direction.values[0] == pytest.approx((180 + np.degrees(np.arctan2(3, 4))) % 360)
    assert np.isnan(speed.values[1]) and np.isnan(direction.values[1])


def test_wind_speed_is_not_magnitude_of_daily_mean():
    u = series(np.r_[np.ones(12) * 3, np.ones(12) * -3])
    v = series(np.zeros(24))
    speed, direction = daily_wind(u, v)
    assert speed.values[0] == pytest.approx(3)
    assert np.isnan(direction.values[0])


def test_era5_accumulation_end_time_and_no_single_hour_daily_substitution():
    tp = series(np.ones(24) * 0.001, start="2024-06-01 01:00", attrs={"GRIB_stepType": "accum", "units": "m"})
    rain = daily_accumulation(tp, "tp")
    assert pd.Timestamp(rain.time.values[0]) == pd.Timestamp("2024-06-01")
    assert rain.values[0] == pytest.approx(24)
    incomplete = series([0.001, 0.001], freq="D", attrs=tp.attrs)
    assert np.isnan(daily_accumulation(incomplete, "tp")).all()
    ssrd = series(np.ones(24) * 3600, start="2024-06-01 01:00",
                  attrs={"GRIB_stepType": "accum", "units": "J m**-2"})
    assert daily_accumulation(ssrd, "ssrd").values[0] == pytest.approx(1)
    with pytest.raises(ValueError, match="not interchangeable"):
        daily_accumulation(ssrd, "avg_snswrf")


def test_no_geographical_extrapolation():
    field = xr.DataArray(np.ones((2, 2)), dims=("latitude", "longitude"),
                         coords={"latitude": [33, 33.25], "longitude": [120, 120.25]})
    result = remap(field, np.array([33, 33.25, 35]), np.array([120, 120.25, 122]))
    assert result.values[0, 0] == 1
    assert np.isnan(result.values[2]).all() and np.isnan(result.values[:, 2]).all()


def test_sampling_layer_is_not_bathymetry():
    ds = xr.Dataset({"no3": (("time", "depth", "latitude", "longitude"), np.ones((1, 2, 1, 1)))},
                    coords={"time": [pd.Timestamp("2024-06-01")], "depth": [10, 0.494025],
                            "latitude": [33], "longitude": [120]})
    ds["deptho"] = xr.DataArray([[40.0]], dims=("latitude", "longitude"))
    chosen = surface_field(ds, "no3", 2024)
    assert "depth" not in chosen.dims
    with pytest.raises(ValueError, match="outside"):
        surface_field(ds, "no3", 2023)


def test_paper_profile_disallows_unmentioned_architecture_changes():
    path = config_path("pilot_2022_2024.json", "reproduce_2022_2024.json")
    config = read_config(path)
    assert config["variables"] == PAPER_VARIABLES
    model = STVGLGFormer(config)
    assert model.observation_embedding is None
    assert model.env_embedding[0].in_features == 1
    assert len(model.env_blocks) == 4 and len(model.query_blocks) == 2
    for section, key, value in [("model", "observation_quality_embedding", True),
                                ("training", "env_dropout", 0.5)]:
        changed = copy.deepcopy(config)
        changed[section][key] = value
        with pytest.raises(ValueError):
            validate_paper_structure(changed)


def test_paper_stopping_rule_has_no_added_peak_condition():
    dates = pd.date_range("2024-06-01", periods=5)
    assert disappearance_date(dates, [1, 2, 3, 20, 40], 1, require_post_peak=False) == dates[2]
    assert disappearance_date(dates, [1, 2, 3, 20, 40], 1, require_post_peak=True) is None


def test_empty_validation_is_explicit_and_forbidden_for_paper_experiments(tmp_path):
    import json
    path = config_path("pilot_two_years.json", "fit_2022_2023_test_2024.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    assert read_config(path)["val_years"] == []
    config["require_paper_data"] = True
    broken = tmp_path / "invalid.json"
    broken.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Empty validation"):
        read_config(broken)


def test_full_paper_training_stops_before_model_creation_when_data_missing(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from stv_glg import training
    config = read_config(config_path("paper_architecture.json", "reproduce_2022_2024.json"))
    config["require_paper_data"] = True
    config["output_dir"] = str(tmp_path / "blocked")
    normalizer = SimpleNamespace(cloud_active=False, env_active=np.zeros((8, 10), bool))
    monkeypatch.setattr(training, "prepare", lambda _: ({}, normalizer, {"warnings": []}))

    def unexpected_model(*args):
        raise AssertionError("Model must not be created with incomplete strict-profile inputs.")

    monkeypatch.setattr(training, "STVGLGFormer", unexpected_model)
    with pytest.raises(ValueError, match="Required input coverage or product alignment failed"):
        training.train(config, "cpu")
    assert (tmp_path / "blocked" / "data_audit.json").exists()
    assert not (tmp_path / "blocked" / "best.pt").exists()
