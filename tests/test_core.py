from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import torch

from stv_glg.data import Normalizer, Season, assign_regions, calendar_features, load_seasons, make_batch, preceding_batch
from stv_glg.evaluation import disappearance_date, metrics
from stv_glg.loss import progressive_objective
from stv_glg.model import STVGLGFormer, glg_basis, glg_parameters


@pytest.fixture
def config():
    return {"variables": ["no3", "po4", "si"], "region_rows": 4, "region_cols": 2,
            "domain": {"lat_min": 33, "lat_max": 37, "lon_min": 119, "lon_max": 123},
            "max_days": 30,
            "model": {"d_model": 16, "heads": 4, "encoder_layers": 1, "query_layers": 1,
                      "ff_multiplier": 2, "dropout": 0.0}}


@pytest.fixture
def season(config):
    dates = pd.date_range("2024-05-15", periods=30)
    target = np.arange(30, dtype=np.float32)
    observed = np.ones(30, dtype=bool)
    observed[5:8] = False
    return Season(2024, dates, target, np.ones(30, dtype=bool), observed, target.copy(), np.zeros(30, np.float32),
                  np.zeros(30, np.float32), np.zeros(30, bool), np.zeros((30, 8, 3), np.float32),
                  np.zeros((30, 8, 3), bool), calendar_features(dates, 30), pd.DataFrame())


@pytest.fixture
def norm():
    return Normalizer(10.0, np.zeros(3, np.float32), np.ones(3, np.float32), np.ones((8, 3), bool),
                      False, [1.0, 0.3, 0.08, 0.05, 10.0])


def test_full_network_and_backward_with_missing_environment(config, season, norm):
    torch.set_num_threads(2)
    batch = make_batch([season, season], 9, norm, torch.device("cpu"))
    model = STVGLGFormer(config)
    prediction = model(batch)
    assert prediction["prediction"].shape == (2, 21)
    assert torch.isfinite(prediction["prediction"]).all()
    assert (prediction["prediction"] >= 0).all()
    assert (prediction["environment_gate"] == 0).all()
    assert (prediction["tau"].diff(dim=1) > 0).all()
    assert torch.allclose(prediction["tau"][:, -1], torch.ones(2))
    loss, _ = progressive_objective(prediction, model(preceding_batch(batch)), batch, 0.2, 0)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_partial_environment_and_all_missing_axial_slices(config, season, norm):
    season.env_mask[:9, 6, 1] = True
    season.env[:9, 6, 1] = 0.5
    batch = make_batch([season], 9, norm, torch.device("cpu"))
    model = STVGLGFormer(config)
    result = model(batch)
    result["prediction"].sum().backward()
    assert torch.isfinite(result["prediction"]).all()
    assert model.env_embedding[0].weight.grad is not None
    assert torch.isfinite(model.env_embedding[0].weight.grad).all()


def test_adjacent_prefix_alignment_and_causal_interpolation(config, season, norm):
    current = make_batch([season], 9, norm, torch.device("cpu"))
    previous = preceding_batch(current)
    independent = make_batch([season], 8, norm, torch.device("cpu"))
    assert torch.allclose(previous["area"], independent["area"])
    assert previous["area"][0, -1] == pytest.approx(0.4)
    assert current["area"][0, 6] == pytest.approx(0.6)
    assert torch.equal(previous["future_calendar"][:, 1:], current["future_calendar"])
    previous_predictions = torch.cat((torch.ones(1, 1) * 999, current["target"]), dim=1)
    _, record = progressive_objective({"prediction": current["target"]}, {"prediction": previous_predictions}, current, 1, 0)
    assert record["previous"] == 0


def test_stop_gradient_and_hinge():
    cur = torch.tensor([[3.0, 3.0]], requires_grad=True)
    pre = torch.tensor([[999.0, 1.0, 1.0]], requires_grad=True)
    batch = {"target": torch.zeros(1, 2), "target_mask": torch.ones(1, 2, dtype=torch.bool)}
    loss, record = progressive_objective({"prediction": cur}, {"prediction": pre}, batch, 0.5, 2)
    assert record["progressive"] == pytest.approx(6)
    loss.backward()
    assert pre.grad is None
    assert torch.allclose(cur.grad, torch.tensor([[4.5, 4.5]]))


def test_glg_peak_normalization_unimodality_and_extreme_numerics():
    params = glg_parameters(torch.zeros(2, 8))
    peak = glg_basis(params["mu"][:, None], params)
    assert torch.allclose(peak, torch.ones_like(peak), atol=1e-6)
    tau = torch.linspace(0, 1, 2001)[None].expand(2, -1)
    mixture = (glg_basis(tau, params) * params["weights"][:, None]).sum(-1)
    assert (mixture[:, :1001].diff(dim=1) >= -1e-7).all()
    assert (mixture[:, 1000:].diff(dim=1) <= 1e-7).all()
    logits = torch.full((1, 8), 10000.0, requires_grad=True)
    values = glg_basis(tau[:1], glg_parameters(logits))
    assert torch.isfinite(values).all()
    values.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_train_only_normalizer(config, season):
    held_out = copy.deepcopy(season)
    held_out.year = 2025
    held_out.target = held_out.target * 1000
    fitted = Normalizer.fit({2024: season, 2025: held_out}, [2024], config["variables"])
    assert fitted.area_scale < 30
    assert not fitted.env_active.any()


def test_metrics_and_operational_stop():
    result = metrics([1, 2, 3], [2, 2, 2])
    assert result["MSE_km4"] == pytest.approx(result["RMSE_km2"] ** 2)
    assert result["WAPE_percent"] == pytest.approx(100 / 3)
    dates = pd.date_range("2025-05-01", periods=9)
    assert disappearance_date(dates, [0, 1, 2, 10, 20, 6, 3, 2, 1], 0) == dates[-1]
    assert disappearance_date(dates, np.ones(9), 0) is None
    assert np.isnan(metrics([0, 0], [0, 0])["WAPE_percent"])


def test_region_edges(config):
    ids = assign_regions(np.array([37, 36.5, 33, 32.99]), np.array([119, 123, 121, 121]), config)
    assert ids.tolist() == [0, 1, 7, -1]


def test_daily_average_interpolation_and_blank_future_rows(config, tmp_path):
    workbook = tmp_path / "area.xlsx"
    frame = pd.DataFrame({"date": ["2024-05-15", "2024-05-15", "2024-05-17", "2024-05-20"],
                          "area": [1, 3, 6, np.nan]})
    frame.to_excel(workbook, sheet_name="2024", index=False)
    c = {**config, "area_file": str(workbook), "target_column": "area", "date_column": "date",
         "env_files": [], "train_years": [2024], "val_years": [], "test_years": [], "env_max_carry_days": 3}
    seasons, audit = load_seasons(c)
    s = seasons[2024]
    assert s.target[:3].tolist() == [2, 4, 6]
    assert s.target_mask[:3].all() and not s.target_mask[3:].any()
    assert not s.observed_mask[1]
    assert audit["years"][0]["same_date_extra_rows_averaged"] == 1
    assert audit["years"][0]["blank_area_rows"] == 1


def test_future_area_and_environment_cannot_change_inputs(config, season, norm):
    before = make_batch([season], 9, norm, torch.device("cpu"))
    season.target[9:] += 10000
    season.env[9:] += 10000
    season.env_mask[9:] = True
    after = make_batch([season], 9, norm, torch.device("cpu"))
    for key in before:
        if key not in ("target", "target_mask"):
            assert torch.equal(before[key], after[key])


def test_zero_peak_position_initialization_is_finite(config):
    model = STVGLGFormer(config, [0.0, 0.0, 0.03, 0.02, 30])
    assert torch.isfinite(model.parameter_head[-1].bias).all()
