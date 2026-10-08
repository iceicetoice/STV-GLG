import copy

import numpy as np
import pandas as pd
import pytest
import torch

from stv_glg.data import Normalizer, PrefixSampler, Season, calendar_features, make_batch, preceding_batch
from stv_glg.loss import masked_mse, progressive_objective
from stv_glg.model import STVGLGFormer


def inputs():
    dates = pd.date_range("2024-05-15", periods=30)
    a = np.arange(30, dtype=np.float32)
    yes = np.ones(30, bool)
    season = Season(2024, dates, a, yes, yes, a, np.zeros(30, np.float32),
                    np.zeros(30, np.float32), np.zeros(30, bool),
                    np.ones((30, 8, 3), np.float32), np.ones((30, 8, 3), bool),
                    calendar_features(dates, 30, 0), pd.DataFrame())
    config = {"variables": ["no3", "po4", "si"], "region_rows": 4, "region_cols": 2, "max_days": 30,
              "min_prefix": 7, "training": {"lambda_prog": 0.2, "min_future_labels": 3},
              "model": {"d_model": 16, "heads": 4, "encoder_layers": 1, "query_layers": 1,
                        "ff_multiplier": 2, "dropout": 0.0}}
    norm = Normalizer(10, np.zeros(3), np.ones(3), np.ones((8, 3), bool), False, [1, 0.3, 0.08, 0.05, 10])
    return config, season, norm


def test_eq15_has_no_scoring_parameters_and_normalizes_over_dates():
    config, season, norm = inputs()
    model = STVGLGFormer(config)
    batch = make_batch([season], 8, norm, torch.device("cpu"))
    states, _ = model.encode(batch)
    result = model(batch)
    expected = torch.softmax(torch.tanh(states), dim=1)
    assert result["pool_weights"].shape == (1, 8, 16)
    assert torch.equal(result["pool_weights"], expected)
    assert torch.allclose(result["pool_weights"].sum(1), torch.ones(1, 16))
    assert not any("pool" in key for key in model.state_dict())


def test_adjacent_sampler_never_uses_six_day_reference():
    config, season, _ = inputs()
    sampler = PrefixSampler({2024: season}, [2024], config, 42)
    assert min(sampler.pools) == 8
    assert all(sampler.sample(4)[1] >= 8 for _ in range(500))
    config["training"]["lambda_prog"] = 0
    assert min(PrefixSampler({2024: season}, [2024], config, 42).pools) == 7


def test_model_and_reference_enforce_minimum_prefix():
    config, season, norm = inputs()
    batch = make_batch([season], 7, norm, torch.device("cpu"))
    assert STVGLGFormer(config)(batch)["prediction"].shape[1] == 23
    with pytest.raises(ValueError, match="Both adjacent"):
        preceding_batch(batch, min_prefix=7)
    with pytest.raises(ValueError, match="at least 7"):
        STVGLGFormer(config)(make_batch([season], 6, norm, torch.device("cpu")))


def test_masked_global_mean_matches_eq41_and_ignores_missing_nan():
    target = torch.tensor([[1.0, 2.0], [3.0, float("nan")]])
    prediction = torch.tensor([[2.0, 4.0], [6.0, float("nan")]], requires_grad=True)
    mask = torch.tensor([[True, True], [True, False]])
    loss = masked_mse(prediction, target, mask)
    assert float(loss.detach()) == pytest.approx(14 / 3)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[1, 1] == 0
    full = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert masked_mse(full, torch.zeros_like(full), torch.ones_like(full, dtype=torch.bool)) == full.square().mean()


def test_deterministic_prefix_forwards_and_bound():
    config, season, norm = inputs()
    model = STVGLGFormer(config).train()
    batch = make_batch([season], 8, norm, torch.device("cpu"))
    first, second = model(batch), model(batch)
    assert torch.equal(first["prediction"], second["prediction"])
    previous = model(preceding_batch(batch, min_prefix=7))
    _, values = progressive_objective(first, previous, batch, 0.2, 0.01)
    assert values["area"] - values["previous"] <= 0.01 + values["progressive"] + 1e-6


def test_all_future_measurements_are_excluded_from_forecast():
    config, season, norm = inputs()
    model = STVGLGFormer(config).eval()
    before = model(make_batch([season], 8, norm, torch.device("cpu")))["prediction"]
    modified = copy.deepcopy(season)
    modified.target[8:] *= 10000
    modified.env[8:] *= 10000
    modified.cloud[8:] = 1
    modified.cloud_mask[8:] = True
    after = model(make_batch([modified], 8, norm, torch.device("cpu")))["prediction"]
    assert torch.equal(before, after)
