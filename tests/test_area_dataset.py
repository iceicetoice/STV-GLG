import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from stv_glg.config import read_config
from stv_glg.data import Normalizer, historical_area, load_seasons, make_batch, preceding_batch
from stv_glg.loss import progressive_objective
from stv_glg.model import STVGLGFormer
from stv_glg.training import load_checkpoint, save_checkpoint


ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def dataset():
    config = read_config(ROOT / "configs" / "reproduce_2022_2024.json")
    seasons, audit = load_seasons(config)
    normalizer = Normalizer.fit(seasons, config["train_years"], config["variables"])
    return config, seasons, audit, normalizer


def test_workbook_hash_and_area_only_columns():
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = ROOT / "data" / item["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        with pd.ExcelFile(path) as book:
            for year in (2022, 2023, 2024):
                assert list(book.parse(str(year)).columns) == ["date", "unmixed_area_km2"]
    assert manifest["environment_values_released"] == manifest["cloud_values_released"] == 0


def test_counts_dimensions_and_no_invented_covariates(dataset):
    _, seasons, _, normalizer = dataset
    assert sorted(seasons) == [2022, 2023, 2024]
    assert [int(s.observed_mask.sum()) for s in seasons.values()] == [36, 63, 73]
    assert [int(s.target_mask.sum()) for s in seasons.values()] == [62, 93, 96]
    assert not normalizer.env_active.any() and not normalizer.cloud_active
    for season in seasons.values():
        assert season.env.shape == (177, 8, 10)
        assert not season.env_mask.any() and not season.cloud_mask.any()
        assert (season.env == 0).all() and (season.cloud == 0).all()
        final = np.flatnonzero(season.observed_mask)[-1]
        assert not season.target_mask[final + 1:].any()


def test_daily_reference_and_training_loader_agree(dataset):
    config, seasons, _, _ = dataset
    with pd.ExcelFile(config["area_file"]) as book:
        for year, season in seasons.items():
            daily = book.parse(f"daily_{year}")
            stored = daily.target_area_km2.fillna(0).to_numpy(np.float32)
            np.testing.assert_array_equal(stored, season.target)
            np.testing.assert_array_equal(daily.target_valid.to_numpy(bool), season.target_mask)
            np.testing.assert_array_equal(daily.is_observed.to_numpy(bool), season.observed_mask)


def test_all_adjacent_prefixes_are_causal(dataset):
    config, seasons, _, normalizer = dataset
    for season in seasons.values():
        for n in range(8, 177):
            batch = make_batch([season], n, normalizer, torch.device("cpu"))
            previous = preceding_batch(batch, config["min_prefix"])
            expected = historical_area(season, n - 1) / normalizer.area_scale
            np.testing.assert_allclose(previous["area"][0].numpy(), expected, rtol=1e-6, atol=1e-7)
            assert torch.equal(previous["future_calendar"][:, 1:], batch["future_calendar"])
            assert not batch["env_mask"].any() and not batch["cloud_mask"].any()


def test_finite_area_only_loss_and_backward(dataset):
    torch.set_num_threads(2)
    config, seasons, _, normalizer = dataset
    model = STVGLGFormer(config, normalizer.glg_initial, normalizer.area_scale)
    batch = make_batch([seasons[2022]], 28, normalizer, torch.device("cpu"))
    with torch.no_grad():
        previous = model(preceding_batch(batch, config["min_prefix"]))
    current = model(batch)
    loss, _ = progressive_objective(current, previous, batch, 0.2, 25 / normalizer.area_scale ** 2)
    assert torch.isfinite(loss)
    assert torch.isfinite(current["prediction"]).all()
    assert (current["prediction"] >= 0).all()
    assert (current["tau"][:, 1:] > current["tau"][:, :-1]).all()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


def test_area_checkpoint_is_portable(dataset, tmp_path):
    config, _, _, normalizer = dataset
    model = STVGLGFormer(config, normalizer.glg_initial, normalizer.area_scale)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    path = tmp_path / "test.pt"
    save_checkpoint(path, model, optimizer, scheduler, normalizer, config, 1, 0.1)
    raw = torch.load(path, weights_only=False, map_location="cpu")
    assert raw["config"]["area_file"] == "STV_GLG.xlsx"
    _, _, resolved, _ = load_checkpoint(path, torch.device("cpu"), ROOT / "data")
    assert Path(resolved["area_file"]).resolve() == Path(config["area_file"]).resolve()
    assert sorted(load_seasons(resolved)[0]) == [2022, 2023, 2024]
