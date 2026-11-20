import json
from dataclasses import asdict

import pytest
import torch

from mech_interp.features import build_feature_index
from mech_interp.sparse_autoencoder import SAEConfig, SparseAutoencoder, evaluate
from mech_interp.training import train_sae


def test_decoder_constraint_and_tangent_gradient():
    torch.manual_seed(1)
    sae = SparseAutoencoder(SAEConfig(d_model=8, hidden_dim=16))
    loss, _, features = sae(torch.randn(10, 8))
    assert features.min() >= 0
    loss.backward()
    sae.project_decoder_gradient()
    assert (sae.W_dec.grad * sae.W_dec).sum(-1).abs().max() < 1e-5
    torch.optim.Adam(sae.parameters()).step()
    sae.normalize_decoder()
    torch.testing.assert_close(sae.W_dec.norm(dim=-1), torch.ones(16))


def test_objective_l1_coefficient_and_dimension_reduction():
    sae = SparseAutoencoder(SAEConfig(d_model=3, hidden_dim=5, l1_coefficient=0.2))
    x = torch.randn(4, 3)
    loss, recon, feats = sae(x)
    expected = (recon - x).square().sum(-1).mean() + 0.2 * feats.sum(-1).mean()
    torch.testing.assert_close(loss, expected)


def test_variance_explained_uses_training_dimension_means():
    sae = SparseAutoencoder(SAEConfig(d_model=2, hidden_dim=2))
    with torch.no_grad():
        sae.W_enc.zero_()
        sae.b_dec.copy_(torch.tensor([10.0, -10.0]))
    x = torch.tensor([[9.0, -9.0], [11.0, -11.0]])
    metric = evaluate(sae, x, scale=1, mean=torch.tensor([10.0, -10.0]))
    assert metric["variance_explained"] == pytest.approx(0)
    assert metric["inactive_fraction"] == 1


def test_training_reduces_error_and_checkpoint_reloads(tmp_path):
    torch.manual_seed(14)
    acts = torch.randn(192, 8) @ torch.diag(torch.arange(1.0, 9.0))
    path = tmp_path / "activations.pt"
    metadata = {"model": {"model_id": "synthetic"}, "hook_name": "synthetic"}
    torch.save({"train": acts[:128], "validation": acts[128:], "metadata": metadata}, path)
    config = SAEConfig(d_model=8, hidden_dim=16, steps=100, lr=0.01, eval_every=100)
    report = train_sae(path, config, tmp_path / "run", threads=2)
    assert (
        report["final_validation"]["mse_per_dimension"]
        < report["initial_validation"]["mse_per_dimension"]
    )
    ckpt = torch.load(tmp_path / "run/sae_final.pt", weights_only=True)
    loaded = SparseAutoencoder(SAEConfig(**ckpt["config"]))
    loaded.load_state_dict(ckpt["state"])
    measured = evaluate(loaded, acts[128:], ckpt["scale"], ckpt["train_mean"])
    assert measured == report["final_validation"]
    with pytest.raises(FileExistsError):
        train_sae(path, config, tmp_path / "run", threads=2)
    json.dumps(report, allow_nan=False)


def test_feature_index_uses_saved_scale_and_global_top_examples(tmp_path):
    config = SAEConfig(d_model=2, hidden_dim=2)
    sae = SparseAutoencoder(config)
    with torch.no_grad():
        sae.W_enc.copy_(torch.eye(2))
        sae.b_dec.zero_()
    path = tmp_path / "sae.pt"
    torch.save(
        {
            "config": asdict(config),
            "state": sae.state_dict(),
            "scale": 2.0,
            "activation_metadata": {"model": {}, "hook_name": "layer8"},
        },
        path,
    )
    acts = torch.tensor([[1.0, 0], [8.0, 0], [0.0, 6.0]])
    rows = [{"document": i, "before": "", "token": str(i), "after": ""} for i in range(3)]
    result = build_feature_index(path, acts, rows, "test", top_k=2)
    assert result["features"][0]["max_activation"] == 4
    assert result["features"][0]["examples"][0]["document"] == 1
    assert result["features"][1]["max_activation"] == 3


@pytest.mark.parametrize("kwargs", [{"steps": 0}, {"hidden_dim": -1}, {"l1_coefficient": -1}])
def test_invalid_configs_rejected(kwargs):
    with pytest.raises(ValueError):
        SAEConfig(**kwargs)
