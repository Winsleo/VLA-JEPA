"""CPU contracts for I4 teacher selection and post-checkpoint predictor reinitialization."""

import ast
from pathlib import Path

import torch
from omegaconf import OmegaConf

from starVLA.model.framework.VLM4A.VLA_JEPA import reinitialize_predictor
from starVLA.model.modules.world_model.vj2_predictor import VisionTransformerPredictorAC


def _predictor():
    return VisionTransformerPredictorAC(
        img_size=(32, 32),
        patch_size=16,
        num_frames=2,
        tubelet_size=1,
        embed_dim=8,
        predictor_embed_dim=8,
        depth=1,
        num_heads=1,
        action_embed_dim=4,
        num_add_tokens=2,
        grid_size=(2, 2),
        num_temporal_blocks=2,
    )


def test_reinitialization_replaces_checkpoint_predictor_values_only():
    predictor = _predictor()
    with torch.no_grad():
        for parameter in predictor.parameters():
            parameter.zero_()

    count = reinitialize_predictor(predictor)

    assert count == sum(parameter.numel() for parameter in predictor.parameters())
    assert any(torch.count_nonzero(parameter) for parameter in predictor.parameters())


def test_i2_config_keeps_the_pinned_teacher_and_does_not_reinitialize():
    config = OmegaConf.load(Path(__file__).parents[1] / "configs" / "i1_libero_local.yaml")
    assert config.framework.vj2_model.teacher == "vjepa2"
    assert config.framework.vj2_model.input_size == 256
    assert config.framework.vj2_model.reinit_predictor is False


def test_framework_uses_the_shared_loader_and_resets_after_super_load():
    """Static ordering keeps default I2 behavior while making I4 reset checkpoint-derived state."""
    source = Path(__file__).parents[1] / "starVLA" / "model" / "framework" / "VLM4A" / "VLA_JEPA.py"
    tree = ast.parse(source.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "load_state_dict" and node.args.args[0].arg == "self"
    )
    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
    names = [getattr(call.func, "id", None) for call in calls]
    assert "reinitialize_predictor" in names
    super_load = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load_state_dict"
        and isinstance(node.func.value, ast.Call)
    )
    reset = next(node for node in ast.walk(method) if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "reinitialize_predictor")
    assert super_load.lineno < reset.lineno


def test_trainer_only_resets_after_a_partial_checkpoint_load():
    """A full load resets in VLA_JEPA; partial loading bypasses that method."""
    source = Path(__file__).parents[1] / "starVLA" / "training" / "train_starvla.py"
    tree = ast.parse(source.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_training" and node.args.args[0].arg == "self"
    )
    resets = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "reinitialize_world_predictor"
    ]
    assert len(resets) == 1
    reset = resets[0]
    parents = [node for node in ast.walk(method) if isinstance(node, ast.If) and reset in ast.walk(node)]
    assert any("reload_modules" in ast.unparse(parent.test) for parent in parents)
