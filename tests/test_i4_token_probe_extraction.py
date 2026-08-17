"""The extraction shortcut must equal what `VLA_JEPA.forward` computes.

`action_tokens` is a local inside `forward`, so `run_i4_token_probe.capture_action_tokens`
re-derives it: same prompt, same last hidden layer, same gather, same regrouping. Nothing here
loads a 6 GB checkpoint; these tests pin the derivation against `forward`'s own source so the
shortcut cannot drift away from it silently.

Pure source/logic checks, CPU-only.
"""

import ast
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_i4_token_probe.py"
FRAMEWORK = Path(__file__).resolve().parents[1] / "starVLA" / "model" / "framework" / "VLA_JEPA.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_i4_token_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _function(source: str, name: str) -> ast.FunctionDef:
    return next(
        node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_capture_gathers_the_same_positions_as_forward():
    """Both must slice on `action_token_ids`, not on the embodied-action token."""
    captured = ast.dump(_function(SCRIPT.read_text(), "capture_action_tokens"))
    assert "action_token_ids" in captured
    # `embodied_action_token_id` feeds the policy head, not the depth condition. Selecting it here
    # would probe a different tensor while still producing plausible shapes.
    assert "embodied_action_token_id" not in captured
    assert "hidden_states" in captured


def test_capture_uses_the_prompt_template_that_forward_uses():
    """`forward` passes CoT_prompt and `predict_action` does not; the depth gradient flowed through
    `forward`, so the probe must build `forward`'s prompt."""
    captured = ast.dump(_function(SCRIPT.read_text(), "capture_action_tokens"))
    assert "prompt_template" in captured and "CoT_prompt" in captured
    assert "e_actions" in captured, "both replacement keys are needed to reproduce forward's prompt"
    forward = ast.dump(_function(FRAMEWORK.read_text(), "forward"))
    assert "CoT_prompt" in forward, "forward no longer uses CoT_prompt; the probe must be revisited"


def test_capture_regroups_tokens_by_transition_like_forward():
    """`forward` reshapes to (batch, groups, -1, hidden) before conditioning the depth head."""
    captured = _function(SCRIPT.read_text(), "capture_action_tokens")
    views = [node for node in ast.walk(captured) if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "view"]
    assert views, "the capture must reshape the gathered tokens"
    source = ast.dump(captured)
    assert "num_temporal_blocks" in source, "the transition count must come from the backbone, not a literal"


def test_loader_asserts_the_model_is_frozen_after_setting_it_frozen():
    """Setting `eval()` and clearing `requires_grad` is not enough; the loader must verify both.

    A probe read through a live-gradient or train-mode model would not be measuring the frozen
    representation the report claims, and dropout alone would make the features non-deterministic.
    Checked on the source because the alternative is loading a 6 GB checkpoint in a unit test.
    """
    loader = _function(SCRIPT.read_text(), "_load_frozen_model")
    source = ast.dump(loader)
    assert "eval" in source and "requires_grad_" in source, "the loader must freeze the model"
    raises = [node for node in ast.walk(loader) if isinstance(node, ast.Raise)]
    assert len(raises) >= 2, "both the gradient and the mode guard must be able to fail the run"
    messages = " ".join(
        node.value
        for raise_node in raises
        for node in ast.walk(raise_node)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    assert "frozen" in messages and "eval" in messages


def test_missing_instruction_table_is_reported_not_guessed():
    """A wrong instruction changes what the action tokens encode, so absence must be loud."""
    module = _module()
    try:
        module._instructions(Path("/nonexistent-clip-root"))
    except SystemExit as error:
        assert "task_language.json" in str(error)
        return
    raise AssertionError("a missing instruction table must stop the run")
