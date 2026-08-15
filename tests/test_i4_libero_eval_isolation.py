"""Source-level checks for the I4 LIBERO renderer-isolation entry point.

The simulator environment is intentionally not imported here: these checks only pin the CLI
contract that lets the external evaluator put one task or one episode in a fresh process.
"""

import ast
from pathlib import Path


EVAL_LIBERO = Path(__file__).resolve().parents[1] / "examples" / "LIBERO" / "eval_libero.py"


def _source() -> str:
    return EVAL_LIBERO.read_text()


def test_isolation_arguments_are_optional_and_typed():
    tree = ast.parse(_source())
    args_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Args")
    annotations = {
        node.target.id: ast.unparse(node.annotation)
        for node in args_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    assert annotations["task_id"] == "int | None"
    assert annotations["episode_id"] == "int | None"


def test_task_and_episode_selection_keep_default_ranges():
    source = _source()

    assert "range(num_tasks_in_suite) if args.task_id is None else [args.task_id]" in source
    assert "range(args.num_trials_per_task)\n            if args.episode_id is None\n            else [args.episode_id]" in source


def test_task_cleanup_is_explicit():
    source = _source()
    assert "env.close()" in source
