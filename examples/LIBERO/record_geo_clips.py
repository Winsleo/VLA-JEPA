"""Record RGB + simulator metric depth clips for the I3 geometry probes.

Why a separate entry point instead of reusing the training set: `LEROBOT_LIBERO_DATA` ships only
mp4-encoded RGB, so depth would have to be estimated. LIBERO can render metric depth directly
(`camera_depths=True` + `robosuite.utils.camera_utils.get_real_depth_map`), which is tier 1 of the
depth-target contract (`docs/implementation-plan.md` section 6) and is RGB-aligned by construction.

The rollout is driven by the same policy server as `eval_libero.py`, so the clips come from the same
on-policy distribution the evaluation measures. The loop below mirrors `eval_libero.py:145-228`
(preprocessing, action decoding, stepping). It is duplicated rather than factored out because
`eval_libero.py` is the validated I1/I2 evaluation entry point and must not be restructured; the
parts that can be shared without touching its rollout loop are imported.

Output layout, one directory per episode, one compressed npz per clip (outside the repository):

    <out>/<suite>/<task_id>_<slug>/ep<idx>/clip<k>.npz   rgb / depth_m / valid / extrinsics / actions
    <out>/<suite>/<task_id>_<slug>/ep<idx>/meta.json     recording contract + provenance
    <out>/manifest_<suite>.jsonl                         one line per clip (one file per process)
"""

import dataclasses
import json
import logging
import pathlib
import subprocess
from typing import List, Sequence, Tuple

import numpy as np
import tqdm
import tyro
from libero.libero import benchmark
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
)

from examples.LIBERO.eval_libero import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    _get_libero_env,
    _quat2axisangle,
    short_name,
)
from examples.LIBERO.geo_clip_contract import (
    CLIP_STRIDE,
    MAX_STEPS_BY_SUITE,
    PIXEL_CONVENTION,
    RECORDER_VERSION,
    TARGET_TYPE,
    VALID_RULE,
    VIEW_NAMES,
    clip_starts,
    decode_action,
    depth_valid_mask,
    rotate180,
    split_for_episode,
)
from examples.LIBERO.model2libero_interface import M1Inference


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size: List[int] = dataclasses.field(default_factory=lambda: [224, 224])

    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10  # steps of settling before recording, same as eval
    num_trials_per_task: int = 5
    seed: int = 7

    pretrained_path: str = ""
    with_state: str = "true"

    out_path: str = "geo_clips"
    clip_frames: int = 8  # teacher clip length (framework.vj2_model.num_frames)
    clips_per_episode: int = 8
    # Episode index ranges per split, so every task appears in all three splits (in-domain probe)
    # and no frame is shared across them: [0, num_train) train, then num_val val, the rest test.
    num_train_episodes: int = 3
    num_val_episodes: int = 1


def _git_commit(path: pathlib.Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _stack_views(per_view: Sequence[np.ndarray]) -> np.ndarray:
    """Stack one array per view into a leading view axis, in `VIEW_NAMES` order."""
    return np.stack(per_view, axis=0)


def _observation_frames(env, obs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the recorded tensors for one observation: RGB, metric depth, camera pose."""
    rgb = _stack_views([rotate180(obs[f"{view}_image"]) for view in VIEW_NAMES])
    depth = _stack_views(
        [rotate180(get_real_depth_map(env.sim, obs[f"{view}_depth"])[..., 0]).astype(np.float32) for view in VIEW_NAMES]
    )
    extrinsics = _stack_views([get_camera_extrinsic_matrix(env.sim, view) for view in VIEW_NAMES])
    return rgb, depth, extrinsics


def record_geo_clips(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    np.random.seed(args.seed)

    if args.task_suite_name not in MAX_STEPS_BY_SUITE:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    max_steps = MAX_STEPS_BY_SUITE[args.task_suite_name]

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()

    out_root = pathlib.Path(args.out_path)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / f"manifest_{args.task_suite_name}.jsonl"
    provenance = {
        "recorder_version": RECORDER_VERSION,
        "vla_jepa_commit": _git_commit(pathlib.Path(__file__).resolve().parents[2]),
        "policy_ckpt": args.pretrained_path,
    }

    model = M1Inference(
        policy_ckpt_path=args.pretrained_path,
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
    )

    total_clips = 0
    with manifest_path.open("w") as manifest:
        for task_id in tqdm.tqdm(range(task_suite.n_tasks)):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed, camera_depths=True)

            extent = float(env.sim.model.stat.extent)
            z_near = float(env.sim.model.vis.map.znear) * extent
            z_far = float(env.sim.model.vis.map.zfar) * extent
            intrinsics = {
                view: get_camera_intrinsic_matrix(env.sim, view, LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION).tolist()
                for view in VIEW_NAMES
            }
            task_dir = out_root / args.task_suite_name / f"{task_id:02d}_{short_name(task.name)}"

            for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                model.reset(task_description=task_description)
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])

                rgb_frames: List[np.ndarray] = []
                depth_frames: List[np.ndarray] = []
                extrinsic_frames: List[np.ndarray] = []
                action_frames: List[np.ndarray] = []
                done = False
                t, step = 0, 0

                while t < max_steps + args.num_steps_wait:
                    if t < args.num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    rgb, depth, extrinsics = _observation_frames(env, obs)
                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )
                    obs_input = {
                        "images": [rgb[0], rgb[1]],
                        "task_description": str(task_description),
                        "step": step,
                    }
                    if args.with_state == "true":
                        obs_input["state"] = np.expand_dims(state, axis=0)

                    delta_action = decode_action(model.step(**obs_input)["raw_action"])

                    rgb_frames.append(rgb)
                    depth_frames.append(depth)
                    extrinsic_frames.append(extrinsics)
                    action_frames.append(delta_action)

                    obs, _, done, _ = env.step(delta_action.tolist())
                    if done:
                        break
                    t += 1
                    step += 1

                starts = clip_starts(len(rgb_frames), args.clip_frames, args.clips_per_episode)
                if not starts:
                    logging.warning(
                        f"task {task_id} episode {episode_idx}: only {len(rgb_frames)} frames, "
                        f"fewer than clip_frames={args.clip_frames}; no clip written"
                    )
                    continue

                episode_dir = task_dir / f"ep{episode_idx:02d}"
                episode_dir.mkdir(parents=True, exist_ok=True)
                split = split_for_episode(episode_idx, args.num_train_episodes, args.num_val_episodes)

                for clip_index, start in enumerate(starts):
                    stop = start + args.clip_frames
                    depth_clip = np.stack(depth_frames[start:stop], axis=0)
                    valid_clip = depth_valid_mask(depth_clip, z_near, z_far)
                    clip_path = episode_dir / f"clip{clip_index:02d}.npz"
                    np.savez_compressed(
                        clip_path,
                        rgb=np.stack(rgb_frames[start:stop], axis=0),
                        depth_m=depth_clip,
                        valid=valid_clip,
                        extrinsics=np.stack(extrinsic_frames[start:stop], axis=0),
                        actions=np.stack(action_frames[start:stop], axis=0),
                    )
                    manifest.write(
                        json.dumps(
                            {
                                "path": str(clip_path.relative_to(out_root)),
                                "suite": args.task_suite_name,
                                "task_id": task_id,
                                "episode_index": episode_idx,
                                "clip_index": clip_index,
                                "start": start,
                                "split": split,
                                "target_type": TARGET_TYPE,
                                "success": bool(done),
                                "valid_fraction": float(valid_clip.mean()),
                            }
                        )
                        + "\n"
                    )
                    manifest.flush()
                    total_clips += 1

                meta = {
                    "target_type": TARGET_TYPE,
                    "depth_units": "meter",
                    "z_near": z_near,
                    "z_far": z_far,
                    "mujoco_extent": extent,
                    "valid_rule": VALID_RULE,
                    "suite": args.task_suite_name,
                    "task_id": task_id,
                    "task_name": task.name,
                    "task_description": str(task_description),
                    "episode_index": episode_idx,
                    "init_state_index": episode_idx,
                    "seed": args.seed,
                    "split": split,
                    "success": bool(done),
                    "num_steps_wait": args.num_steps_wait,
                    "max_steps": max_steps,
                    "recorded_frames": len(rgb_frames),
                    "resolution": LIBERO_ENV_RESOLUTION,
                    "views": list(VIEW_NAMES),
                    "pixel_convention": PIXEL_CONVENTION,
                    "clip_frames": args.clip_frames,
                    "clip_stride": CLIP_STRIDE,
                    "clip_starts": starts,
                    "intrinsics": intrinsics,
                    "provenance": provenance,
                }
                (episode_dir / "meta.json").write_text(json.dumps(meta, indent=2))
                logging.info(
                    f"task {task_id} episode {episode_idx} ({split}): "
                    f"{len(rgb_frames)} frames, {len(starts)} clips, success={done}"
                )

            env.close()

    logging.info(f"Wrote {total_clips} clips and {manifest_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(record_geo_clips)
