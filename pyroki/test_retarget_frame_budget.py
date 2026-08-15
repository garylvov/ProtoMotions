"""Guard test: the h1_2 retargeter must never shorten a clip on its own.

The defect this locks down: `--target-raw-frames` defaulted to 450 and
`load_motion_data` did `raw_positions[:target_raw_frames]`, so a 600-frame
(20 s @ 30 fps) clip came out as 450 subsampled frames (15 s) with exit 0 and
no warning. Five seconds of motion vanished per clip, silently, at scale.

Run standalone (no GPU needed, the code under test is pure numpy):

    JAX_PLATFORMS=cpu \
    /oscar/data/stellex/glvov/glvov-envs/pyroki/bin/python \
        pyroki/test_retarget_frame_budget.py
"""

import inspect
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
import tempfile
from pathlib import Path

import numpy as onp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from batch_retarget_to_h1_2_from_keypoints import load_motion_data  # noqa: E402

N_KEYPOINTS = 18
FPS = 30.0


def _write_clip(directory, raw_frames, fps=FPS):
    """A synthetic keypoint pack in the layout load_motion_data expects."""
    rng = onp.random.default_rng(0)
    positions = rng.normal(scale=0.3, size=(raw_frames, N_KEYPOINTS, 3))
    orientations = onp.tile(
        onp.eye(3)[None, None, :, :], (raw_frames, N_KEYPOINTS, 1, 1)
    )
    payload = {
        "positions": positions,
        "orientations": orientations,
        "left_foot_contacts": onp.zeros((raw_frames, 2)),
        "right_foot_contacts": onp.zeros((raw_frames, 2)),
        "fps": fps,
    }
    path = Path(directory) / f"clip{raw_frames}.npy"
    onp.save(path, payload, allow_pickle=True)
    return str(path)


def _load(path, subsample_factor=1, target_raw_frames=450, truncate=False):
    # The opt-in argument is what the fix adds. Call without it when it is
    # absent so this test reports the actual duration defect rather than a
    # TypeError on the signature.
    kwargs = {}
    if "truncate_to_target" in inspect.signature(load_motion_data).parameters:
        kwargs["truncate_to_target"] = truncate
    return load_motion_data(
        path,
        "rigv1",
        subsample_factor,
        target_raw_frames,
        FPS,
        **kwargs,
    )


def test_long_clip_keeps_its_full_duration():
    """20 s in must be 20 s out with the shipped --target-raw-frames of 450."""
    with tempfile.TemporaryDirectory() as tmp:
        raw_frames = 600  # 20.0 s at 30 fps, past the 450-frame default buffer
        path = _write_clip(tmp, raw_frames)
        keypoints, orientations, left, right, num_timesteps, input_fps = _load(path)

        source_duration = raw_frames / FPS
        output_duration = num_timesteps / input_fps
        assert abs(output_duration - source_duration) < 1e-6, (
            f"output duration {output_duration:.3f}s != source {source_duration:.3f}s; "
            f"{raw_frames - num_timesteps} raw frames were discarded"
        )
        assert num_timesteps == raw_frames

        # The solver buffer must cover the clip, and it must be a whole multiple
        # of the quantum so the @jdc.jit solve does not recompile per clip.
        assert keypoints.shape[0] >= raw_frames
        assert keypoints.shape[0] % 450 == 0
        assert keypoints.shape[0] == 900
        for array in (orientations, left, right):
            assert array.shape[0] == keypoints.shape[0]


def test_short_clip_still_pads_to_the_quantum():
    """The pre-existing pad path is untouched for clips under the quantum."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_clip(tmp, 300)
        keypoints, _, _, _, num_timesteps, _ = _load(path)
        assert num_timesteps == 300
        assert keypoints.shape[0] == 450


def test_truncation_requires_the_explicit_opt_in():
    """--truncate-to-target still discards frames, but only when asked."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_clip(tmp, 600)
        keypoints, _, _, _, num_timesteps, _ = _load(path, truncate=True)
        assert num_timesteps == 450
        assert keypoints.shape[0] == 450


def test_subsampled_long_clip_keeps_its_full_duration():
    """Duration must survive the subsampling path too."""
    with tempfile.TemporaryDirectory() as tmp:
        raw_frames = 600
        path = _write_clip(tmp, raw_frames)
        _, _, _, _, num_timesteps, input_fps = _load(path, subsample_factor=2)
        expected = len(range(0, raw_frames, 2))
        assert num_timesteps == expected
        output_duration = num_timesteps * 2 / input_fps
        assert abs(output_duration - raw_frames / FPS) < 0.1


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
