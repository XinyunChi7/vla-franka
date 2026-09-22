# vla-franka

Robot-control-side scripts for running the `realrobot-plug-smolvla` checkpoint
rollout on a Franka arm: arm control, gripper debouncing, the SmolVLA policy
bridge, and the GelSight tactile node that feeds it. Extracted from the full
`vlm-franka` workspace, scoped to just what's needed to run a rollout.

## Layout

- `src/franky_ros2/` — ROS2 package: arm control node
  (`franky_control_node_orientation.py`), gripper debouncing
  (`gripper_debouncer.py`), the SmolVLA policy bridge
  (`smolvla_policy_node.py`), and the launch file (`launch/vla_test.launch.py`).
- `src/tactile_ros/` — ROS2 package: GelSight tactile sensor node
  (`gelsight_node.py`) with FEATS force estimation (`feats_model.py`).
- `policy_deploy/` — `policy_runner.py` and `recorder.py`, imported directly by
  `smolvla_policy_node.py` via `sys.path` (not a ROS package; matches the
  layout of the checkpoint's deploy directory).

## Prerequisites not included here

- **Checkpoint weights + vendored `lerobot_fork/`** — hosted on Hugging Face at
  `tliangucl/realrobot-plug-smolvla`. `policy_deploy/` expects to sit next to
  (or be pointed at, via `smolvla_policy_node.py --deploy-pkg-dir`) a checkout
  of that repo's `lerobot_fork/` and a chosen `checkpoints/.../<ckpt>/` dir.
- **`gsrobotics` SDK** — a separate third-party repo that `gelsight_node.py`
  sys.path-injects at runtime (`GSROBOTICS_PATH` env var), including the FEATS
  model weights (`unet_09042025_124903_80.pt`) and normalization file.
- **`franky`/`libfranka`** — the Franka arm's Python bindings, external install.
- **A Python venv with torch/transformers/lerobot** (referred to elsewhere as
  `lerobot-env`) — `smolvla_policy_node.py` can't run under `ros2 run`; it
  needs to be launched directly with that interpreter (see the docstring at
  the top of the file) while `rclpy` stays importable from the ROS install.

## Running

```
colcon build --packages-select franky_ros2 tactile_ros
source install/setup.bash
ros2 launch franky_ros2 vla_test.launch.py \
    ckpt_dir:=<path to a checkpoint dir> \
    task:="insert the plug into the power strip"
```

Defaults to dry-run (logs actions, publishes nothing). Pass `live:=true` to
actually drive the arm — read `smolvla_policy_node.py`'s docstring first and
do the sign-convention check it describes before ever running live.
