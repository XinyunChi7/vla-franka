# vla-franka

Robot-control-side scripts for running the `realrobot-plug-smolvla` checkpoint
rollout on a Franka arm: arm control, gripper debouncing, the SmolVLA policy
bridge, and the GelSight tactile node that feeds it. Extracted from the full
`vlm-franka` workspace, scoped to just what's needed to run a rollout.

## Layout

- `src/franky_ros2/` — ROS2 package: arm control node
  (`franky_control_node_orientation.py`), gripper debouncing
  (`gripper_debouncer.py`), the SmolVLA policy bridge
  (`smolvla_policy_node.py`), keyboard jog/correction
  (`keyboard_teleop_pose_orientation.py`), rollout episode recording
  (`episode_recorder_node.py`), and the launch files
  (`launch/vla_test.launch.py`, `launch/vla_test_teleop.launch.py`).
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
- **`pynput`** (for `keyboard_teleop_pose_orientation.py`) and **`h5py`** (for
  `episode_recorder_node.py`) — installed in whichever Python environment runs
  those nodes via `ros2 run`.

Both launch files hardcode `REPO_ROOT = '/home/xinyun/vlm-franka'` (used as
the policy process's `cwd` and to resolve `--deploy-pkg-dir`) — update that
path for your own checkout before running.

## (3) VLA rollouts

### (3.1) VLA policy rollout (SmolVLA), no teleop

Same sensors/arm as above, plus `smolvla_policy_node.py` (run under the `~/lerobot-env`
venv, not the ROS-managed Python) driving the arm directly.

```
ros2 launch franky_ros2 vla_test.launch.py ckpt_dir:=checkpoints/realrobot-plug-smolvla/ckpt_vision task:="insert the plug into the power strip" live:=true

ros2 launch franky_ros2 vla_test.launch.py ckpt_dir:=checkpoints/realrobot-plug-smolvla/ckpt_tactile_D93_recovery_v2_20k task:="insert the plug into the power strip"  record_dir:=runs/eval_tactile_v2 live:=true dynamics_factor:=0.08
```

Key args: `venv_python` (default `/home/xinyun/lerobot-env/bin/python3`), `ckpt_dir`, `task`,
`live` (default `false` = dry-run/log-only, no motion/gripper commands published),
`record_dir` (if set, records rollouts under this dir), `visualize_cameras`.

Once launched, start the policy loop from another terminal:

```
ros2 topic pub /teleop_command std_msgs/String "data: start_policy" --once
```

Stop it again with:

```
ros2 topic pub /teleop_command std_msgs/String "data: stop_policy" --once
```

```
ros2 topic pub --once /teleop_command std_msgs/String "data: 'recover'"
```

### (3.2) VLA policy rollout with live teleop correction

Same as above, but also brings up `keyboard_teleop_pose_orientation.py` and
`episode_recorder_node`, so an operator can jog/correct the arm mid-rollout instead of only
watching it.

```
ros2 launch franky_ros2 vla_test_teleop.launch.py ckpt_dir:=checkpoints/realrobot-plug-smolvla/ckpt_vision task:="insert the plug into the power strip" live:=true
```

Same args as `vla_test.launch.py`, plus `output_dir`/`sample_rate_hz` for
`episode_recorder_node` and `dynamics_factor` (default `0.10` here, higher than the pure-teleop
default since the policy needs it).

Holding a movement/gripper key auto-publishes `pause_policy` on `/teleop_command` (see
`keyboard_teleop_pose_orientation.py`); releasing it for ~0.5s auto-publishes `resume_policy`.
This only stops `smolvla_policy_node.py` from sending new pose/gripper commands for that
window — it does not e-stop the arm — so the keyboard's own commands land without the policy
re-issuing a conflicting target on the next control tick. Press `T` to fully toggle
teleop-only mode instead (holds the policy paused regardless of the debounce above); press `T`
again to hand control back to the policy.

Start the policy loop from another terminal, same as above:

```
ros2 topic pub /teleop_command std_msgs/String "data: start_policy" --once
```

Defaults to dry-run (logs actions, publishes nothing). Pass `live:=true` to
actually drive the arm — read `smolvla_policy_node.py`'s docstring first and
do the sign-convention check it describes before ever running live.
