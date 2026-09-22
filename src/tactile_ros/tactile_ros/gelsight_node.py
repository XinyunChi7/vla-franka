#!/usr/bin/env python3
"""
ROS2 node for the GelSight Mini tactile sensor.

RUN:
GSROBOTICS_PATH=/home/xinyun/vlm-franka/gsrobotics \
ros2 run tactile_ros gelsight_node \
  --ros-args \
  -p use_feats:=true \
  -p visualize:=true \
  -p feats_model_path:=/home/xinyun/vlm-franka/gsrobotics/feats/src/feats/models/unet_09042025_124903_80.pt \
  -p feats_norm_path:=/home/xinyun/vlm-franka/gsrobotics/feats/src/feats/data/labels/normalization_08042025_122519.npy


Scale with the 'force_gain' parameter; calibrate against a reference scale

Published topics '/tactile/gelsight':
  image_raw            (sensor_msgs/Image)            raw camera frame (rgb8)
  image_annotated       (sensor_msgs/Image)            frame with marker arrows
  force                 (geometry_msgs/WrenchStamped)  proxy force estimate
  force_map              fxyz estimated using FEAST

"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import WrenchStamped

try:
    import torch
    from tactile_ros.feats_model import load_feats
    from tactile_ros.feats_model import preprocess as _feats_preprocess
    from tactile_ros.feats_model import postprocess as _feats_postprocess
    _FEATS_AVAILABLE = True
except ImportError:
    _FEATS_AVAILABLE = False


def _find_gsrobotics_path() -> str:
    """Locate the vendored gsrobotics SDK.

    Honors the GSROBOTICS_PATH env var if set, otherwise walks up from this
    file looking for a 'gsrobotics/utilities' sibling. This works both when
    running from the source tree and from the colcon install tree, since
    both are nested under the same workspace root that also holds
    'gsrobotics'.
    """
    env_path = os.environ.get('GSROBOTICS_PATH')
    if env_path and os.path.isdir(env_path):
        return env_path

    here = Path(os.path.realpath(__file__))
    for ancestor in here.parents:
        candidate = ancestor / 'gsrobotics'
        if (candidate / 'utilities').is_dir():
            return str(candidate)
    return ''


_GSROBOTICS_PATH = _find_gsrobotics_path()
if _GSROBOTICS_PATH and _GSROBOTICS_PATH not in sys.path:
    sys.path.insert(0, _GSROBOTICS_PATH)

try:
    from utilities.gelsightmini import GelSightMini, Camera  # noqa: E402
    from utilities.marker_tracker import MarkerTracker  # noqa: E402
    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - exercised only when SDK missing
    GelSightMini = None
    Camera = None
    MarkerTracker = None
    _IMPORT_ERROR = exc

DEF_TARGET_WIDTH = 320
DEF_TARGET_HEIGHT = 240
DEF_BORDER_FRACTION = 0.15
DEF_PUBLISH_RATE_HZ = 30.0
DEF_FORCE_GAIN = 0.001  # N per pixel of marker displacement; uncalibrated default
DEF_DEVICE_NAME_FILTER = 'GelSight'


class GelsightNode(Node):
    def __init__(self):
        super().__init__('gelsight_node')

        if GelSightMini is None:
            raise RuntimeError(
                "Could not import the GelSight SDK from "
                f"'{_GSROBOTICS_PATH or '<not found>'}'. Set the "
                "GSROBOTICS_PATH environment variable to the gsrobotics "
                f"checkout. Original import error: {_IMPORT_ERROR}"
            )

        # -1 = auto-detect via 'device_name_filter' (recommended: the order
        # of /dev/v4l/by-id/* is not stable across runs/replugs, so a fixed
        # numeric index can silently land on the wrong USB interface, e.g. a
        # metadata-only node that fails to open for capture). Set to >= 0 to
        # force a specific by-id index instead.
        self.declare_parameter('device_index', -1)
        self.declare_parameter('device_name_filter', DEF_DEVICE_NAME_FILTER)
        self.declare_parameter('topic_prefix', 'tactile/gelsight')
        self.declare_parameter('frame_id', 'gelsight')
        self.declare_parameter('target_width', DEF_TARGET_WIDTH)
        self.declare_parameter('target_height', DEF_TARGET_HEIGHT)
        self.declare_parameter('border_fraction', DEF_BORDER_FRACTION)
        self.declare_parameter('publish_rate_hz', DEF_PUBLISH_RATE_HZ)
        self.declare_parameter('force_gain', DEF_FORCE_GAIN)
        self.declare_parameter('draw_markers', True)
        # If the fraction of markers still tracked drops below this for
        # 'reinit_after_bad_frames' consecutive frames (e.g. lighting change,
        # heavy deformation, marker pattern moved out of the LK search
        # window), the marker grid is reinitialized from a fresh frame
        # instead of spamming "did not converge" forever like the upstream
        # demo does.
        self.declare_parameter('min_tracked_ratio', 0.5)
        self.declare_parameter('reinit_after_bad_frames', 5)
        self.declare_parameter('use_feats', True)
        self.declare_parameter('feats_model_path', '')
        self.declare_parameter('feats_norm_path', '')
        self.declare_parameter('visualize', False)

        device_index = self.get_parameter('device_index').value
        device_name_filter = self.get_parameter('device_name_filter').value
        topic_prefix = self.get_parameter('topic_prefix').value
        self._frame_id = self.get_parameter('frame_id').value
        target_width = self.get_parameter('target_width').value
        target_height = self.get_parameter('target_height').value
        border_fraction = self.get_parameter('border_fraction').value
        publish_rate = self.get_parameter('publish_rate_hz').value
        self._force_gain = self.get_parameter('force_gain').value
        self._draw_markers = self.get_parameter('draw_markers').value
        self._min_tracked_ratio = self.get_parameter('min_tracked_ratio').value
        self._reinit_after_bad_frames = self.get_parameter('reinit_after_bad_frames').value
        self._use_feats = self.get_parameter('use_feats').value
        feats_model_path = self.get_parameter('feats_model_path').value
        feats_norm_path = self.get_parameter('feats_norm_path').value
        self._visualize = self.get_parameter('visualize').value

        self._bridge = CvBridge()

        ns = topic_prefix.strip('/')
        self._pub_image = self.create_publisher(Image, f'{ns}/image_raw', 10)
        self._pub_annotated = self.create_publisher(Image, f'{ns}/image_annotated', 10)
        self._pub_markers = self.create_publisher(
            Float32MultiArray, f'{ns}/marker_displacement', 10)
        self._pub_force = self.create_publisher(WrenchStamped, f'{ns}/force', 10)
        self._pub_data = self.create_publisher(String, f'{ns}/data', 10)
        self._pub_force_map = self.create_publisher(Image, f'{ns}/force_map', 10)

        # FEATS markerless force estimation
        self._feats_model = None
        self._feats_norm = None
        self._feats_device = None
        if self._use_feats:
            if not _FEATS_AVAILABLE:
                raise RuntimeError(
                    'use_feats=True but torch or tactile_ros.feats_model could not be imported')
            if not feats_model_path or not feats_norm_path:
                raise RuntimeError(
                    'use_feats=True requires feats_model_path and feats_norm_path parameters')
            self._feats_model, self._feats_norm, self._feats_device = load_feats(
                feats_model_path, feats_norm_path)
            self.get_logger().info(f'FEATS model loaded from {feats_model_path}')

        self._cam = GelSightMini(
            target_width=target_width,
            target_height=target_height,
            border_fraction=border_fraction,
        )
        if device_index < 0:
            device_index = self._discover_device_index(device_name_filter)
        self._cam.select_device(device_index)
        self._cam.start()

        # Marker-tracking state, set up lazily once the first frame arrives.
        self._tracker_initialized = False
        self._Ox = None
        self._Oy = None
        self._old_gray = None
        self._p0 = None
        self._bad_frame_count = 0
        self._lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        period = 1.0 / publish_rate if publish_rate > 0 else 1.0 / DEF_PUBLISH_RATE_HZ
        self._timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"Gelsight node started (device {device_index}, "
            f"publishing under '/{ns}')")

    # ------------------------------------------------------------------
    # Device discovery
    # ------------------------------------------------------------------

    def _discover_device_index(self, name_filter: str) -> int:
        """Resolve the by-id index for the GelSight camera by USB product
        name instead of trusting glob ordering of /dev/v4l/by-id/*, which is
        not guaranteed stable across runs/replugs. A fixed numeric index can
        otherwise land on the wrong interface of the device (e.g. a
        metadata-only node that enumerates but fails to open for capture).
        """
        devices = Camera.list_devices()
        candidates = sorted(
            ((idx, path) for idx, path in devices.items()
             if name_filter.lower() in path.lower()),
            key=lambda item: item[1],
        )
        if not candidates:
            self.get_logger().warn(
                f"No /dev/v4l/by-id device matched '{name_filter}' "
                f"(seen: {list(devices.values())}); falling back to index 0.")
            return 0

        # by-id names end in '-video-indexN'; the lowest N is the device's
        # primary video-streaming interface, higher ones are typically
        # metadata/extension nodes. Sorting candidates by path puts indexN
        # in ascending order, so the first match is the one we want.
        chosen_idx, chosen_path = candidates[0]
        self.get_logger().info(f'Auto-selected device {chosen_idx}: {chosen_path}')
        return chosen_idx

    # ------------------------------------------------------------------
    # Marker tracking
    # ------------------------------------------------------------------

    def _init_tracker(self, frame: np.ndarray) -> bool:
        """Try to detect the marker grid in `frame` and seed tracking state.
        Returns False (without raising) on a frame where too few markers were
        detected to form a grid, so the caller can just retry on the next
        frame instead of crashing the node."""
        img = np.float32(frame) / 255.0
        try:
            marker_tracker = MarkerTracker(img)
        except (IndexError, ValueError) as e:
            self.get_logger().warn(f'Marker detection failed on this frame, retrying: {e}')
            return False
        marker_centers = marker_tracker.initial_marker_center
        if marker_centers.ndim != 2 or marker_centers.shape[0] == 0:
            self.get_logger().warn('No markers detected on this frame, retrying.')
            return False
        self._Ox = marker_centers[:, 1]
        self._Oy = marker_centers[:, 0]
        self._old_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        self._p0 = np.stack([self._Ox, self._Oy], axis=1).astype(np.float32).reshape(-1, 1, 2)
        self._bad_frame_count = 0
        self.get_logger().info(f'Initialized marker tracking with {len(marker_centers)} markers')
        return True

    def _track_markers(self, frame: np.ndarray) -> np.ndarray:
        """Run LK optical flow on the marker grid and draw arrows (if
        enabled) directly onto `frame`. Returns per-marker (dx, dy)
        displacement relative to each marker's rest position, shape (N, 2)
        where N is the number of markers still being tracked this frame."""
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        p1, st, _err = cv2.calcOpticalFlowPyrLK(
            self._old_gray, frame_gray, self._p0, None, **self._lk_params)
        self._old_gray = frame_gray

        mask = st.flatten() == 1
        tracked = p1.reshape(-1, 2)
        rest_x, rest_y = self._Ox[mask], self._Oy[mask]
        displacement = tracked[mask] - np.stack([rest_x, rest_y], axis=1)

        tracked_ratio = mask.mean() if len(mask) else 0.0
        if mask.all():
            self._p0 = p1.reshape(-1, 1, 2)
        else:
            self.get_logger().debug(f'{(~mask).sum()} marker(s) lost tracking this frame')

        if tracked_ratio < self._min_tracked_ratio:
            self._bad_frame_count += 1
        else:
            self._bad_frame_count = 0

        if self._draw_markers:
            for (a, b), ix, iy in zip(tracked[mask], rest_x.astype(int), rest_y.astype(int)):
                cv2.arrowedLine(
                    frame, (ix, iy), (int(a), int(b)), (255, 255, 255),
                    thickness=1, line_type=cv2.LINE_8, tipLength=0.15)

        return displacement

    # ------------------------------------------------------------------
    # Main loop / publishing
    # ------------------------------------------------------------------

    def _on_timer(self) -> None:
        frame = self._cam.update(dt=0)
        if frame is None:
            return

        stamp = self.get_clock().now().to_msg()

        if self._use_feats:
            self._publish_image(self._pub_image, frame, stamp)
            self._publish_feats_force(frame, stamp)
            return

        if not self._tracker_initialized:
            self._tracker_initialized = self._init_tracker(frame)
            return

        annotated = frame.copy()
        displacement = self._track_markers(annotated)

        if self._bad_frame_count >= self._reinit_after_bad_frames:
            self.get_logger().warn(
                f'Lost tracking on too many markers for '
                f'{self._bad_frame_count} frames in a row; reinitializing '
                f'marker grid from the current frame.')
            self._tracker_initialized = False

        self._publish_image(self._pub_image, frame, stamp)
        self._publish_image(self._pub_annotated, annotated, stamp)
        self._publish_markers_and_force(displacement, stamp)

    def _publish_feats_force(self, frame: np.ndarray, stamp) -> None:
        tensor = _feats_preprocess(frame, self._feats_device)
        with torch.no_grad():
            output = self._feats_model(tensor)
        gx, gy, gz = _feats_postprocess(output, self._feats_norm)

        fx, fy, fz = float(gx.sum()), float(gy.sum()), float(gz.sum())

        wrench = WrenchStamped()
        wrench.header.stamp = stamp
        wrench.header.frame_id = self._frame_id
        wrench.wrench.force.x = fx
        wrench.wrench.force.y = fy
        wrench.wrench.force.z = fz
        self._pub_force.publish(wrench)

        force_map = self._render_force_map(frame, gx, gy, gz, fx, fy, fz)
        self._publish_image(self._pub_force_map, force_map, stamp)

        if self._visualize:
            cv2.imshow('FEATS Force Estimation',
                       cv2.cvtColor(force_map, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        combined = String()
        combined.data = json.dumps({
            'force_estimate': {'x': fx, 'y': fy, 'z': fz},
            'method': 'feats',
        })
        self._pub_data.publish(combined)

    def _render_force_map(
        self,
        frame: np.ndarray,
        gx: np.ndarray,
        gy: np.ndarray,
        gz: np.ndarray,
        fx: float,
        fy: float,
        fz: float,
    ) -> np.ndarray:
        """Camera feed + Fx/Fy/Fz colormaps in one 4-panel RGB image."""
        font = cv2.FONT_HERSHEY_SIMPLEX

        def labeled_panel(img_rgb: np.ndarray, title: str) -> np.ndarray:
            panel = img_rgb.copy()
            cv2.putText(panel, title, (6, 22), font, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(panel, title, (6, 22), font, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            return panel

        def force_panel(arr: np.ndarray, title: str) -> np.ndarray:
            lo, hi = arr.min(), arr.max()
            norm = ((arr - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            colored = cv2.resize(colored, (320, 240), interpolation=cv2.INTER_NEAREST)
            return labeled_panel(colored, title)

        cam_panel = labeled_panel(
            cv2.resize(frame, (320, 240)),
            f'Camera  F=({fx:.3f},{fy:.3f},{fz:.3f})')
        fx_panel = force_panel(gx, f'Fx  sum={fx:.4f}')
        fy_panel = force_panel(gy, f'Fy  sum={fy:.4f}')
        fz_panel = force_panel(gz, f'Fz  sum={fz:.4f}')

        return np.concatenate([cam_panel, fx_panel, fy_panel, fz_panel], axis=1)

    def _publish_image(self, publisher, frame: np.ndarray, stamp) -> None:
        msg = self._bridge.cv2_to_imgmsg(frame, encoding='rgb8')
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id
        publisher.publish(msg)

    def _publish_markers_and_force(self, displacement: np.ndarray, stamp) -> None:
        marker_msg = Float32MultiArray()
        marker_msg.data = [float(v) for v in displacement.flatten()]
        self._pub_markers.publish(marker_msg)

        if len(displacement):
            mean_dx, mean_dy = displacement.mean(axis=0)
            mean_mag = float(np.linalg.norm(displacement, axis=1).mean())
        else:
            mean_dx, mean_dy, mean_mag = 0.0, 0.0, 0.0

        wrench = WrenchStamped()
        wrench.header.stamp = stamp
        wrench.header.frame_id = self._frame_id
        wrench.wrench.force.x = float(mean_dx) * self._force_gain
        wrench.wrench.force.y = float(mean_dy) * self._force_gain
        wrench.wrench.force.z = mean_mag * self._force_gain
        self._pub_force.publish(wrench)

        combined = String()
        combined.data = json.dumps({
            'marker_displacement': [float(v) for v in displacement.flatten()],
            'force_estimate': {
                'x': wrench.wrench.force.x,
                'y': wrench.wrench.force.y,
                'z': wrench.wrench.force.z,
            },
        })
        self._pub_data.publish(combined)

    # ------------------------------------------------------------------

    def destroy_node(self):
        if getattr(self._cam, 'camera', None):
            self._cam.camera.release()
        if self._visualize:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = GelsightNode()
    except RuntimeError as exc:
        print(f'[gelsight_node] {exc}', file=sys.stderr)
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
