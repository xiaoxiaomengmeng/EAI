"""Qt views (source panels, MuJoCo view) and the main window driving them."""
from __future__ import annotations

import json
import sys
import threading
import time
import traceback

import mujoco
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.spatial.transform import Rotation

from sonic_config import (
    AXIS_COLORS,
    BODY_EDGES,
    CAMERA_FPS,
    FPS,
    FUTURE_FRAMES,
    HAND_EDGES,
    HAND_TRACKER_NAMES,
    JOINT_NAMES,
    ROTATION_OFFSET_JOINTS,
    ROTATION_OFFSETS_PATH,
)
from sonic_policy import Simulator
from sonic_recording import load_rotation_offsets, load_skeleton
from sonic_smpl import apply_rotation_sources, convert
from sonic_source import LatestFrame, publish_frames

from sim_scenes import LockChain, SceneView, scene_render_options
from sim_scenes.capture_panel import CapturePanel
from sim_scenes.mouse_view import MouseCameraView
from sim_scenes.recording_controls import RecordingControls
from sim_scenes.robot_cameras import (
    CAMERA_MOUNTS,
    RobotCameraPanel,
    render_camera_images,
)
from sim_scenes.source_streams import SourceViewStreams
from sim_scenes.streaming_controls import StreamingControls
from sim_scenes.viewport import fit_renderer



def source_basis(pose: np.ndarray) -> np.ndarray:
    left = np.mean([
        pose[JOINT_NAMES.index(f"left_{joint}"), :3] - pose[JOINT_NAMES.index(f"right_{joint}"), :3]
        for joint in ("shoulder", "hip")
    ], axis=0)
    left[1] = 0.0
    left /= max(np.linalg.norm(left), 1e-8)
    up = np.array([0.0, 1.0, 0.0])
    return np.stack((np.cross(left, up), left, up))


class SourceView(SourceViewStreams, QtWidgets.QWidget):
    body_fraction = .62

    def __init__(self):
        super().__init__()
        self.pose = None
        self.trackers = None
        self.gloves = None
        self.basis = np.eye(3)
        self.adjustment_joint = None
        self.rotation_offsets = None
        self.setMinimumHeight(450)

    def set_frame(
        self,
        pose: np.ndarray,
        trackers: np.ndarray,
        gloves: dict[str, np.ndarray],
    ) -> None:
        self.pose, self.trackers = pose, trackers
        self.gloves, self.basis = gloves, source_basis(pose)
        self.update()

    def set_adjustment(
        self,
        joint: str | None,
        rotation_offsets: dict[str, np.ndarray] | None,
    ) -> None:
        self.adjustment_joint = joint
        self.rotation_offsets = rotation_offsets
        self.update()

    @staticmethod
    def project(points: np.ndarray, rect: QtCore.QRectF) -> list[QtCore.QPointF]:
        area = rect.adjusted(12, 27, -12, -10)
        low, high = points.min(axis=0), points.max(axis=0)
        span = np.maximum(high - low, 1e-4)
        scale = 0.9 * min(area.width() / span[0], area.height() / span[1])
        center = 0.5 * (low + high)
        return [QtCore.QPointF(area.center().x() + (point[0] - center[0]) * scale,
                               area.center().y() - (point[1] - center[1]) * scale) for point in points]

    @staticmethod
    def panel(painter: QtGui.QPainter, rect: QtCore.QRectF, title: str) -> None:
        painter.setPen(QtGui.QPen(QtGui.QColor("#343b45"), 1))
        painter.setBrush(QtGui.QColor("#1a2028"))
        painter.drawRoundedRect(rect, 7, 7)
        painter.setPen(QtGui.QColor("#cbd5e1"))
        painter.drawText(rect.adjusted(9, 6, -9, -6), QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop, title)

    def draw_body(self, painter: QtGui.QPainter, rect: QtCore.QRectF, axes: tuple[int, int], title: str) -> None:
        self.panel(painter, rect, title)
        if self.pose is None:
            return
        waist = JOINT_NAMES.index("waist")
        tracker_rot = Rotation.from_quat(self.trackers[:, 3:]).as_matrix()
        tracker_axes = self.trackers[:, None, :3] + 0.1 * np.transpose(
            tracker_rot, (0, 2, 1)
        )
        raw_points = np.concatenate(
            (self.pose[:, :3], self.trackers[:, :3], tracker_axes.reshape(-1, 3))
        )
        adjustment_start = len(raw_points)
        adjustment_origin = None
        if self.adjustment_joint is not None:
            if self.adjustment_joint.endswith("wrist"):
                side = 0 if self.adjustment_joint.startswith("left") else 1
                adjustment_origin = self.trackers[side, :3]
                raw_rotation = tracker_rot[side]
            else:
                joint = JOINT_NAMES.index(self.adjustment_joint)
                adjustment_origin = self.pose[joint, :3]
                raw_rotation = Rotation.from_quat(
                    self.pose[joint, 3:]
                ).as_matrix()
            correction = Rotation.from_euler(
                "XYZ", self.rotation_offsets[self.adjustment_joint], degrees=True
            ).as_matrix()
            raw_axes = adjustment_origin + 0.14 * raw_rotation.T
            corrected_axes = adjustment_origin + 0.14 * (raw_rotation @ correction).T
            raw_points = np.concatenate((raw_points, raw_axes, corrected_axes))
        aligned = (raw_points - self.pose[waist, :3]) @ self.basis.T
        points = self.project(aligned[:, axes], rect)
        for start, end in BODY_EDGES:
            names = JOINT_NAMES[start], JOINT_NAMES[end]
            color = "#48c6b8" if any("left_" in name for name in names) else "#f0a75e" if any("right_" in name for name in names) else "#d4d8df"
            painter.setPen(QtGui.QPen(QtGui.QColor(color), 2.2))
            painter.drawLine(points[start], points[end])
        for side, color in enumerate(("#48c6b8", "#f0a75e")):
            tracker = points[len(JOINT_NAMES) + side]
            wrist = points[JOINT_NAMES.index(f"{'left' if side == 0 else 'right'}_wrist")]
            painter.setPen(QtGui.QPen(
                QtGui.QColor(color), 1.2, QtCore.Qt.PenStyle.DashLine
            ))
            painter.drawLine(wrist, tracker)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(color))
            painter.drawEllipse(tracker, 4.0, 4.0)
            if self.adjustment_joint == f"{'left' if side == 0 else 'right'}_wrist":
                continue
            for axis, axis_color in enumerate(AXIS_COLORS):
                painter.setPen(QtGui.QPen(QtGui.QColor(axis_color), 1.7))
                painter.drawLine(
                    tracker, points[len(JOINT_NAMES) + len(HAND_TRACKER_NAMES) + 3 * side + axis]
                )
        if adjustment_origin is not None:
            if self.adjustment_joint.endswith("wrist"):
                side = 0 if self.adjustment_joint.startswith("left") else 1
                origin = points[len(JOINT_NAMES) + side]
            else:
                origin = points[JOINT_NAMES.index(self.adjustment_joint)]
            for axis, color in enumerate(AXIS_COLORS):
                painter.setPen(QtGui.QPen(
                    QtGui.QColor(color), 1.3, QtCore.Qt.PenStyle.DashLine
                ))
                painter.drawLine(origin, points[adjustment_start + axis])
                painter.setPen(QtGui.QPen(QtGui.QColor(color), 2.8))
                painter.drawLine(origin, points[adjustment_start + 3 + axis])

    def draw_hand(self, painter: QtGui.QPainter, rect: QtCore.QRectF, side: str) -> None:
        self.panel(painter, rect, f"{'左' if side == 'left' else '右'}手原始关键点  21/21")
        if self.gloves is None:
            return
        points = self.gloves[side][..., :3]
        screen = self.project((points - points[0])[:, (0, 2)], rect)
        painter.setPen(QtGui.QPen(QtGui.QColor("#48c6b8" if side == "left" else "#f0a75e"), 2.0))
        for start, end in HAND_EDGES:
            painter.drawLine(screen[start], screen[end])

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor("#11161c"))
        panels = self.panel_rects()
        self.draw_body(painter, panels['source_body_front'], (1, 2), "身体正视  24 + 手 tracker 2")
        self.draw_body(painter, panels['source_body_side'], (0, 2), "身体侧视  24 + 手 tracker 2")
        self.draw_hand(painter, panels['source_left_hand'], "left")
        self.draw_hand(painter, panels['source_right_hand'], "right")


class MuJoCoView(MouseCameraView):
    def __init__(self):
        super().__init__("正在加载场景…")
        self.image = None
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(620, 620)
        self.setStyleSheet("background:#07090c;color:#8792a2;font-size:16px;")

    def set_image(self, image: np.ndarray) -> None:
        height, width = image.shape[:2]
        self.image = QtGui.QImage(image.data, width, height, image.strides[0], QtGui.QImage.Format.Format_RGB888).copy()
        self.update_pixmap()

    def update_pixmap(self) -> None:
        if self.image is not None:
            self.setPixmap(QtGui.QPixmap.fromImage(self.image).scaled(
                self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.update_pixmap()


class Window(QtWidgets.QMainWindow):
    def __init__(self, frames, xixi_scene: bool = False, *, scene: str | None = None,
                 render_shadows: bool = False,
                 rotation_offsets: dict[str, np.ndarray] | None = None,
                 pause_event: threading.Event | None = None,
                 stream_config=None, stream_enabled: bool = False):
        super().__init__()
        self.sim = Simulator(xixi_scene, scene=scene)
        self.scene_option = scene_render_options(self.sim.pulse, self.sim.model)
        self.rotation_offsets = load_rotation_offsets() if rotation_offsets is None else rotation_offsets
        self.pause_event = pause_event
        self.adjusting = False
        self.current_frame = None
        self.selected_joint = ROTATION_OFFSET_JOINTS[0][0]
        self.rest, self.parents = load_skeleton()
        self.latest = LatestFrame()
        self.failed = False
        self.finished_reported = False
        self.sim.model.vis.global_.offwidth = 800
        self.sim.model.vis.global_.offheight = 800
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.lookat[:] = [0.15, 0.0, 0.8]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 2.5, 135, -15
        self.scene_view = SceneView(self.sim.model, self.sim.data, self.camera,
                                    render_shadows=render_shadows)
        # Finalize scene bounds and shadow resolution before allocating Renderer.
        self.renderer = mujoco.Renderer(self.sim.model, height=800, width=800)
        # Asynchronous rendering: `renderer` stays on the GUI thread for
        # interactive calls; the worker builds its own GL context, because a
        # MuJoCo Renderer may only be used from the thread that created it.
        # scene_lock keeps worker snapshots out of sim.step; render_lock guards
        # the shared model state those snapshots read and rewrite (camera
        # framing, clipping, shadow fit). Both are held for one snapshot only:
        # mjr_render reads the renderer's own scene, so the draw runs outside
        # them and a drag never waits for a whole batch to finish.
        self.scene_lock = threading.Lock()
        self.render_lock = threading.RLock()
        self.render_snapshot = LockChain(self.render_lock, self.scene_lock)
        self.render_request = threading.Condition()
        self.render_pending = False
        self.render_size = (800, 800)
        self.render_sources = None
        self.render_stop = False
        self.async_lock = threading.Lock()
        self.async_result = None
        self.worker_renderer = None
        # Body cameras render on their own threads, one renderer per view, so
        # every preview runs at CAMERA_FPS while the worker only draws the main
        # view. These slots hold the newest images; the Qt tick and the stream
        # publisher consume them without waiting for a render request.
        self.camera_lock = threading.Lock()
        self.camera_images = {}
        self.camera_rendered_at = {}
        self.camera_stop = threading.Event()
        self.publish_lock = threading.Lock()
        self.publish_frame = None
        self.publish_generation = 0
        self.publish_stop = threading.Event()
        # A worker thread may not touch Qt widgets, so its failures are queued
        # here and reported by the GUI tick: a dead render/camera/publish thread
        # otherwise shows up only as a view that silently stops updating.
        self.thread_error_lock = threading.Lock()
        self.thread_errors = []
        self.render_ms = 0.0
        self.display_fps = 0.0
        self.last_display_at = None
        self.streaming = StreamingControls(stream_config)
        self.build_ui()
        self.streaming.started.connect(self.render_frame)
        self.streaming.toggle.setChecked(stream_enabled)
        self.render_frame()
        self.statusBar().showMessage("场景已就绪 · 等待数据连接或第一帧…")
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.consume_frame)
        self.timer.start(round(1000 / FPS))
        threading.Thread(target=publish_frames, args=(frames, self.latest), daemon=True).start()
        self.render_thread = threading.Thread(
            target=self.render_loop, name="sonic-render", daemon=True)
        self.render_thread.start()
        self.camera_threads = [
            threading.Thread(target=self.camera_loop, args=(name,),
                             name=f"sonic-camera-{name}", daemon=True)
            for name, _, _, _ in CAMERA_MOUNTS
        ]
        for thread in self.camera_threads:
            thread.start()
        self.publish_thread = threading.Thread(
            target=self.publish_loop, name="sonic-publish", daemon=True)
        self.publish_thread.start()

    def build_ui(self) -> None:
        self.setWindowTitle("G1 SONIC v1.1")
        self.resize(1500, 900)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        self.resize_timer = QtCore.QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.render_frame)
        self.mujoco_view = MuJoCoView()
        self.mujoco_view.viewport_resized.connect(lambda: self.resize_timer.start(80))
        self.mujoco_view.orbit_requested.connect(self.orbit_camera)
        self.mujoco_view.zoom_requested.connect(self.zoom_camera)
        self.mujoco_view.setMinimumSize(480, 400)
        self.mujoco_view.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                       QtWidgets.QSizePolicy.Policy.Ignored)
        simulation = QtWidgets.QVBoxLayout()
        simulation.setSpacing(8)
        simulation.addWidget(self.mujoco_view, 3)
        self.robot_cameras = RobotCameraPanel(self.sim.model)
        self.recording = RecordingControls(
            self.sim.model, source_fps=FPS, render_shadows=self.scene_view.shadows.enabled,
            action_semantics="Clipped SONIC position targets for body joints; clipped hand retargeting targets for fingers.",
        )
        self._closing_after_export = False
        self.recording.settled.connect(self.finish_recording_close)
        self.capture_panel = CapturePanel(self.robot_cameras, self.recording, self.mujoco_view)
        self.capture_panel.expanded_changed.connect(
            lambda expanded: self.render_frame() if expanded else None
        )
        simulation.addWidget(self.capture_panel, 1)
        layout.addLayout(simulation, 1)
        right = QtWidgets.QWidget()
        right.setFixedWidth(470)
        right_layout = QtWidgets.QVBoxLayout(right)
        self.source_view = SourceView()
        right_layout.addWidget(self.source_view, 1)
        self.info = QtWidgets.QLabel(
            "SONIC v1.1 ONNX（CPU）\n本体：29 DoF RL · 手部：Wuji retargeting"
        )
        self.info.setStyleSheet(
            "padding:10px;background:#1a2028;border:1px solid #343b45;border-radius:6px;"
        )
        right_layout.addWidget(self.info)
        self.adjust_button = QtWidgets.QPushButton("进入姿态调节")
        self.adjust_button.clicked.connect(self.enter_adjustment)
        self.adjust_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        right_layout.addWidget(self.adjust_button)
        self.adjustment_panel = self.make_adjustment_panel()
        self.adjustment_panel.hide()
        right_layout.addWidget(self.adjustment_panel)
        self.camera_selector = QtWidgets.QComboBox()
        self.camera_selector.addItems(["场景总览", "跟随机器人"])
        self.camera_selector.setCurrentIndex(1)
        self.camera_selector.setAccessibleName("相机视角")
        self.camera_selector.currentIndexChanged.connect(self.select_camera)
        right_layout.addWidget(self.camera_selector)
        right_layout.addWidget(self.streaming)
        reset = QtWidgets.QPushButton("重置机器人")
        reset.clicked.connect(self.reset)
        reset.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        right_layout.addWidget(reset)
        layout.addWidget(right)
        self.setCentralWidget(central)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_InputMethodEnabled, False)
        for widget in self.findChildren(QtWidgets.QWidget):
            widget.setAttribute(
                QtCore.Qt.WidgetAttribute.WA_InputMethodEnabled, False
            )
        self.statusBar().showMessage("等待第一帧…")
        self.setStyleSheet(
            "QMainWindow,QWidget{background:#11161c;color:#e4e9ef;}"
            "QPushButton{background:#252d38;border:1px solid #3b4654;"
            "border-radius:5px;padding:8px;}"
            "QPushButton:checked{background:#385777;border-color:#65a7e8;}"
        )

    def make_adjustment_panel(self) -> QtWidgets.QGroupBox:
        panel = QtWidgets.QGroupBox("当前帧姿态调节")
        layout = QtWidgets.QVBoxLayout(panel)
        help_text = QtWidgets.QLabel(
            "局部坐标轴：X 红 · Y 绿 · Z 蓝\n虚线是原始方向，实线是修正后方向"
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        joints = QtWidgets.QHBoxLayout()
        self.joint_buttons = {}
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        for name, label in ROTATION_OFFSET_JOINTS:
            button = QtWidgets.QPushButton(label.replace(" tracker", ""))
            button.setCheckable(True)
            button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            button.clicked.connect(
                lambda checked, name=name: checked and self.select_joint(name)
            )
            group.addButton(button)
            joints.addWidget(button)
            self.joint_buttons[name] = button
        layout.addLayout(joints)
        self.axis_sliders = []
        self.axis_values = []
        for axis, (name, color) in enumerate(zip("XYZ", AXIS_COLORS)):
            row = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(f"{name}")
            label.setStyleSheet(f"color:{color};font-weight:700;")
            row.addWidget(label)
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setRange(-1800, 1800)
            slider.setSingleStep(10)
            slider.setPageStep(50)
            slider.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            slider.setStyleSheet(
                f"QSlider::handle:horizontal{{background:{color};width:14px;"
                "margin:-5px 0;border-radius:7px;}"
            )
            value = QtWidgets.QLabel("0.0°")
            value.setFixedWidth(52)
            reset_axis = QtWidgets.QPushButton("归零")
            reset_axis.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            reset_axis.clicked.connect(lambda _, axis=axis: self.reset_axis(axis))
            slider.valueChanged.connect(
                lambda raw, axis=axis: self.change_axis(axis, raw / 10.0)
            )
            row.addWidget(slider, 1)
            row.addWidget(value)
            row.addWidget(reset_axis)
            layout.addLayout(row)
            self.axis_sliders.append(slider)
            self.axis_values.append(value)
        actions = QtWidgets.QHBoxLayout()
        reset_all = QtWidgets.QPushButton("全部归零")
        reset_all.clicked.connect(self.reset_all_offsets)
        finish = QtWidgets.QPushButton("保存并继续")
        finish.clicked.connect(self.finish_adjustment)
        for button in (reset_all, finish):
            button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            actions.addWidget(button)
        layout.addLayout(actions)
        return panel

    def render_main_view(self) -> np.ndarray:
        """Draw the main view now: short snapshot under the locks, draw outside."""
        rect = self.mujoco_view.contentsRect()
        with self.render_snapshot:
            self.renderer = fit_renderer(self.renderer, self.sim.model, rect)
            self.render_size = (rect.width(), rect.height())
            self.scene_view.prepare(self.renderer, self.scene_option)
        return self.renderer.render()

    def render_frame(self) -> None:
        """Render synchronously on the GUI thread (setup and interactive calls)."""
        self.apply_async_result()
        captured_at = time.time()
        main_image = self.render_main_view()
        self.mujoco_view.set_image(main_image)
        if self.capture_panel.expanded or self.streaming.active:
            images = self.robot_cameras.render_frame(self.sim.data, self.scene_option,
                                                     self.scene_view.shadows,
                                                     snapshot=self.render_snapshot)
            if self.streaming.active:
                self.streaming.publish({"main": main_image, **images,
                                        **self.source_view.stream_frames()}, captured_at)

    def request_async_render(self) -> None:
        """Queue one worker render of the latest state; returns immediately."""
        if self.streaming.active and self.render_sources is None:
            # QWidget painting stays on the GUI thread; the worker pairs it
            # with its render and publishes from its own thread. consume_frame
            # captures once per tick, so a drag only repaints the panels after
            # the previous capture has been consumed. The capture itself stays
            # outside publish_lock (QWidget.render costs milliseconds); only the
            # hand-off is locked, so the publisher's take-and-clear cannot drop
            # a panel set the GUI thread has just stored.
            panels = self.source_view.stream_frames()
            with self.publish_lock:
                self.render_sources = panels
        rect = self.mujoco_view.contentsRect()
        with self.render_request:
            self.render_size = (rect.width(), rect.height())
            self.render_pending = True
            self.render_request.notify()

    def apply_async_result(self) -> None:
        """Show the newest completed worker frame; Qt widgets stay on the GUI thread."""
        with self.async_lock:
            result, self.async_result = self.async_result, None
        if result is None:
            return
        main_image, camera_images, self.render_ms = result
        self.mujoco_view.set_image(main_image)
        if camera_images is not None and self.capture_panel.expanded:
            for name, image in camera_images.items():
                view = self.robot_cameras.views.get(name)
                if view is not None:
                    view.set_image(image)
        now = time.time()
        if self.last_display_at is not None:
            instant = 1.0 / max(1e-3, now - self.last_display_at)
            self.display_fps = instant if not self.display_fps else .9 * self.display_fps + .1 * instant
        self.last_display_at = now

    def report_thread_error(self, what: str) -> None:
        """Queue a worker-thread failure for the GUI tick to report.

        Called from the render, camera and publish threads; the traceback is
        formatted here because it is only current inside the except block.
        """
        detail = traceback.format_exc()
        with self.thread_error_lock:
            self.thread_errors.append((what, detail))

    def take_thread_errors(self) -> list:
        with self.thread_error_lock:
            errors, self.thread_errors = self.thread_errors, []
        return errors

    def render_loop(self) -> None:
        """Worker: renders the latest simulation state on its own GL context."""
        try:
            while True:
                with self.render_request:
                    while not self.render_stop and not self.render_pending:
                        self.render_request.wait()
                    if self.render_stop:
                        break
                    self.render_pending = False
                    size = self.render_size
                captured_at = time.time()
                with self.render_lock:
                    self.worker_renderer = fit_renderer(self.worker_renderer, self.sim.model, size)
                    with self.scene_lock:
                        self.scene_view.prepare(self.worker_renderer, self.scene_option)
                # One array, two deferred consumers: render() allocates per call
                # (SceneView.render), so the publish grace period below cannot see
                # the next frame overwrite this one.
                main_image = self.worker_renderer.render()
                with self.camera_lock:
                    complete = all(name in self.camera_images for name, _, _, _ in CAMERA_MOUNTS)
                    camera_images = dict(self.camera_images) if complete else None
                # The publisher pairs this frame with the newest body camera
                # images on its own thread, so the main loop never waits for them.
                with self.publish_lock:
                    self.publish_generation += 1
                    self.publish_frame = (self.publish_generation, main_image, captured_at)
                with self.async_lock:
                    self.async_result = (main_image, camera_images,
                                         (time.time() - captured_at) * 1000.0)
        except Exception:
            # Without this the thread dies on the default excepthook (stderr
            # only). A dead worker freezes the main view while zoom keeps
            # working -- zoom redraws synchronously on the GUI thread -- which
            # looks exactly like the input problem it is not.
            self.report_thread_error("主画面渲染")
        finally:
            if self.worker_renderer is not None:
                self.worker_renderer.close()
            self.worker_renderer = None

    def camera_loop(self, name: str) -> None:
        """Render one body camera on its own thread, paced at CAMERA_FPS.

        The renderer is created and closed on this thread because a MuJoCo
        Renderer may only be used from the thread that made it current.
        """
        renderer = None
        try:
            deadline = time.perf_counter()
            while not self.camera_stop.is_set():
                if not (self.capture_panel.expanded or self.streaming.active):
                    # Nothing shows this camera: release the GL context until a
                    # preview panel or the stream needs it again.
                    if renderer is not None:
                        renderer.close()
                        renderer = None
                    self.camera_stop.wait(.1)
                    deadline = time.perf_counter()
                    continue
                if renderer is None:
                    renderer = mujoco.Renderer(self.sim.model, height=180, width=320)
                    deadline = time.perf_counter()
                gap = deadline - time.perf_counter()
                while gap > .002:
                    # macOS sleeps overshoot by ~50%; approach the deadline in
                    # shrinking steps instead of sleeping the whole gap.
                    time.sleep(gap * .55)
                    gap = deadline - time.perf_counter()
                images = render_camera_images(
                    renderer, self.sim.model, self.sim.data, self.scene_option,
                    self.scene_view.shadows, snapshot=self.render_snapshot, names=[name])
                with self.camera_lock:
                    self.camera_images.update(images)
                    self.camera_rendered_at[name] = time.time()
                deadline += 1 / CAMERA_FPS
                if deadline < time.perf_counter() - 1 / CAMERA_FPS:
                    deadline = time.perf_counter()
        except Exception:
            # The publisher keeps sending this view's last image and now burns
            # the whole grace period every frame, so name the camera that died.
            self.report_thread_error(f"{name} 相机")
        finally:
            if renderer is not None:
                renderer.close()

    def publish_loop(self) -> None:
        """Pair main frames with body cameras and source panels for the stream.

        Body cameras get a short grace period to catch up with the main
        snapshot, so a published frame never mixes images older than its own
        main view; the waiting happens here, never on a render thread. The
        source panels are captured on the GUI thread, so they are taken with the
        frame under one lock: the grace period below must not let a later tick
        swap in panels from another moment, and a frame this loop drops must not
        leave its capture behind.
        """
        try:
            published = 0
            while not self.publish_stop.is_set():
                with self.publish_lock:
                    frame, self.publish_frame = self.publish_frame, None
                    # Only a frame this loop actually took may claim the panels:
                    # clearing them while the slot is empty would steal the capture
                    # that the next frame is about to publish.
                    sources = None
                    if frame is not None:
                        sources, self.render_sources = self.render_sources, None
                if frame is None:
                    self.publish_stop.wait(.005)
                    continue
                generation, main_image, captured_at = frame
                if generation <= published or not self.streaming.active:
                    continue
                deadline = time.time() + .1
                while time.time() < deadline:
                    with self.camera_lock:
                        fresh = all(self.camera_rendered_at.get(name, 0) >= captured_at
                                    for name, _, _, _ in CAMERA_MOUNTS)
                    if fresh:
                        break
                    time.sleep(.002)
                with self.camera_lock:
                    cameras = dict(self.camera_images)
                if len(cameras) != len(CAMERA_MOUNTS) or sources is None:
                    continue
                published = generation
                self.streaming.publish({"main": main_image, **cameras, **sources}, captured_at)
        except Exception:
            # The stream would simply stop delivering frames while the controls
            # still show a healthy address, so surface it instead.
            self.report_thread_error("推流发布")

    def orbit_camera(self, dx: float, dy: float) -> None:
        # Drags arrive at mouse-event rate, so the GUI thread only applies the
        # state change; frames come from the worker like they do every tick.
        with self.render_lock:
            self.scene_view.orbit(dx, dy)
        self.request_async_render()

    def zoom_camera(self, steps: float) -> None:
        # A wheel notch is discrete: redraw the main view right away for direct
        # feedback, leaving cameras and the network to the worker's batch.
        with self.render_lock:
            self.scene_view.zoom(steps)
        self.mujoco_view.set_image(self.render_main_view())
        self.request_async_render()

    def select_camera(self, index: int) -> None:
        with self.render_lock:
            self.scene_view.select(index)
        self.render_frame()

    @QtCore.Slot()
    def reset(self) -> None:
        self.recording.stop_recording()
        with self.scene_lock:
            self.sim.reset()
        with self.render_lock:
            self.scene_view.select(self.camera_selector.currentIndex())
            self.render_frame()
        self.statusBar().showMessage("机器人和策略历史已重置" + (" · " + self.sim.pulse.label() if self.sim.pulse else ""))

    @QtCore.Slot()
    def enter_adjustment(self) -> None:
        if self.current_frame is None:
            self.statusBar().showMessage("等待收到第一帧后再进入姿态调节")
            return
        self.recording.set_capture_enabled(False)
        self.adjusting = True
        if self.pause_event is not None:
            self.pause_event.set()
        self.info.hide()
        self.adjust_button.hide()
        self.adjustment_panel.show()
        self.joint_buttons[self.selected_joint].setChecked(True)
        self.show_selected_offset()
        self.source_view.set_adjustment(
            self.selected_joint, self.rotation_offsets
        )
        self.render_frame()
        self.statusBar().showMessage("已锁定当前输入帧，拖动彩色滑杆即可调节")

    def select_joint(self, name: str) -> None:
        self.selected_joint = name
        self.show_selected_offset()
        self.source_view.set_adjustment(name, self.rotation_offsets)
        self.render_frame()

    def show_selected_offset(self) -> None:
        for slider, label, value in zip(
            self.axis_sliders,
            self.axis_values,
            self.rotation_offsets[self.selected_joint],
        ):
            blocker = QtCore.QSignalBlocker(slider)
            slider.setValue(round(float(value) * 10.0))
            label.setText(f"{value:.1f}°")

    def change_axis(self, axis: int, value: float) -> None:
        self.rotation_offsets[self.selected_joint][axis] = value
        self.axis_values[axis].setText(f"{value:.1f}°")
        self.source_view.update()
        self.render_frame()

    def reset_axis(self, axis: int) -> None:
        self.axis_sliders[axis].setValue(0)

    def reset_all_offsets(self) -> None:
        for value in self.rotation_offsets.values():
            value.fill(0.0)
        self.show_selected_offset()
        self.source_view.update()
        self.render_frame()
        self.statusBar().showMessage("全部局部旋转修正已重置")

    def save_offsets(self) -> bool:
        try:
            ROTATION_OFFSETS_PATH.write_text(json.dumps(
                {name: value.tolist() for name, value in self.rotation_offsets.items()},
                ensure_ascii=False,
                indent=2,
            ) + "\n")
            return True
        except OSError as error:
            self.statusBar().showMessage(f"保存失败：{error}")
            return False

    @QtCore.Slot()
    def finish_adjustment(self) -> None:
        if not self.save_offsets():
            return
        self.adjusting = False
        self.recording.set_capture_enabled(True)
        self.source_view.set_adjustment(None, None)
        self.adjustment_panel.hide()
        self.info.show()
        self.adjust_button.show()
        self.render_frame()
        if self.pause_event is not None:
            self.pause_event.clear()
        self.statusBar().showMessage(
            f"修正已保存到 {ROTATION_OFFSETS_PATH.name}，继续接收动作"
        )

    @QtCore.Slot()
    def consume_frame(self) -> None:
        # The worker's newest frame is applied on every tick, even when no input
        # arrived: after a replay ends (or a stream fails) the pointer can still
        # orbit the camera, and nothing else would repaint the view.
        self.apply_async_result()
        for what, detail in self.take_thread_errors():
            # A render or publish thread has stopped. `failed` stays untouched on
            # purpose: recording stores qpos/action only and re-renders its three
            # videos in finalize(), so a dead preview must not abort a good take.
            print(f"{what}线程已停止：\n{detail}", file=sys.stderr)
            self.statusBar().showMessage(f"{what}线程已停止 · 详细信息已输出到终端")
        if self.adjusting:
            frame, error, finished = self.current_frame, None, False
        else:
            frame, error, finished = self.latest.take()
            if frame is not None:
                self.current_frame = frame
        if error and not self.failed:
            self.failed = True
            self.recording.stop_recording()
            print(error, file=sys.stderr)
            self.statusBar().showMessage("数据流错误 · 场景仍可查看，详细信息已输出到终端")
        if frame is not None and not self.failed:
            try:
                reference, pose, trackers, gloves, hands = frame
                if self.adjusting:
                    control_pose = apply_rotation_sources(
                        pose[None], trackers[None], self.rotation_offsets
                    )
                    converted = convert(control_pose, self.rest, self.parents)
                    reference = {
                        name: np.repeat(value, FUTURE_FRAMES, axis=0)
                        for name, value in converted.items()
                    }
                target, inference_ms = self.sim.plan(reference)
                with self.scene_lock:
                    self.sim.advance(target, hands)
                pelvis = self.sim.data.xpos[self.sim.pelvis].copy()
                self.source_view.set_frame(pose, trackers, gloves)
                if self.streaming.active:
                    # One capture per tick feeds every render request until the
                    # worker consumes it; the panels are painted here and never
                    # per drag event. Locked hand-off, unlocked capture: see
                    # request_async_render.
                    panels = self.source_view.stream_frames()
                    with self.publish_lock:
                        self.render_sources = panels
                self.request_async_render()
                if not self.adjusting:
                    self.recording.capture(self.sim.data, self.sim.commanded_qpos)
                render = f"渲染 {self.render_ms:.0f} ms · {self.display_fps:.0f} fps"
                if self.adjusting:
                    label = dict(ROTATION_OFFSET_JOINTS)[self.selected_joint]
                    status = f"姿态调节 · {label} · ONNX {inference_ms:.2f} ms · {render}"
                else:
                    status = f"50 Hz · ONNX {inference_ms:.2f} ms · pelvis z={pelvis[2]:.3f} m · {render}"
                self.statusBar().showMessage(status + (" · " + self.sim.pulse.label() if self.sim.pulse else ""))
            except Exception as exc:
                self.recording.stop_recording()
                self.failed = True
                print(traceback.format_exc(), file=sys.stderr)
                self.statusBar().showMessage(f"仿真停止：{type(exc).__name__}: {exc}")
        if finished and not self.finished_reported and not self.failed:
            self.finished_reported = True
            self.recording.stop_recording()
            self.statusBar().showMessage("回放结束" + (" · " + self.sim.pulse.label() if self.sim.pulse else ""))

    def finish_recording_close(self) -> None:
        if self._closing_after_export:
            self.close()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.resize_timer.stop()
        self.timer.stop()
        if self.recording.active or self.recording.busy:
            self._closing_after_export = True
            self.recording.stop_recording()
            if self.recording.busy:
                event.ignore()
                return
        if self.pause_event is not None:
            self.pause_event.clear()
        self.timer.stop()
        with self.render_request:
            self.render_stop = True
            self.render_request.notify()
        self.camera_stop.set()
        self.publish_stop.set()
        for thread in (self.render_thread, self.publish_thread, *self.camera_threads):
            thread.join(timeout=2)
        self.streaming.close_stream()
        self.robot_cameras.close_renderer()
        self.renderer.close()
        super().closeEvent(event)
