"""界面:源数据视图(SourceView)、MuJoCo 画布(MuJoCoView)与主窗口(TuningWindow)。"""

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

from g1_config import (
    AXIS_COLORS, BODY_EDGES, FPS, HAND_EDGES, HAND_TRACKER_NAMES,
    JOINT_NAMES, TUNING_DEFAULTS, TUNING_PATH,
)
from g1_ik import FullBodyIK, set_hands
from g1_model import object_id
from g1_source import LatestFrame, publish_frames
from sim_scenes import PulseSession, SceneView, scene_render_options
from sim_scenes.capture_panel import CapturePanel
from sim_scenes.mouse_view import MouseCameraView
from sim_scenes.recording_controls import RecordingControls
from sim_scenes.robot_cameras import RobotCameraPanel
from sim_scenes.source_streams import SourceViewStreams
from sim_scenes.streaming_controls import StreamingControls
from sim_scenes.viewport import fit_renderer


class SourceView(SourceViewStreams, QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.pose = None
        self.trackers = None
        self.gloves = None
        self.basis = None
        self.calibrating = False
        self.setMinimumHeight(390)

    def set_calibrating(self, calibrating: bool) -> None:
        self.calibrating = calibrating
        self.update()

    def set_frame(
        self,
        pose: np.ndarray,
        trackers: np.ndarray,
        gloves: dict[str, np.ndarray],
        basis: np.ndarray,
    ) -> None:
        self.pose = pose
        self.trackers = trackers
        self.gloves = gloves
        self.basis = basis
        self.update()

    @staticmethod
    def project(points: np.ndarray, rect: QtCore.QRectF) -> list[QtCore.QPointF]:
        area = rect.adjusted(12, 27, -12, -10)
        low, high = points.min(axis=0), points.max(axis=0)
        span = np.maximum(high - low, 1e-4)
        scale = 0.9 * min(area.width() / span[0], area.height() / span[1])
        center = 0.5 * (low + high)
        return [
            QtCore.QPointF(
                area.center().x() + (point[0] - center[0]) * scale,
                area.center().y() - (point[1] - center[1]) * scale,
            )
            for point in points
        ]

    @staticmethod
    def panel(painter: QtGui.QPainter, rect: QtCore.QRectF, title: str) -> None:
        painter.setPen(QtGui.QPen(QtGui.QColor("#343b45"), 1))
        painter.setBrush(QtGui.QColor("#1a2028"))
        painter.drawRoundedRect(rect, 7, 7)
        painter.setPen(QtGui.QColor("#cbd5e1"))
        painter.drawText(
            rect.adjusted(9, 6, -9, -6),
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop,
            title,
        )

    @staticmethod
    def draw_edges(
        painter: QtGui.QPainter,
        points: list[QtCore.QPointF],
        edges: tuple[tuple[int, int], ...],
        color: QtGui.QColor,
    ) -> None:
        painter.setPen(QtGui.QPen(color, 2.2, QtCore.Qt.PenStyle.SolidLine))
        for start, end in edges:
            painter.drawLine(points[start], points[end])
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(color.lighter(125))
        for point in points[:len(JOINT_NAMES)]:
            painter.drawEllipse(point, 2.6, 2.6)

    def draw_body(
        self, painter: QtGui.QPainter, rect: QtCore.QRectF, axes: tuple[int, int], title: str
    ) -> None:
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
        rotation_guide_names = ("spine3", "left_foot", "right_foot")
        rotation_guide_start = len(raw_points)
        if self.calibrating:
            guide_ids = [JOINT_NAMES.index(name) for name in rotation_guide_names]
            guide_rot = Rotation.from_quat(self.pose[guide_ids, 3:]).as_matrix()
            guide_axes = self.pose[guide_ids, None, :3] + 0.1 * np.transpose(
                guide_rot, (0, 2, 1)
            )
            raw_points = np.concatenate((raw_points, guide_axes.reshape(-1, 3)))
        aligned = (raw_points - self.pose[waist, :3]) @ self.basis.T
        points = self.project(aligned[:, axes], rect)
        for start, end in BODY_EDGES:
            names = (JOINT_NAMES[start], JOINT_NAMES[end])
            color = (
                QtGui.QColor("#48c6b8") if any("left_" in name for name in names)
                else QtGui.QColor("#f0a75e") if any("right_" in name for name in names)
                else QtGui.QColor("#d4d8df")
            )
            painter.setPen(QtGui.QPen(color, 2.2))
            painter.drawLine(points[start], points[end])
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor("#eef2f7"))
        for point in points[:len(JOINT_NAMES)]:
            painter.drawEllipse(point, 2.6, 2.6)
        if self.calibrating:
            painter.setPen(QtGui.QPen(QtGui.QColor("#f4d35e"), 2.2))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            for side in ("left", "right"):
                for joint in ("knee", "ankle", "elbow", "wrist"):
                    painter.drawEllipse(
                        points[JOINT_NAMES.index(f"{side}_{joint}")], 6.0, 6.0
                    )
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
            for axis, axis_color in enumerate(AXIS_COLORS):
                painter.setPen(QtGui.QPen(QtGui.QColor(axis_color), 1.7))
                painter.drawLine(
                    tracker,
                    points[len(JOINT_NAMES) + len(HAND_TRACKER_NAMES) + 3 * side + axis],
                )
        if self.calibrating:
            for guide, name in enumerate(rotation_guide_names):
                origin = points[JOINT_NAMES.index(name)]
                for axis, color in enumerate(AXIS_COLORS):
                    painter.setPen(QtGui.QPen(QtGui.QColor(color), 1.7))
                    painter.drawLine(
                        origin, points[rotation_guide_start + 3 * guide + axis]
                    )

    def draw_hand(
        self, painter: QtGui.QPainter, rect: QtCore.QRectF, side: str
    ) -> None:
        label = "左手" if side == "left" else "右手"
        self.panel(painter, rect, f"{label}原始关键点  21/21")
        if self.gloves is None or side not in self.gloves:
            return
        points = self.gloves[side][..., :3]
        points = points - points[0]
        screen = self.project(points[:, (0, 2)], rect)
        color = QtGui.QColor("#48c6b8" if side == "left" else "#f0a75e")
        self.draw_edges(painter, screen, HAND_EDGES, color)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor("#11161c"))
        panels = self.panel_rects()
        suffix = "  · 黄圈位置 / RGB旋转" if self.calibrating else ""
        self.draw_body(painter, panels['source_body_front'], (1, 2), f"身体正视{suffix}")
        self.draw_body(painter, panels['source_body_side'], (0, 2), f"身体侧视{suffix}")
        self.draw_hand(painter, panels['source_left_hand'], "left")
        self.draw_hand(painter, panels['source_right_hand'], "right")


class MuJoCoView(MouseCameraView):
    def __init__(self):
        super().__init__("正在加载场景…")
        self.image = None
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(620, 620)
        self.setStyleSheet("background: #07090c; color: #8792a2; font-size: 16px;")

    def set_image(self, image: np.ndarray) -> None:
        height, width = image.shape[:2]
        self.image = QtGui.QImage(
            image.data,
            width,
            height,
            image.strides[0],
            QtGui.QImage.Format.Format_RGB888,
        ).copy()
        self.update_pixmap()

    def update_pixmap(self) -> None:
        if self.image is not None:
            self.setPixmap(
                QtGui.QPixmap.fromImage(self.image).scaled(
                    self.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
            )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.update_pixmap()


class TuningWindow(QtWidgets.QMainWindow):
    GROUPS = (
        ("身体比例", (
            ("臂长", "arm_scale", 0.7, 1.3, 0.01),
            ("腿长", "leg_scale", 0.7, 1.3, 0.01),
            ("肩宽", "shoulder_width", 0.7, 1.3, 0.01),
            ("髋宽", "hip_width", 0.7, 1.3, 0.01),
        )),
        ("平滑与求解", (
            ("手部响应", "hand_response", 0.05, 1.0, 0.01),
            ("阻尼", "damping", 0.005, 0.2, 0.005),
        )),
        ("IK 任务权重", (
            ("手腕位置", "wrist_pos", 0.0, 2.0, 0.05),
            ("脚踝位置", "ankle_pos", 0.0, 2.0, 0.05),
            ("膝部位置", "knee_pos", 0.0, 2.0, 0.05),
            ("肘部位置", "elbow_pos", 0.0, 2.0, 0.05),
            ("躯干旋转", "torso_rot", 0.0, 1.0, 0.05),
            ("手腕旋转", "wrist_rot", 0.0, 1.0, 0.05),
            ("脚踝旋转", "ankle_rot", 0.0, 1.0, 0.05),
        )),
        ("手腕目标局部旋转（度）", (
            ("左腕 X（红）", "left_wrist_x", -180.0, 180.0, 1.0),
            ("左腕 Y（绿）", "left_wrist_y", -180.0, 180.0, 1.0),
            ("左腕 Z（蓝）", "left_wrist_z", -180.0, 180.0, 1.0),
            ("右腕 X（红）", "right_wrist_x", -180.0, 180.0, 1.0),
            ("右腕 Y（绿）", "right_wrist_y", -180.0, 180.0, 1.0),
            ("右腕 Z（蓝）", "right_wrist_z", -180.0, 180.0, 1.0),
        )),
    )

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        frames,
        tuning: dict[str, float],
        pause_event: threading.Event | None = None,
        *,
        render_shadows: bool = False,
        stream_config=None, stream_enabled: bool = False,
    ):
        super().__init__()
        self.model = model
        self.pulse = PulseSession.for_model(model)
        self.scene_option = scene_render_options(self.pulse, model)
        self.data = data
        self.tuning = tuning
        self.pause_event = pause_event
        self.calibrating = False
        self.ik = FullBodyIK(model, data, tuning)
        self.latest = LatestFrame()
        self.last_error = float("nan")
        self.failed = False
        self.finished_reported = False
        self.current_frame = None
        self.controls = {}
        model.vis.global_.offwidth = 800
        model.vis.global_.offheight = 800
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.lookat[:] = [0.15, 0.0, 0.8]
        self.camera.distance = 2.5
        self.camera.azimuth = 135
        self.camera.elevation = -15
        self.camera_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.scene_view = SceneView(model, data, self.camera, render_shadows=render_shadows)
        # Finalize scene bounds and shadow resolution before allocating Renderer.
        self.renderer = mujoco.Renderer(model, height=800, width=800)
        self.streaming = StreamingControls(stream_config)
        self.build_ui()
        self.streaming.started.connect(self.render_frame)
        self.streaming.toggle.setChecked(stream_enabled)
        self.render_frame()
        self.statusBar().showMessage("场景已就绪 · 等待数据连接或第一帧…")
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.consume_frame)
        self.timer.start(round(1000 / FPS))
        threading.Thread(
            target=publish_frames, args=(frames, self.latest), daemon=True
        ).start()

    def build_ui(self) -> None:
        self.setWindowTitle("G1 Ego Retargeting")
        self.resize(1500, 900)
        self.setMinimumSize(1120, 720)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
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
        self.robot_cameras = RobotCameraPanel(self.model)
        self.recording = RecordingControls(
            self.model, source_fps=FPS, render_shadows=self.scene_view.shadows.enabled,
            action_semantics="Kinematic IK and hand retargeting targets; state equals target at source samples; exported state is interpolated and action is held.",
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
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        self.source_view = SourceView()
        right_layout.addWidget(self.source_view, 1)

        self.info = QtWidgets.QLabel(
            "轻量全身 IK · 8 个位置任务 + 5 个旋转任务\n"
            "进入参数标定后锁定当前动作，再调整比例、阻尼和任务权重"
        )
        self.info.setWordWrap(True)
        self.info.setStyleSheet(
            "padding:10px;background:#1a2028;border:1px solid #343b45;border-radius:6px;"
        )
        right_layout.addWidget(self.info)
        self.calibration_button = QtWidgets.QPushButton("进入参数标定")
        self.calibration_button.clicked.connect(self.enter_calibration)
        right_layout.addWidget(self.calibration_button)
        self.camera_selector = QtWidgets.QComboBox()
        self.camera_selector.addItems(["场景总览", "跟随机器人"])
        self.camera_selector.setCurrentIndex(1)
        self.camera_selector.setAccessibleName("相机视角")
        self.camera_selector.currentIndexChanged.connect(self.select_camera)
        right_layout.addWidget(self.camera_selector)
        right_layout.addWidget(self.streaming)

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 4, 0)
        help_text = QtWidgets.QLabel(
            "当前输入帧已锁定\n黄圈：8 个位置目标 · 红绿蓝轴：5 个旋转目标"
        )
        help_text.setWordWrap(True)
        controls_layout.addWidget(help_text)
        for title, specs in self.GROUPS:
            controls_layout.addWidget(self.make_group(title, specs))
        buttons = QtWidgets.QHBoxLayout()
        for text, callback in (
            ("恢复默认", self.reset_defaults),
            ("重新标定身体", self.recalibrate),
            ("保存并继续", self.finish_calibration),
        ):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        controls_layout.addLayout(buttons)
        controls_layout.addStretch()
        self.calibration_panel = QtWidgets.QScrollArea()
        self.calibration_panel.setWidgetResizable(True)
        self.calibration_panel.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.calibration_panel.setWidget(controls)
        self.calibration_panel.hide()
        right_layout.addWidget(self.calibration_panel, 1)
        layout.addWidget(right)
        self.setCentralWidget(central)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_InputMethodEnabled, False)
        for widget in self.findChildren(QtWidgets.QWidget):
            widget.setAttribute(QtCore.Qt.WidgetAttribute.WA_InputMethodEnabled, False)
            if isinstance(
                widget,
                (QtWidgets.QAbstractButton, QtWidgets.QAbstractSlider),
            ):
                widget.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.statusBar().showMessage("等待第一帧…")
        self.setStyleSheet(
            "QMainWindow, QWidget { background: #11161c; color: #e4e9ef; }"
            "QGroupBox { border: 1px solid #343b45; border-radius: 6px; "
            "margin-top: 9px; padding-top: 8px; font-weight: 600; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 9px; padding: 0 4px; }"
            "QPushButton { background: #252d38; border: 1px solid #3b4654; "
            "border-radius: 5px; padding: 7px; }"
            "QPushButton:hover { background: #303a47; }"
        )

    def make_group(self, title: str, specs: tuple) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(title)
        form = QtWidgets.QFormLayout(group)
        for label, key, minimum, maximum, step in specs:
            value = float(np.clip(self.tuning[key], minimum, maximum))
            self.tuning[key] = value
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setRange(0, round((maximum - minimum) / step))
            slider.setValue(round((value - minimum) / step))
            value_label = QtWidgets.QLabel(f"{value:.3f}" if step < 0.01 else f"{value:.2f}")
            value_label.setFixedWidth(48)
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(slider, 1)
            row_layout.addWidget(value_label)
            slider.valueChanged.connect(
                lambda index, k=key, lo=minimum, delta=step, output=value_label: self.change_value(
                    k, lo + index * delta, delta, output
                )
            )
            self.controls[key] = (slider, value_label, minimum, maximum, step)
            form.addRow(label, row)
        return group

    def render_frame(self) -> None:
        self.renderer = fit_renderer(self.renderer, self.model, self.mujoco_view.contentsRect())
        captured_at = time.time()
        main_image = self.scene_view.render(self.renderer, self.scene_option)
        self.mujoco_view.set_image(main_image)
        if self.capture_panel.expanded or self.streaming.active:
            images = self.robot_cameras.render_frame(self.data, self.scene_option, self.scene_view.shadows)
            if self.streaming.active:
                self.streaming.publish({"main": main_image, **images,
                                        **self.source_view.stream_frames()}, captured_at)

    def orbit_camera(self, dx: float, dy: float) -> None:
        self.scene_view.orbit(dx, dy)
        self.render_frame()

    def zoom_camera(self, steps: float) -> None:
        self.scene_view.zoom(steps)
        self.render_frame()

    def select_camera(self, index: int) -> None:
        self.scene_view.select(index)
        self.render_frame()

    def change_value(
        self, key: str, value: float, step: float, label: QtWidgets.QLabel
    ) -> None:
        self.tuning[key] = value
        label.setText(f"{value:.3f}" if step < 0.01 else f"{value:.2f}")

    @QtCore.Slot()
    def enter_calibration(self) -> None:
        if self.current_frame is None:
            self.statusBar().showMessage("等待收到第一帧后再进入参数标定")
            return
        self.recording.set_capture_enabled(False)
        self.calibrating = True
        if self.pause_event is not None:
            self.pause_event.set()
        self.source_view.set_calibrating(True)
        self.info.hide()
        self.calibration_button.hide()
        self.calibration_panel.show()
        self.render_frame()
        self.statusBar().showMessage("已锁定当前输入帧，拖动滑杆即可观察 G1 变化")

    @QtCore.Slot()
    def consume_frame(self) -> None:
        if self.calibrating:
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
            previous_qpos = self.data.qpos.copy()
            try:
                pose, raw_pose, trackers, raw_gloves, hands = frame
                self.last_error = self.ik.solve(pose)
                set_hands(self.model, self.data, hands)
                if self.pulse:
                    self.pulse.step_kinematic(self.data, 1.0 / FPS)
                mujoco.mj_forward(self.model, self.data)
                self.source_view.set_frame(raw_pose, trackers, raw_gloves, self.ik.source_to_g1)
                self.render_frame()
                if not self.calibrating:
                    self.recording.capture(self.data)
                if self.calibrating:
                    status = f"参数标定 · 当前帧已锁定 · IK 残差 {self.last_error:.4f}"
                else:
                    status = f"50 Hz · IK 残差 {self.last_error:.4f}"
                self.statusBar().showMessage(status + (" · " + self.pulse.label() if self.pulse else ""))
            except Exception as error:
                self.recording.stop_recording()
                self.data.qpos[:] = previous_qpos
                mujoco.mj_forward(self.model, self.data)
                print(traceback.format_exc(), file=sys.stderr)
                self.statusBar().showMessage(
                    f"本帧已回滚：{type(error).__name__}: {error}"
                )
        if finished and not self.finished_reported and not self.failed:
            self.finished_reported = True
            self.recording.stop_recording()
            self.statusBar().showMessage(f"回放结束 · 最后 IK 残差 {self.last_error:.4f}" + (" · " + self.pulse.label() if self.pulse else ""))

    def reset_defaults(self) -> None:
        for key, value in TUNING_DEFAULTS.items():
            slider, _, minimum, maximum, step = self.controls[key]
            slider.setValue(round((np.clip(value, minimum, maximum) - minimum) / step))
        self.statusBar().showMessage("已恢复默认参数")

    def recalibrate(self) -> None:
        self.ik.reset_calibration()
        self.statusBar().showMessage("将以当前锁定帧重新标定身体方向和比例")

    def save_tuning(self) -> bool:
        try:
            TUNING_PATH.write_text(
                json.dumps(self.tuning, ensure_ascii=False, indent=2) + "\n"
            )
            return True
        except OSError as error:
            self.statusBar().showMessage(f"保存失败：{error}")
            return False

    @QtCore.Slot()
    def finish_calibration(self) -> None:
        if not self.save_tuning():
            return
        self.calibrating = False
        self.recording.set_capture_enabled(True)
        self.source_view.set_calibrating(False)
        self.calibration_panel.hide()
        self.info.show()
        self.calibration_button.show()
        self.render_frame()
        if self.pause_event is not None:
            self.pause_event.clear()
        self.statusBar().showMessage(
            f"参数已保存到 {TUNING_PATH.name}，继续接收动作"
        )

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
        self.streaming.close_stream()
        self.robot_cameras.close_renderer()
        self.renderer.close()
        super().closeEvent(event)
