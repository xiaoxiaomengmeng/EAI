"""Compact Qt controls for capture and background LeRobot video export."""
from datetime import datetime
from pathlib import Path
import threading
import uuid

from PySide6 import QtCore, QtGui, QtWidgets


class RecordingControls(QtWidgets.QWidget):
    export_progress = QtCore.Signal(int, int)
    export_done = QtCore.Signal(str)
    export_error = QtCore.Signal(str)
    settled = QtCore.Signal()
    status_changed = QtCore.Signal(str)

    def __init__(self, model, *, source_fps=50, render_shadows=False, action_semantics):
        super().__init__()
        self.model = model
        self.source_fps = source_fps
        self.render_shadows = render_shadows
        self.action_semantics = action_semantics
        self.output_parent = Path(__file__).resolve().parents[1] / "recordings"
        self.session = None
        self.joint_addresses = None
        self.last_output = None
        self.last_error = ""
        self._capture_error = ""
        self._exporting = False
        self._capture_enabled = True
        self._worker = None
        self.task_input = QtWidgets.QLineEdit("G1 机器人动作采集")
        self.task_input.setAccessibleName("录制任务描述")
        self.task_input.setPlaceholderText("任务描述，例如：拿起桌上的物块")
        self.directory_button = QtWidgets.QPushButton("保存位置")
        self.directory_button.setToolTip(str(self.output_parent))
        self.directory_button.clicked.connect(self.choose_directory)
        self.start_button = QtWidgets.QPushButton("开始录制")
        self.start_button.clicked.connect(self.toggle_recording)
        self.open_button = QtWidgets.QPushButton("打开目录")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self.open_output)
        self.status = QtWidgets.QLabel("三路视频 + 关节轨迹 · 720p / 30 fps")
        self.status.setMinimumWidth(0)
        self.status.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                  QtWidgets.QSizePolicy.Policy.Preferred)
        self.status.setAccessibleName("录制状态")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.task_input, 1)
        row.addWidget(self.directory_button)
        row.addWidget(self.start_button)
        layout.addLayout(row)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.open_button)
        layout.addLayout(bottom)
        self.export_progress.connect(self._progress)
        self.export_done.connect(self._finished)
        self.export_error.connect(self._failed)

    @property
    def busy(self):
        return self._exporting

    @property
    def active(self):
        return self.session is not None and not self.busy

    def _status(self, text):
        self.status.setText(text)
        self.status.setToolTip(text)
        self.status_changed.emit(text)

    def _buttons(self):
        self.task_input.setEnabled(not self.active and not self.busy)
        self.directory_button.setEnabled(not self.active and not self.busy)
        self.start_button.setEnabled(not self.busy and self._capture_enabled)
        self.start_button.setText("正在导出…" if self.busy else "停止并保存" if self.active else "开始录制")
        self.open_button.setEnabled(self.last_output is not None)

    def choose_directory(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择录制保存位置", str(self.output_parent.parent if not self.output_parent.exists() else self.output_parent)
        )
        if directory:
            self.output_parent = Path(directory)
            self.directory_button.setToolTip(directory)

    def open_output(self):
        if self.last_output is not None:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.last_output)))

    def toggle_recording(self):
        if self.active:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self, output_dir=None):
        if self.active or self.busy or not self._capture_enabled:
            return False
        self.last_error = ""
        self._capture_error = ""
        task = self.task_input.text().strip()
        if not task:
            self._status("请填写任务描述后开始录制")
            return False
        try:
            from .lerobot_recording import RecordingSession, robot_joint_layout
            _, addresses = robot_joint_layout(self.model)
            if output_dir is None:
                name = f"g1_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
                output_dir = self.output_parent / name
            session = RecordingSession(
                self.model, Path(output_dir), task, source_fps=self.source_fps,
                render_shadows=self.render_shadows, action_semantics=self.action_semantics,
            )
            self.session = session
            self.joint_addresses = addresses
            self.last_output = Path(output_dir)
        except Exception as error:
            self.last_error = str(error)
            self._status(f"无法开始录制：{error}")
            return False
        self._buttons()
        self._status("录制中 · 等待下一帧动作")
        return True

    def capture(self, data, commanded_qpos=None):
        if not self.active or not self._capture_enabled:
            return
        try:
            target = data.qpos if commanded_qpos is None else commanded_qpos
            self.session.append(data, target[self.joint_addresses].copy())
            count = self.session.frame_count
            self._status(f"录制中 · {count} 帧 · {count / self.source_fps:.2f} 秒仿真时间")
        except Exception as error:
            self._capture_error = str(error)
            self.stop_recording()

    def set_capture_enabled(self, enabled):
        if not enabled:
            self.stop_recording()
        self._capture_enabled = enabled
        self._buttons()

    def stop_recording(self):
        if not self.active:
            return
        session = self.session
        try:
            session.seal()
        except Exception as error:
            self._failed(str(error))
            return
        self._exporting = True
        self._buttons()
        self._status("正在生成三路高清视频…")

        def export():
            try:
                output = session.finalize(progress=self.export_progress.emit)
            except Exception as error:
                self.export_error.emit(str(error))
            else:
                self.export_done.emit(str(output) if output is not None else "")

        self._worker = threading.Thread(target=export, name="lerobot-export", daemon=False)
        self._worker.start()

    @QtCore.Slot(int, int)
    def _progress(self, done, total):
        self._status(f"正在导出 · {done}/{total} 帧 · {done / max(1, total):.0%}")

    @QtCore.Slot(str)
    def _finished(self, output):
        self._exporting = False
        self.session = None
        self.last_output = Path(output) if output else None
        self.last_error = self._capture_error
        self._buttons()
        if output:
            prefix = f"录制中断（{self._capture_error}），已有帧已保存" if self._capture_error else "已保存"
            self._status(f"{prefix} · {output}")
        else:
            self._status("没有录到动作帧，未生成数据集")
        self.settled.emit()

    @QtCore.Slot(str)
    def _failed(self, error):
        self._exporting = False
        self.session = None
        self.last_error = error
        self._buttons()
        self._status(f"导出失败：{error} · 原始轨迹保留在 {self.last_output}")
        self.settled.emit()
