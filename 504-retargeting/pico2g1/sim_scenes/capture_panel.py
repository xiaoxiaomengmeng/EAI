"""Capture content with a small disclosure icon overlaid on the main viewport."""
from PySide6 import QtCore, QtWidgets


class CapturePanel(QtWidgets.QWidget):
    expanded_changed = QtCore.Signal(bool)

    def __init__(self, cameras, recording, viewport):
        super().__init__()
        self.cameras, self.recording, self.viewport = cameras, recording, viewport
        self._status_text = "三路相机 · LeRobot 录制"
        self._color = None
        # The icon is a viewport child, so it survives hiding this entire section
        # and never occupies any space in the camera/recording layouts.
        self.toggle = QtWidgets.QToolButton(viewport)
        self.toggle.setFixedSize(26, 26)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        self.toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.toggle.toggled.connect(self.set_expanded)
        viewport.installEventFilter(self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(cameras, 1)
        layout.addWidget(recording)
        recording.status_changed.connect(self.set_status)
        self._update_icon()
        self._place_icon()

    @property
    def expanded(self):
        return self.toggle.isChecked()

    def _place_icon(self):
        self.toggle.move(max(0, self.viewport.width() - 34), 8)
        self.toggle.raise_()

    def eventFilter(self, watched, event):
        if watched is self.viewport and event.type() == QtCore.QEvent.Type.Resize:
            self._place_icon()
        return super().eventFilter(watched, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Give the three 16:9 views their natural height at the available width;
        # the main viewport gets all remaining space.
        height = self.cameras.preferred_height(self.width()) + self.recording.sizeHint().height() + 8
        if self.minimumHeight() != height or self.maximumHeight() != height:
            self.setFixedHeight(height)

    @QtCore.Slot(bool)
    def set_expanded(self, expanded):
        if self.toggle.isChecked() != expanded:
            self.toggle.setChecked(expanded)
            return
        self.setVisible(expanded)
        self._update_icon()
        self.expanded_changed.emit(expanded)

    @QtCore.Slot(str)
    def set_status(self, text):
        self._status_text = text
        self._update_icon()

    def _update_icon(self):
        action = "收起数据采集" if self.expanded else "展开数据采集"
        self.toggle.setArrowType(QtCore.Qt.ArrowType.UpArrow if self.expanded else QtCore.Qt.ArrowType.DownArrow)
        self.toggle.setAccessibleName(action)
        self.toggle.setAccessibleDescription(self._status_text)
        self.toggle.setToolTip(f"{action}（不影响录制）\n{self._status_text}")
        color = "#f87171" if self._status_text.startswith("录制中") else (
            "#fbbf24" if self._status_text.startswith(("正在", "导出失败", "录制中断")) else "#cbd5e1"
        )
        if color != self._color:
            self._color = color
            self.toggle.setStyleSheet(
                "QToolButton { background:rgba(26,32,40,220); color:" + color + "; "
                "border:1px solid " + color + "; border-radius:5px; padding:3px; }"
                "QToolButton:hover { background:#303a47; }"
            )
