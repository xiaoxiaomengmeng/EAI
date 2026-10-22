"""Mouse event handling shared by both rendered Qt views."""
from PySide6 import QtCore, QtGui, QtWidgets

class MouseCameraView(QtWidgets.QLabel):
    orbit_requested = QtCore.Signal(float, float)
    zoom_requested = QtCore.Signal(float)
    viewport_resized = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._drag_position = None
        self.setToolTip('左键拖动：旋转视角；滚轮：缩放镜头')

    def _end_drag(self) -> None:
        self._drag_position = None
        self.unsetCursor()

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self.viewport_resized.emit()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_position = event.position()
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self._drag_position is not None and event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            delta = event.position() - self._drag_position
            self._drag_position = event.position()
            self.orbit_requested.emit(delta.x(), delta.y())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._end_drag()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        steps = event.pixelDelta().y() / 100 if not event.pixelDelta().isNull() else event.angleDelta().y() / 120
        if steps:
            self.zoom_requested.emit(steps)
            event.accept()
        else:
            event.ignore()

    def event(self, event):
        if event.type() in (QtCore.QEvent.Type.WindowDeactivate, QtCore.QEvent.Type.Hide):
            self._end_drag()
        return super().event(event)
