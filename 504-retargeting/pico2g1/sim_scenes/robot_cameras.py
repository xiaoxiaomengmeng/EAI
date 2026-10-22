"""Body-mounted robot cameras and their synchronized Qt preview strip."""
import contextlib
import xml.etree.ElementTree as ET

import mujoco
from PySide6 import QtCore, QtGui, QtWidgets

# G1's head is a mesh on torso_link, with no independently actuated neck.
# Head looks along body +X. Wrist cameras look forward/up from below the palm;
# all orientations remain relative to their mounting body, including wrist roll.
CAMERA_MOUNTS = (
    ("robot_head", "头部", "torso_link", "0.10 0 0.40"),
    ("robot_left_wrist", "左手腕", "left_wrist_yaw_link", "0.015 0 -0.12"),
    ("robot_right_wrist", "右手腕", "right_wrist_yaw_link", "0.015 0 -0.12"),
)


def add_robot_cameras(root: ET.Element) -> None:
    bodies = {body.get("name"): body for body in root.findall(".//body")}
    for name, _, body, position in CAMERA_MOUNTS:
        wrist = name != "robot_head"
        # 40 degrees upward, with a tighter field of view for finger contact detail.
        axes = "0 -1 0 -0.642787610 0 0.766044443" if wrist else "0 -1 0 0 0 1"
        ET.SubElement(bodies[body], "camera", name=name, mode="fixed", pos=position,
                      xyaxes=axes, fovy="55" if wrist else "75", ipd="0")


def render_camera_images(renderer, model, data, option, shadows, snapshot=None, names=None) -> dict:
    """Render the mounted cameras with one renderer, without touching Qt.

    `names` limits the batch to a subset (a drag refreshes one camera per
    batch). `snapshot` (a context manager) holds the model state each camera
    needs: the scene copy and the clipping override. The GL render runs after
    it is released, so another thread only waits for the copy, never for the
    draw.
    """
    images = {}
    for name, _, _, _ in CAMERA_MOUNTS:
        if names is not None and name not in names:
            continue
        with snapshot or contextlib.nullcontext():
            # Overview clipping can be metres from the eye. Wrists need
            # centimetres. Restore the main view's settings even if it fails.
            near = model.vis.map.znear
            model.vis.map.znear = .01 / model.stat.extent
            try:
                renderer.update_scene(data, camera=name, scene_option=option)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadows.enabled
                shadows.prepare(renderer, data)
            finally:
                model.vis.map.znear = near
        # No copy: render() allocates per call, so this image stays correct even
        # while a preview or the stream consumes it later (SceneView.render).
        images[name] = renderer.render()
    return images


class CameraImage(QtWidgets.QWidget):
    """Paint to the available space without a pixmap inflating layout minimums."""
    def __init__(self, title: str):
        super().__init__()
        self.image = None
        self.setAccessibleName(title + "相机画面")
        self.setMinimumSize(160, 90)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Expanding)

    def set_image(self, pixels) -> None:
        height, width = pixels.shape[:2]
        self.image = QtGui.QImage(pixels.data, width, height, pixels.strides[0],
                                 QtGui.QImage.Format.Format_RGB888).copy()
        self.update()

    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#07090c"))
        if self.image is not None:
            size = self.image.size().scaled(self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            target = QtCore.QRect(QtCore.QPoint(), size)
            target.moveCenter(self.rect().center())
            painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(target, self.image)


class RobotCameraPanel(QtWidgets.QWidget):
    """One small renderer reused for three views of the current simulation state."""
    def __init__(self, model: mujoco.MjModel):
        super().__init__()
        self.model = model
        self.renderer = mujoco.Renderer(model, height=180, width=320)
        self.views = {}
        self.titles = []
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        for name, title, _, _ in CAMERA_MOUNTS:
            card = QtWidgets.QWidget()
            column = QtWidgets.QVBoxLayout(card)
            column.setContentsMargins(1, 1, 1, 1)
            column.setSpacing(0)
            label = QtWidgets.QLabel(title)
            label.setStyleSheet("background:#1a2028;color:#cbd5e1;padding:6px 8px;")
            column.addWidget(label)
            self.titles.append(label)
            view = CameraImage(title)
            column.addWidget(view, 1)
            self.views[name] = view
            layout.addWidget(card, 1)

    def preferred_height(self, width):
        card_width = max(162, (width - 2 * self.layout().spacing()) / 3)
        image_height = round((card_width - 2) * 9 / 16)
        return image_height + max(title.sizeHint().height() for title in self.titles) + 2

    def render_frame(self, data, option, shadows, snapshot=None) -> dict:
        images = render_camera_images(self.renderer, self.model, data, option, shadows,
                                      snapshot=snapshot)
        # Publish only after the entire set is rendered; Qt paints after this call.
        for name, image in images.items():
            self.views[name].set_image(image)
        return images

    def close_renderer(self) -> None:
        self.renderer.close()
