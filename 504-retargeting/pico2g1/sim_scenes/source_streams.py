"""Mirror the four source panels using their actual QWidget painting."""
import numpy as np
from PySide6 import QtCore, QtGui


class SourceViewStreams:
    """Shared geometry for on-screen painting and synchronized panel capture."""
    body_fraction = .61

    def panel_rects(self):
        margin = 8.0
        width = (self.width() - 3 * margin) / 2
        body_height = (self.height() - 3 * margin) * self.body_fraction
        hand_height = self.height() - body_height - 3 * margin
        return {
            'source_body_front': QtCore.QRectF(margin, margin, width, body_height),
            'source_body_side': QtCore.QRectF(2 * margin + width, margin, width, body_height),
            'source_left_hand': QtCore.QRectF(margin, 2 * margin + body_height, width, hand_height),
            'source_right_hand': QtCore.QRectF(2 * margin + width, 2 * margin + body_height, width, hand_height),
        }

    def stream_frames(self):
        # QWidget.render paints the current data now, including overlays, without
        # waiting for update() to reach the screen or reading the desktop pixels.
        # The caller runs on the GUI thread; JPEG encoding remains in the worker.
        image = QtGui.QImage(self.size(), QtGui.QImage.Format.Format_RGB32)
        image.fill(QtGui.QColor('#11161c'))
        self.render(image)
        images = {}
        for name, rect in self.panel_rects().items():
            crop = image.copy(rect.toAlignedRect().intersected(image.rect()))
            rgb = crop.convertToFormat(QtGui.QImage.Format.Format_RGB888)
            pixels = np.frombuffer(rgb.constBits(), dtype=np.uint8).reshape(rgb.height(), rgb.bytesPerLine())
            images[name] = pixels[:, :rgb.width() * 3].reshape(rgb.height(), rgb.width(), 3).copy()
        return images
