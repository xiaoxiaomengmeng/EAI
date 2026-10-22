"""Small desktop controls for the local HTTP preview server."""
from PySide6 import QtCore, QtGui, QtNetwork, QtWidgets
from .live_stream import LiveStreamServer, StreamConfig


class StreamingControls(QtWidgets.QWidget):
    started = QtCore.Signal()

    def __init__(self, config=None):
        super().__init__()
        self.config = config or StreamConfig()
        self.server = None
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.toggle = QtWidgets.QCheckBox('实时推流')
        self.toggle.setToolTip('共享主画面、三路相机、身体正侧视及左右手原始关键点，共八路画面')
        self.toggle.toggled.connect(self.set_enabled)
        layout.addWidget(self.toggle)
        self.address = QtWidgets.QLineEdit()
        self.address.setReadOnly(True)
        self.address.setFrame(False)
        self.address.setAccessibleName('推流访问地址或状态')
        self.address.setPlaceholderText('开启后显示局域网地址')
        self.address.setMinimumWidth(0)
        self.address.setStyleSheet('color:#94a3b8;font-size:11px;background:transparent;')
        layout.addWidget(self.address, 1)
        self.open_button = QtWidgets.QPushButton('打开预览')
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self.open_preview)
        self.open_button.setStyleSheet('padding:3px 7px;')
        layout.addWidget(self.open_button)
        self.health_timer = QtCore.QTimer(self)
        self.health_timer.setInterval(1000)
        self.health_timer.timeout.connect(self.check_health)

    @property
    def active(self):
        return self.server is not None and not self.server.stopped

    def set_enabled(self, enabled):
        if not enabled:
            self.close_stream()
            self.address.setText('推流已关闭')
            return
        try:
            self.server = LiveStreamServer(self.config).start()
        except (OSError, ValueError) as exc:
            self.server = None
            self.toggle.setChecked(False)
            self.address.setText(f'推流启动失败：{exc}')
            return
        host, port = self.server.address
        hosts = [host]
        if host == '0.0.0.0':
            hosts = sorted({entry.ip().toString() for interface in QtNetwork.QNetworkInterface.allInterfaces()
                            if interface.flags() & QtNetwork.QNetworkInterface.InterfaceFlag.IsUp
                            for entry in interface.addressEntries()
                            if entry.ip().protocol() == QtNetwork.QAbstractSocket.NetworkLayerProtocol.IPv4Protocol
                            and not entry.ip().isLoopback()}) or ['127.0.0.1']
        self.urls_text = '  '.join(f'http://{host}:{port}/' for host in hosts)
        self.address.setText(self.urls_text)
        self.address.setCursorPosition(0)
        self.address.setToolTip(self.urls_text)
        self.local_url = f'http://{"127.0.0.1" if host == "0.0.0.0" else host}:{port}/'
        self.open_button.setEnabled(True)
        self.health_timer.start()
        print(f'Eight-view live preview: {self.urls_text}', flush=True)
        self.started.emit()

    def check_health(self):
        if self.active:
            text = f'推流编码失败：{self.server.error}' if self.server.error else self.urls_text
            if self.address.text() != text:
                self.address.setText(text)
                self.address.setCursorPosition(0)

    def open_preview(self):
        if self.active:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(self.local_url))

    def publish(self, images, captured_at):
        # Local reference: this runs on the publisher thread while close_stream()
        # clears self.server on the GUI thread, and `active` is only a snapshot.
        # A server closed in between is harmless: its publish() checks stopped.
        server = self.server
        if server is not None and not server.stopped:
            server.publish(images, captured_at)

    def close_stream(self):
        self.health_timer.stop()
        if self.server is not None:
            self.server.close()
            self.server = None
        self.open_button.setEnabled(False)
