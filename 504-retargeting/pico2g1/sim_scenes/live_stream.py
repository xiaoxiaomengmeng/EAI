"""Bounded, latest-frame LAN streaming; no rendering or networking on Qt's thread."""
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import struct
import threading
import time
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PySide6 import QtCore, QtGui

STREAM_IDS = ('main', 'robot_head', 'robot_left_wrist', 'robot_right_wrist',
              'source_body_front', 'source_body_side', 'source_left_hand', 'source_right_hand')
TITLES = ('主画面', '头部视角', '左腕部相机', '右腕部相机',
          '身体正视', '身体侧视', '左手原始关键点', '右手原始关键点')


@dataclass(frozen=True)
class StreamConfig:
    host: str = '0.0.0.0'
    port: int = 8876
    fps: int = 30
    quality: int = 80

    def __post_init__(self):
        if not (0 <= self.port <= 65535 and 1 <= self.fps <= 60 and 1 <= self.quality <= 95):
            raise ValueError('port must be 0–65535, fps 1–60, quality 1–95')


@dataclass(frozen=True)
class EncodedFrame:
    sequence: int
    captured_at: float
    captured_monotonic: float
    images: dict
    encode_ms: float


class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, owner):
        self.owner = owner
        self.slots = threading.BoundedSemaphore(32)
        super().__init__(address, _Handler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            request.settimeout(2)
            request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            request.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class LiveStreamServer:
    def __init__(self, config=None):
        self.config = config or StreamConfig()
        self.condition = threading.Condition()
        self.pending = None
        self.latest = None
        self.stopped = True
        self.error = None
        self.http = self.worker = self.thread = None
        self.sequence = 0

    def start(self):
        self.http = _HTTPServer((self.config.host, self.config.port), self)
        self.address = self.http.server_address
        self.stopped = False
        self.worker = threading.Thread(target=self._encode_loop, name='stream-jpeg', daemon=True)
        self.thread = threading.Thread(target=self.http.serve_forever, kwargs={'poll_interval': .1},
                                       name='stream-http', daemon=True)
        self.worker.start()
        self.thread.start()
        return self

    def publish(self, images, captured_at=None):
        """Own the RGB memory at this boundary: the encoder thread reads these
        arrays later, and any producer -- a Qt crop or a renderer that reuses its
        buffer -- may otherwise hand us a view it is still writing to.
        """
        if self.stopped:
            return
        captured_at = time.time() if captured_at is None else captured_at
        captured_monotonic = time.monotonic() - max(0, time.time() - captured_at)
        pixels = {key: np.array(images[key], dtype=np.uint8, order='C', copy=True) for key in STREAM_IDS}
        with self.condition:
            self.sequence = max(self.sequence + 1, time.time_ns() // 1000)
            self.pending = (self.sequence, captured_at, captured_monotonic, pixels)
            self.condition.notify_all()

    def _encode_loop(self):
        next_encode = 0
        while True:
            with self.condition:
                while not self.stopped:
                    delay = next_encode - time.monotonic()
                    if self.pending is not None and delay <= 0:
                        break
                    self.condition.wait(max(.001, delay) if self.pending is not None else None)
                if self.stopped:
                    return
                sequence, captured_at, captured_monotonic, pixels = self.pending
                self.pending = None
            start = time.monotonic()
            next_encode = start + 1 / self.config.fps
            try:
                images = {}
                for key, rgb in pixels.items():
                    height, width = rgb.shape[:2]
                    image = QtGui.QImage(rgb.data, width, height, rgb.strides[0], QtGui.QImage.Format.Format_RGB888)
                    buffer = QtCore.QBuffer()
                    buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
                    if not image.save(buffer, 'JPEG', self.config.quality):
                        raise RuntimeError('JPEG encoder failed')
                    images[key] = bytes(buffer.data())
                frame = EncodedFrame(sequence, captured_at, captured_monotonic, images,
                                     (time.monotonic() - start) * 1000)
                with self.condition:
                    self.latest = frame
                    self.error = None
                    self.condition.notify_all()
            except Exception as exc:
                with self.condition:
                    self.error = str(exc)
                    self.condition.notify_all()

    def wait_frame(self, after=0, timeout=1):
        with self.condition:
            self.condition.wait_for(lambda: self.stopped or (self.latest is not None and
                                    self.latest.sequence > after), timeout)
            return self.latest if not self.stopped and self.latest and self.latest.sequence > after else None

    def close(self):
        with self.condition:
            self.stopped = True
            self.pending = None
            self.condition.notify_all()
        if self.http is not None:
            self.http.shutdown()
            self.http.server_close()
            self.http = None
        for thread in (self.worker, self.thread):
            if thread is not None:
                thread.join(timeout=1)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def _headers(self, status, content_type, length=None, extra=None):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        self.send_header('X-Accel-Buffering', 'no')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if self.server.owner.stopped:
            # Idle HTTP/1.1 handlers can outlive server_close(). Never let an
            # old long-poll connection keep serving 204s after a server restart.
            self.send_header('Connection', 'close')
            self.close_connection = True
        if length is not None:
            self.send_header('Content-Length', str(length))
        for name, value in (extra or {}).items():
            self.send_header(name, str(value))
        self.end_headers()

    def _send(self, status, content_type, body=b'', extra=None):
        self._headers(status, content_type, len(body), extra)
        self.wfile.write(body)

    def do_GET(self):
        try:
            self._get()
        except (OSError, ConnectionError):
            self.close_connection = True

    def _get(self):
        owner = self.server.owner
        if owner.stopped:
            self._send(503, 'text/plain', b'Stream stopped')
            return
        url = urlsplit(self.path)
        if url.path == '/':
            self._send(200, 'text/html; charset=utf-8', Path(__file__).with_name('web').joinpath('live.html').read_bytes())
            return
        if url.path == '/api/streams':
            with owner.condition:
                frame = owner.latest
                body = {'streams': [dict(id=key, title=title, mjpeg=f'/stream/{key}.mjpg',
                                        snapshot=f'/snapshot/{key}.jpg') for key, title in zip(STREAM_IDS, TITLES)],
                        'fps_limit': owner.config.fps, 'quality': owner.config.quality,
                        'sequence': frame.sequence if frame else None,
                        'encode_ms': frame.encode_ms if frame else None, 'error': owner.error}
            self._send(200, 'application/json', json.dumps(body).encode())
            return
        if url.path == '/api/frames':
            try:
                after = int(parse_qs(url.query).get('after', ['0'])[0])
                if not 0 <= after <= 2**63 - 1:
                    raise ValueError()
            except ValueError:
                self._send(400, 'text/plain', b'Invalid after sequence')
                return
            frame = owner.wait_frame(after)
            if frame is None:
                self._send(503 if owner.stopped else 204, 'application/octet-stream')
                return
            header = json.dumps({'sequence': frame.sequence, 'captured_at': frame.captured_at,
                                 'age_ms': (time.monotonic() - frame.captured_monotonic) * 1000,
                                 'encode_ms': frame.encode_ms,
                                 'frames': [dict(id=key, size=len(frame.images[key])) for key in STREAM_IDS]}).encode()
            body = struct.pack('!I', len(header)) + header + b''.join(frame.images.values())
            self._send(200, 'application/octet-stream', body)
            return
        for key in STREAM_IDS:
            if url.path == f'/snapshot/{key}.jpg':
                frame = owner.wait_frame()
                if frame is None:
                    self._send(503, 'text/plain', b'Waiting for rendered frames')
                else:
                    self._send(200, 'image/jpeg', frame.images[key], self._frame_headers(frame))
                return
            if url.path == f'/stream/{key}.mjpg':
                self._headers(200, 'multipart/x-mixed-replace; boundary=frame', extra={'Connection': 'close'})
                self.close_connection = True
                sequence = 0
                while not owner.stopped:
                    frame = owner.wait_frame(sequence)
                    if frame is None:
                        continue
                    jpeg = frame.images[key]
                    self.wfile.write((f'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n'
                                      f'X-Sequence: {frame.sequence}\r\nX-Captured-At: {frame.captured_at}\r\n\r\n').encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b'\r\n')
                    self.wfile.flush()
                    sequence = frame.sequence
                return
        self._send(404, 'text/plain', b'Not found')

    @staticmethod
    def _frame_headers(frame):
        return {'X-Sequence': frame.sequence, 'X-Captured-At': frame.captured_at}


def add_stream_arguments(parser):
    parser.add_argument('--stream', action='store_true', help='enable LAN robot and source-view preview')
    parser.add_argument('--stream-host', default='0.0.0.0', help='stream listen address')
    parser.add_argument('--stream-port', type=int, default=8876, help='stream HTTP port (default 8876)')
    parser.add_argument('--stream-fps', type=int, choices=range(1, 61), default=30, metavar='1–60')
    parser.add_argument('--stream-quality', type=int, choices=range(1, 96), default=80, metavar='1–95')


def stream_config_from_args(args):
    return StreamConfig(args.stream_host, args.stream_port, args.stream_fps, args.stream_quality)
