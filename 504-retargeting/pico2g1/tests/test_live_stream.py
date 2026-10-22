"""Exercise actual HTTP frames, freshness, backpressure, and server lifetime."""
import concurrent.futures
import http.client
import json
import os
import socket
import struct
import time
import unittest

import numpy as np
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6 import QtGui, QtWidgets

from sim_scenes import live_stream


def unpack(payload):
    length = struct.unpack('!I', payload[:4])[0]
    header = json.loads(payload[4:4 + length])
    offset, images = 4 + length, {}
    for frame in header['frames']:
        images[frame['id']] = QtGui.QImage.fromData(payload[offset:offset + frame['size']])
        offset += frame['size']
    assert offset == len(payload)
    return header, images


class StreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.server = live_stream.LiveStreamServer(live_stream.StreamConfig(host='127.0.0.1', port=0, fps=60))
        self.server.start()
        self.addCleanup(self.server.close)

    def get(self, path):
        connection = http.client.HTTPConnection(*self.server.address, timeout=4)
        try:
            connection.request('GET', path)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def publish(self, value):
        self.server.publish({key: np.full((72, 128, 3), value, np.uint8)
                             for key in live_stream.STREAM_IDS})

    def test_eight_view_bundle_and_individual_jpeg_share_frame(self):
        self.publish(71)
        status, headers, body = self.get('/api/frames?after=0')
        self.assertEqual(status, 200)
        self.assertIn('no-store', headers['Cache-Control'])
        meta, images = unpack(body)
        self.assertEqual(set(images), {'main', 'robot_head', 'robot_left_wrist', 'robot_right_wrist',
                                      'source_body_front', 'source_body_side', 'source_left_hand', 'source_right_hand'})
        for name, image in images.items():
            self.assertEqual((image.width(), image.height()), (128, 72))
            self.assertAlmostEqual(image.pixelColor(60, 30).red(), 71, delta=2)
            status, headers, body = self.get('/snapshot/' + name + '.jpg')
            self.assertEqual(status, 200)
            self.assertEqual(int(headers['X-Sequence']), meta['sequence'])
            self.assertEqual(QtGui.QImage.fromData(body), image)
        self.assertLess(meta['age_ms'], 1000)

    def test_slow_viewer_skips_to_latest_and_waiter_wakes(self):
        self.publish(10)
        first, _ = unpack(self.get('/api/frames')[2])
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pending = pool.submit(self.get, '/api/frames?after=' + str(first['sequence']))
            time.sleep(.05)
            self.assertFalse(pending.done())
            for value in range(20, 90):
                self.publish(value)
            self.assertEqual(pending.result()[0], 200)
        # No consumer drains intermediate sets; a later request sees the newest.
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            last, images = unpack(self.get('/api/frames')[2])
            if abs(images['main'].pixelColor(30, 30).red() - 89) <= 1:
                break
        self.assertAlmostEqual(images['main'].pixelColor(30, 30).red(), 89, delta=1)
        self.assertGreater(last['sequence'], first['sequence'])
        self.assertEqual(self.get('/api/frames?after=' + str(last['sequence']))[0], 204)

    def test_routes_mjpeg_and_shutdown_with_open_client(self):
        self.assertEqual(self.get('/api/frames?after=broken')[0], 400)
        self.assertEqual(self.get('/stream/unknown.mjpg')[0], 404)
        self.assertEqual(self.get('/../../README.md')[0], 404)
        self.publish(150)
        manifest = json.loads(self.get('/api/streams')[2])
        self.assertEqual(len(manifest['streams']), 8)
        connection = http.client.HTTPConnection(*self.server.address, timeout=3)
        connection.request('GET', '/stream/robot_left_wrist.mjpg')
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertIn('multipart/x-mixed-replace', response.getheader('Content-Type'))
        self.assertEqual(response.fp.readline(), b'--frame\r\n')
        headers = {}
        while (line := response.fp.readline()) != b'\r\n':
            key, value = line.decode().strip().split(': ', 1)
            headers[key] = value
        jpeg = response.fp.read(int(headers['Content-Length']))
        self.assertFalse(QtGui.QImage.fromData(jpeg).isNull())
        start = time.monotonic()
        self.server.close()
        self.assertLess(time.monotonic() - start, 1.5)
        connection.close()
        with socket.socket() as probe:
            self.assertNotEqual(probe.connect_ex(self.server.address), 0)

    def test_occupied_port_does_not_start_worker(self):
        other = live_stream.LiveStreamServer(live_stream.StreamConfig(host='127.0.0.1', port=self.server.address[1]))
        with self.assertRaises(OSError):
            other.start()
        other.close()

    def test_nonreading_tcp_client_does_not_hold_back_other_viewers(self):
        # A large JPEG exceeds the slow socket's receive window. The producer and
        # another viewer must progress while its HTTP handler is blocked writing.
        noise = np.random.default_rng(4).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        self.server.publish({key: noise for key in live_stream.STREAM_IDS})
        self.assertIsNotNone(self.server.wait_frame(timeout=3))
        slow = socket.socket()
        self.addCleanup(slow.close)
        slow.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        slow.settimeout(3)
        slow.connect(self.server.address)
        slow.sendall(b'GET /stream/main.mjpg HTTP/1.1\r\nHost: localhost\r\n\r\n')
        self.assertTrue(slow.recv(1))
        first_sequence = self.server.latest.sequence
        start = time.monotonic()
        self.publish(42)
        status, _, payload = self.get('/api/frames?after=' + str(first_sequence))
        self.assertEqual(status, 200)
        self.assertLess(time.monotonic() - start, 1)
        _, images = unpack(payload)
        self.assertAlmostEqual(images['main'].pixelColor(20, 20).red(), 42, delta=1)

    def test_stop_closes_waiting_keepalive_so_browser_can_reconnect(self):
        self.publish(30)
        frame = self.server.wait_frame(timeout=2)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            waiting = pool.submit(self.get, '/api/frames?after=' + str(frame.sequence))
            time.sleep(.05)
            self.assertFalse(waiting.done())
            self.server.close()
            status, headers, _ = waiting.result()
        self.assertEqual(status, 503)
        self.assertEqual(headers.get('Connection'), 'close')


if __name__ == '__main__':
    unittest.main()
