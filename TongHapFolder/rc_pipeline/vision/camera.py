import time
import threading
import subprocess
import cv2
import numpy as np


class RPiMJPEGCamera:
    """
    rpicam-vid MJPEG stdout을 백그라운드 스레드에서 계속 읽고,
    항상 최신 프레임 1장만 유지한다.
    """

    def __init__(self, width=640, height=480, framerate=10):
        self.width = width
        self.height = height
        self.framerate = framerate

        self.buffer = bytearray()
        self.latest_frame = None
        self.latest_frame_id = 0
        self.latest_rx_done_mono = None

        self.lock = threading.Lock()
        self.alive = True

        cmd = [
            "rpicam-vid",
            "-t",
            "0",
            "--nopreview",
            "--codec",
            "mjpeg",
            "--width",
            str(width),
            "--height",
            str(height),
            "--framerate",
            str(framerate),
            "-o",
            "-",
        ]

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

        if self.proc.stdout is None:
            raise RuntimeError("failed to open rpicam-vid stdout pipe")

        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with self.lock:
                if self.latest_frame is not None:
                    break
            time.sleep(0.01)

    def _reader_loop(self):
        while self.alive:
            chunk = self.proc.stdout.read(4096)
            if not chunk:
                break

            self.buffer.extend(chunk)

            frames = []
            search_pos = 0

            while True:
                start = self.buffer.find(b"\xff\xd8", search_pos)
                if start == -1:
                    break

                end = self.buffer.find(b"\xff\xd9", start + 2)
                if end == -1:
                    break

                frames.append((start, end + 2))
                search_pos = end + 2

            if not frames:
                if len(self.buffer) > 2 * 1024 * 1024:
                    self.buffer = self.buffer[-65536:]
                continue

            last_start, last_end = frames[-1]
            jpg = self.buffer[last_start:last_end]
            self.buffer = self.buffer[last_end:]

            arr = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            rx_done = time.monotonic()

            with self.lock:
                self.latest_frame = frame
                self.latest_frame_id += 1
                self.latest_rx_done_mono = rx_done

    def read(self, wait_timeout=2.0):
        deadline = time.monotonic() + wait_timeout

        while self.alive and time.monotonic() < deadline:
            with self.lock:
                if self.latest_frame is not None:
                    return True, {
                        "frame": self.latest_frame.copy(),
                        "frame_id": self.latest_frame_id,
                        "rx_done_mono": self.latest_rx_done_mono,
                    }
            time.sleep(0.005)

        return False, None

    def release(self):
        self.alive = False
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(timeout=2.0)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
