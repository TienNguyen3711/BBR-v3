"""Refresh commands independently of policy computation, with a watchdog."""
import math
import threading
import time
from qbbr.action import bbr_hook
from qbbr.action.kernel_contract import HEARTBEAT_S, quantize_request


class CommandHeartbeat:
    def __init__(self, interval_s=HEARTBEAT_S):
        if not 0 < interval_s < 1:
            raise ValueError('Heartbeat interval must be below the one-second lease')
        self.interval_s = interval_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._gain = None
        self._deadline = 0.
        self._error = None

    def check(self):
        if self._error is not None:
            raise RuntimeError('Kernel command heartbeat failed') from self._error

    def submit(self, gain, valid_for_s):
        quantize_request(gain)
        if not math.isfinite(valid_for_s) or valid_for_s <= 0:
            raise ValueError('Command validity must be positive')
        with self._lock:
            self.check()
            if self._stop.is_set():
                raise RuntimeError('Heartbeat is closed')
            bbr_hook.set_pacing_gain(gain)
            self._gain = gain
            self._deadline = time.monotonic() + valid_for_s
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()

    def _run(self):
        while not self._stop.wait(self.interval_s):
            with self._lock:
                try:
                    # A live thread cannot keep a stalled policy's action forever.
                    bbr_hook.set_pacing_gain(self._gain if time.monotonic() < self._deadline else None)
                except Exception as exc:
                    self._error = exc
                    self._stop.set()
                    try:
                        bbr_hook.clear_pacing_gain_override()
                    except OSError:
                        pass  # the kernel lease remains the final fallback

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        bbr_hook.clear_pacing_gain_override()
