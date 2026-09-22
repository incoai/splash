"""Fixed-size latency histograms shared by HTTP preparation and token delivery."""

import math
import threading
import time
from bisect import bisect_left
from contextlib import contextmanager

BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1,
    2.5,
    5,
    10,
    30,
    60,
    120,
    300,
    900,
    1800,
)
STAGES = (
    "http_request",
    "upload",
    "preparation_queue",
    "preparation",
    "template",
    "tokenization",
    "grammar",
    "images",
    "native_queue",
    "ttft",
    "output_interval",
)


class LatencyMetrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._counts = {name: [0] * (len(BUCKETS) + 1) for name in STAGES}
        self._sums = dict.fromkeys(STAGES, 0.0)

    def observe(self, stage, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            return
        with self._lock:
            self._counts[stage][bisect_left(BUCKETS, seconds)] += 1
            self._sums[stage] += seconds

    @contextmanager
    def measure(self, stage):
        started = time.monotonic()
        try:
            yield
        finally:
            self.observe(stage, time.monotonic() - started)

    def snapshot(self):
        with self._lock:
            result = {}
            for stage, counts in self._counts.items():
                cumulative = 0
                buckets = {}
                for bound, count in zip((*BUCKETS, "+Inf"), counts):
                    cumulative += count
                    buckets[str(bound)] = cumulative
                result[stage] = {
                    "buckets": buckets,
                    "count": cumulative,
                    "sum": self._sums[stage],
                }
            return result


class RequestLatency:
    def __init__(self, metrics, received_at):
        self.metrics = metrics
        self.received_at = received_at
        self.last_output = None

    def tokens(self):
        # Token callbacks for a request are serialized by the native transport.
        now = time.monotonic()
        if self.last_output is None:
            self.metrics.observe("ttft", now - self.received_at)
        else:
            self.metrics.observe("output_interval", now - self.last_output)
        self.last_output = now


def prometheus_latency(snapshot):
    lines = []
    for stage in STAGES:
        values = snapshot.get(stage)
        if values is None:
            continue
        name = f"splash_{stage}_seconds"
        lines.append(f"# TYPE {name} histogram")
        for bound, count in values["buckets"].items():
            lines.append(f'{name}_bucket{{le="{bound}"}} {count}')
        lines.append(f"{name}_count {values['count']}")
        lines.append(f"{name}_sum {values['sum']}")
    return lines
