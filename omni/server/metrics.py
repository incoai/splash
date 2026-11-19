"""Prometheus text rendering of the server's status document."""

from __future__ import annotations

import math

from .latency import prometheus_latency


def is_finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


# Each gauge or counter Prometheus reads from the status, by the path of
# keys to its value: the audio engine's document (runtime/omni/OmniServer.cpp),
# then what the server adds to it (backend.NativeBackend.status,
# http_server.FrontendServer.status).
PROMETHEUS_SERIES = {
    "splash_ready": ("ready",),
    "splash_lanes_active": ("lanes", "active"),
    "splash_lanes_total": ("lanes", "total"),
    "splash_queued": ("queued",),
    "splash_requests_received_total": ("requests", "received"),
    "splash_requests_completed_total": ("requests", "completed"),
    "splash_requests_failed_total": ("requests", "failed"),
    "splash_memory_allocated_bytes": ("memory", "allocated_bytes"),
    "splash_memory_peak_allocated_bytes": ("memory", "peak_allocated_bytes"),
    "splash_engine_uptime_seconds": ("uptime_seconds",),
    "splash_transport_pending": ("transport", "pending"),
    "splash_transport_restarts_total": ("transport", "restarts"),
    "splash_http_requests_active": ("http", "requests", "active"),
    "splash_http_requests_capacity": ("http", "requests", "capacity"),
    "splash_http_request_body_bytes_active": ("http", "request_body_bytes", "active"),
    "splash_http_connections_active": ("http", "connections", "active"),
}


def prometheus_metrics(status):
    """Render low-cardinality metrics directly from the status document."""

    def value(path):
        current = status
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        if isinstance(current, bool):
            return 1 if current else 0
        if is_finite_number(current):
            return current
        return None

    lines = [
        "# HELP splash_info Splash runtime metrics.",
        "# TYPE splash_info gauge",
        'splash_info{runtime="native"} 1',
    ]
    for name, path in PROMETHEUS_SERIES.items():
        metric_value = value(path)
        if metric_value is not None:
            lines.append(f"{name} {metric_value}")
    lines.extend(prometheus_latency(status.get("latency", {})))
    return "\n".join(lines) + "\n"
