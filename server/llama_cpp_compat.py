"""llama.cpp-compatible probes and slot lanes, behind --llama-cpp-compat.

WHY: a proxy such as llama-swap supervises a child the way it supervises
llama-server: it polls llama.cpp's GET /slots and GET /props for the child's
window and per-slot state, and pins a session to one slot by sending
``id_slot`` in the generation body and reading the same number back from the
response (slot_affinity.go injects it, metrics.go learns the lane only from a
response that carries it). Splash has neither route and no llama.cpp slots, so
this module answers both from Splash's own status document and keeps one lane
per in-flight generation, which is what its /slots rows are keyed by and what
every generation response stamps back as ``id_slot``.

Off by default: without the option this module adds no routes, accepts no
``id_slot``, and stamps nothing.
"""

import json

# The llama-server status endpoints a proxy polls. They answer quietly: a poll
# per second would bury the Started/Done lines, so a probe stays as silent in
# the console as the 404 it replaced.
QUIET_PROBES = frozenset({"/slots", "/props"})
# The generation routes whose requests occupy a lane. count_tokens, tokenize
# and apply-template answer from the prompt alone and never reach the engine,
# so they take no lane.
GENERATION_ROUTES = frozenset(
    {
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/v1/messages",
    }
)
# The field a proxy names its chosen lane with, in the body and stamped back
# onto the response.
ID_SLOT = "id_slot"


def lane_of(body):
    """The lane `body` asks for with its id_slot: an int in the lane range,
    else None for a request that names none. A bool is not a lane."""
    value = body.get(ID_SLOT) if isinstance(body, dict) else None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def request_row(request, lane, window):
    """One /slots row for one active_requests entry on a known lane."""
    started = request.get("phase", "") != "queued"
    return {
        # The lane, never Splash's own request id (active_requests[].id, e.g.
        # 172): the proxy joins /slots rows to sessions by slot id and sizes
        # its slot table max(id)+1 wide - a request-id row would join nothing
        # and size the table 173.
        "id": lane,
        "state": "busy" if started else "idle",
        "n_ctx": window,
        "is_processing": started,
        "n_prompt_tokens": int(request.get("prompt_tokens", 0)),
        "n_prompt_tokens_processed": int(request.get("prompt_processed", 0)),
        "n_prompt_tokens_cache": 0,
        # llama.cpp leaves next_token empty until the first token is decoded;
        # a prefill row with n_decoded 0 would read as decode to any reader.
        "next_token": (
            [{"n_decoded": int(request.get("generated_tokens", 0))}]
            if started and int(request.get("generated_tokens", 0)) > 0
            else []
        ),
    }


def idle_row(lane, window):
    return {
        "id": lane,
        "state": "idle",
        "n_ctx": window,
        "is_processing": False,
        "n_prompt_tokens": 0,
        "n_prompt_tokens_processed": 0,
        "n_prompt_tokens_cache": 0,
        "next_token": [],
    }


def slots_body(status, lanes=None, queue_size=1):
    """A llama.cpp-shaped /slots body from Splash's own status document.

    Splash has no llama.cpp slots; the rows are translated from its
    active_requests. One row per active request, up to queue_size, each keyed
    by its lane (`lanes` maps Splash request id -> lane): the id is stable for
    a request's whole life even though the status lists requests in the
    scheduler's order, which can flip between polls. An active request the
    lane map does not know gets no row rather than a lying one; without a map
    at all rows fall back to list position capped at queue_size, which for a
    one-lane child is exactly the single row at id 0. No active request yields
    one idle row per lane so n_ctx stays the window a reader of /slots reads.

    n_ctx is maximum_context_tokens - the window. n_prompt_tokens_cache is
    always 0: the prefix cache's kv_hit_tokens is global to the whole cache,
    there is no per-request split, and inventing one would misreport the
    session.

    A request the engine has not STARTED (phase "queued": accepted, waiting
    for its turn on a lane) must NOT be reported as a processing slot. A
    proxy's prefill-stall guard reads /slots and cuts any slot that is
    processing with nothing decoded and a flat n_prompt_tokens_processed
    across its budget; a queued request reports processed = 0 forever while
    it waits, and a wait longer than the budget reads as a frozen prefill.
    The status says which: active_requests[].phase is "queued", "prefill" or
    "decode". Only prefill and decode are work the engine has started; queued
    keeps the row (so n_ctx stays the window) but with is_processing false,
    exactly like the idle row. If phase were ever missing the same rule holds
    by content: processed = 0 with nothing decoded is not started.
    """
    window = int(status.get("maximum_context_tokens", 0))
    requests = status.get("active_requests") or []
    rows = []
    for position, request in enumerate(requests[: max(int(queue_size), 1)]):
        if lanes is not None:
            lane = lanes.get(request.get("id"))
            if lane is None:
                continue
        else:
            lane = position
        rows.append(request_row(request, lane, window))
    if not rows:
        rows = [idle_row(lane, window) for lane in range(max(int(queue_size), 1))]
    return json.dumps({"slots": rows})


def props_body(status, queue_size=1):
    """A minimal llama.cpp-shaped /props body: the fields a native status
    reader wants (model_path, default_generation_settings.n_ctx, total_slots,
    build_info). total_slots is the configured --queue-size - the truth for
    the status display; a proxy sizes its slot table from the /slots rows,
    not from here."""
    plan = status.get("memory_plan") or {}
    model = plan.get("model") or {}
    instance = status.get("instance") or {}
    return json.dumps(
        {
            "total_slots": max(int(queue_size), 1),
            "default_generation_settings": {
                "n_ctx": int(status.get("maximum_context_tokens", 0))
            },
            "model_alias": instance.get("model", ""),
            "model_path": model.get("model_name", ""),
            "build_info": f"splash llama-cpp-compat schema_version={status.get('schema_version', 0)}",
        }
    )


def stamp_id_slot(document, job):
    """`document` with the generation's lane stamped as id_slot: a proxy that
    pinned the request to a slot learns the lane only from a response that
    carries it, so every generation response names the lane its /slots row is
    keyed by. An unmapped request - one past the lanes - gets no field rather
    than a lying one. The generation response builders call this; without the
    option the job carries no map and nothing is stamped."""
    lanes = getattr(job, "lane_map", None)
    if lanes is None:
        return document
    lane = lanes.stamp_id_slot(job.request_id)
    if lane is None:
        return document
    document["id_slot"] = lane
    return document


class LaneMap:
    """Splash request id -> the lane its /slots row uses, for the request's
    whole life.

    WHY this exists at all (a child with --queue-size above 1 runs several
    requests at once): the proxy joins a session to a /slots row by the lane
    id it assigned, and Splash's status lists active_requests in ITS scheduler
    order, which is not arrival order and can flip between polls. A row id
    following list position would show session A with session B's phase and
    counters - the lane governor, the liveness monitor and the menus all read
    those. So each request gets ONE lane for its whole life, and /slots rows
    are keyed by it.

    The lane is the PROXY's number, not the engine's: the proxy's affinity
    middleware injects id_slot into the request body before it reaches
    Splash, and the proxy learns a lane only from a response that carries
    id_slot. note_slot records the injected id_slot; the response builders
    stamp the same number back (stamp_id_slot), so the proxy's store and this
    map agree by construction. A request with no id_slot (housekeeping, a
    curl) takes the lowest free lane. The engine's own lane (StartEvent.lane)
    is recorded as a cross-check: if it ever differs from the proxy's,
    note_start logs one line naming both - the proxy's number wins, because
    that is what the proxy joins by.

    Lanes are 0..queue_size-1, so the proxy's slot table stays queue_size
    wide. A request that cannot get a lane (more in flight than queue_size,
    which the proxy's own cap bounds) maps to None and gets no row.
    """

    def __init__(self, queue_size, log=lambda _message: None):
        self.queue_size = max(int(queue_size), 0)
        self.log = log
        self._lane = {}
        # The lane each request held while it was mapped. note_done frees the
        # lane, but a response built after the engine's terminal record still
        # stamps from this: the lane is the request's for its whole life, and
        # the proxy learns it only from a response that carries it.
        self._held = {}
        self._engine_lane = {}
        self._warned = set()

    def _claim(self, request_id, lane):
        # A lane re-assigned by the proxy belongs to the newest request. The
        # loser keeps no lane at all: its response stamps nothing, and it
        # gets no /slots row, rather than one naming a lane another request
        # now holds.
        for other, held in list(self._lane.items()):
            if held == lane and other != request_id:
                del self._lane[other]
                self._held.pop(other, None)
        self._lane[request_id] = lane
        self._held[request_id] = lane

    def note_slot(self, request_id, id_slot):
        """The request arrived with the proxy's id_slot (or None)."""
        if self.queue_size <= 0:
            return
        if id_slot is not None and 0 <= id_slot < self.queue_size:
            self._claim(request_id, id_slot)
            return
        taken = set(self._lane.values())
        for candidate in range(self.queue_size):
            if candidate not in taken:
                self._claim(request_id, candidate)
                return
        # More requests in flight than lanes (the proxy's own cap bounds
        # this): no lane, and no /slots row for it. A request that never
        # held a lane keeps none (stamp_id_slot reads _held).

    def note_start(self, request_id, engine_lane):
        """Splash started the request on its own engine lane (StartEvent)."""
        self._engine_lane[request_id] = engine_lane
        mine = self._lane.get(request_id)
        if mine is not None and engine_lane != mine and request_id not in self._warned:
            self._warned.add(request_id)
            self.log(
                f"lane cross-check: request {request_id} on proxy lane {mine}, "
                f"engine lane {engine_lane}; the proxy's lane is what /slots reports"
            )

    def note_done(self, request_id):
        """The request left the engine (done, error or cancelled): its lane
        is free for the next request, but the request keeps it for its own
        response (stamp_id_slot)."""
        self._lane.pop(request_id, None)
        self._engine_lane.pop(request_id, None)
        self._warned.discard(request_id)

    def stamp_id_slot(self, request_id):
        """The lane to stamp onto the response, or None when unmapped. The
        lane is read even after note_done freed it: the engine's terminal
        record runs before the response is built, and the request's lane is
        its own for its whole life."""
        return self._held.get(request_id)

    def snapshot(self):
        return dict(self._lane)


def lane_map(app, queue_size):
    """The lane map --llama-cpp-compat keeps on the frontend, which the
    generation response builders read and the backend log records free.

    The map frees a lane from the backend's own Done/Error/Closed log records
    (NativeBackend._record runs once per submitted job, whatever ended it), so
    a lane is held from the request's arrival until the engine reports it
    done, errored or cancelled. The engine's lane arrives as a Started record
    and is only a cross-check.

    `app.log_request` is the logger the backend runs its records through
    (NativeBackend.request_logger), so the hooks run before the record reaches
    the console."""
    lanes = LaneMap(queue_size, log=app.log_compat)
    app.lane_map = lanes

    def record_compat(record):
        if record["outcome"] == "started":
            lanes.note_start(record["request_id"], record["lane"])
        else:
            lanes.note_done(record["request_id"])
        app.log_request(record)

    app.log_request = record_compat
    return lanes


def attach_probes(server, lanes, queue_size):
    """The two quiet probe routes on `server`, answering from the server's own
    status document and `lanes`."""

    def probes(self):
        # self is the FrontendHandler answering the GET.
        status = server.status()
        if self.route == "/slots":
            payload = json.loads(slots_body(status, lanes.snapshot(), queue_size))
        else:
            payload = json.loads(props_body(status, queue_size))
        self._json(200, payload)

    server.compat_probes = probes
