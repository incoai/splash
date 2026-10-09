"""HTTP ingress bounds: the connections the server serves at once and the
threads that serve them, the requests and request body bytes it admits, and
the closing of connections it refuses."""

import queue
import re
import select
import socket
import threading
import time
import weakref

from . import json_codec
from .errors import APIError

# The answer to a connection that gets no slot, or gives up its slot before
# its request is read. With no request path, no API dialect is known: a
# generic server error with its stable diagnostic code.
_CONNECTION_OVERLOADED_PAYLOAD = (
    b'{"error":{"type":"server_error","code":"frontend_overloaded",'
    b'"message":"HTTP connection capacity is exhausted"}}'
)
CONNECTION_OVERLOADED_RESPONSE = (
    b"HTTP/1.1 503 Service Unavailable\r\n"
    b"Content-Type: application/json\r\nConnection: close\r\nRetry-After: 1\r\n"
    b"Content-Length: %d\r\n\r\n%s"
    % (len(_CONNECTION_OVERLOADED_PAYLOAD), _CONNECTION_OVERLOADED_PAYLOAD)
)
# The most of a request's head a connection waits for without a thread: the
# kernel holds it unread, in a receive buffer of 128 KiB on macOS unless
# tuned, so its client never waits for room. A thread reads a longer head,
# under http.server's own limits.
HEAD_WAIT_BYTES = 64 * 1024
# The last of a request line's three words when BaseHTTPRequestHandler reads
# header lines after the line: an HTTP version it takes, below 2.0.
_HEADER_VERSION = re.compile(r"HTTP/([0-9]{1,10})\.([0-9]{1,10})")


class HttpAdmission:
    """Nonwaiting capacity gate, in request counts or input bytes."""

    def __init__(self, capacity):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("HTTP admission capacity must be a positive integer")
        self.capacity = capacity
        self.active = 0
        # Input finalizers can run during a stats snapshot on this thread.
        self.lock = threading.RLock()

    def acquire(self, amount=1):
        with self.lock:
            if self.active + amount > self.capacity:
                return False
            self.active += amount
            return True

    def release(self, amount=1):
        with self.lock:
            if amount > self.active:
                raise RuntimeError("HTTP admission slot released without acquisition")
            self.active -= amount

    def stats(self):
        with self.lock:
            return {"active": self.active, "capacity": self.capacity}


def refuse_connection(connection):
    """Send CONNECTION_OVERLOADED_RESPONSE without waiting: the server has
    read no request from the connection and written it no response, so the
    response fits in its send buffer."""
    try:
        connection.send(CONNECTION_OVERLOADED_RESPONSE, socket.MSG_DONTWAIT)
    except OSError:
        pass


def _ready(connection):
    """Whether `connection` is ready for its thread, by what its client has
    sent, looked at without being read: the head of its request, as
    BaseHTTPRequestHandler reads it, has arrived (the request line and,
    after one that names an HTTP version below 2.0, header lines up to an
    empty one), or the client has closed, or has sent HEAD_WAIT_BYTES
    without ending the head."""
    try:
        sent = connection.recv(HEAD_WAIT_BYTES, socket.MSG_PEEK | socket.MSG_DONTWAIT)
    except BlockingIOError:
        return False
    except OSError:
        # Reset, or closed once it gave its slot away.
        return True
    if not sent or len(sent) == HEAD_WAIT_BYTES:
        return True
    line, newline, _ = sent.partition(b"\n")
    if not newline:
        return False
    words = str(line, "iso-8859-1").split()
    version = _HEADER_VERSION.fullmatch(words[-1]) if len(words) == 3 else None
    if version is None or (int(version[1]), int(version[2])) >= (2, 0):
        return True
    return b"\n\n" in sent or b"\n\r\n" in sent


def _drain(connection):
    """Read and drop what the client of `connection` has sent; False once it
    has closed."""
    while True:
        try:
            if not connection.recv(65536, socket.MSG_DONTWAIT):
                return False
        except BlockingIOError:
            return True
        except OSError:
            return False


def _close_at_once(connection):
    """Close `connection` without waiting on its client, what it has sent
    read first: closed with input unread, the connection would be reset."""
    _drain(connection)
    connection.close()


def _watch(events, descriptor, change):
    try:
        events.control([select.kevent(descriptor, select.KQ_FILTER_READ, change)], 0)
    except OSError:
        # Closed once it gave its slot away.
        pass


class Workers:
    """`capacity` threads, started with the server, that serve connections
    with `serve(connection, address)`, each reused from one connection to the
    next. Starting a thread waits for it to run, and threads started while
    a burst arrives hold off the accept loop until the kernel's queue of
    connections overflows, which macOS resets: so none starts later, and
    handing a connection over is a queue put."""

    def __init__(self, capacity, serve):
        self.serve = serve
        self.queue = queue.SimpleQueue()
        self.threads = [
            threading.Thread(target=self._run, name="connection worker", daemon=True)
            for _ in range(capacity)
        ]
        for thread in self.threads:
            thread.start()

    def put(self, connection, address):
        self.queue.put((connection, address))

    def stop(self):
        """End each worker once the connections handed over are served."""
        for _ in self.threads:
            self.queue.put(None)

    def _run(self):
        while (item := self.queue.get()) is not None:
            self.serve(*item)


class WaitingConnections:
    """Connections that wait on their clients without a thread, all on one
    thread: ones given a slot, until the heads of their requests arrive, and
    ones answered without their requests read, until their clients close.

    One given a slot is looked at as it is handed over and again each time
    its client sends more, which kqueue reports once per arrival; what the
    client sent is left unread, for the connection's thread to read. Once it
    is _ready, it keeps its slot (`slots.arrived`) and goes to
    `dispatch(connection, address)`. One not ready by the deadline it was
    given at its accept is closed without a response; one that gives its
    slot away comes back here answered.

    Closing a connection with request bytes unread resets it, and the reset
    can destroy the answer before the client reads it, as it does when the
    client sends its request only after the close. An answered one is
    half-closed instead; what its client still sends is read and dropped,
    and it is closed once the client has closed or `linger` seconds after
    its answer. At most `capacity` linger at once: one more takes the place
    of the longest lingering, whose client has had the longest to read its
    answer. At most `capacity` more wait for the thread to take them; any
    beyond them are closed at once.
    """

    # The most events the thread takes at once.
    EVENTS = 64

    def __init__(self, slots, dispatch, capacity, linger):
        self.slots = slots
        self.dispatch = dispatch
        self.capacity = capacity
        self.linger = linger
        self.lock = threading.Lock()
        # Connections the thread has yet to take: given a slot, as
        # (connection, address, deadline), and answered, as (connection,
        # deadline).
        self.given = []
        self.answered = []
        self.stopped = False
        # A byte sent on the second end wakes the thread to take them.
        self.wakeup = socket.socketpair()
        for end in self.wakeup:
            end.setblocking(False)
        self.thread = threading.Thread(
            target=self._run, name="waiting connections", daemon=True
        )
        self.thread.start()

    def wait(self, connection, address, deadline):
        """Hold `connection`, given a slot, until the head of its request
        arrives, which is due by `deadline`."""
        with self.lock:
            stopped = self.stopped
            if not stopped:
                wake = self._queue(self.given, (connection, address, deadline))
        if stopped:
            self.slots.expire(connection)
            _close_at_once(connection)
        elif wake:
            self._wake()

    def close(self, connection):
        """Close `connection`, answered without its request read, as the
        class does; one held for its request stops waiting for it."""
        try:
            connection.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        deadline = time.monotonic() + self.linger
        with self.lock:
            linger = not self.stopped and len(self.answered) < self.capacity
            if linger:
                wake = self._queue(self.answered, (connection, deadline))
        if not linger:
            _close_at_once(connection)
        elif wake:
            self._wake()

    def stop(self):
        """Close every connection held, freeing the slots of those that wait
        for their requests, and end the thread."""
        with self.lock:
            self.stopped = True
        self._wake()
        self.thread.join()
        for end in self.wakeup:
            end.close()

    def _queue(self, items, item):
        # Under the lock. Until the thread takes what it was woken for, it
        # needs no other wakeup.
        wake = not (self.given or self.answered)
        items.append(item)
        return wake

    def _wake(self):
        try:
            self.wakeup[1].send(b"\0")
        except OSError:
            # Wakeups the thread has yet to read fill its buffer, or it has
            # stopped.
            pass

    def _run(self):
        events = select.kqueue()
        _watch(events, self.wakeup[0].fileno(), select.KQ_EV_ADD)
        # Each connection held, by descriptor: ones given a slot, as
        # [connection, address, deadline], and answered ones, as
        # [connection, deadline]. Another thread closes a connection given a
        # slot only once it has given its slot away, with `capacity` answered
        # ones already waiting for this thread: what is read of it then
        # fails, or comes from the connection its descriptor went to next,
        # which is looked at when it is taken, and as its slot is gone it is
        # dropped here, at the latest by its deadline.
        waiting, lingering = {}, {}
        watch = select.KQ_EV_ADD | select.KQ_EV_CLEAR
        while True:
            with self.lock:
                given, self.given = self.given, []
                answered, self.answered = self.answered, []
                stopped = self.stopped
            if stopped:
                for connection, *_ in (*waiting.values(), *given):
                    if self.slots.expire(connection):
                        _close_at_once(connection)
                for connection, _ in (*lingering.values(), *answered):
                    _close_at_once(connection)
                events.close()
                return
            for connection, address, deadline in given:
                descriptor = connection.fileno()
                if _ready(connection):
                    self._serve(events, connection, address)
                else:
                    waiting[descriptor] = [connection, address, deadline]
                    _watch(events, descriptor, watch)
            for connection, deadline in answered:
                descriptor = connection.fileno()
                # It gave its slot away from there, or one closed before did.
                waiting.pop(descriptor, None)
                _watch(events, descriptor, watch)
                if not _drain(connection):
                    _close_at_once(connection)
                    continue
                if len(lingering) == self.capacity:
                    # Lingering in the order answered, the longest first.
                    _close_at_once(lingering.pop(next(iter(lingering)))[0])
                lingering[descriptor] = [connection, deadline]
            deadlines = [
                entry[-1] for entry in (*waiting.values(), *lingering.values())
            ]
            timeout = None
            if deadlines:
                timeout = max(0.0, min(deadlines) - time.monotonic())
            for event in events.control(None, self.EVENTS, timeout):
                descriptor = event.ident
                if descriptor == self.wakeup[0].fileno():
                    try:
                        self.wakeup[0].recv(4096)
                    except BlockingIOError:
                        pass
                elif descriptor in waiting:
                    connection, address, _ = waiting[descriptor]
                    if event.flags & select.KQ_EV_EOF or _ready(connection):
                        del waiting[descriptor]
                        self._serve(events, connection, address, descriptor)
                elif descriptor in lingering:
                    connection, _ = lingering[descriptor]
                    if not _drain(connection):
                        del lingering[descriptor]
                        _close_at_once(connection)
            now = time.monotonic()
            for descriptor, (connection, _, deadline) in list(waiting.items()):
                if deadline <= now:
                    del waiting[descriptor]
                    # One whose slot is gone has given it away.
                    if self.slots.expire(connection):
                        _close_at_once(connection)
            for descriptor, (connection, deadline) in list(lingering.items()):
                if deadline <= now:
                    del lingering[descriptor]
                    _close_at_once(connection)

    def _serve(self, events, connection, address, watched=None):
        """Hand `connection`, ready, to its thread with its slot, no longer
        watching `watched`, its descriptor, if it was watched. One that has
        given its slot away comes back answered, still watched, or has been
        closed."""
        if not self.slots.arrived(connection):
            return
        if watched is not None:
            _watch(events, watched, select.KQ_EV_DELETE)
        self.dispatch(connection, address)


class ConnectionSlots:
    """The connections the server holds for requests, at most `capacity`.

    A connection waits for its request without a thread, held by `waiting`,
    until the head of its request has arrived; it then keeps its slot, and
    one of `workers`, a thread per slot started with the server, reads and
    serves the request with `serve(connection, address)`. No thread starts
    while connections arrive: they are taken as fast as they arrive, before
    the kernel's queue overflows, and one whose client sends nothing, or
    only part of a head, ties up no thread.

    One whose request's head has yet to arrive, or that drains an upload
    refused unread, gives its slot to a new connection when no slot is free,
    the longest waiting first, so stalled connections, however many and from
    however many addresses, cannot keep others out. When every slot has a
    request whose head has arrived, whether or not the waiting thread has
    looked at it yet, the new connection is refused. One that gives way
    awaiting its request gets the 503 of a connection refused at the accept,
    as its request may be on its way, and is closed as a refused one is,
    lingering, among at most `refused_capacity`, for up to `linger` seconds.
    One draining a refused upload has its response already: it is shut
    down, for its thread to see the end of its input.
    """

    # What a connection with a slot is doing.
    WAITING = "waiting for its request"
    SERVING = "serving"
    DRAINING = "draining"

    def __init__(self, capacity, serve, refused_capacity, linger):
        self.capacity = capacity
        # Each connection with a slot, mapped to what it is doing; those that
        # wait on their client in the order they began to.
        self.holders = {}
        self.lock = threading.Lock()
        self.idle = threading.Event()
        self.idle.set()
        self.workers = Workers(capacity, serve)
        self.waiting = WaitingConnections(
            self, self.workers.put, refused_capacity, linger
        )

    def admit(self, connection, address, deadline):
        """Give `connection` a slot, to wait for its request, whose head is
        due by `deadline`; False when every slot has a request whose head
        has arrived."""
        with self.lock:
            if len(self.holders) >= self.capacity:
                # One still waiting whose head has arrived, which the waiting
                # thread has yet to look at, keeps its slot.
                giving_way = next(
                    (
                        held
                        for held, state in self.holders.items()
                        if state == self.DRAINING
                        or (state == self.WAITING and not _ready(held))
                    ),
                    None,
                )
                if giving_way is None:
                    return False
                if self.holders.pop(giving_way) == self.WAITING:
                    self.refuse(giving_way)
                else:
                    # Under the lock, which a connection's release takes
                    # before the connection is closed, so the descriptor is
                    # still its own.
                    try:
                        giving_way.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
            self.holders[connection] = self.WAITING
            self.idle.clear()
        self.waiting.wait(connection, address, deadline)
        return True

    def refuse(self, connection):
        """Answer `connection`, which has no slot, with
        CONNECTION_OVERLOADED_RESPONSE, and close it without reading its
        request."""
        refuse_connection(connection)
        self.waiting.close(connection)

    def arrived(self, connection):
        """Keep the slot of `connection`, the head of whose request has
        arrived, until it is released; False once it has given its slot
        away."""
        with self.lock:
            if connection not in self.holders:
                return False
            self.holders[connection] = self.SERVING
            return True

    def expire(self, connection):
        """Free the slot of `connection`, whose request's head is due, for the
        caller to close it; False once it has given its slot away."""
        with self.lock:
            if self.holders.pop(connection, None) is None:
                return False
            if not self.holders:
                self.idle.set()
            return True

    def draining(self, connection):
        """`connection`, answered, drains an upload refused unread: it waits
        on its client again, last in line."""
        with self.lock:
            if self.holders.pop(connection, None) is not None:
                self.holders[connection] = self.DRAINING

    def release(self, connection):
        """Give back the slot of `connection`, if it still has one, before
        the connection is closed."""
        with self.lock:
            self.holders.pop(connection, None)
            if not self.holders:
                self.idle.set()

    def stop(self):
        """Close every connection held without a thread, and end each worker
        once the connections handed to it are served."""
        self.waiting.stop()
        self.workers.stop()

    def stats(self):
        with self.lock:
            return {"active": len(self.holders), "capacity": self.capacity}


class RequestBodyReservation:
    """Account input bytes until preparation and any retained input are released."""

    def __init__(self, admission, size):
        if not admission.acquire(size):
            raise APIError(
                503,
                "request body capacity is exhausted; retry shortly",
                "frontend_overloaded",
            )
        self.admission = admission
        self.size = size

    def release(self):
        self.admission.release(self.size)
        self.size = 0

    def grow(self, size):
        if not self.admission.acquire(size):
            raise APIError(
                503,
                "retained input capacity is exhausted; retry shortly",
                "frontend_overloaded",
            )
        self.size += size

    def retain_for(self, job):
        # Generation retains schemas and, for Responses, conversation history.
        # Text/image prompts have otherwise become tokens and prepared pixels.
        policy = job.tool_policy
        retained = (
            job.response_history_items,
            job.response_format,
            policy.schemas if policy else None,
            policy.namespaces if policy else None,
            job.stop_sequences,
        )
        retained = [value for value in retained if value]
        size = json_codec.encoded_size(retained) if retained else 0
        if size > self.size:
            self.grow(size - self.size)
        else:
            self.admission.release(self.size - size)
        self.size = size
        if size:
            weakref.finalize(job, self.release)
