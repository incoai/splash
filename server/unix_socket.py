"""The UNIX socket a server listens on in place of a TCP port: how --host
names it, and how its path stays one server's.

A TCP port is one listener's because the kernel refuses its second bind. A
socket's path is not: bind refuses any path that exists, a socket a killed
server left behind as much as a live one, and a connection cannot tell them
apart either, since a socket that is bound but does not listen yet, as a
server's is while its model loads, refuses one as a leftover does. So a lock
beside the socket, PATH.lock, says whose the path is: a server holds it from
before it binds until it has removed its socket, and the kernel gives it back
when a server dies without removing one.

Standard library only: the launcher claims the path before .venv exists.
"""

import contextlib
import errno
import fcntl
import os
import socket

# How --host names a socket: unix:PATH.
PREFIX = "unix:"


def path(host):
    """The socket path of a --host of unix:PATH, else None."""
    if host.startswith(PREFIX):
        return host[len(PREFIX) :]
    return None


class Claim:
    """`path` held for one listener, by its lock: OSError, as of an address in
    use, where another holds it. The lock file stays when the claim closes:
    one removed could be locked by a server that opened it before, beside a
    new one of the same name."""

    def __init__(self, path):
        self.path = path
        self._bound = False
        self._lock = open(f"{path}.lock", "a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise OSError(errno.EADDRINUSE, os.strerror(errno.EADDRINUSE)) from None

    def bind(self, listener):
        """Bind `listener` at the path, in place of a socket there that
        refuses a connection: with the lock held it is no server's that still
        loads, so it is a leftover. Anything else there stays, a socket that
        answers or cannot be asked and a file of another kind, and the bind
        fails as on an address in use."""
        with socket.socket(socket.AF_UNIX) as probe:
            if probe.connect_ex(self.path) == errno.ECONNREFUSED:
                os.unlink(self.path)
        listener.bind(self.path)
        self._bound = True

    def close(self):
        """Remove the socket this claim bound, which with the lock held is
        still the one at the path, then give the path up."""
        if self._bound:
            self._bound = False
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self.path)
        self._lock.close()
