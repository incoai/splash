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
        """Bind `listener` at the path, in place of a socket there that no
        Splash server is behind. With the lock held none is, one that still
        loads its model included, so what a connection to the socket meets
        decides. Refused, nothing listens on it: it is a leftover, as a
        killed server's. Not permitted (EPERM), the server's sandbox gives it
        no outbound connection, as a profile that lets it only bind and
        accept does, and says so of every socket, a leftover among them,
        which no start could then replace: it goes too. Anything else there
        stays, and the bind fails as on an address in use: a socket that
        accepts the connection is another program's listener, one whose
        permissions refuse this user (EACCES) is not this user's to remove,
        and a file of another kind (ENOTSOCK) is no socket."""
        with socket.socket(socket.AF_UNIX) as probe:
            if probe.connect_ex(self.path) in (errno.ECONNREFUSED, errno.EPERM):
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
