"""No network in the engine worker. The Hugging Face variables (HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE) only switch off
the Hugging Face libraries; ACE-Step 1.5 also downloads from ModelScope, and probes www.google.com with a raw socket
to choose between the two (acestep/model_downloader.py `_can_access_google`, `_smart_download`). So the worker
installs this guard before any engine is imported: every socket connect and name lookup to anything but the loopback
interface (or a Unix socket) fails with NetworkBlocked, whichever library tries it. Stdlib only."""
from __future__ import annotations

import ipaddress
import socket
from typing import Any, Callable

LOCAL_NAMES = ("localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback")


class NetworkBlocked(ConnectionRefusedError):
    """Raised instead of reaching the network from the engine worker."""


def is_local_host(host: Any) -> bool:
    if host is None or host == "":
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    host = str(host).strip("[]").split("%", 1)[0]
    if host.lower() in LOCAL_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _address_is_local(sock: socket.socket, address: Any) -> bool:
    if getattr(socket, "AF_UNIX", None) is not None and sock.family == socket.AF_UNIX:
        return True
    host = address[0] if isinstance(address, tuple) and address else address
    return is_local_host(host)


def install() -> Callable[[], None]:
    """Block non-loopback connects and lookups in this process. Returns a function that undoes it (for tests)."""
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    real_getaddrinfo, real_gethostbyname = socket.getaddrinfo, socket.gethostbyname

    def connect(self: socket.socket, address: Any) -> Any:
        if not _address_is_local(self, address):
            raise NetworkBlocked(f"sound-design-skill: network access refused in the engine worker ({address!r})")
        return real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> Any:
        if not _address_is_local(self, address):
            raise NetworkBlocked(f"sound-design-skill: network access refused in the engine worker ({address!r})")
        return real_connect_ex(self, address)

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if not is_local_host(host):
            raise NetworkBlocked(f"sound-design-skill: name lookup refused in the engine worker ({host!r})")
        return real_getaddrinfo(host, *args, **kwargs)

    def gethostbyname(host: Any) -> Any:
        if not is_local_host(host):
            raise NetworkBlocked(f"sound-design-skill: name lookup refused in the engine worker ({host!r})")
        return real_gethostbyname(host)

    socket.socket.connect = connect            # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex      # type: ignore[method-assign]
    socket.getaddrinfo = getaddrinfo           # type: ignore[assignment]
    socket.gethostbyname = gethostbyname       # type: ignore[assignment]

    def undo() -> None:
        socket.socket.connect = real_connect          # type: ignore[method-assign]
        socket.socket.connect_ex = real_connect_ex    # type: ignore[method-assign]
        socket.getaddrinfo = real_getaddrinfo         # type: ignore[assignment]
        socket.gethostbyname = real_gethostbyname     # type: ignore[assignment]
    return undo
