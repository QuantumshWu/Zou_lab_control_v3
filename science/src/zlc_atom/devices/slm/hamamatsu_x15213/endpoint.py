"""The socket chores of the SLM server: its bind rule, telling this machine's
clients from a peer's, listing the addresses a peer can reach, and dropping
a connection. The pulse server's equivalents live in the native core.
"""

from __future__ import annotations

from collections.abc import Iterable
import errno
import ipaddress
import os
import socket


def bind_exclusive(sock: socket.socket, address: tuple[str, int]) -> None:
    """Bind a server's socket as the only one on its port, or say why not.

    The rule every listening socket here binds by.  A second listener on a
    held port must be refused at its bind -- on the wildcard or on the same
    address.  Two different specific addresses, 127.0.0.1 and 127.0.0.2,
    may still share a port: each client reaches the one it dialled, and
    every bench server binds the wildcard.
    Windows refuses a plain bind only on the very same address: a listener
    on 127.0.0.1 binds beside one on 0.0.0.0 and, as the more specific one,
    takes every later loopback connection, so a second bench's server takes
    the first bench's own clients and its board.  SO_EXCLUSIVEADDRUSE
    refuses that bind in either order and still rebinds a port whose server
    has closed; SO_REUSEADDR would instead let a second socket bind a port a
    live listener holds.  POSIX keeps SO_REUSEADDR for what it means there,
    rebinding past TIME_WAIT -- on a stream socket only: on a datagram one it
    lets two bind the same port.  Linux still refuses a specific address
    beside a wildcard listener with it; BSD and macOS let that bind through,
    so the overlap above is refused on Windows and Linux only.  The refusal
    names the port, not the socket layer's words for it (Windows answers
    "forbidden by its access permissions" when a wildcard listener holds
    the port).
    """

    if os.name == "nt":
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    elif sock.type == socket.SOCK_STREAM:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(address)
    except OSError as error:
        if error.errno not in (errno.EADDRINUSE, errno.EACCES):
            raise
        raise OSError(
            error.errno,
            f"port {address[1]} is taken: another server on this machine already "
            f"listens on it, or the system reserves it ({error.strerror})",
        ) from error


def is_loopback_host(host: str) -> bool:
    """Whether ``host`` is THIS machine talking to itself.

    A server that listens on every interface sees its own machine's
    clients arrive from 127.0.0.0/8 or ``::1`` (or the IPv4-mapped form of
    either), and a peer's from that peer's address; this is the one test
    both the pulse and the SLM server apply to tell them apart.  Text that
    is not an address is not loopback.
    """

    try:
        address = ipaddress.ip_address(str(host).strip())
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return bool((mapped or address).is_loopback)


def local_ipv4_addresses() -> tuple[str, ...]:
    """Discover non-loopback IPv4 addresses without requiring a network request."""

    addresses: list[str] = []
    packed_addresses: set[bytes] = set()

    def add(value: object) -> None:
        try:
            address = str(value).strip()
            packed = socket.inet_aton(address)
        except (OSError, ValueError):
            return
        if address == "0.0.0.0" or address.startswith("127.") or packed in packed_addresses:
            return
        packed_addresses.add(packed)
        addresses.append(address)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            add(sock.getsockname()[0])
    except OSError:
        pass

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM):
            add(info[4][0])
    except OSError:
        pass

    try:
        for address in socket.gethostbyname_ex(socket.gethostname())[2]:
            add(address)
    except OSError:
        pass

    return tuple(addresses)


def drop_connection(connection: socket.socket) -> None:
    """End a connection nobody wants any more, from this side.

    Windows can leave a blocking recv waiting after shutdown alone. Close
    the revoked socket here instead of waiting for its peer to speak again.
    """

    try:
        connection.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    finally:
        connection.close()


def drop_peer_connections(connections: Iterable[socket.socket]) -> int:
    """Drop every connection from another machine; return how many went.

    What a server does when it stops being on offer: this machine's own
    clients stay connected.
    """

    dropped = 0
    for connection in connections:
        try:
            peer = connection.getpeername()[0]
        except OSError:
            continue
        if not is_loopback_host(peer):
            drop_connection(connection)
            dropped += 1
    return dropped


__all__ = [
    "bind_exclusive",
    "drop_connection",
    "drop_peer_connections",
    "is_loopback_host",
    "local_ipv4_addresses",
]
