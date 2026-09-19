# SPDX-License-Identifier: Apache-2.0
"""Bind worker-local Gloo groups without depending on macOS hostname resolution."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from ipaddress import IPv4Address
from typing import Any

import torch.distributed.distributed_c10d as c10d

_UNSET = object()


@contextmanager
def bind_gloo_to_ipv4(ip: str) -> Iterator[None]:
    """Bind all groups created during synchronous worker initialization to ``ip``.

    PyTorch's built-in Gloo path currently ignores ``pg_options``. Temporarily
    specialize its native constructor instead, covering vLLM's world group and
    hardcoded Gloo subgroups. A subclass preserves c10d's native type checks.
    The original constructor is restored even if initialization fails.
    """
    try:
        address = IPv4Address(ip) if isinstance(ip, str) else None
    except ValueError:
        address = None
    if address is None or address.is_unspecified or address.is_multicast:
        raise ValueError("Gloo binding requires an explicit unicast IPv4 address")

    original = c10d.ProcessGroupGloo

    class BoundGloo(original):  # type: ignore[valid-type, misc]
        def __init__(
            self,
            store: Any,
            rank: int,
            size: int,
            options: Any = _UNSET,
            *,
            timeout: Any = _UNSET,
        ) -> None:
            if options is not _UNSET and timeout is not _UNSET:
                raise TypeError("Gloo accepts either options or timeout, not both")
            supplied = timeout if timeout is not _UNSET else options
            bound = original._Options()
            if isinstance(supplied, original._Options):
                bound._timeout = supplied._timeout
                bound._threads = supplied._threads
                bound.global_ranks_in_group = list(supplied.global_ranks_in_group)
                bound.group_name = supplied.group_name
            elif supplied is not _UNSET:
                if not isinstance(supplied, timedelta):
                    raise TypeError("Gloo expects a timedelta timeout or Gloo options")
                bound._timeout = supplied
            bound._devices = [original.create_device(hostname=str(address))]
            super().__init__(store, rank, size, bound)

    c10d.ProcessGroupGloo = BoundGloo
    try:
        yield
    finally:
        c10d.ProcessGroupGloo = original
