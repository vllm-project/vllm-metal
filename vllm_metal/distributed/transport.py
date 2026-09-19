# SPDX-License-Identifier: Apache-2.0
"""Explicit pipeline transport selection and strict JACCL ring bootstrap.

Device-matrix rows follow pipeline rank order. Lane order is significant:
lane i on one Mac connects to lane i in the reciprocal matrix entry.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlx.core.distributed import Group

DeviceEntry = str | tuple[str, ...] | None
DeviceMatrix = tuple[tuple[DeviceEntry, ...], ...]
logger = logging.getLogger(__name__)


def _lanes(entry: DeviceEntry) -> tuple[str, ...]:
    if entry is None:
        return ()
    return (entry,) if isinstance(entry, str) else entry


def _device_matrix(value: Any, world_size: int) -> DeviceMatrix:
    label = "pipeline_transport.device_matrix"
    if (
        not isinstance(value, list)
        or len(value) != world_size
        or any(not isinstance(row, list) or len(row) != world_size for row in value)
    ):
        raise ValueError(f"{label} must be a square matrix matching world_size")

    rows: list[tuple[DeviceEntry, ...]] = []
    for rank, row in enumerate(value):
        if row[rank] is not None:
            raise ValueError(f"{label} diagonal entries must be null")
        entries: list[DeviceEntry] = []
        devices: set[str] = set()
        for entry in row:
            if entry is not None and not isinstance(entry, (str, list)):
                raise ValueError(
                    f"{label} entries must be null, a string or string list"
                )
            frozen = tuple(entry) if isinstance(entry, list) else entry
            names = _lanes(frozen)
            if entry is not None and not 1 <= len(names) <= 4:
                raise ValueError(f"{label} supports 1..4 lanes per connection")
            for name in names:
                if (
                    not isinstance(name, str)
                    or not name
                    or any(
                        character.isspace()
                        or ord(character) < 32
                        or ord(character) == 127
                        for character in name
                    )
                ):
                    raise ValueError(
                        f"{label} requires nonempty device names without whitespace"
                    )
                if name in devices:
                    raise ValueError(f"{label} rank {rank} repeats device {name!r}")
                devices.add(name)
            entries.append(frozen)
        rows.append(tuple(entries))

    matrix = tuple(rows)
    width = len(_lanes(matrix[0][1]))
    for rank, row in enumerate(matrix):
        # In a two-rank ring the left and right neighbor are the same peer.
        neighbors = {(rank - 1) % world_size, (rank + 1) % world_size}
        for peer, entry in enumerate(row):
            count = len(_lanes(entry))
            if count != len(_lanes(matrix[peer][rank])):
                raise ValueError(
                    f"{label} reciprocal connections need equal lane counts"
                )
            if peer in neighbors and (count == 0 or count != width):
                raise ValueError(
                    f"{label} ring edges need the same positive lane count"
                )
    return matrix


def _validate_peers(rank: int, peer_ips: list[str], world_size: int) -> None:
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("JACCL pipeline rank must be within world_size")
    if not isinstance(peer_ips, list) or len(peer_ips) != world_size:
        raise ValueError("JACCL peer addresses must match pipeline world_size")
    addresses = []
    for ip in peer_ips:
        try:
            # Native JACCL's coordinator uses AF_INET and splits ip:port at the
            # first colon, so IPv6/hostnames must not reach native bootstrap.
            address = ipaddress.IPv4Address(ip) if isinstance(ip, str) else None
        except ipaddress.AddressValueError:
            address = None
        if address is None or (
            address.is_unspecified
            or address.is_loopback
            or address.is_multicast
            or address.is_reserved
        ):
            raise ValueError("JACCL peers require usable, non-loopback IPv4 addresses")
        addresses.append(address)
    if len(set(addresses)) != world_size:
        raise ValueError("JACCL pipeline requires distinct peer IPs, one stage per Mac")


@dataclass(frozen=True)
class PipelineTransportConfig:
    """Validated transport options; importing this module does not import MLX."""

    backend: str = "ring"
    device_matrix: DeviceMatrix | None = None
    coordinator_port: int = 59451
    world_size: int = 1

    @classmethod
    def from_additional_config(
        cls, additional_config: dict[str, Any] | None, world_size: int
    ) -> PipelineTransportConfig:
        if type(world_size) is not int or world_size < 1:
            raise ValueError("pipeline_transport world_size must be a positive integer")
        if additional_config is None:
            additional_config = {}
        if not isinstance(additional_config, dict):
            raise ValueError("pipeline_transport requires an additional_config object")
        options = additional_config.get("pipeline_transport", {})
        if not isinstance(options, dict):
            raise ValueError("pipeline_transport must be an object")
        if set(options) - {"backend", "device_matrix", "coordinator_port"}:
            raise ValueError("pipeline_transport contains unknown options")
        backend = options.get("backend", "ring")
        if backend not in ("ring", "jaccl"):
            raise ValueError("pipeline_transport.backend must be 'ring' or 'jaccl'")
        if backend == "ring":
            if "device_matrix" in options or "coordinator_port" in options:
                raise ValueError(
                    "pipeline_transport device_matrix/coordinator_port require jaccl"
                )
            return cls(world_size=world_size)
        if world_size < 2:
            raise ValueError("pipeline_transport jaccl requires at least two stages")
        port = options.get("coordinator_port", 59451)
        if type(port) is not int or not 1024 <= port <= 65535:
            raise ValueError(
                "pipeline_transport.coordinator_port must be an integer in 1024..65535"
            )
        matrix = _device_matrix(options.get("device_matrix"), world_size)
        return cls(backend, matrix, port, world_size)

    def bootstrap_jaccl(self, rank: int, peer_ips: list[str]) -> Group:
        """Initialize RDMA with strict JACCL selection; never fall back to TCP."""
        if self.backend != "jaccl":
            raise ValueError(
                "bootstrap_jaccl requires the jaccl transport configuration"
            )
        _validate_peers(rank, peer_ips, self.world_size)
        coordinator = f"{peer_ips[0]}:{self.coordinator_port}"
        names = (
            "JACCL_RANK",
            "MLX_RANK",
            "JACCL_IBV_DEVICES",
            "MLX_IBV_DEVICES",
            "JACCL_COORDINATOR",
            "MLX_JACCL_COORDINATOR",
            "JACCL_RING",
            "MLX_JACCL_RING",
            "MLX_HOSTFILE",
        )
        previous = {name: os.environ.get(name) for name in names}
        fd, path = tempfile.mkstemp(prefix=f"mlx_jaccl_rank{rank}_", suffix=".json")
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(self.device_matrix, stream)
            os.environ.update(
                {
                    "JACCL_RANK": str(rank),
                    "MLX_RANK": str(rank),
                    "JACCL_IBV_DEVICES": path,
                    "MLX_IBV_DEVICES": path,
                    "JACCL_COORDINATOR": coordinator,
                    "MLX_JACCL_COORDINATOR": coordinator,
                    "JACCL_RING": "1",
                    "MLX_JACCL_RING": "1",
                }
            )
            os.environ.pop("MLX_HOSTFILE", None)
            import mlx.core as mx

            logger.info(
                "MLX JACCL bootstrap: rank=%d/%d coordinator=%s devices=%s",
                rank,
                self.world_size,
                coordinator,
                self.device_matrix[rank],
            )
            group = mx.distributed.init(backend="jaccl", strict=True)
            if group.rank() != rank or group.size() != self.world_size:
                raise RuntimeError(
                    "JACCL formed the wrong pipeline group: "
                    f"rank={group.rank()}, size={group.size()}; "
                    f"expected rank={rank}, size={self.world_size}"
                )
            return group
        finally:
            try:
                Path(path).unlink(missing_ok=True)
            finally:
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
