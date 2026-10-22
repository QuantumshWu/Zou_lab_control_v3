"""The test device family's SPI, as a capability Protocol."""

from typing import Protocol, runtime_checkable

import numpy as np

import zlc_plugin.capabilities

zlc_plugin.capabilities.CAPABILITIES.setdefault("test.echo", "zlc_test_plugins.devices.test.contract:Echo")


@runtime_checkable
class Echo(Protocol):
    @property
    def opened_with(self) -> dict: ...

    def frame(self, size: int, value: int) -> np.ndarray: ...

    def total(self, array: np.ndarray) -> int: ...

    def fire(self) -> None: ...
