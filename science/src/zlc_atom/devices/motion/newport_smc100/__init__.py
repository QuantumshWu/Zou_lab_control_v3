"""A Newport SMC100CC/PP controller and its stage as one motion axis."""

from .axis import Smc100Axis, Smc100Config, discover_smc100
from .protocol import (
    BAUD,
    COMMAND_ERRORS,
    STATES,
    ControllerStatus,
    Smc100CommandError,
    Smc100ConnectionError,
    Smc100Error,
    Smc100Link,
    Smc100NoReply,
)

__all__ = [
    "BAUD",
    "COMMAND_ERRORS",
    "ControllerStatus",
    "STATES",
    "Smc100Axis",
    "Smc100CommandError",
    "Smc100ConnectionError",
    "Smc100Config",
    "Smc100Error",
    "Smc100Link",
    "Smc100NoReply",
    "discover_smc100",
]
