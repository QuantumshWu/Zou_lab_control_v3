"""The WitMotion HWT3100-485 magnetometer as a waveform source."""

from .modbus import (
    BAUD_CODES,
    SUPPORTED_BAUDS,
    ModbusError,
    ModbusTimeout,
    crc16,
    decode_field,
    establish_baud,
    max_poll_rate_hz,
)
from .source import (
    DEFAULT_BAUD,
    DEFAULT_RATE_HZ,
    HWT3100_OUTPUTS,
    Hwt3100Config,
    Hwt3100WaveformSource,
    discover_hwt3100,
)

__all__ = [
    "BAUD_CODES",
    "DEFAULT_BAUD",
    "DEFAULT_RATE_HZ",
    "HWT3100_OUTPUTS",
    "Hwt3100Config",
    "Hwt3100WaveformSource",
    "ModbusError",
    "ModbusTimeout",
    "SUPPORTED_BAUDS",
    "crc16",
    "decode_field",
    "discover_hwt3100",
    "establish_baud",
    "max_poll_rate_hz",
]
