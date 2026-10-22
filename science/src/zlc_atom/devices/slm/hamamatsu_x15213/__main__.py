"""Serve this machine's X15213 to local or LAN clients, or warm its GPU code.

    python -m zlc_atom.devices.slm.hamamatsu_x15213 [--port N --transport dvi|usb ...]
    python -m zlc_atom.devices.slm.hamamatsu_x15213 warm-gpu
"""

import sys

from .device_types import main, warm_gpu

if __name__ == "__main__":
    if sys.argv[1:2] == ["warm-gpu"]:
        sys.exit(warm_gpu())
    sys.exit(main(sys.argv[1:]))
