"""``python -m zlc_plugin host``: the plugin host process."""

import sys

from zlc_plugin import host

if __name__ == "__main__":
    if sys.argv[1:] != ["host"]:
        sys.stderr.write("usage: python -m zlc_plugin host\n")
        sys.exit(2)
    sys.exit(host.main())
