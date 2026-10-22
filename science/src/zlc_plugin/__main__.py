"""``python -m zlc_plugin host`` (one plugin host process) or ``describe <package>...``."""

import sys

from zlc_plugin import describe, host

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "host":
        sys.exit(host.main())
    if command == "describe":
        sys.exit(describe.main(sys.argv[2:]))
    sys.stderr.write("usage: python -m zlc_plugin host | describe <package>...\n")
    sys.exit(2)
