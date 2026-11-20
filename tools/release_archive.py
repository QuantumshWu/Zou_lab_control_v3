"""The engine release archive (tools/release.ps1): every file under the named folders of bin as
<folder>/<path>, in one ustar tar compressed as one xz stream -- the x86 branch converter, then
LZMA2 at preset 9e, a CRC64 check. The same files give the same bytes: names in sorted order, no
owner, mode 644, each file's own time. tools/tar_xz.cs unpacks it on the PC that installs it.

Arguments: the archive to write, the bin folder, the folder names.
"""
import lzma
import os
import sys
import tarfile

FILTERS = [{"id": lzma.FILTER_X86}, {"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]


def main(archive, bin_folder, *folders):
    partial = archive + ".part"
    with lzma.open(partial, "wb", format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, filters=FILTERS) as stream, \
            tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for folder in folders:
            top = os.path.join(bin_folder, folder)
            names = sorted(os.path.relpath(os.path.join(at, name), top).replace(os.sep, "/")
                           for at, _, files in os.walk(top) for name in files)
            for name in names:
                path = os.path.join(top, name)
                info = tarfile.TarInfo(f"{folder}/{name}")
                stat = os.stat(path)
                info.size, info.mtime, info.mode = stat.st_size, int(stat.st_mtime), 0o644
                with open(path, "rb") as file:
                    tar.addfile(info, file)
    os.replace(partial, archive)


if __name__ == "__main__":
    main(*sys.argv[1:])
