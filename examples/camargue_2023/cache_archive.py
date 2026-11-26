"""Restore archived PlantGL precomputations before a simulation."""

from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory


def restore_cache(cache: Path) -> None:
    """Restore a missing cache from a Camargue release asset, if available."""
    if cache.is_dir():
        return

    name = f"camargue-2023-{cache.name.lstrip('.')}.tar.zst"
    # Support assets stored with the case and archives made from the repository root.
    candidates = (cache.parent / name, cache.parent.parent.parent / name)
    archive = next((path for path in candidates if path.is_file()), None)
    if archive is None:
        return

    zstd = shutil.which("zstd")
    if zstd is None:
        raise RuntimeError(f"Install zstd to restore the PlantGL cache from {archive}.")

    with TemporaryDirectory(prefix=".pglcache-restore-", dir=cache.parent) as temporary:
        staged = Path(temporary) / cache.name
        staged.mkdir()
        with subprocess.Popen([zstd, "-dc", "--", str(archive)], stdout=subprocess.PIPE) as process:
            with process.stdout as stream, tarfile.open(fileobj=stream, mode="r|") as contents:
                for member in contents:
                    path = PurePosixPath(member.name)
                    if member.isdir() and path.parts == (cache.name,):
                        continue
                    if (not member.isfile() or path.parts != (cache.name, path.name)
                            or not re.fullmatch(r"-?[0-9]+\.pkl", path.name)):
                        raise ValueError(f"Unexpected entry in PlantGL cache archive: {member.name}")
                    # Only write validated pickle filenames into the staging directory.
                    with contents.extractfile(member) as source, (staged / path.name).open("xb") as target:
                        shutil.copyfileobj(source, target)
                # Read through the end of the compressed stream so zstd can check it.
                while stream.read(65536):
                    pass
            if process.wait():
                raise RuntimeError(f"Unable to decompress PlantGL cache archive {archive}.")
        if not any(staged.iterdir()):
            raise ValueError(f"PlantGL cache archive is empty: {archive}")
        try:
            staged.rename(cache)
        except OSError:
            if not cache.is_dir():
                raise
        else:
            print(f"Restored PlantGL cache from {archive}")
