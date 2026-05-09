"""On-demand fetcher for MIPLIB benchmark instances.

MIPLIB hosts publicly downloadable instances at miplib.zib.de. The
:func:`fetch_miplib` helper downloads a named instance, decompresses
``.mps.gz`` into a local ``.mps``, and caches the result under
``~/.cache/qubots/miplib/`` (or ``$QUBOTS_MIPLIB_CACHE``).

Network access is required only on first fetch. Subsequent calls hit the
cache.
"""

from __future__ import annotations

import gzip
import os
import shutil
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


_DEFAULT_BASE_URL = "https://miplib.zib.de/WebData/instances"


def _ssl_context() -> ssl.SSLContext:
    """Build an SSL context using certifi's bundle if available.

    Python on Windows doesn't read the OS certificate store, so the system
    default context often fails on legitimate HTTPS endpoints. Using
    certifi (when installed) sidesteps this. If certifi isn't available we
    fall back to the system default, which works on most Linux/macOS
    setups.
    """
    try:
        import certifi  # type: ignore[import-untyped]

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _cache_root() -> Path:
    custom = os.environ.get("QUBOTS_MIPLIB_CACHE")
    if custom:
        return Path(custom).expanduser().resolve()
    return Path.home() / ".cache" / "qubots" / "miplib"


def fetch_miplib(
    name: str,
    *,
    cache_dir: str | Path | None = None,
    base_url: str | None = None,
    force: bool = False,
) -> Path:
    """Fetch a named MIPLIB instance and return the path to a local ``.mps`` file.

    ``name`` is the instance basename without extension (e.g. ``"gen-ip002"``).
    Each fetch downloads ``<base_url>/<name>.mps.gz`` and decompresses it.
    Already-cached instances are reused unless ``force=True``.
    """
    if "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"Invalid MIPLIB instance name: {name!r}")

    root = Path(cache_dir).expanduser().resolve() if cache_dir is not None else _cache_root()
    root.mkdir(parents=True, exist_ok=True)

    target_mps = root / f"{name}.mps"
    if target_mps.exists() and not force:
        return target_mps

    url = f"{(base_url or _DEFAULT_BASE_URL).rstrip('/')}/{name}.mps.gz"
    gz_path = root / f"{name}.mps.gz"

    context = _ssl_context()
    try:
        with urllib.request.urlopen(url, timeout=60, context=context) as resp:
            with gz_path.open("wb") as out:
                shutil.copyfileobj(resp, out)
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if "CERTIFICATE_VERIFY_FAILED" in str(reason):
            raise RuntimeError(
                f"Failed to fetch {url}: TLS certificate verification failed. "
                "On Windows, run: pip install certifi  (qubots will pick it up "
                "automatically). Or set SSL_CERT_FILE to a valid CA bundle."
            ) from exc
        raise RuntimeError(f"Failed to fetch {url}: {reason}") from exc

    with gzip.open(gz_path, "rb") as fin, target_mps.open("wb") as fout:
        shutil.copyfileobj(fin, fout)

    try:
        gz_path.unlink()
    except OSError:
        pass

    return target_mps


def is_miplib_cached(name: str, cache_dir: str | Path | None = None) -> bool:
    root = Path(cache_dir).expanduser().resolve() if cache_dir is not None else _cache_root()
    return (root / f"{name}.mps").exists()
