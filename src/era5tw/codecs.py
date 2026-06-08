"""Map config codec specs -> numcodecs compressor instances.

Spec examples (from config.yaml):
    {id: none}
    {id: blosc, cname: zstd, clevel: 5, shuffle: shuffle}
    {id: zstd,  level: 5}
    {id: gzip,  level: 5}
"""
from __future__ import annotations

from typing import Any

_SHUFFLE = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}


def get_compressor(spec: dict[str, Any]):
    """Return a numcodecs codec instance, or None for uncompressed storage."""
    cid = str(spec.get("id", "none")).lower()
    if cid in ("none", "null", "raw"):
        return None

    import numcodecs

    if cid == "blosc":
        shuffle = spec.get("shuffle", "shuffle")
        shuffle = _SHUFFLE.get(str(shuffle).lower(), 1) if isinstance(shuffle, str) else int(shuffle)
        return numcodecs.Blosc(
            cname=spec.get("cname", "zstd"),
            clevel=int(spec.get("clevel", 5)),
            shuffle=shuffle,
        )
    if cid == "zstd":
        return numcodecs.Zstd(level=int(spec.get("level", 3)))
    if cid == "gzip":
        return numcodecs.GZip(level=int(spec.get("level", 5)))
    if cid == "lz4":
        return numcodecs.LZ4(acceleration=int(spec.get("acceleration", 1)))
    if cid == "zlib":
        return numcodecs.Zlib(level=int(spec.get("level", 5)))

    raise ValueError(f"Unknown codec id: {cid!r}")


def codec_label(spec: dict[str, Any]) -> str:
    cid = str(spec.get("id", "none")).lower()
    if cid == "blosc":
        return f"blosc-{spec.get('cname','zstd')}-l{spec.get('clevel',5)}-{spec.get('shuffle','shuffle')}"
    if cid in ("zstd", "gzip", "zlib"):
        return f"{cid}-l{spec.get('level', 5)}"
    return cid
