"""Resolve config chunking strategies into concrete chunk tuples.

A strategy is given as {time: T, lat: Y, lon: X} where any value of -1 means
"use the full size of that dimension". Values are clamped to the array size.
"""
from __future__ import annotations

from typing import Any


def resolve_chunks(strategy: dict[str, Any], sizes: dict[str, int]) -> dict[str, int]:
    """Return {dim: chunk_size} clamped to `sizes` (the dataset dim lengths)."""
    out: dict[str, int] = {}
    for dim, size in sizes.items():
        want = strategy.get(dim, -1)
        if want is None or int(want) <= 0:      # -1 / 0 / missing -> full dim
            out[dim] = int(size)
        else:
            out[dim] = min(int(want), int(size))
    return out


def chunk_tuple(strategy: dict[str, Any], dims: tuple[str, ...], sizes: dict[str, int]) -> tuple[int, ...]:
    resolved = resolve_chunks(strategy, sizes)
    return tuple(resolved[d] for d in dims)


def n_chunks(chunks: dict[str, int], sizes: dict[str, int]) -> int:
    import math
    total = 1
    for d, s in sizes.items():
        total *= math.ceil(s / chunks[d])
    return total
