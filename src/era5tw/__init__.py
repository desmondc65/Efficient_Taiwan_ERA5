"""Efficient Taiwan ERA5 -> Zarr toolkit.

A small library backing the scripts/ that download regional ERA5, convert it to
chunked+compressed Zarr, and benchmark chunking/codec/dataloader/DDP performance.
"""
__version__ = "0.1.0"
