"""
Preprocessing package for public transport network generation.

This package provides modular functions for:
    - downloading and preprocessing transport lines (bus, tram, trolleybus),
    - aggregating and projecting stops,
    - constructing and simplifying transport graphs,
    - computing stop-to-stop time and OD matrices,
    - executing the full preprocessing pipeline with a single entry point.

Typical usage:
---------------
>>> from preprocess import preprocess_data, Modality
>>> result = preprocess_data(blocks, [Modality.BUS, Modality.TRAM])
>>> stops_gdf, time_matrix, graph = result[Modality.BUS]
"""

import importlib

from .types import Modality
from .preprocess_data import preprocess
from .od import get_OD