from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Polygon, MultiPolygon
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .types import Modality, MODALITY_STOP_TAGS


def _preprocess_stop_name(name: str) -> set:
    """Normalize and tokenize a public transport stop name.

    The function takes a stop name as input, converts it to lowercase,
    splits it into separate words, and returns them as a set.
    Empty or missing values are handled gracefully by returning an empty set.

    Parameters
    ----------
    name : str
        The stop name string. Can be None or NaN.

    Returns
    -------
    set of str
        A set of unique, lowercase tokens extracted from the stop name.
        Returns an empty set if the input is empty or NaN.

    Examples
    --------
    >>> _preprocess_stop_name("Main Square Stop")
    {'main', 'square', 'stop'}

    >>> _preprocess_stop_name(None)
    set()

    >>> _preprocess_stop_name("улица Ленина")
    {'улица', 'ленина'}
    
    """
    if not name or pd.isna(name):
        return set()
    words = set(name.lower().split())

    return words


def _calculate_name_similarity(name1: str, name2: str) -> float:
    """Calculate the similarity between two public transport stop names.

    This function computes a normalized similarity score between two stop names
    based on the overlap of their tokenized word sets (Jaccard-like measure).  
    Stop names are first preprocessed using `_preprocess_stop_name`, which lowercases
    and splits them into unique words. The resulting similarity value ranges
    from 0.0 (no common words) to 1.0 (identical word sets).

    Missing or empty names are handled gracefully, returning 0.0 similarity.

    Parameters
    ----------
    name1 : str
        The first stop name string. Can be None or NaN.
    name2 : str
        The second stop name string. Can be None or NaN.

    Returns
    -------
    float
        A float value in the range [0.0, 1.0] representing the similarity
        between the two names based on shared words. Returns 0.0 if either
        name is missing or has no valid tokens.

    Examples
    --------
    >>> _calculate_name_similarity("Main Square Stop", "Main Square")
    0.67

    >>> _calculate_name_similarity("улица Ленина", "Ленина улица")
    1.0

    >>> _calculate_name_similarity("Central Station", "North Station")
    0.5

    >>> _calculate_name_similarity("Central Station", None)
    0.0
    """

    if pd.isna(name1) or pd.isna(name2):
        return 0.0
    words1 = _preprocess_stop_name(name1)
    words2 = _preprocess_stop_name(name2)
    
    if not words1 or not words2:
        return 0.0
        
    common_words = words1.intersection(words2)
    # Jaccard Words Similarity 
    similarity = len(common_words) / max(len(words1), len(words2))

    return similarity


def _calculate_median_distance(stops_gdf: gpd.GeoDataFrame) -> float:
    """Calculate the median spatial distance between stops with similar names.

    This function estimates a characteristic merging distance for public transport stops
    by analyzing all pairs of stops whose names are highly similar (Jaccard similarity ≥ 0.9).
    It computes pairwise geometric distances between such stops and returns the median
    of those distances, excluding pairs that are farther apart than the maximum threshold.

    The result can be used as an adaptive distance threshold when aggregating
    nearby stops with the same or similar names.

    Parameters
    ----------
    stops_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing stop geometries and their names.
        Must include a 'name' column and a valid geometry column in a projected CRS (meters).

    Returns
    -------
    float
        Median distance (in meters) between stops with similar names.
        Returns a default value of 100 if no valid pairs are found.

    Notes
    -----
    - Only pairs with name similarity ≥ 0.9 are considered.
    - Distances greater than 100 meters are ignored.
    - The function assumes geometries are projected in meters.

    Examples
    --------
    >>> _calculate_median_distance(stops_gdf)
    42.5
    """
    MAX_DISTANCE = 100  # maximum allowed distance in meters
    distances = []
    
    # Filter out stops with empty names
    valid_stops = stops_gdf[stops_gdf['name'].notna() & (stops_gdf['name'] != '')]
    
    # Compare all pairs of stops
    names = valid_stops['name'].values
    geoms = valid_stops.geometry.values
    
    for i in range(len(valid_stops)):
        for j in range(i + 1, len(valid_stops)):
            # Calculate name similarity
            similarity = _calculate_name_similarity(names[i], names[j])
            
            # If names are similar enough, calculate distance
            if similarity >= 0.9:
                dist = geoms[i].distance(geoms[j])
                # Only append if distance is within threshold
                if dist <= MAX_DISTANCE:
                    distances.append(dist)
    
    if not distances:  # If no valid pairs found
        return 100  # Default value
        
    return np.median(distances)


def _get_stops(polygon: Polygon | MultiPolygon, modality_tags: dict) -> gpd.GeoDataFrame:
    # Download public transport stops for given modality type from OpenStreetMap
    stops = ox.features_from_polygon(polygon, tags = modality_tags)
    stops_points = stops[stops.geom_type == "Point"].copy()
    stops_points.reset_index(inplace=True)
    local_crs = stops_points.estimate_utm_crs()
    stops_points.to_crs(local_crs, inplace=True)

    return stops_points


def aggregate_stops(stops_gdf: gpd.GeoDataFrame, distance_threshold: int = None) -> gpd.GeoDataFrame:
    """Aggregate nearby public transport stops based on spatial and name similarity.

    This function merges stops that are either spatially close or have similar names,
    producing a simplified set of representative stops (centroids).  
    The merging rules combine geometric proximity and semantic similarity of names,
    helping to unify duplicate or overlapping stop records from OSM or other datasets.

    If no `distance_threshold` is provided, it is automatically estimated using
    `_calculate_median_distance()` — the median distance between stops with similar names.

    Parameters
    ----------
    stops_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing stop geometries and their attributes.
        Must include a 'name' column and be in a projected CRS (meters).
    distance_threshold : int, optional
        Maximum spatial distance (in meters) for merging nearby stops.
        If None, computed automatically from data.

    Returns
    -------
    geopandas.GeoDataFrame
        A new GeoDataFrame of aggregated stops, where each row represents
        a merged stop group. The following columns are included:
        - `geometry`: centroid of merged stops.
        - `name`: representative name (first in group).
        - `group_name`: unique group identifier.
        - `original_stops`: number of original stops merged.
        - `original_ids`: list of original stop indices.

    Notes
    -----
    Stops are merged if **any** of the following conditions are true:
    1. Their buffered geometries (within `distance_threshold`) intersect.
    2. Their names are similar (`name_similarity ≥ 0.8`) and they are closer
       than `distance_threshold`, or are identical in name and within 300 m.
    3. Pairs farther than 300 m are never merged.

    Connected components in the adjacency matrix define merged stop groups.

    Examples
    --------
    >>> aggregated = aggregate_stops(stops_gdf)
    Automatically calculated distance_threshold: 45.00m
    >>> aggregated.head()
           name   group_name  original_stops geometry
    0  Central   Group_0     3              POINT(....)
    1  Lenina    Group_1     2              POINT(....)
    """
    if distance_threshold is None:
        # Calculate automatic threshold
        median_dist = _calculate_median_distance(stops_gdf)
        distance_threshold = int(median_dist)
        print(f"Automatically calculated distance_threshold: {distance_threshold:.2f}m")

    n_stops = len(stops_gdf)
    # Adjacency matrix initialization
    adjacency = np.zeros((n_stops, n_stops))
    # rows, cols = [], []
    stops_buffered = stops_gdf.copy()
    stops_buffered.geometry = stops_gdf.geometry.buffer(distance_threshold)

    # Fill adjacency matrix based on rules
    for i in range(n_stops):
        for j in range(i+1, n_stops):
            # Buffer intersection
            buffers_intersect = stops_buffered.iloc[i].geometry.intersects(
                stops_buffered.iloc[j].geometry
            )
            
            # Calculate name similarity
            name_similarity = _calculate_name_similarity(
                stops_gdf.iloc[i]['name'],
                stops_gdf.iloc[j]['name']
            )
            
            # Determine actual distance between points
            distance = stops_gdf.iloc[i].geometry.distance(stops_gdf.iloc[j].geometry)
            
            #  Merging rules:
            # 1. If buffers overlap
            # 2. Or if names are similar AND distance < 100m
            # 3. Do not merge if distance > 300m
            should_merge = (
                buffers_intersect or 
                (name_similarity >= 0.8 and (
                    distance <= distance_threshold or              # close stops points
                    (name_similarity == 1.0 and distance <= 300)  # similar names up to 300m
                ))
            )
            
            if should_merge:
                adjacency[i,j] = adjacency[j,i] = 1


    # Find connected components
    sparse_matrix = csr_matrix(adjacency)
    n_components, labels = connected_components(sparse_matrix, directed=False)


    groups = {}
    for i in range(n_components):
        groups[i] = stops_gdf.index[labels == i].tolist()
        

    aggregated_stops = []
    for group in groups.values():
        group_stops = stops_gdf.loc[group]
        
        # Calculate centroid as new stop location
        centroid = group_stops.dissolve().geometry.iloc[0].centroid
        
        # Get original names if they exist
        names = group_stops['name'].tolist() if 'name' in group_stops.columns else []
        
        # Create group name
        group_name = f'Group_{len(aggregated_stops)}'
        
        aggregated_stops.append({
            'geometry': centroid,
            'name': names[0] if names else None, 
            'group_name': group_name,
            'original_stops': len(group),
            'original_ids': group
        })
        result = gpd.GeoDataFrame(aggregated_stops, crs=stops_gdf.crs)
        
    return result


def get_agg_stops(polygon: Polygon | MultiPolygon, modalities: list[Modality] ) -> Dict[Modality, gpd.GeoDataFrame]:
    """Download and aggregate public transport stops by modality within a polygon.

    This function retrieves raw public transport stop points from OpenStreetMap for one or more
    transport modalities (e.g., bus, tram, trolleybus) and merges nearby or duplicate stops
    into representative aggregated stop locations.

    Parameters
    ----------
    polygon : shapely.Polygon or shapely.MultiPolygon
        The geographic boundary within which to download stop locations.
    modalities : list of Modality
        List of transport types to process (e.g., `[Modality.BUS, Modality.TRAM]`).

    Returns
    -------
    dict of {Modality: geopandas.GeoDataFrame}
        A dictionary mapping each transport modality to its aggregated stop dataset.
        Each GeoDataFrame includes:
        - `geometry`: centroid of aggregated stops,
        - `name`: representative stop name,
        - `group_name`: unique group identifier,
        - `original_stops`: number of original stops merged,
        - `original_ids`: list of merged stop indices,
        - `modality`: transport mode (e.g., `"bus"`, `"tram"`).

    Notes
    -----
    - Downloads raw stop data using `_get_stops()` and the modality-specific OSM tags
      defined in `MODALITY_STOP_TAGS`.
    - Applies `aggregate_stops()` to merge stops that are spatially close or have
      similar names, reducing redundancy in the dataset.
    - Skips modalities for which no stop points are found.

    Examples
    --------
    >>> stops_by_mode = get_agg_stops(city_boundary, [Modality.BUS, Modality.TRAM])
    >>> stops_by_mode[Modality.BUS].head()
           name   group_name  original_stops  modality                   geometry
    0  Central   Group_0     3               bus       POINT (445122.1 6139458.2)
    1  Lenina    Group_1     2               bus       POINT (445315.7 6139430.1)
    """
    result: Dict[Modality, gpd.GeoDataFrame] = {}

    for modality in modalities:
        modality_tag = MODALITY_STOP_TAGS[modality]
        if modality_tag is None:
            raise ValueError(f"For transport type {modality} no available Tags.")

        stops_points = _get_stops(polygon, modality_tag)
        if stops_points.empty:
            continue

        agg = aggregate_stops(stops_points)
        agg["modality"] = modality.value  
        result[modality] = agg.reset_index(drop=True)

    return result