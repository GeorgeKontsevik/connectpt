from typing import Union
import networkx
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx  # for type hints
from sklearn.preprocessing import MinMaxScaler

from iduedu import get_adj_matrix_gdf_to_gdf


def get_OD(blocks: gpd.GeoDataFrame, stops: gpd.GeoDataFrame, walk_graph: networkx.Graph, crs: int):
    """Compute an origin-destination (OD) matrix between transport stops using a gravity-based model.

    This function estimates potential travel demand between stops based on the spatial
    distribution of population and urban attractiveness across city blocks.
    Each block is connected to its nearest stops (within a walking network), and its
    population and attractiveness are proportionally distributed among them.  
    The final OD matrix between stops is then computed using the classical
    **gravity model** formulation:

        OD_ij = (Pop_i * Att_j) / d_ij

    where `Pop_i` and `Att_j` represent population and attractiveness at stops,
    and `d_ij` is the walking distance (in meters) between stops.

    Parameters
    ----------
    blocks : geopandas.GeoDataFrame
        GeoDataFrame of spatial blocks with at least the following columns:
        - `population`: block population
        - `density`: normalized density index
        - `diversity`: normalized diversity index
        - `land_use`: categorical land use type (used to compute land-use coefficient)
        Must have valid geometries.
    stops : geopandas.GeoDataFrame
        GeoDataFrame of public transport stops. Must have point geometries.
    walk_graph : networkx.Graph
        Walking network graph used to compute distances between blocks and stops.
        Must contain an edge attribute `"time_min"` or `"length_m"`.
    crs : int or pyproj.CRS
        Target projected coordinate reference system (meters).

    Returns
    -------
    pandas.DataFrame
        Square OD matrix (stops × stops) representing estimated travel demand.
        Each cell `OD_ij` gives the relative interaction intensity between stop *i* and stop *j*.

    Notes
    -----
    - Block attractiveness is computed as the sum of normalized density, diversity,
      and a land-use coefficient based on block land-use category.
    - Each block is connected to stops reachable within 10 minutes walking time.
      If no stop is reachable, the nearest stop is used.
    - The gravity model assumes distance decay: interaction decreases with higher distance.
    - Distances equal to 0 are replaced by NaN to avoid division by zero.

    Examples
    --------
    >>> od_matrix = get_OD(blocks, stops, walk_graph, crs=32636)
    >>> od_matrix.shape
    (120, 120)
    >>> od_matrix.iloc[:3, :3]
                 0         1         2
    0     0.000000  0.001231  0.000982
    1     0.001542  0.000000  0.001117
    2     0.000893  0.000776  0.000000
    """
    # Normalize CRS and fill missing density/diversity values
    blocks.to_crs(crs, inplace=True)
    blocks.loc[blocks['density'].isna(), 'density'] = 0.0
    blocks.loc[blocks['diversity'].isna(), 'diversity'] = 0.0
    scaler = MinMaxScaler()
    blocks[["density", "diversity"]] = scaler.fit_transform(blocks[["density", "diversity"]])
    blocks = blocks.set_geometry('geometry')

    # Assign land-use coefficients and compute block attractiveness
    landuse_coeff = {
        None: 0.06,
        'LandUse.INDUSTRIAL': 0.25,
        'LandUse.BUSINESS': 0.3,
        'LandUse.SPECIAL': 0.1,
        'LandUse.TRANSPORT': 0.1,
        'LandUse.RESIDENTIAL': 0.1,
        'LandUse.AGRICULTURE': 0.05,
        'LandUse.RECREATION': 0.05
    }
    blocks['lu_coeff'] = blocks['land_use'].apply(lambda x: landuse_coeff.get(x, 0))
    blocks['attractiveness'] = blocks['density'] + blocks['diversity'] + blocks['lu_coeff']

    # Reproject stops and compute walking matrix (blocks → stops)
    stops.to_crs(crs, inplace=True)
    walk_mx = get_adj_matrix_gdf_to_gdf(
        blocks,
        stops,
        walk_graph,
        weight="time_min",
        dtype=np.float64,
    )

    # Associate each block with nearby stops within 10 min walking distance
    walk_dict = {}
    for i, row in walk_mx.iterrows():
        walk_dict[i] = []
        for j, value in row.items():
            if value <= 10:  # Max 10 minutes walking time
                walk_dict[i].append((j, value))
        if len(walk_dict[i]) == 0:
            walk_dict[i].append((row.idxmin(), row.min()))

    # Map blocks to stops
    block_to_stops = walk_dict.copy()

    # Reverse mapping: stops → weighted contributions from blocks
    block_to_weights = {}
    for block1, stops1 in block_to_stops.items():
        stop_ids = np.array([stop[0] for stop in stops1])
        distances = np.array([stop[1] if stop[1] > 0 else 0.1 for stop in stops1], dtype=np.float64)
        weights = 1 / distances  # Closer stops get higher weights
        weights_normalized = weights / weights.sum()
        block_to_weights[block1] = list(zip(stop_ids, weights_normalized))

    # Initialize stop-level attributes
    stops_dict = {s: {'att': 0, 'pop': 0} for s in list(stops.index)}

    # Aggregate population and attractiveness from blocks to stops
    for key, value in block_to_weights.items():
        for stop_id, k in value:
            stops_dict[stop_id]['att'] += blocks.iloc[key]['attractiveness'] * k
            stops_dict[stop_id]['pop'] += blocks.iloc[key]['population'] * k

    stops['att'] = [v['att'] for _, v in stops_dict.items()]
    stops['pop'] = [v['pop'] for _, v in stops_dict.items()]

    # Compute stop-to-stop adjacency matrix (distance matrix)
    mx_stopstop = get_adj_matrix_gdf_to_gdf(stops, stops, walk_graph, 'length_m', dtype=np.float64)
    adj_mx = mx_stopstop.replace(0, np.nan)  # Avoid division by zero

    # Apply gravity model to compute OD matrix
    od_matrix = pd.DataFrame(
        np.outer(stops["pop"], stops["att"]) / adj_mx,
        index=adj_mx.index,
        columns=adj_mx.columns
    ).fillna(0)

    return od_matrix