from typing import Dict, List, Tuple, Optional
import numpy as np
import geopandas as gpd
import networkx as nx
import momepy as mp
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge


def _normalize_road_geometries(roads_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    roads = roads_gdf.copy()
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()

    def _normalize_geom(geom):
        if geom is None or geom.is_empty:
            return None
        if isinstance(geom, LineString):
            return geom
        if isinstance(geom, MultiLineString):
            merged = linemerge(geom)
            if isinstance(merged, LineString):
                return merged
            return geom
        return None

    roads["geometry"] = roads.geometry.map(_normalize_geom)
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
    roads = roads.explode(index_parts=False).reset_index(drop=True)
    roads = roads[roads.geometry.geom_type == "LineString"].copy()
    return roads.reset_index(drop=True)


def roads_to_graph(roads_gdf: gpd.GeoDataFrame, stops_gdf: gpd.GeoDataFrame) -> nx.Graph:
    """Convert road geometries into a NetworkX graph and mark stop nodes.

    This function builds an undirected NetworkX graph from a GeoDataFrame of road
    geometries using `momepy.gdf_to_nx`, and then identifies the nearest network
    nodes for all public transport stops. The corresponding nodes are marked with
    the attribute `"is_stop" = True`, allowing further operations that depend on
    stop locations (e.g., subgraph extraction or route generation).

    Parameters
    ----------
    roads_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing road geometries (LineString). Should be in a
        projected CRS (meters) and represent a connected street network.
    stops_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing stop point geometries. Must use the same CRS as
        `roads_gdf` (or be pre-aligned before calling the function).

    Returns
    -------
    networkx.Graph
        An undirected NetworkX graph derived from the road network, where nodes
        corresponding to the nearest locations of stops are annotated with
        `is_stop=True`.

    Notes
    -----
    - The graph is created using `momepy.gdf_to_nx`, which infers node positions
      from road segment endpoints.
    - The nearest node for each stop is determined via spatial join
      (`geopandas.sjoin_nearest`).
    - The resulting graph can be directly used for topological or routing analyses.

    Examples
    --------
    >>> G = roads_to_graph(roads_gdf, stops_gdf)
    >>> list(nx.get_node_attributes(G, "is_stop").items())[:3]
    [((435245.5, 6138021.3), True), ((435321.8, 6138120.7), True), ...]
    """
    roads_gdf = _normalize_road_geometries(roads_gdf)
    roads_graph = mp.gdf_to_nx(roads_gdf, multigraph=False, directed=False)
    nodes_gdf, _ = mp.nx_to_gdf(roads_graph)

    sjoin = stops_gdf.sjoin_nearest(nodes_gdf)
    for node in zip(sjoin["x"], sjoin["y"]):
        roads_graph.nodes[node]["is_stop"] = True

    return roads_graph


# def stop_complete_then_prune(
#     G: nx.Graph,
#     stop_flag: str = "is_stop",
#     weight_attr: str = "mm_len",
#     node_x: str = "x",
#     node_y: str = "y",
#     min_weight: float | None = None,
#     max_weight: float | None = None,
#     speed_kmh: float | None = None,
#     max_hops: int | None = None,
#     max_detour_factor: float | None = None,
# ) -> nx.Graph:
#     def is_stop(n):
#         return bool(G.nodes[n].get(stop_flag, False))

#     def edge_w(u, v):
#         d = G.get_edge_data(u, v, {})
#         return d.get(weight_attr)

#     def seg_geom(u, v):
#         d = G.get_edge_data(u, v, {}) or {}
#         g = d.get("geometry")
#         if g is not None:
#             return g
#         ux, uy = G.nodes[u].get(node_x), G.nodes[u].get(node_y)
#         vx, vy = G.nodes[v].get(node_x), G.nodes[v].get(node_y)
#         if None not in (ux, uy, vx, vy):
#             return LineString([(ux, uy), (vx, vy)])
#         return None

#     stops = [n for n in G.nodes if is_stop(n)]
#     SG = nx.Graph()
#     if "crs" in G.graph:
#         SG.graph["crs"] = G.graph["crs"]
#     for s in stops:
#         SG.add_node(s, **G.nodes[s])

#     seen_paths: set[tuple] = set()

#     for i, s in enumerate(stops):
#         dist, paths = nx.single_source_dijkstra(G, s, weight=weight_attr)

#         for t in stops[i + 1:]:
#             d = dist.get(t)
#             if d is None:
#                 continue
#             if min_weight is not None and d < min_weight:
#                 continue
#             if max_weight is not None and d > max_weight:
#                 continue

#             path = paths[t]
#             if max_hops and len(path) - 1 > max_hops:
#                 continue

#             stop_count = sum(1 for n in path if is_stop(n))
#             if stop_count != 2:
#                 continue

#             geoms = []
#             total_check = 0.0
#             path_signature = []
#             ok = True

#             for u, v in zip(path[:-1], path[1:]):
#                 seg_key = frozenset((u, v))
#                 path_signature.append(seg_key)

#                 w = edge_w(u, v)
#                 if w is None:
#                     ok = False
#                     break
#                 total_check += w

#                 g = seg_geom(u, v)
#                 if g is not None:
#                     geoms.append(g)

#             if not ok:
#                 continue

#             signature = tuple(sorted(path_signature))
#             if signature in seen_paths:
#                 continue
#             seen_paths.add(signature)

#             if max_detour_factor is not None and geoms:
#                 merged = linemerge(geoms)
#                 if merged.length > 0:
#                     straight = LineString([merged.coords[0], merged.coords[-1]])
#                     if straight.length > 0 and total_check > max_detour_factor * straight.length:
#                         continue

#             attrs = {"weight": total_check, "original_path": path}

#             if speed_kmh:
#                 attrs["time_min"] = total_check * 60.0 / (1000.0 * float(speed_kmh))

#             if geoms:
#                 try:
#                     attrs["geometry"] = linemerge(geoms)
#                 except Exception:
#                     pass

#             SG.add_edge(s, t, **attrs)

#     return nx.convert_node_labels_to_integers(SG)


def stop_complete_then_prune(
    G: nx.Graph,
    stop_flag: str = "is_stop",
    weight_attr: str = "mm_len",
    node_x: str = "x",
    node_y: str = "y",
    min_weight: float | None = None,
    max_weight: float | None = None,
    speed_kmh: float | None = None,
) -> nx.Graph:
    """Build a complete stop-to-stop graph and prune intermediate-stop connections.

    This function constructs a new graph where each pair of stops in the input graph `G`
    is connected by its shortest path, computed using Dijkstra’s algorithm.
    Only connections that contain exactly two stops (the start and end stops)
    are retained — any path containing intermediate stops is discarded.

    The resulting graph contains:
      - `weight`: total length of the shortest path (using `weight_attr`)
      - `original_path`: the list of original node IDs in the path
      - `geometry`: LineString or MultiLineString geometry (if available)
      - `time_min`: travel time in minutes (if `speed_kmh` is provided)

    Parameters
    ----------
    G : networkx.Graph
        Input undirected graph where edges have a numeric weight attribute (e.g., length).
    stop_flag : str, optional
        Node attribute key indicating whether a node represents a stop. Default is "is_stop".
    weight_attr : str, optional
        Edge attribute to use as path weight (e.g., "mm_len" or "length_m"). Default is "mm_len".
    node_x, node_y : str, optional
        Node attributes containing x and y coordinates (used to reconstruct missing geometries).
    min_weight, max_weight : float or None, optional
        Optional filters to skip edges shorter or longer than these thresholds.
    speed_kmh : float or None, optional
        Average travel speed (km/h). If provided, travel time (`time_min`) is computed.

    Returns
    -------
    networkx.Graph
        A simplified undirected graph connecting only pairs of stops via their shortest paths.
        Each edge represents a direct connection between two stops with geometry and attributes.

    Notes
    -----
    - Internal non-stop nodes are removed from the resulting connections.
    - The function assumes the input graph is weighted and connected.
    - CRS information is preserved if available in `G.graph['crs']`.

    Examples
    --------
    >>> SG = stop_complete_then_prune(G, weight_attr="length_m", speed_kmh=20)
    >>> list(SG.edges(data=True))[:2]
    [((1, 3), {'weight': 250.5, 'time_min': 0.75, 'original_path': [...]}), ...]
    """
    def is_stop(n): 
        return bool(G.nodes[n].get(stop_flag, False))

    def edge_w(u, v):
        d = G.get_edge_data(u, v, {})
        return d.get(weight_attr, None)

    def seg_geom(u, v):
        d = G.get_edge_data(u, v, {}) or {}
        g = d.get("geometry")
        if g is not None:
            return g
        # Fallback: build a segment from node coordinates if geometry is missing
        ux, uy = G.nodes[u].get(node_x), G.nodes[u].get(node_y)
        vx, vy = G.nodes[v].get(node_x), G.nodes[v].get(node_y)
        if None not in (ux, uy, vx, vy):
            return LineString([(ux, uy), (vx, vy)])
        return None

    stops = [n for n in G.nodes if is_stop(n)]

    SG = nx.Graph()
    if "crs" in G.graph:
        SG.graph["crs"] = G.graph["crs"]
    for s in stops:
        SG.add_node(s, **G.nodes[s])

    # Run single-source Dijkstra for each stop to get distances and paths
    for i, s in enumerate(stops):
        dist, paths = nx.single_source_dijkstra(G, s, weight=weight_attr)

        for t in stops[i + 1:]:  # Avoid duplicates (t > s)
            d = dist.get(t)
            if d is None:
                continue

            # Optional filters by path length
            if min_weight is not None and d < min_weight:
                continue
            if max_weight is not None and d > max_weight:
                continue

            path = paths[t]
            # Count how many stops are on the path
            stop_count = sum(1 for n in path if is_stop(n))
            if stop_count != 2:
                # Path includes intermediate stops → skip
                continue

            # Reconstruct geometry and validate total weight
            geoms = []
            total_check = 0.0
            ok = True
            for u, v in zip(path[:-1], path[1:]):
                w = edge_w(u, v)
                if w is None:
                    ok = False
                    break
                total_check += w
                g = seg_geom(u, v)
                if g is not None:
                    geoms.append(g)
            if not ok:
                continue

            attrs = {
                "weight": total_check,
                "original_path": path
            }
            if speed_kmh:
                attrs["time_min"] = total_check * 60.0 / (1000.0 * float(speed_kmh))
            if geoms:
                try:
                    attrs["geometry"] = linemerge(geoms)
                except Exception:
                    # If geometry merge fails, skip geometry assignment
                    pass

            SG.add_edge(s, t, **attrs)

    # Always return a graph object, even when no valid stop-to-stop edges were created.
    simple_graph = nx.convert_node_labels_to_integers(SG)
    return simple_graph


def build_time_matrix(G: nx.Graph, attr: str = "time_min"):
    """Build a travel time matrix between nodes of a graph.

    This function constructs a square matrix representing pairwise travel times
    between all nodes in a given NetworkX graph `G`, using an edge attribute
    such as `"time_min"` or any other numeric field representing time or distance.

    Parameters
    ----------
    G : networkx.Graph or networkx.MultiGraph or networkx.DiGraph
        Input graph where edges contain a numeric time or distance attribute.
    attr : str, optional
        Name of the edge attribute used to extract time values. Default is `"time_min"`.

    Returns
    -------
    np.ndarray
        A 2D array of string values representing the time matrix between nodes.
        - `"Inf"` indicates that no direct edge exists between the corresponding nodes.
        - Diagonal elements are set to `0.00`.

    Notes
    -----
    - For `MultiGraph` objects, only the first edge between nodes is used.
    - Values are rounded to two decimal places.
    - The matrix is ordered according to the sorted list of graph nodes.

    Examples
    --------
    >>> matrix_str = build_time_matrix(G, attr="time_min")
    >>> matrix_str[:3, :3]
    array([['0.00', '1.25', 'Inf'],
           ['1.25', '0.00', '0.80'],
           ['Inf', '0.80', '0.00']], dtype='<U5')
    """
    nodes = sorted(G.nodes())
    n = len(nodes)
    matrix = np.full((n, n), np.inf)

    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if G.has_edge(u, v):
                edge_data = G.get_edge_data(u, v)
                # Handle MultiGraph: edge_data = {0: {...}, 1: {...}, ...}
                if isinstance(edge_data, dict) and 0 in edge_data:
                    edge = next(iter(edge_data.values()))
                else:
                    edge = edge_data
                if edge is not None:
                    val = edge.get(attr, np.inf)
                    matrix[i, j] = val

    np.fill_diagonal(matrix, 0.0)
    matrix = np.round(matrix, 2)
    matrix_str = np.where(np.isinf(matrix), "Inf", np.char.mod("%.2f", matrix))

    return matrix_str
