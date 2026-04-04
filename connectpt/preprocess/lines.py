import warnings
import geopandas as gpd
import pygeoops
import osmnx as ox
import momepy as mp
import neatnet
import shapely

from shapely.geometry import Polygon, MultiPolygon
from typing import Dict

from .types import Modality, MODALITY_LINE_TAGS
from .utils import _as_geodataframe, _close_gaps, restore_linestrings

ox.settings.useful_tags_way.append("railway")
warnings.filterwarnings("ignore", category=UserWarning)



def _get_electric_lines(polygon: Polygon | MultiPolygon, filter_tags: str) -> gpd.GeoDataFrame:
    # Download and extract electric or rail-based transport lines within a given polygon
    custom_filter = filter_tags
    G = ox.graph_from_polygon(polygon, network_type='all', custom_filter=custom_filter, retain_all=True, truncate_by_edge=True)
    n,e = mp.nx_to_gdf(G)
    n = _as_geodataframe(n, crs=getattr(n, "crs", None) or 4326)
    e = _as_geodataframe(e, crs=getattr(e, "crs", None) or getattr(n, "crs", None) or 4326)
    edges = restore_linestrings(e, n)
    edges = edges.clip(polygon)
    local_crs = edges.estimate_utm_crs()
    edges.to_crs(local_crs, inplace=True)
    
    return edges


def _get_drive_lines(polygon: Polygon | MultiPolygon) -> gpd.GeoDataFrame:
    # Download and extract drivable road network edges within a polygon (OSMnx-only path)
    graph = ox.graph_from_polygon(
        polygon,
        network_type="drive",
        retain_all=True,
        truncate_by_edge=True,
    )
    _, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True)
    edges = edges[edges.geometry.geom_type == "LineString"]

    return edges


def _preprocess_electric_lines(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Preprocess electric or rail-based transport lines to create clean, continuous geometries
    local_crs = edges.estimate_utm_crs()
    edges.to_crs(local_crs, inplace=True)
    buffer_edges = edges.geometry.buffer(25, cap_style="round", join_style="mitre")
    edges_union = buffer_edges.union_all()
    centerline = pygeoops.centerline(edges_union, densify_distance=58, simplifytolerance=-0.15)
    lines_gdf = gpd.GeoDataFrame(geometry=[centerline], crs=local_crs)
    lines_gdf.reset_index(drop=True, inplace=True)
    lines_gdf = lines_gdf.explode("geometry")
    geom_lines = _close_gaps(lines_gdf.geometry, tolerance=2)
    lines_gdf = gpd.GeoDataFrame(geometry=geom_lines)
    
    return lines_gdf


def _preprocess_drive_lines(
    edges: gpd.GeoDataFrame,
    boundary: Polygon | MultiPolygon,
    preloaded_buildings: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    # Preprocess driveable network lines to create clean, continuous geometries
    if preloaded_buildings is not None and not preloaded_buildings.empty:
        buildings = preloaded_buildings.copy()
        buildings = buildings[buildings.geometry.notna() & ~buildings.geometry.is_empty].copy()
        if "building" in buildings.columns:
            buildings = buildings.query('building != "roof"').copy()
        if buildings.crs is None:
            buildings = buildings.set_crs(edges.crs)
        elif buildings.crs != edges.crs:
            buildings = buildings.to_crs(edges.crs)
    else:
        buildings = (ox.features_from_polygon(boundary, tags={"building": True})
                        .query('building != "roof"')
                        .to_crs(edges.crs))
    simplified = neatnet.neatify(edges, exclusion_mask=buildings.geometry , max_segment_length = 1)
    simplified_geom = _close_gaps(simplified, tolerance=0.5)
    lines_gdf = gpd.GeoDataFrame(geometry=simplified_geom)

    return lines_gdf


def get_lines(
    polygon: Polygon | MultiPolygon,
    modalities: list[Modality],
    preloaded_drive_lines: gpd.GeoDataFrame | None = None,
    preloaded_buildings: gpd.GeoDataFrame | None = None,
) -> Dict[Modality, gpd.GeoDataFrame]:
    """Download and preprocess transport network lines for multiple transport modalities.

    This function retrieves, processes, and unifies transport line geometries (bus, tram, trolleybus, etc.)
    within a given polygon. Depending on the modality, it either downloads drivable roads (for buses)
    or electric network lines (for trams and trolleybuses), then applies the appropriate preprocessing
    steps to produce simplified and continuous line geometries.

    Parameters
    ----------
    polygon : shapely.Polygon or shapely.MultiPolygon
        The geographic boundary within which transport lines are extracted and processed.
    modalities : list of Modality
        List of transport types to process (e.g., `[Modality.BUS, Modality.TRAM]`).

    Returns
    -------
    dict of {Modality: geopandas.GeoDataFrame}
        A dictionary mapping each transport modality to its corresponding processed
        GeoDataFrame of line geometries. Each GeoDataFrame includes:
        - `geometry`: simplified `LineString` geometries,
        - `modality`: modality name (e.g., `"bus"`, `"tram"`).

    Notes
    -----
    - For buses, the function downloads the drivable street network using `_get_drive_lines()`
      and simplifies it via `_preproccess_drive_lines()`.
    - For trams and trolleybuses, it downloads rail or electric lines using `_get_electric_lines()`
      and processes them with `_preproccess_electric_lines()`.
    - All resulting geometries are reprojected to local UTM CRS for consistency and spatial accuracy.

    Examples
    --------
    >>> result = get_lines(city_boundary, [Modality.BUS, Modality.TRAM])
    >>> result[Modality.TRAM].head()
      modality                                           geometry
    0     tram  LINESTRING (445122.1 6139458.2, 445215.4 6139420.7)
    1     tram  LINESTRING (445310.7 6139381.9, 445402.5 6139342.4)
    """
    result: Dict[Modality, gpd.GeoDataFrame] = {}

    for modality in modalities:
        modality_tag = MODALITY_LINE_TAGS[modality]
        if modality_tag is None:
            # For bus, we use driving roads
            if preloaded_drive_lines is not None and not preloaded_drive_lines.empty:
                roads_gdf = preloaded_drive_lines.copy()
                if roads_gdf.crs is None:
                    roads_gdf = roads_gdf.set_crs(4326)
                local_crs = roads_gdf.estimate_utm_crs()
                if local_crs is not None:
                    roads_gdf = roads_gdf.to_crs(local_crs)
            else:
                roads_gdf = _get_drive_lines(polygon)
            processed_lines = _preprocess_drive_lines(
                roads_gdf,
                boundary=polygon,
                preloaded_buildings=preloaded_buildings,
            )
            processed_lines["modality"] = modality.value
        else:
            lines_gdf = _get_electric_lines(polygon, modality_tag)
            processed_lines = _preprocess_electric_lines(lines_gdf)
            processed_lines["modality"] = modality.value

        result[modality] = processed_lines.reset_index(drop=True)

    return result
