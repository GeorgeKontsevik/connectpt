import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import Point, LineString
from shapely import line_locate_point, union_all
from shapely.ops import linemerge


def _as_geodataframe(frame, crs=None):
    if isinstance(frame, gpd.GeoDataFrame):
        result = frame.copy()
        if result.crs is None and crs is not None:
            result = result.set_crs(crs)
        return result
    result = gpd.GeoDataFrame(frame.copy(), geometry="geometry", crs=crs)
    return result


def _close_gaps(gdf, tolerance) -> gpd.GeoSeries:
    """Function from momepy. Close gaps in LineString geometry where it should be contiguous.

    Snaps both lines to a centroid of a gap in between.

    Parameters
    ----------
    gdf : GeoDataFrame, GeoSeries
        GeoDataFrame  or GeoSeries containing LineString representation of a network.
    tolerance : float
        nodes within a tolerance will be snapped together

    Returns
    -------
    GeoSeries

    """
    geom = gdf.geometry.array
    coords = shapely.get_coordinates(geom)
    indices = shapely.get_num_coordinates(geom)

    # generate a list of start and end coordinates and create point geometries
    edges = [0]
    i = 0
    for ind in indices:
        ix = i + ind
        edges.append(ix - 1)
        edges.append(ix)
        i = ix
    edges = edges[:-1]
    points = shapely.points(np.unique(coords[edges], axis=0))

    buffered = shapely.buffer(points, tolerance / 2)

    dissolved = shapely.union_all(buffered)

    exploded = [
        shapely.get_geometry(dissolved, i)
        for i in range(shapely.get_num_geometries(dissolved))
    ]

    centroids = shapely.centroid(exploded)

    snapped = shapely.snap(geom, shapely.union_all(centroids), tolerance)

    return gpd.GeoSeries(snapped, crs=gdf.crs)


def restore_linestrings(edges_gdf, nodes_gdf):
    # Restore None geometries for roads
    nodes_crs = getattr(nodes_gdf, "crs", None)
    edges = _as_geodataframe(edges_gdf, crs=getattr(edges_gdf, "crs", None) or nodes_crs)
    nodes = _as_geodataframe(nodes_gdf, crs=nodes_crs or getattr(edges, "crs", None))
    geometry = gpd.GeoSeries(edges["geometry"], crs=edges.crs)
    mask = geometry.isna() | geometry.is_empty
    
    for idx in edges[mask].index:
        start_node = edges.loc[idx, 'node_start']
        end_node = edges.loc[idx, 'node_end']
        
        start_point = nodes.loc[start_node, 'geometry']
        end_point = nodes.loc[end_node, 'geometry']
        
        line = LineString([
            (start_point.x, start_point.y),
            (end_point.x, end_point.y)
        ])
        
        edges.loc[idx, 'geometry'] = line
    
    return gpd.GeoDataFrame(edges, geometry="geometry", crs=edges.crs)


def _cut(line: LineString, distance: float) -> list[LineString]:
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0 or distance >= line.length:

        return [LineString(line)]
    
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [LineString(coords[: i + 1]), LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)

            return [LineString(coords[:i] + [(cp.x, cp.y)]), LineString([(cp.x, cp.y)] + coords[i:])]
        

def _project_stop_on_road(road_geom: LineString, stop_geom: Point) -> list[LineString]:
    # Project stops on road geometry and return LineStrings of splitted road
    distance = line_locate_point(road_geom, stop_geom)
    splitted_road = _cut(road_geom, distance)

    return splitted_road

