from typing import Dict, Union, Literal

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

from iduedu import get_adj_matrix_gdf_to_gdf


def _prepare_blocks(blocks: gpd.GeoDataFrame, crs: Union[int, str]) -> gpd.GeoDataFrame:
    blocks = blocks.copy()
    blocks.to_crs(crs, inplace=True)
    blocks.loc[blocks["density"].isna(), "density"] = 0.0
    blocks.loc[blocks["diversity"].isna(), "diversity"] = 0.0

    scaler = MinMaxScaler()
    blocks[["density", "diversity"]] = scaler.fit_transform(blocks[["density", "diversity"]])
    blocks = blocks.set_geometry("geometry")

    landuse_coeff = {
        None: 0.06,
        "LandUse.INDUSTRIAL": 0.25,
        "LandUse.BUSINESS": 0.3,
        "LandUse.SPECIAL": 0.1,
        "LandUse.TRANSPORT": 0.1,
        "LandUse.RESIDENTIAL": 0.1,
        "LandUse.AGRICULTURE": 0.05,
        "LandUse.RECREATION": 0.05,
    }
    blocks["lu_coeff"] = blocks["land_use"].apply(lambda x: landuse_coeff.get(x, 0))
    blocks["attractiveness"] = blocks["density"] + blocks["diversity"] + blocks["lu_coeff"]
    return blocks


def _block_to_stop_weights(
    blocks: gpd.GeoDataFrame,
    stops: gpd.GeoDataFrame,
    walk_graph: nx.Graph,
    max_walk_time: float = 10.0,
) -> Dict[int, list[tuple[int, float]]]:
    """Вернёт словарь: {block_idx: [(stop_idx, weight_norm), ...]} для одной модальности."""
    # матрица время_ходьбы от блоков к остановкам
    walk_mx = get_adj_matrix_gdf_to_gdf(
        blocks, stops, walk_graph, weight="time_min", dtype=np.float64
    )

    mapping: Dict[int, list[tuple[int, float]]] = {}
    for i, row in walk_mx.iterrows():
        # кандидаты ≤ порога по времени
        candidates = [(j, v) for j, v in row.items() if v <= max_walk_time]
        if not candidates:
            # иначе ближайшая
            jmin = row.idxmin()
            candidates = [(jmin, float(row.loc[jmin]))]

        # веса = 1/время (с защитой от 0)
        dists = np.array([c[1] if c[1] > 0 else 0.1 for c in candidates], dtype=float)
        w = 1.0 / dists
        w = w / w.sum()
        mapping[i] = list(zip([c[0] for c in candidates], w))
    return mapping


def get_multi_OD(
    blocks: gpd.GeoDataFrame,
    stops_by_modality: Dict[object, gpd.GeoDataFrame],
    walk_graph: nx.Graph,
    crs: Union[int, str],
    *,
    max_walk_time: float = 10.0,
    allocation: Literal["independent", "exclusive", "proportional"] = "independent",
) -> Dict[object, pd.DataFrame]:
    """Считает OD-матрицы отдельно для каждой модальности без межмодальных пар.

    Parameters
    ----------
    blocks : GeoDataFrame
        Блоки с колонками: population, density, diversity, land_use, geometry.
    stops_by_modality : dict
        Словарь {modality: GeoDataFrame(точки-остановки)}.
    walk_graph : nx.Graph
        Пеший граф с атрибутами 'time_min' (для блок→остановка) и 'length_m' (для остановка→остановка).
    crs : int|str
        Проецированный CRS (метры).
    max_walk_time : float, default 10.0
        Порог времени ходьбы (мин) при ассоциации блоков со стопами.
    allocation : {"independent","exclusive","proportional"}, default "independent"
        Стратегия распределения вклада блоков между модальностями:
          - "independent": каждая модальность нормирует блоки независимо (потенциальный спрос по видам).
          - "exclusive": блок идёт только в модальность с минимальным T_min до ближайшей остановки.
          - "proportional": вклад блока делится между модальностями пропорционально 1/T_min(модальность).

    Returns
    -------
    dict
        {modality: OD DataFrame} — квадратная матрица только для остановок данной модальности.
    """
    # подготовим блоки один раз
    blocks_prep = _prepare_blocks(blocks, crs=crs)

    # приведём остановки к CRS и подготовим веса блок→стоп для каждой модальности
    stops_prepped: Dict[object, gpd.GeoDataFrame] = {}
    block_weights_by_mod: Dict[object, Dict[int, list[tuple[int, float]]]] = {}
    min_time_by_mod: Dict[object, pd.Series] = {}

    for m, stops in stops_by_modality.items():
        st = stops.copy()
        st.to_crs(crs, inplace=True)
        stops_prepped[m] = st

        # block→stop распределения внутри модальности
        block_weights_by_mod[m] = _block_to_stop_weights(
            blocks_prep, st, walk_graph, max_walk_time=max_walk_time
        )

        # Для стратегий "exclusive"/"proportional" — минимум времени до любой остановки этой модальности
        walk_mx = get_adj_matrix_gdf_to_gdf(
            blocks_prep, st, walk_graph, weight="time_min", dtype=np.float64
        )
        min_time_by_mod[m] = walk_mx.min(axis=1)

    # коэффициенты распределения блоков между модальностями (только для exclusive/proportional)
    if allocation == "exclusive":
        # модальность с минимальным T_min выигрывает весь вклад
        best_mod = pd.concat(min_time_by_mod, axis=1).idxmin(axis=1)  # index=block, value=modality
        alloc_coef = {
            m: (best_mod == m).astype(float)  # 1.0 если лучший, иначе 0.0
            for m in stops_by_modality.keys()
        }
    elif allocation == "proportional":
        # делим пропорционально 1/T_min по модальностям
        inv = pd.concat({m: 1.0 / min_time_by_mod[m].replace(0, 0.1) for m in stops_by_modality}, axis=1)
        alloc_sum = inv.sum(axis=1).replace(0, np.nan)
        alloc = inv.div(alloc_sum, axis=0).fillna(0.0)
        alloc_coef = {m: alloc[m] for m in stops_by_modality}
    else:
        # independent: коэффициент 1 для всех модальностей (внутренняя нормировка по модальности)
        alloc_coef = {m: pd.Series(1.0, index=blocks_prep.index) for m in stops_by_modality}

    # теперь считаем OD по каждой модальности отдельно
    result: Dict[object, pd.DataFrame] = {}

    for m, stops in stops_prepped.items():
        weights_map = block_weights_by_mod[m]
        coef_series = alloc_coef[m].reindex(blocks_prep.index).fillna(0.0)

        # агрегируем к стопам с учётом коэффициента распределения между модальностями
        acc = {s_idx: {"att": 0.0, "pop": 0.0} for s_idx in list(stops.index)}
        for blk_idx, pairs in weights_map.items():
            coef_m = float(coef_series.loc[blk_idx])
            if coef_m == 0.0:
                continue
            for stop_id, w in pairs:
                acc[stop_id]["att"] += float(blocks_prep.iloc[blk_idx]["attractiveness"]) * float(w) * coef_m
                acc[stop_id]["pop"] += float(blocks_prep.iloc[blk_idx]["population"]) * float(w) * coef_m

        stops = stops.copy()
        stops["att"] = [v["att"] for _, v in acc.items()]
        stops["pop"] = [v["pop"] for _, v in acc.items()]

        # дистанции стоп↔стоп только внутри модальности
        dmx = get_adj_matrix_gdf_to_gdf(stops, stops, walk_graph, "length_m", dtype=np.float64)
        dmx = dmx.replace(0, np.nan)

        # OD по гравитации
        od = pd.DataFrame(
            np.outer(stops["pop"], stops["att"]) / dmx,
            index=dmx.index,
            columns=dmx.columns,
        ).fillna(0.0)

        result[m] = od

    return result