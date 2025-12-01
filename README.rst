ConnectPT
=========

.. logo-start

.. figure:: https://psv4.userapi.com/s/v1/d2/aE8kEC2MYzxzxQgGbG4SIXKijfv-ouCe9jNMag7ONZ8TdctZo5IKBe-MR2OTRxVEWMIaq7yxqnPSpyKEK4HMDw_yf5_XLgYa-7MQxgABQBIUzMCtFT7G5FsrWZN7GbfnTQUP-1X-NSqK/connectpt_logo_gen_gpt_v4.png
   :alt: ConnectPT

.. logo-end

|PythonVersion|

.. readme-start 

Overview
--------
ConnectPT is a research toolkit for building public transport networks and generating route plans. It combines preprocessing utilities for stop-level graphs with optimization and learning-based route generators.
The tool is based on the `transit_learning <https://github.com/AHolliday/transit_learning/tree/master>`__ project, which has been modified. Improvements include the addition of a transport-weighted connectivity metric for route generation. In the future, we plan to adapt the generation process to urban planning implementations. 

Installation
------------
The library can be installed with ``pip``:

::

   pip install git+https://github.com/alexandermorozzov/connectpt@main


How to use
----------
Preprocess public transport data for selected modes:

.. code-block:: python

   import geopandas as gpd
   from connectpt.preprocess import preprocess, Modality

   blocks = gpd.read_file("path/to/blocks.geojson") # City blocks are obtained via BlocksNet library
   result, graph = preprocess(blocks, [Modality.BUS])
   stops_gdf, time_matrix, stop_graph = result[Modality.BUS]

Generate and evaluate routes with Bee Colony Optimization using a provided config:

.. code-block:: python

   from omegaconf import OmegaConf
   from connectpt.routes_generator import (
       RouteGenBatchState,
       bee_colony,
       get_cost_module_from_cfg,
       get_dataset_from_config,
       init_from_cfg,
   )

   cfg = OmegaConf.load("connectpt/routes_generator/cfg/bco_mumford.yaml")
   dataset = get_dataset_from_config(cfg.data)
   batch = dataset[:1]
   cost = get_cost_module_from_cfg(cfg.cost)
   state = RouteGenBatchState(batch, cost, n_routes_to_plan=cfg.experiment.n_routes)
   init_routes = init_from_cfg(state, cfg.init)
   best = bee_colony(state, cost, init_routes, n_bees=cfg.experiment.n_bees)

   print(best.shape)  # batch_size x n_routes x max_nodes

For neural route generators, see `examples/route_generator/experiment.ipynb` and configs under `connectpt/routes_generator/cfg/`.

Data and Examples
-----------------
Benchmark data for evaluation – graphs and demand matrices: `data/` (Mandl, Mumford, blocks/graphs/routes).

You can contact us:
-------------------
- Alexander Morozov — alexandermorozzov@gmail.com
- Ruslan Kozlyak — rkozliak@gmail.com

Acknowledgments:
----------------
This work supported by the Ministry of Economic Development of the Russian Federation (IGK 000000C313925P4C0002), agreement No139-15-2025-010.




.. readme-end

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.11+-blue
   :target: https://www.python.org/