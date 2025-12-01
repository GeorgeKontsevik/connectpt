ConnectPT
=========

.. logo-start

.. logo-end

|PythonVersion| |LicenseBadge|

.. readme-start 

Overview
--------
ConnectPT is a research toolkit for building public transport networks and generating route plans. It combines preprocessing utilities for stop-level graphs with optimization and learning-based route generators derived from the Transit Learning project.

Features
--------
- End-to-end preprocessing pipeline (`connectpt.preprocess`) that downloads stops/lines by modality, projects them onto road networks, and returns stop coordinates plus stop-to-stop time matrices.
- Route-generation stack (`connectpt.routes_generator`) with Bee Colony Optimization, path-combining neural models, PPO/REINFORCE-style training loops, and evaluation helpers.
- Config-driven experiments with Hydra/OmegaConf; ready-to-use configs live in `connectpt/routes_generator/cfg/`.
- Benchmark scenarios (Mandl, Mumford) and cached datasets for quick experiments in `data/` and `examples/cache/`.
- Jupyter notebooks demonstrating preprocessing and route generation in `examples/`.

Installation
------------
Python 3.11+ is required. Install PyTorch and torch-geometric suitable for your platform, then install ConnectPT:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install -e .               # library only
   pip install -e .[dev,tests,docs]  # optional extras

Quickstart
----------
Preprocess public transport data for selected modes:

.. code-block:: python

   import geopandas as gpd
   from connectpt.preprocess import preprocess, Modality

   blocks = gpd.read_file("path/to/blocks.geojson")
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
- Benchmark graphs and demand matrices: `data/` (Mandl, Mumford, blocks/graphs/routes).
- Notebooks for preprocessing and route generation: `examples/preprocess/` and `examples/route_generator/`.
- Cached datasets for quick experimentation: `examples/cache/`.

Project Structure
-----------------
- `connectpt/preprocess`: stop/line preprocessing, OD/time matrices, modality handling.
- `connectpt/routes_generator`: datasets, cost modules, optimization methods (BCO, RL), and evaluation utilities; configs in `cfg/`.
- `data`: small benchmark scenarios used in examples/tests.
- `examples`: runnable notebooks for preprocessing and training/evaluation.
- `docs`: Sphinx sources (includes this README).
- `tests`: minimal tests for core components.

Development
-----------
- Run tests: `pytest`.
- Format/lint: `black . && isort .`.
- Build docs: `sphinx-build -b html docs/source docs/build`.

License
-------
BSD-3-Clause (see `LICENSE`). Route-generation components are adapted from the Transit Learning project by Andrew Holliday (GPLv3 in upstream); retain notices when redistributing.

Contacts
--------
- Alexander Morozov — alexandermorozzov@gmail.com
- Ruslan Kozlyak — rkozliak@gmail.com

.. readme-end

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.11+-blue
   :target: https://www.python.org/

.. |LicenseBadge| image:: https://img.shields.io/badge/license-BSD--3--Clause-green
   :target: LICENSE
