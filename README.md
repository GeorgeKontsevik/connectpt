# connectpt

---

[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

Built with:

![numpy](https://img.shields.io/badge/NumPy-013243.svg?style={0}&logo=NumPy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458.svg?style={0}&logo=pandas&logoColor=white)
![pytest](https://img.shields.io/badge/Pytest-0A9EDC.svg?style={0}&logo=Pytest&logoColor=white)
![sphinx](https://img.shields.io/badge/Sphinx-000000.svg?style={0}&logo=Sphinx&logoColor=white)
![tqdm](https://img.shields.io/badge/tqdm-FFC107.svg?style={0}&logo=tqdm&logoColor=black)

---

## Table of Contents

- [Overview](#overview)
- [Core Features](#core-features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)

---

## Overview

connectpt is a Python toolkit for public-transport preprocessing and route-generation experiments. It is aimed at researchers and developers working on transit network analysis, route planning, and related machine-learning workflows. The repository provides preprocessing and route-generation surfaces, along with notebook-based examples and YAML-driven experiment configuration for exploratory and reproducible work. If you are new to the project, start with Getting Started for the runnable path and the recommended entry point into the workflow.

---

## Core Features

- Preprocess public-transport stop and line data end to end for one or more transport modalities, producing modality-specific stop data, travel-time matrices, and simplified stop graphs for downstream analysis.
- Project stops onto route geometries and convert the resulting network into a stop-to-stop graph, which gives developers a transit-ready representation for routing and evaluation workflows.
- Build scenario datasets for route-generation experiments in HDF5, including candidate routes, sampled route sets, per-stop outputs, and scenario-level summary values for training or analysis.
- Support iterative route-network search with a bee-colony optimizer, enabling developers to explore and improve candidate transit networks against a cost objective.
- Provide inductive route-learning and evaluation tooling around graph-based transit datasets, helping developers train models, compare baselines, and assess generated routes.

---

## Installation

**Prerequisites:** requires Python >=3.11

Install connectpt using one of the following methods:

**Build from source:**

1. Clone the connectpt repository:
```sh
git clone https://github.com/GeorgeKontsevik/connectpt
```

2. Navigate to the project directory:
```sh
cd connectpt
```

3. Install the project dependencies:

```sh
pip install -r requirements.txt
```

---

## Getting Started

**Prerequisites**

- Python environment with the project dependencies installed. The repository includes a docs/development install path via `pip install -e '.[docs]'` in the documentation workflow.
- Input data for preprocessing: a `geopandas.GeoDataFrame` of polygonal blocks with a valid geometry column.

**Quick start**

1. Install the package in editable mode from the repository root.
```bash
   pip install -e '.[docs]'
```

2. Prepare your block polygons as a `GeoDataFrame`. The preprocessing pipeline reprojects them to EPSG:4326 internally and derives a boundary from the union of the blocks.

3. Run the preprocessing workflow from `connectpt.preprocess.preprocess_data` with the blocks and the transport modalities you want to process.
```python
   result, simplified_graph_largest = preprocess(blocks_gdf, [Modality.BUS, Modality.TRAM])
```

4. Use the returned modality results for downstream route-generation experiments. The preprocessing output includes stop GeoDataFrames, a stop-to-stop time matrix, and the largest connected component of the simplified stop graph.

5. If you want to build the documentation locally, run the Sphinx build from the repository root.
```bash
   sphinx-build docs/source docs/build
```

---

## API Reference

- `connectpt.preprocess.preprocess_data.preprocess(blocks, modalities)` — end-to-end preprocessing pipeline that builds stop/line data per modality, projects stops onto roads, constructs a stop graph, and returns modality-specific outputs including stop GeoDataFrames, time matrices, and the largest connected component graph.
- `connectpt.routes_generator.build_dataset.sample_batch(...)` — samples route scenarios under a budget constraint and returns the chosen route indices, padding mask, and budgets.
- `connectpt.routes_generator.build_dataset.build_dataset(...)` — generates an HDF5 dataset of sampled scenarios and simulated outputs from a route generator/simulator pair.
- `connectpt.routes_generator.bee_colony.bee_colony(...)` — implements the bee colony optimization routine for route network planning.
- `connectpt.routes_generator.inductive_route_learning.Baseline` — abstract baseline interface with `update(...)` and `get_baseline(...)` methods.
- `connectpt.routes_generator.inductive_route_learning.FixedBaseline` — baseline implementation that stores a fixed value and updates it from observed costs.
- `connectpt.routes_generator.inductive_route_learning.RollingBaseline` — baseline implementation that maintains an exponentially weighted rolling average of costs.

---

## Examples

Examples of how this should work and how it should be used are available [here](https://github.com/GeorgeKontsevik/connectpt/tree/main/docs/source/examples).

---

## Documentation

A detailed connectpt description is available [here](https://github.com/GeorgeKontsevik/connectpt/tree/main/.github/workflows/documentation.yml).

---

## Contributing

- **[Report Issues](https://github.com/GeorgeKontsevik/connectpt/issues)**: Submit bugs found or log feature requests for the project.

- **[Submit Pull Requests](https://github.com/GeorgeKontsevik/connectpt/tree/main/CONTRIBUTING.md)**: To learn more about making a contribution to connectpt.

---

## Citation

If you use this software, please cite it as below.

### APA format:

    GeorgeKontsevik (2026). connectpt repository [Computer software]. https://github.com/GeorgeKontsevik/connectpt

### BibTeX format:

    @misc{connectpt,

        author = {GeorgeKontsevik},

        title = {connectpt repository},

        year = {2026},

        publisher = {github.com},

        journal = {github.com repository},

        howpublished = {\url{https://github.com/GeorgeKontsevik/connectpt}},

        url = {https://github.com/GeorgeKontsevik/connectpt}

    }

---