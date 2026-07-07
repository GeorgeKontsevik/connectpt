# connectpt

Public-transport preprocessing and route-generation toolkit.

## Scheme

```mermaid
flowchart LR
    A[Inputs] --> B[Run: examples/preprocess/example_preprocess.ipynb]
    B --> C[Checked outputs]
    C --> D[Paper / thesis use]
```

## Main Result

![Main result](docs/readme_result.svg)

## Run

Entrypoint: `examples/preprocess/example_preprocess.ipynb`

Human:

```bash
pip install -e . && jupyter notebook examples/preprocess/example_preprocess.ipynb
```

Agent:

Use iduedu-derived stops when available; do not force route diversity with fallback paths.

## Publication

No tracked paper/preprint in this repo.

## Next Steps / Heuristics

Heuristic: gravity demand is preferred for real training data; synthetic demand must be labeled.
