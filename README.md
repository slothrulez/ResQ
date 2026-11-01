# Q-Rescue — milestone 1

Hybrid ambulance dispatch prototype: Dijkstra routing, a QUBO assignment
formulation, QAOA sampled on Qiskit Aer, and independent exact/Hungarian
validation. This milestone has no frontend and makes no quantum-advantage
claim.

## Setup

Python 3.10 is supported. Install the pinned dependencies into the existing
virtual environment:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

## CLI

```bash
.venv/bin/python -m qrescue.cli
.venv/bin/python -m qrescue.cli --closure
.venv/bin/python -m qrescue.cli --three-incidents
.venv/bin/python -m qrescue.cli --reps 3 --shots 4096 --seed 7 --json
```
