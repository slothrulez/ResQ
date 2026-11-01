# Q-Rescue

Q-Rescue is a small hybrid quantum-classical ambulance-dispatch prototype.
It models a synthetic road network, routes ambulances to incident locations,
and formulates the assignment as a QUBO. A Streamlit dashboard and command-line
report expose the same backend, including user-selectable road closures,
assignment routes, sampled QAOA candidates, and independent classical checks.
The demonstration uses schematic Kochi-flavoured names and is not a real
geographic or emergency-response system.

## What the quantum part actually does

The assignment QUBO is converted to an Ising cost operator and passed to
QAOA implemented with Qiskit Aer. Aer evaluates the variational circuit's
statevector expectation, a classical COBYLA optimizer selects its parameters,
and an Aer sampler produces candidate bitstrings. The application decodes each
sample and checks its assignment constraints before presenting it. This is a
quantum-algorithm simulation on a classical computer, not execution on quantum
hardware. Exact enumeration and a Hungarian assignment run independently as
classical validation and baseline; they are not fallbacks for QAOA failure.
The project makes no quantum-advantage claim.

## Setup

Python 3.10 is supported. With the standard library's `venv` and pip:

```bash
python -m venv .venv
source .venv/bin/activate
.venv/bin/python -m pip install --upgrade pip
pip install -r requirements.txt
```

Or create and install with uv:

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Run

Run the tests:

```bash
.venv/bin/python -m pytest -q
```

Run the CLI demo, apply the named bridge closure, or use the 3-by-3 variant:

```bash
.venv/bin/python -m qrescue.cli
.venv/bin/python -m qrescue.cli --closure
.venv/bin/python -m qrescue.cli --three-incidents
.venv/bin/python -m qrescue.cli --reps 3 --shots 4096 --seed 7 --json
```

Start the dashboard:

```bash
streamlit run app.py
```

## Reproducibility

The CLI defaults to seed `7`; the Aer estimator, sampler, and optimizer use
explicit seeded settings. The demo graph and edge weights are fixed. Package
versions are pinned in `requirements.txt`, including Python-facing core
dependencies Qiskit 2.6.1, Qiskit Aer 0.17.2, NumPy 2.2.6, SciPy 1.15.3,
NetworkX 3.4.2, Streamlit 1.65.0, and Plotly 7.1.0. Different CPU/library
environments may still affect runtime.

## Repository layout

```text
app.py                    Streamlit dashboard entry point
qrescue/network.py        Road graph, closures, Dijkstra routes
qrescue/scenario.py       Deterministic synthetic scenarios
qrescue/qubo.py           Dispatch problem, QUBO, constraints, decoding
qrescue/ising.py          QUBO-to-Ising conversion
qrescue/qaoa.py           QAOA circuit, Aer execution, sample validation
qrescue/classical.py      Exact enumeration and Hungarian baseline
qrescue/pipeline.py       Prepared backend stages and solve report
qrescue/cli.py            Command-line report
tests/                    Unit and Aer integration tests
docs/                     Formulation, scenario, architecture, limitations
artifacts/                Recorded milestone experiments
```

See [the architecture](docs/ARCHITECTURE.md), [the scenario details](docs/DEMO_SCENARIO.md),
and [limitations](docs/LIMITATIONS.md) for more.
