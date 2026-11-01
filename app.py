from __future__ import annotations

import platform
from datetime import datetime
from importlib.metadata import version

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from qrescue.classical import BaselineResult, ExactResult
from qrescue.ising import qubo_to_ising
from qrescue.qaoa import QAOAResult, run_qaoa
from qrescue import pipeline
from qrescue.qubo import InfeasibleInstance, check_constraints
from qrescue.pipeline import Prepared
from qrescue.scenario import DEMO_CLOSURE, Scenario, demo_scenario

SEVERITY_LABEL = {3: "Critical", 2: "Serious", 1: "Minor"}
SOURCE_QAOA = "QAOA · quantum circuit on Qiskit Aer simulator"
SOURCE_EXACT = "Exact · classical brute-force validation"
SOURCE_BASELINE = "Baseline · classical Hungarian algorithm"
SOURCE_ROUTING = "Routing · classical Dijkstra"

COLOR_QAOA = "#6A3FB5"
COLOR_EXACT = "#1F4E79"
COLOR_EDGE = "#B8C2CC"
COLOR_CLOSED = "#C0392B"
COLOR_AMBULANCE = "#1F7A4D"
COLOR_INCIDENT = {3: "#C0392B", 2: "#D9822B", 1: "#C9A227"}

st.set_page_config(page_title="Q-Rescue", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1400px;}
    .qr-tag {display:inline-block; padding:2px 10px; border-radius:999px;
             font-size:0.75rem; font-weight:600; letter-spacing:0.02em;
             margin-bottom:0.4rem;}
    .qr-quantum {background:#EEE7FA; color:#4B2A8A;}
    .qr-classical {background:#E3ECF5; color:#1F4E79;}
    .qr-validation {background:#E6F2EC; color:#1F6B45;}
    .qr-muted {color:#5B6573; font-size:0.9rem;}
    div[data-testid="stMetricValue"] {font-size:1.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def tag(kind: str, text: str) -> None:
    st.markdown(f'<span class="qr-tag qr-{kind}">{text}</span>', unsafe_allow_html=True)


def edge_key(u: str, v: str) -> tuple[str, str]:
    return tuple(sorted((u, v)))


def edge_label(scenario: Scenario, u: str, v: str) -> str:
    data = scenario.network.graph.edges[u, v]
    name = data.get("name")
    base = f"{u} – {v} ({data['minutes']} min)"
    return f"{name}: {base}" if name else base


@st.cache_resource(show_spinner=False)
def load_scenario(three_incidents: bool) -> Scenario:
    return demo_scenario(three_incidents=three_incidents)


@st.cache_resource(show_spinner=False)
def prepare(three_incidents: bool, closure: tuple[tuple[str, str], ...]) -> Prepared:
    return pipeline.prepare(load_scenario(three_incidents), closure)


@st.cache_resource(show_spinner=False)
def classical(
    three_incidents: bool, closure: tuple[tuple[str, str], ...]
) -> tuple[ExactResult, BaselineResult]:
    return pipeline.run_classical(prepare(three_incidents, closure))


def route_rows(prepared: Prepared, scenario: Scenario, chosen) -> list[dict]:
    return [
        {
            "ambulance": r["ambulance_id"],
            "incident": r["incident_id"],
            "nodes": r["nodes"],
            "minutes": r["travel_minutes"],
        }
        for r in pipeline.routes_for(prepared.network, scenario, chosen)
    ]


def network_figure(
    scenario: Scenario,
    closure: tuple[tuple[str, str], ...],
    qaoa_routes: list[dict] | None,
    exact_routes: list[dict],
) -> go.Figure:
    coords = scenario.coordinates
    closed = {edge_key(*e) for e in closure}
    fig = go.Figure()

    for u, v, data in scenario.network.graph.edges(data=True):
        is_closed = edge_key(u, v) in closed
        (x0, y0), (x1, y1) = coords[u], coords[v]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(
                    color=COLOR_CLOSED if is_closed else COLOR_EDGE,
                    width=4 if is_closed else 3,
                    dash="dot" if is_closed else "solid",
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        label = f"{data['minutes']}"
        if is_closed:
            label = "CLOSED"
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2,
            text=label,
            showarrow=False,
            font=dict(size=10, color=COLOR_CLOSED if is_closed else "#5B6573"),
            bgcolor="rgba(255,255,255,0.85)",
        )

    def add_routes(routes: list[dict], color: str, name: str, dash: str, width: int, offset: float):
        first = True
        for row in routes:
            nodes = row["nodes"] or []
            xs = [coords[n][0] + offset for n in nodes]
            ys = [coords[n][1] + offset for n in nodes]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=color, width=width, dash=dash),
                    name=name,
                    legendgroup=name,
                    showlegend=first,
                    hovertemplate=f"{name}<br>{row['ambulance']} → {row['incident']}"
                    f"<br>{row['minutes']:.0f} min<extra></extra>",
                )
            )
            first = False

    add_routes(exact_routes, COLOR_EXACT, "Exact optimum route", "dash", 3, -0.06)
    if qaoa_routes:
        add_routes(qaoa_routes, COLOR_QAOA, "QAOA candidate route", "solid", 5, 0.06)

    plain = [n for n in scenario.network.graph.nodes]
    fig.add_trace(
        go.Scatter(
            x=[coords[n][0] for n in plain],
            y=[coords[n][1] for n in plain],
            mode="markers",
            marker=dict(size=9, color="#FFFFFF", line=dict(color="#5B6573", width=1.5)),
            text=plain,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    for n in plain:
        fig.add_annotation(
            x=coords[n][0], y=coords[n][1], text=n, showarrow=False, yshift=22,
            font=dict(size=11, color="#1B2430"), bgcolor="rgba(255,255,255,0.8)",
        )
    fig.add_trace(
        go.Scatter(
            x=[coords[a.base_node][0] for a in scenario.ambulances],
            y=[coords[a.base_node][1] for a in scenario.ambulances],
            mode="markers",
            marker=dict(size=18, color=COLOR_AMBULANCE, symbol="square", line=dict(color="white", width=2)),
            name="Ambulance base",
            text=[f"{a.id} · {a.name}" for a in scenario.ambulances],
            hovertemplate="%{text}<extra></extra>",
        )
    )
    for severity in sorted({i.severity for i in scenario.incidents}, reverse=True):
        group = [i for i in scenario.incidents if i.severity == severity]
        fig.add_trace(
            go.Scatter(
                x=[coords[i.node][0] for i in group],
                y=[coords[i.node][1] for i in group],
                mode="markers",
                marker=dict(size=20, color=COLOR_INCIDENT[severity], symbol="diamond", line=dict(color="white", width=2)),
                name=f"Incident · {SEVERITY_LABEL[severity]}",
                text=[f"{i.id} · {i.description}" for i in group],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#FAFBFC",
        paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    )
    return fig


def assignment_text(chosen) -> str:
    return ", ".join(f"{a} → {i}" for a, i in chosen) if chosen else "—"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Scenario")
    variant = st.radio(
        "Instance",
        ["3 ambulances × 2 incidents (6 qubits)", "3 ambulances × 3 incidents (9 qubits)"],
        index=0,
    )
    three_incidents = variant.startswith("3 ambulances × 3")
    scenario = load_scenario(three_incidents)

    st.markdown("### Road closure")
    edges = sorted(edge_key(u, v) for u, v in scenario.network.graph.edges)
    labels = {e: edge_label(scenario, *e) for e in edges}
    default_closure = [edge_key(*e) for e in DEMO_CLOSURE]
    if "closure" not in st.session_state:
        st.session_state.closure = []
    c1, c2 = st.columns(2)
    if c1.button("Demo closure", use_container_width=True):
        st.session_state.closure = default_closure
    if c2.button("Open all roads", use_container_width=True):
        st.session_state.closure = []
    st.multiselect(
        "Closed roads",
        options=edges,
        format_func=lambda e: labels[e],
        key="closure",
    )
    closure = tuple(sorted(st.session_state.closure))

    st.markdown("### QAOA settings")
    reps = st.select_slider("Layers p", options=[1, 2, 3], value=2)
    shots = st.select_slider("Shots", options=[1024, 2048, 4096, 8192], value=4096)
    seed = int(st.number_input("Seed", min_value=0, max_value=10_000, value=7, step=1))
    maxiter = st.select_slider("COBYLA max iterations per start", options=[20, 40, 60, 100], value=40)
    run_clicked = st.button("Run quantum optimisation", type="primary", use_container_width=True)

config = (three_incidents, closure, reps, shots, seed, maxiter)

# ---------------------------------------------------------------- header
st.markdown("## Q-Rescue")
st.markdown(
    '<p class="qr-muted">Hybrid quantum-classical ambulance dispatch. Classical Dijkstra '
    "computes travel times on a synthetic road network; QAOA, simulated with Qiskit Aer, "
    "searches for the ambulance-to-incident assignment; classical solvers validate the result.</p>",
    unsafe_allow_html=True,
)

try:
    prepared = prepare(three_incidents, closure)
except (InfeasibleInstance, ValueError) as error:
    st.error(
        f"This closure makes the dispatch problem infeasible: {error}. "
        "No optimisation was run. Reopen a road to continue."
    )
    st.stop()

problem, qubo = prepared.problem, prepared.qubo
exact, baseline = classical(three_incidents, closure)

if run_clicked:
    with st.status("Running QAOA on the Qiskit Aer simulator…", expanded=False) as status:
        result = run_qaoa(problem, reps=reps, shots=shots, seed=seed, maxiter=maxiter, optimal_bitstring=exact.bitstring)
        status.update(label=f"QAOA finished in {result.runtime_seconds:.2f} s", state="complete")
    runs = st.session_state.get("run_counter", 0) + 1
    st.session_state.run_counter = runs
    st.session_state.qaoa_run = {
        "config": config,
        "result": result,
        "finished_at": datetime.now().strftime("%H:%M:%S"),
        "number": runs,
    }

run = st.session_state.get("qaoa_run")
current: QAOAResult | None = run["result"] if run and run["config"] == config else None
stale = run is not None and current is None

qaoa_routes = route_rows(prepared, scenario, current.best_valid.assignment) if current and current.best_valid else None
exact_routes = route_rows(prepared, scenario, exact.assignment)

m = st.columns(5)
m[0].metric("Ambulances", len(problem.ambulances))
m[1].metric("Incidents", len(problem.incidents))
m[2].metric("Closed roads", len(closure))
m[3].metric("Qubits", problem.n_qubits)
m[4].metric("Search space", f"{2 ** problem.n_qubits} bitstrings")

tab_overview, tab_qubo, tab_qaoa, tab_validate, tab_about = st.tabs(
    ["Scenario", "QUBO formulation", "QAOA run", "Validation", "How it works"]
)

# ---------------------------------------------------------------- scenario
with tab_overview:
    left, right = st.columns([3, 2])
    with left:
        tag("classical", SOURCE_ROUTING)
        st.plotly_chart(
            network_figure(scenario, closure, qaoa_routes, exact_routes),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption(
            "Edge labels are travel minutes. Dashed blue: exact optimum (classical validation). "
            "Solid purple: best valid QAOA candidate from the latest run with these settings."
        )
    with right:
        st.markdown("#### Ambulances")
        st.dataframe(
            pd.DataFrame([{"ID": a.id, "Unit": a.name, "Base": a.base_node} for a in problem.ambulances]),
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("#### Incidents")
        st.dataframe(
            pd.DataFrame(
                [
                    {"ID": i.id, "Location": i.node, "Severity": f"{SEVERITY_LABEL[i.severity]} (w={i.severity})", "Description": i.description}
                    for i in problem.incidents
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("#### Dispatch cost matrix")
        tag("classical", SOURCE_ROUTING)
        cost = pd.DataFrame(
            [
                [
                    "unreachable" if not np.isfinite(problem.c[a, i])
                    else f"{problem.c[a, i] / problem.incidents[i].severity:.0f} min × {problem.incidents[i].severity} = {problem.c[a, i]:.0f}"
                    for i in range(len(problem.incidents))
                ]
                for a in range(len(problem.ambulances))
            ],
            index=[a.id for a in problem.ambulances],
            columns=[i.id for i in problem.incidents],
        )
        st.dataframe(cost, use_container_width=True)
        st.caption(
            "Cost c[a,i] = severity weight × shortest travel time. Dijkstra recomputes every entry "
            "when roads are closed; this matrix is the only input the quantum step receives."
        )

# ---------------------------------------------------------------- qubo
with tab_qubo:
    st.markdown("#### Binary variables")
    st.markdown(
        "One qubit per reachable pair: x[a,i] = 1 means ambulance *a* is sent to incident *i*."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {"Qubit": k, "Variable": f"x[{problem.ambulances[a].id}, {problem.incidents[i].id}]",
                 "Cost c (weighted min)": f"{problem.c[a, i]:.0f}", "Normalised ĉ": f"{problem.chat[a, i]:.4f}"}
                for k, (a, i) in enumerate(problem.K)
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.markdown("#### Energy function")
    st.latex(
        r"E(x)=\underbrace{\sum_{(a,i)} \hat c_{ai}\,x_{ai}}_{\text{dispatch cost}}"
        r"+P\underbrace{\sum_i\Big(1-\sum_a x_{ai}\Big)^2}_{\text{each incident: exactly one}}"
        r"+P\underbrace{\sum_a\sum_{i<j}x_{ai}x_{aj}}_{\text{each ambulance: at most one}}"
    )
    st.latex(
        r"\hat c_{ai}=\frac{c_{ai}}{c_{\max}},\qquad P = 1+\sum_i \max_a \hat c_{ai}"
        + rf"= {problem.P:.4f},\qquad c_{{\max}} = {problem.c_max:.0f}"
    )
    st.markdown(
        "Any constraint violation adds at least P, while every valid assignment costs less than P, "
        "so the lowest-energy bitstring is the best valid dispatch. Substituting "
        r"$x_k=(1-Z_k)/2$ turns E into a diagonal Ising Hamiltonian, which is the QAOA cost operator."
    )
    operator, offset = qubo_to_ising(qubo, problem.n_qubits)
    c1, c2, c3 = st.columns(3)
    c1.metric("Linear QUBO terms", len(qubo.linear))
    c2.metric("Quadratic QUBO terms", len(qubo.quadratic))
    c3.metric("Ising Pauli terms", len(operator))
    with st.expander("Ising Hamiltonian terms"):
        st.dataframe(
            pd.DataFrame(
                [{"Pauli (qubit 0 rightmost)": str(p), "Coefficient": float(np.real(c))} for p, c in operator.to_list()]
                + [{"Pauli (qubit 0 rightmost)": "offset", "Coefficient": offset}]
            ),
            hide_index=True,
            use_container_width=True,
        )

# ---------------------------------------------------------------- qaoa
with tab_qaoa:
    tag("quantum", SOURCE_QAOA)
    if current is None:
        if stale:
            st.info(
                "The settings changed since the last QAOA run, so its result is not shown. "
                "Press Run quantum optimisation to execute the circuit for the current settings."
            )
        else:
            st.info("QAOA has not been run for these settings yet. Press Run quantum optimisation in the sidebar.")
    else:
        st.markdown(
            f'<p class="qr-muted">Run #{run["number"]} executed live at {run["finished_at"]} on '
            f"{current.backend_name} · {current.backend_method}. Results are never cached between runs.</p>",
            unsafe_allow_html=True,
        )
        if current.status == "error":
            st.error(f"QAOA execution failed: {current.message}. No classical result has been substituted.")
        elif current.status == "no_valid_sample":
            st.warning(
                "QAOA produced no valid assignment in any sample. No classical result has been substituted; "
                "try more layers, more shots or another seed."
            )
        else:
            best = current.best_valid
            k = st.columns(4)
            k[0].metric("Best valid candidate", f"{best.objective_minutes:.0f} weighted min")
            gap = best.objective_minutes - exact.objective_minutes
            k[1].metric("Gap to exact optimum", f"{gap:+.0f} min", delta_color="off")
            k[2].metric("Valid samples", f"{current.valid_fraction:.1%}")
            k[3].metric(
                "Samples equal to optimum",
                f"{current.prob_of_exact_optimum:.1%}" if current.prob_of_exact_optimum is not None else "—",
            )
            st.markdown(f"**Assignment:** {assignment_text(best.assignment)}  ·  bitstring `{best.bitstring}`")
            report = check_constraints(problem, best.x)
            checks = [
                {"Constraint": f"{inc} receives exactly one ambulance", "Count": n, "Satisfied": "Yes" if n == 1 else "No"}
                for inc, n in report.incident_counts.items()
            ] + [
                {"Constraint": f"{amb} serves at most one incident", "Count": n, "Satisfied": "Yes" if n <= 1 else "No"}
                for amb, n in report.ambulance_counts.items()
            ]
            st.markdown("#### Constraint validation of the QAOA candidate")
            st.dataframe(pd.DataFrame(checks), hide_index=True, use_container_width=True)

        if current.status != "error":
            st.markdown("#### Measurement distribution (top bitstrings)")
            top = pd.DataFrame(list(current.top_k))
            fig = go.Figure(
                go.Bar(
                    x=top["bitstring"],
                    y=top["probability"],
                    marker_color=[COLOR_QAOA if v else "#C7CDD4" for v in top["valid"]],
                    hovertemplate="%{x}<br>p=%{y:.3f}<extra></extra>",
                )
            )
            fig.update_layout(
                height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Probability",
                xaxis_title="Bitstring (qubit 0 rightmost)", xaxis_type="category", plot_bgcolor="#FAFBFC",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.caption("Purple bars satisfy every constraint; grey bars violate at least one.")

        st.markdown("#### Circuit and optimiser")
        params = current.optimal_params or ()
        p = current.reps
        st.dataframe(
            pd.DataFrame(
                [
                    {"Setting": "Qubits", "Value": current.n_qubits},
                    {"Setting": "QAOA layers p", "Value": p},
                    {"Setting": "Transpiled depth", "Value": current.circuit_depth},
                    {"Setting": "Two-qubit gates", "Value": current.two_qubit_gate_count},
                    {"Setting": "Optimiser", "Value": "COBYLA, 3 deterministic starts"},
                    {"Setting": "Objective evaluations", "Value": current.optimiser_evals},
                    {"Setting": "Final ⟨H⟩", "Value": f"{current.final_expected_energy:.4f}" if current.final_expected_energy is not None else "—"},
                    {"Setting": "γ", "Value": ", ".join(f"{v:.4f}" for v in params[:p]) or "—"},
                    {"Setting": "β", "Value": ", ".join(f"{v:.4f}" for v in params[p:]) or "—"},
                    {"Setting": "Shots / seed", "Value": f"{current.shots} / {current.seed}"},
                    {"Setting": "Runtime", "Value": f"{current.runtime_seconds:.2f} s"},
                ]
            ).astype(str),
            hide_index=True,
            use_container_width=True,
        )

# ---------------------------------------------------------------- validation
with tab_validate:
    rows = []
    if current is not None and current.best_valid is not None:
        rows.append(
            {"Method": "QAOA best valid sample", "Source": SOURCE_QAOA,
             "Assignment": assignment_text(current.best_valid.assignment),
             "Weighted minutes": current.best_valid.objective_minutes,
             "Valid": "Yes", "Runtime (s)": round(current.runtime_seconds, 4)}
        )
    elif current is not None:
        rows.append(
            {"Method": "QAOA best valid sample", "Source": SOURCE_QAOA,
             "Assignment": "no valid sample" if current.status == "no_valid_sample" else "error",
             "Weighted minutes": None, "Valid": "—", "Runtime (s)": round(current.runtime_seconds, 4)}
        )
    rows += [
        {"Method": "Exact optimum (brute force over 2^n)", "Source": SOURCE_EXACT,
         "Assignment": assignment_text(exact.assignment), "Weighted minutes": exact.objective_minutes,
         "Valid": "Yes" if check_constraints(problem, exact.x).valid else "No", "Runtime (s)": round(exact.runtime_seconds, 4)},
        {"Method": "Hungarian algorithm", "Source": SOURCE_BASELINE,
         "Assignment": assignment_text(baseline.assignment), "Weighted minutes": baseline.objective_minutes,
         "Valid": "Yes" if check_constraints(problem, baseline.x).valid else "No", "Runtime (s)": round(baseline.runtime_seconds, 4)},
    ]
    tag("validation", "Classical validation, computed independently of QAOA")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if current is None:
        st.caption("Run QAOA to add the quantum candidate to this comparison.")
    st.markdown(
        f"Dijkstra routing and problem construction took {(prepared.routing_time_seconds + prepared.qubo_time_seconds) * 1000:.1f} ms. "
        "The classical methods are much faster at this size; QAOA is shown as a working hybrid "
        "pipeline, not as a speed-up."
    )
    st.markdown("#### Routes for the exact optimum")
    tag("classical", SOURCE_ROUTING)
    st.dataframe(
        pd.DataFrame(
            [{"Ambulance": r["ambulance"], "Incident": r["incident"], "Route": " → ".join(r["nodes"]), "Minutes": r["minutes"]} for r in exact_routes]
        ),
        hide_index=True,
        use_container_width=True,
    )

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown("#### What the quantum component actually does")
    st.markdown(
        "- Classical Dijkstra computes the shortest travel time from every ambulance base to every incident.\n"
        "- Those times, weighted by severity, become the coefficients of a QUBO whose lowest-energy "
        "bitstring is the best valid assignment.\n"
        "- The QUBO is converted to an Ising Hamiltonian and used as the cost layer of a QAOA circuit. "
        "A classical optimiser (COBYLA) tunes the circuit angles γ and β to lower the expected energy, "
        "evaluated on the Qiskit Aer simulator.\n"
        "- The tuned circuit is sampled; each measured bitstring is decoded into an assignment, checked "
        "against the constraints and scored. The best valid sample is the QAOA candidate.\n"
        "- QAOA only chooses which ambulance goes where. It does not compute routes."
    )
    st.markdown("#### Limitations")
    st.markdown(
        "- Runs on a classical simulator, not quantum hardware. No quantum advantage is claimed; "
        "the Hungarian algorithm solves this problem exactly in microseconds.\n"
        "- Instances are tiny (6–9 qubits) so the exact optimum can be brute-forced for validation.\n"
        "- The standard X mixer explores invalid bitstrings, so a large share of samples violate constraints.\n"
        "- The objective minimises total severity-weighted response time, not the worst single response.\n"
        "- The road network and incidents are synthetic and static; there is no live traffic.\n"
        "- Assignment is solved, not multi-stop vehicle routing."
    )
    st.markdown("#### Reproducibility")
    st.dataframe(
        pd.DataFrame(
            [
                {"Item": "Python", "Value": platform.python_version()},
                {"Item": "qiskit", "Value": version("qiskit")},
                {"Item": "qiskit-aer", "Value": version("qiskit-aer")},
                {"Item": "Seed (simulator, sampler, transpiler, optimiser starts)", "Value": str(seed)},
                {"Item": "Scenario", "Value": "deterministic synthetic network, see docs/DEMO_SCENARIO.md"},
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
