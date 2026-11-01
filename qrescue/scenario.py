from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from qrescue.network import RoadNetwork


@dataclass(frozen=True)
class Ambulance:
    id: str
    name: str
    base_node: str


@dataclass(frozen=True)
class Incident:
    id: str
    node: str
    severity: int
    description: str


@dataclass(frozen=True)
class Scenario:
    network: RoadNetwork
    ambulances: tuple[Ambulance, ...]
    incidents: tuple[Incident, ...]
    coordinates: dict[str, tuple[float, float]]


NODES = (
    "Vyttila Jn",
    "Palarivattom",
    "Kaloor",
    "MG Road",
    "Edappally",
    "Kakkanad",
    "Thrippunithura",
    "Maradu",
    "Fort Kochi",
    "Thevara",
)

COORDINATES = {
    "Vyttila Jn": (0.0, 0.0),
    "Palarivattom": (1.0, 1.0),
    "Kaloor": (2.0, 2.0),
    "MG Road": (3.0, 1.5),
    "Edappally": (4.0, 2.5),
    "Kakkanad": (5.0, 2.0),
    "Thrippunithura": (6.0, 1.0),
    "Maradu": (7.0, 0.0),
    "Fort Kochi": (8.0, 1.0),
    "Thevara": (9.0, 0.0),
}

DEMO_CLOSURE = (("Thevara", "Vyttila Jn"),)


def demo_scenario(three_incidents: bool = False) -> Scenario:
    graph = nx.Graph()
    graph.add_nodes_from(NODES)
    minutes = (3, 3, 3, 3, 3, 3, 3, 12, 3, 1)
    for index, edge_minutes in enumerate(minutes):
        graph.add_edge(
            NODES[index],
            NODES[(index + 1) % len(NODES)],
            minutes=edge_minutes,
        )

    ambulances = (
        Ambulance("amb-1", "Vyttila Unit", "Vyttila Jn"),
        Ambulance("amb-2", "Kakkanad Unit", "Kakkanad"),
        Ambulance("amb-3", "Maradu Unit", "Maradu"),
    )
    incidents = [
        Incident("inc-1", "Kaloor", 3, "Critical medical emergency"),
        Incident("inc-2", "Fort Kochi", 2, "Serious road incident"),
    ]
    if three_incidents:
        incidents.append(Incident("inc-3", "Edappally", 1, "Minor injury"))

    return Scenario(
        network=RoadNetwork(graph),
        ambulances=ambulances,
        incidents=tuple(incidents),
        coordinates=dict(COORDINATES),
    )


make_demo_scenario = demo_scenario
