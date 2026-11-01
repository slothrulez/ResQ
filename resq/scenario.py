from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from resq.network import RoadNetwork


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
    "Thevara",
    "Edappally",
    "Kakkanad",
    "Thrippunithura",
    "Maradu",
    "Fort Kochi",
    "Willingdon Island",
    "Aluva",
)

COORDINATES = {
    "Vyttila Jn": (3.0, 1.0),
    "Palarivattom": (1.0, 3.0),
    "Kaloor": (3.0, 4.0),
    "MG Road": (2.5, 3.0),
    "Thevara": (3.0, 2.0),
    "Edappally": (1.0, 2.0),
    "Kakkanad": (5.0, 4.0),
    "Thrippunithura": (7.0, 4.0),
    "Maradu": (5.0, 1.0),
    "Fort Kochi": (7.0, 2.0),
    "Willingdon Island": (5.0, 2.0),
    "Aluva": (6.0, 0.0),
}

WEST_BANK = frozenset(
    ("Vyttila Jn", "Palarivattom", "Kaloor", "MG Road", "Thevara", "Edappally")
)
EAST_BANK = frozenset(
    (
        "Kakkanad",
        "Thrippunithura",
        "Maradu",
        "Fort Kochi",
        "Willingdon Island",
        "Aluva",
    )
)

DEMO_CLOSURE = (("Vyttila Jn", "Maradu"),)


def demo_scenario(three_incidents: bool = False) -> Scenario:
    graph = nx.Graph()
    graph.add_nodes_from(NODES)
    roads = (
        ("Vyttila Jn", "Palarivattom", 4, None),
        ("Palarivattom", "Kaloor", 14, None),
        ("Kaloor", "MG Road", 6, None),
        ("MG Road", "Thevara", 9, None),
        ("Thevara", "Vyttila Jn", 9, None),
        ("Palarivattom", "Edappally", 12, None),
        ("Edappally", "Thevara", 4, None),
        ("Edappally", "MG Road", 12, None),
        ("Kakkanad", "Thrippunithura", 4, None),
        ("Thrippunithura", "Fort Kochi", 11, None),
        ("Fort Kochi", "Willingdon Island", 6, None),
        ("Willingdon Island", "Aluva", 12, None),
        ("Aluva", "Maradu", 6, None),
        ("Maradu", "Kakkanad", 9, None),
        ("Maradu", "Fort Kochi", 5, None),
        ("Vyttila Jn", "Maradu", 2, "Venduruthy Bridge"),
        ("Thevara", "Willingdon Island", 9, "Thevara Bridge"),
        ("Kaloor", "Kakkanad", 13, "Kaloor-Kakkanad Bridge"),
    )
    for u, v, minutes, name in roads:
        edge_data = {"minutes": minutes}
        if name is not None:
            edge_data["name"] = name
        graph.add_edge(u, v, **edge_data)

    ambulances = (
        Ambulance("amb-1", "Kaloor Unit", "Kaloor"),
        Ambulance("amb-2", "Vyttila Unit", "Vyttila Jn"),
        Ambulance("amb-3", "Edappally Unit", "Edappally"),
    )
    incidents = [
        Incident("inc-1", "MG Road", 3, "Critical medical emergency"),
        Incident("inc-2", "Aluva", 2, "Serious road incident"),
    ]
    if three_incidents:
        incidents.append(Incident("inc-3", "Maradu", 1, "Minor injury"))

    return Scenario(
        network=RoadNetwork(graph),
        ambulances=ambulances,
        incidents=tuple(incidents),
        coordinates=dict(COORDINATES),
    )


make_demo_scenario = demo_scenario
