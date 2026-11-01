from __future__ import annotations

import math
from collections.abc import Iterable

import networkx as nx


class RoadNetwork:
    def __init__(self, graph: nx.Graph):
        if graph.is_directed():
            raise ValueError("RoadNetwork requires an undirected graph")
        self.graph = graph.copy()
        for u, v, data in self.graph.edges(data=True):
            minutes = data.get("minutes")
            if minutes is None or not math.isfinite(float(minutes)) or float(minutes) <= 0:
                raise ValueError(f"edge {(u, v)!r} must have positive finite minutes")

    def closed(self, edges: Iterable[tuple[str, str]]) -> RoadNetwork:
        graph = self.graph.copy()
        closure_edges = tuple(edges)
        for u, v in closure_edges:
            if not self.graph.has_edge(u, v):
                raise ValueError(f"closure edge {(u, v)!r} does not exist")
        graph.remove_edges_from(closure_edges)
        return RoadNetwork(graph)

    def travel_time(self, u: str, v: str) -> float:
        try:
            return float(nx.dijkstra_path_length(self.graph, u, v, weight="minutes"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return math.inf

    def route(self, u: str, v: str) -> list[str] | None:
        try:
            return list(nx.dijkstra_path(self.graph, u, v, weight="minutes"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
