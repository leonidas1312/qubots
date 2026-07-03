"""NetworkX-backed MaxCut optimizer integration."""

from __future__ import annotations

import random
import time
from typing import Any

import networkx as nx

from qubots.core.optimizer import BaseOptimizer
from qubots.core.types import Result


class NetworkXMaxCutOptimizer(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.capabilities = ["blackbox", "graph"]
        self.max_passes = 20
        self.seed = 0

    @staticmethod
    def _problem_with_edges(problem: Any) -> Any:
        if hasattr(problem, "_load_edges"):
            return problem
        if hasattr(problem, "_problem"):
            loaded = problem._problem()
            if hasattr(loaded, "_load_edges"):
                return loaded
        raise TypeError(
            "NetworkXMaxCutOptimizer requires an edge-list MaxCut problem "
            "created by qubots detect/import, or another problem exposing "
            "_load_edges()."
        )

    @staticmethod
    def _build_graph(problem: Any) -> nx.Graph:
        n_nodes, edges = problem._load_edges()
        graph = nx.Graph()
        graph.add_nodes_from(range(int(n_nodes)))
        for u, v, weight in edges:
            weight = float(weight)
            if graph.has_edge(u, v):
                graph[u][v]["weight"] += weight
            else:
                graph.add_edge(int(u), int(v), weight=weight)
        return graph

    def _initial_partition(self, graph: nx.Graph) -> list[int]:
        rng = random.Random(int(self.seed))
        labels = [0] * graph.number_of_nodes()
        ordered = sorted(
            graph.nodes,
            key=lambda node: (-graph.degree(node, weight="weight"), node),
        )
        for index, node in enumerate(ordered):
            labels[int(node)] = index % 2
        if ordered and rng.random() < 0.5:
            labels[int(ordered[-1])] = 1 - labels[int(ordered[-1])]
        return labels

    @staticmethod
    def _cut_weight(graph: nx.Graph, labels: list[int]) -> float:
        total = 0.0
        for u, v, data in graph.edges(data=True):
            if labels[int(u)] != labels[int(v)]:
                total += float(data.get("weight", 1.0))
        return total

    def optimize(self, problem: Any) -> Result:
        start = time.perf_counter()
        edge_problem = self._problem_with_edges(problem)
        graph = self._build_graph(edge_problem)

        labels = self._initial_partition(graph) if graph.number_of_nodes() else []
        best_value = float(edge_problem.evaluate(labels)) if labels else 0.0
        trace = [best_value]

        ordered = sorted(
            graph.nodes,
            key=lambda node: (-graph.degree(node, weight="weight"), node),
        )
        for _ in range(int(self.max_passes)):
            improved = False
            for node in ordered:
                candidate = list(labels)
                candidate[int(node)] = 1 - candidate[int(node)]
                value = float(edge_problem.evaluate(candidate))
                if value < best_value:
                    labels = candidate
                    best_value = value
                    improved = True
            trace.append(best_value)
            if not improved:
                break

        return Result(
            best_value=float(best_value),
            best_solution=labels,
            runtime_seconds=float(time.perf_counter() - start),
            status="ok",
            trace=trace,
            metadata={
                "framework": "networkx",
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "cut_weight": self._cut_weight(graph, labels),
                "max_passes": int(self.max_passes),
            },
        )
