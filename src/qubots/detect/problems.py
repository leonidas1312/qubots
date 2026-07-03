"""Runnable problem wrappers for detected/imported data."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any

from qubots.core.milp import MILPModel
from qubots.core.problem import BaseProblem
from qubots.detect.models import ProblemSpec


def capabilities_for_family(family: str) -> list[str]:
    return {
        "milp": ["milp_sparse", "milp_dense"],
        "tsp": ["blackbox", "routing"],
        "maxcut": ["blackbox", "graph"],
        "assignment": ["blackbox", "milp_dense", "integer_only"],
        "knapsack": ["blackbox", "milp_dense", "integer_only"],
    }.get(family, ["blackbox"])


def _resolve_source(spec: ProblemSpec, base_dir: str | Path | None) -> Path:
    source = Path(spec.source_path).expanduser()
    if not source.is_absolute():
        if base_dir is None:
            source = Path.cwd() / source
        else:
            source = Path(base_dir) / source
    return source.resolve()


def _read_csv_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [
            [cell.strip() for cell in row]
            for row in csv.reader(f)
            if any(cell.strip() for cell in row)
        ]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class TSPProblem(BaseProblem):
    def __init__(self, tsp_path: str | Path) -> None:
        super().__init__()
        self.tsp_path = str(tsp_path)
        self._coords: list[tuple[float, float]] | None = None

    def _load_coords(self) -> list[tuple[float, float]]:
        if self._coords is not None:
            return self._coords

        path = Path(self.tsp_path)
        coords: list[tuple[float, float]] = []
        in_section = False
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                upper = stripped.upper()
                if upper == "NODE_COORD_SECTION":
                    in_section = True
                    continue
                if upper == "EOF":
                    break
                if not in_section:
                    continue
                parts = stripped.split()
                if len(parts) < 3:
                    continue
                coords.append((float(parts[1]), float(parts[2])))

        if not coords:
            raise ValueError(f"No NODE_COORD_SECTION coordinates found in {path}")
        self._coords = coords
        return coords

    def evaluate(self, solution: list[int]) -> float:
        coords = self._load_coords()
        n = len(coords)
        tour = [int(v) for v in list(solution)[:n]]
        penalty = 0.0
        if len(tour) != n or set(tour) != set(range(n)):
            penalty += 1_000_000.0
            seen: set[int] = set()
            fixed: list[int] = []
            for value in tour:
                if 0 <= value < n and value not in seen:
                    fixed.append(value)
                    seen.add(value)
            fixed.extend(i for i in range(n) if i not in seen)
            tour = fixed[:n]

        length = 0.0
        for i, node in enumerate(tour):
            nxt = tour[(i + 1) % n]
            x1, y1 = coords[node]
            x2, y2 = coords[nxt]
            length += math.hypot(x1 - x2, y1 - y2)
        return float(length + penalty)

    def random_solution(self) -> list[int]:
        n = len(self._load_coords())
        tour = list(range(n))
        random.shuffle(tour)
        return tour


class EdgeListMaxCutProblem(BaseProblem):
    def __init__(self, edge_path: str | Path) -> None:
        super().__init__()
        self.edge_path = str(edge_path)
        self._edges: list[tuple[int, int, float]] | None = None
        self._n_nodes = 0

    def _load_edges(self) -> tuple[int, list[tuple[int, int, float]]]:
        if self._edges is not None:
            return self._n_nodes, self._edges

        path = Path(self.edge_path)
        labels: dict[str, int] = {}
        edges: list[tuple[int, int, float]] = []

        def node_id(label: str) -> int:
            if label not in labels:
                labels[label] = len(labels)
            return labels[label]

        if path.suffix.lower() == ".csv":
            rows = _read_csv_rows(path)
            header = [cell.lower() for cell in rows[0]] if rows else []
            start = 1 if {"source", "target"}.issubset(set(header)) else 0
            for row in rows[start:]:
                if len(row) < 2:
                    continue
                weight = float(row[2]) if len(row) >= 3 and row[2] else 1.0
                edges.append((node_id(row[0]), node_id(row[1]), weight))
        else:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped or stripped.startswith(("#", "%", "//")):
                        continue
                    parts = stripped.split()
                    if parts[0].lower() in {"c", "p"}:
                        continue
                    if parts[0].lower() == "e":
                        parts = parts[1:]
                    if len(parts) < 2:
                        continue
                    weight = float(parts[2]) if len(parts) >= 3 else 1.0
                    edges.append((node_id(parts[0]), node_id(parts[1]), weight))

        self._n_nodes = len(labels)
        self._edges = edges
        return self._n_nodes, edges

    def evaluate(self, solution: list[int]) -> float:
        n_nodes, edges = self._load_edges()
        labels = [int(v) & 1 for v in list(solution)[:n_nodes]]
        labels.extend([0] * max(0, n_nodes - len(labels)))
        cut_weight = sum(weight for u, v, weight in edges if labels[u] != labels[v])
        return -float(cut_weight)

    def random_solution(self) -> list[int]:
        n_nodes, _ = self._load_edges()
        return [random.randint(0, 1) for _ in range(n_nodes)]


class AssignmentProblem(BaseProblem):
    def __init__(self, matrix_path: str | Path) -> None:
        super().__init__()
        self.matrix_path = str(matrix_path)
        self._matrix: list[list[float]] | None = None

    def _load_matrix(self) -> list[list[float]]:
        if self._matrix is not None:
            return self._matrix
        path = Path(self.matrix_path)
        if path.suffix.lower() == ".json":
            raw = _load_json(path)
            if isinstance(raw, dict):
                raw = raw.get("cost_matrix") or raw.get("matrix") or raw.get("costs")
            matrix = [[float(value) for value in row] for row in raw]
        else:
            matrix = [[float(cell) for cell in row] for row in _read_csv_rows(path)]
        if not matrix or any(len(row) != len(matrix[0]) for row in matrix):
            raise ValueError(f"Invalid cost matrix: {path}")
        self._matrix = matrix
        return matrix

    def evaluate(self, solution: list[int]) -> float:
        matrix = self._load_matrix()
        n_workers = len(matrix)
        n_tasks = len(matrix[0])
        assignment = [int(v) for v in list(solution)[:n_workers]]
        penalty = 0.0
        seen: set[int] = set()
        cost = 0.0
        for worker, task in enumerate(assignment):
            if task < 0 or task >= n_tasks or task in seen:
                penalty += 1_000_000.0
                continue
            seen.add(task)
            cost += matrix[worker][task]
        missing = max(0, min(n_workers, n_tasks) - len(seen))
        return float(cost + penalty + missing * 1_000_000.0)

    def random_solution(self) -> list[int]:
        matrix = self._load_matrix()
        n_workers = len(matrix)
        n_tasks = len(matrix[0])
        tasks = list(range(n_tasks))
        random.shuffle(tasks)
        if n_tasks < n_workers:
            tasks.extend(random.choice(range(n_tasks)) for _ in range(n_workers - n_tasks))
        return tasks[:n_workers]

    def as_milp(self) -> MILPModel:
        matrix = self._load_matrix()
        n_workers = len(matrix)
        n_tasks = len(matrix[0])
        n_vars = n_workers * n_tasks
        c = [matrix[i][j] for i in range(n_workers) for j in range(n_tasks)]
        var_names = [f"x_{i}_{j}" for i in range(n_workers) for j in range(n_tasks)]
        A_ub: list[list[float]] = []
        b_ub: list[float] = []
        A_eq: list[list[float]] = []
        b_eq: list[float] = []

        for i in range(n_workers):
            row = [0.0] * n_vars
            for j in range(n_tasks):
                row[i * n_tasks + j] = 1.0
            if n_workers <= n_tasks:
                A_eq.append(row)
                b_eq.append(1.0)
            else:
                A_ub.append(row)
                b_ub.append(1.0)

        for j in range(n_tasks):
            row = [0.0] * n_vars
            for i in range(n_workers):
                row[i * n_tasks + j] = 1.0
            if n_tasks <= n_workers:
                A_eq.append(row)
                b_eq.append(1.0)
            else:
                A_ub.append(row)
                b_ub.append(1.0)

        return MILPModel(
            sense="min",
            c=c,
            var_names=var_names,
            integrality=[True] * n_vars,
            lb=[0.0] * n_vars,
            ub=[1.0] * n_vars,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
        )


class KnapsackDataProblem(BaseProblem):
    def __init__(self, item_path: str | Path) -> None:
        super().__init__()
        self.item_path = str(item_path)
        self._items: tuple[list[float], list[float], float] | None = None

    def _load_items(self) -> tuple[list[float], list[float], float]:
        if self._items is not None:
            return self._items
        path = Path(self.item_path)
        values: list[float] = []
        weights: list[float] = []
        capacity: float | None = None

        if path.suffix.lower() == ".json":
            raw = _load_json(path)
            capacity = float(raw["capacity"])
            for item in raw["items"]:
                values.append(float(item["value"]))
                weights.append(float(item["weight"]))
        else:
            rows = _read_csv_rows(path)
            header = [cell.lower().strip() for cell in rows[0]]
            value_col = "value" if "value" in header else "profit"
            weight_col = "weight" if "weight" in header else "weights"
            value_idx = header.index(value_col)
            weight_idx = header.index(weight_col)
            capacity_idx = header.index("capacity")
            for row in rows[1:]:
                values.append(float(row[value_idx]))
                weights.append(float(row[weight_idx]))
                if capacity is None:
                    capacity = float(row[capacity_idx])

        if capacity is None:
            raise ValueError(f"Knapsack capacity not found in {path}")
        self._items = (values, weights, float(capacity))
        return self._items

    def evaluate(self, solution: list[int]) -> float:
        values, weights, capacity = self._load_items()
        x = [1 if int(v) else 0 for v in list(solution)[: len(values)]]
        x.extend([0] * max(0, len(values) - len(x)))
        total_value = sum(value * xi for value, xi in zip(values, x))
        total_weight = sum(weight * xi for weight, xi in zip(weights, x))
        penalty = max(0.0, total_weight - capacity) * 1_000_000.0
        return float(-total_value + penalty)

    def random_solution(self) -> list[int]:
        values, _, _ = self._load_items()
        return [random.randint(0, 1) for _ in values]

    def as_milp(self) -> MILPModel:
        values, weights, capacity = self._load_items()
        return MILPModel(
            sense="max",
            c=list(values),
            var_names=[f"x_{i}" for i in range(len(values))],
            integrality=[True] * len(values),
            lb=[0.0] * len(values),
            ub=[1.0] * len(values),
            A_ub=[list(weights)],
            b_ub=[float(capacity)],
        )


def problem_from_spec(
    spec_or_path: ProblemSpec | str | Path,
    *,
    base_dir: str | Path | None = None,
) -> BaseProblem:
    if isinstance(spec_or_path, ProblemSpec):
        spec = spec_or_path
    else:
        path = Path(spec_or_path)
        spec = ProblemSpec.from_yaml(path)
        if base_dir is None:
            base_dir = path.parent

    source = _resolve_source(spec, base_dir)
    if spec.family == "milp":
        from qubots.contrib.mps import MPSProblem

        problem = MPSProblem(source, sparse=bool(spec.parameters.get("sparse", True)))
    elif spec.family == "tsp":
        problem = TSPProblem(source)
    elif spec.family == "maxcut":
        problem = EdgeListMaxCutProblem(source)
    elif spec.family == "assignment":
        problem = AssignmentProblem(source)
    elif spec.family == "knapsack":
        problem = KnapsackDataProblem(source)
    else:
        raise ValueError(f"ProblemSpec family {spec.family!r} is not runnable")

    setattr(problem, "_qubots_problem_spec", spec.to_dict())
    setattr(problem, "_qubots_detector", spec.detector)
    setattr(problem, "_qubots_data_hash", spec.data_hash)
    setattr(problem, "capabilities", capabilities_for_family(spec.family))
    setattr(problem, "problem_family", spec.family)
    return problem


class ProblemSpecBackedProblem(BaseProblem):
    def __init__(self, spec_path: str | Path) -> None:
        super().__init__()
        self.spec_path = str(spec_path)
        self._loaded_problem: BaseProblem | None = None

    def _problem(self) -> BaseProblem:
        if self._loaded_problem is None:
            spec_path = Path(self.spec_path).resolve()
            self._loaded_problem = problem_from_spec(spec_path, base_dir=spec_path.parent)
            if self.parameters:
                self._loaded_problem.set_parameters(**self.parameters)
        return self._loaded_problem

    def set_parameters(self, **kwargs: Any) -> None:
        super().set_parameters(**kwargs)
        if self._loaded_problem is not None:
            self._loaded_problem.set_parameters(**kwargs)

    def evaluate(self, solution: Any) -> float:
        return self._problem().evaluate(solution)

    def random_solution(self) -> Any:
        return self._problem().random_solution()

    def as_milp(self) -> Any:
        problem = self._problem()
        if not hasattr(problem, "as_milp"):
            raise NotImplementedError(
                f"{type(problem).__name__} does not expose a MILP form"
            )
        return problem.as_milp()
