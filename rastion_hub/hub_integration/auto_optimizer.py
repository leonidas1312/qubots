# rastion_hub/hub_integration/auto_optimizer.py

import os
import sys
import tempfile
import requests
import zipfile
import importlib
from typing import Optional

class AutoOptimizer:
    @classmethod
    def from_hub(
        cls,
        solver_id: str,
        hub_api_url: str = "http://localhost:8000",
        override_params: dict = None
    ):
        """
        :param solver_id: e.g. "genetic-tsp-1"
        :param hub_api_url: base URL of your FastAPI hub service
        :param override_params: optional dict to override default_params
        """
        # 1) Get solver metadata
        url = f"{hub_api_url}/api/solvers/{solver_id}"
        resp = requests.get(url)
        if resp.status_code == 404:
            raise ValueError(f"Solver '{solver_id}' not found.")
        resp.raise_for_status()
        solver_data = resp.json()

        entry_point = solver_data["entry_point"]
        if ":" not in entry_point:
            raise ValueError("Invalid entry_point in solver data (expect 'module:Class').")

        code_url = solver_data.get("code_url")
        local_dir = None
        if code_url:
            local_dir = cls._download_and_unzip(code_url, solver_id)

        # 2) Dynamic import
        module_path, class_name = entry_point.split(":")
        if local_dir and local_dir not in sys.path:
            sys.path.insert(0, local_dir)

        module = importlib.import_module(module_path)
        SolverClass = getattr(module, class_name)

        # 3) Merge default_params + override_params
        default_params = solver_data.get("default_params") or {}
        if override_params:
            final_params = {**default_params, **override_params}
        else:
            final_params = default_params

        solver_instance = SolverClass(**final_params)
        return solver_instance

    @staticmethod
    def _download_and_unzip(zip_url: str, solver_id: str) -> str:
        """
        Download the solver zip from 'zip_url' into a temp dir, unzip, return local_dir.
        """
        tmp_dir = tempfile.mkdtemp(prefix=f"solver_{solver_id}_")
        zip_path = os.path.join(tmp_dir, f"{solver_id}.zip")

        r = requests.get(zip_url)
        r.raise_for_status()

        with open(zip_path, "wb") as f:
            f.write(r.content)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        return tmp_dir
