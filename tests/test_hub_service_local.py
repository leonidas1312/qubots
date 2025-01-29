# tests/test_hub_service_local.py

import os
import pytest
from fastapi.testclient import TestClient
from rastion_hub.main import app

@pytest.fixture(scope="module")
def local_test_client():
    """
    A pytest fixture that yields a FastAPI test client
    specifically for local (in-memory) usage.
    We ensure SUPABASE_URL/KEY are unset to force local.
    """
    # Clear environment variables
    if "SUPABASE_URL" in os.environ:
        del os.environ["SUPABASE_URL"]
    if "SUPABASE_KEY" in os.environ:
        del os.environ["SUPABASE_KEY"]

    client = TestClient(app)
    yield client

def test_create_solver_in_memory(local_test_client):
    payload = {
        "solver_id": "test-solver-1",
        "description": "Test solver for TSP",
        "entry_point": "my_module:MySolver",
        "default_params": {"population_size": 50}
    }
    resp = local_test_client.post("/api/solvers", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["solver_id"] == "test-solver-1"
    assert data["entry_point"] == "my_module:MySolver"
    assert data["default_params"]["population_size"] == 50

def test_get_solver_in_memory(local_test_client):
    resp = local_test_client.get("/api/solvers/test-solver-1")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["solver_id"] == "test-solver-1"
    assert data["description"] == "Test solver for TSP"

def test_create_problem_in_memory(local_test_client):
    payload = {
        "problem_id": "test-problem-1",
        "description": "Test problem"
    }
    resp = local_test_client.post("/api/problems", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["problem_id"] == "test-problem-1"

def test_get_problem_in_memory(local_test_client):
    resp = local_test_client.get("/api/problems/test-problem-1")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["problem_id"] == "test-problem-1"
    assert data["description"] == "Test problem"
