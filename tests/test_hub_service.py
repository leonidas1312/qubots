# tests/test_hub_service.py

import os
import pytest
from fastapi.testclient import TestClient
from rastion_hub.main import app

client = TestClient(app)

@pytest.fixture(scope="module")
def test_client():
    """
    A pytest fixture that yields a FastAPI test client.
    We assume the app uses in-memory storage if Supabase isn't set.
    """
    # Optionally, set env vars for supabase if you want to test real supabase usage
    # os.environ["SUPABASE_URL"] = "https://my-supabase.co"
    # os.environ["SUPABASE_KEY"] = "my-secret-key"
    yield client

def test_create_solver_in_memory(test_client):
    """
    Test creating a solver with the local in-memory approach (no supabase).
    """
    # Clear env so the hub uses local memory store
    if "SUPABASE_URL" in os.environ:
        del os.environ["SUPABASE_URL"]
    if "SUPABASE_KEY" in os.environ:
        del os.environ["SUPABASE_KEY"]

    # Post a new solver
    payload = {
        "solver_id": "test-solver-1",
        "description": "Test solver for TSP",
        "entry_point": "my_module:MySolver",
        "default_params": {"population_size": 50}
    }
    resp = test_client.post("/api/solvers", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["solver_id"] == "test-solver-1"
    assert data["entry_point"] == "my_module:MySolver"
    assert data["default_params"]["population_size"] == 50

def test_get_solver_in_memory(test_client):
    """
    Test retrieving the solver by ID from local store.
    """
    resp = test_client.get("/api/solvers/test-solver-1")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["solver_id"] == "test-solver-1"
    assert data["description"] == "Test solver for TSP"

def test_create_problem_in_memory(test_client):
    payload = {
        "problem_id": "test-problem-1",
        "description": "Test problem"
    }
    resp = test_client.post("/api/problems", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["problem_id"] == "test-problem-1"

def test_get_problem_in_memory(test_client):
    resp = test_client.get("/api/problems/test-problem-1")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["problem_id"] == "test-problem-1"
    assert data["description"] == "Test problem"

@pytest.mark.skip(reason="Requires real supabase setup & bucket. Uncomment if you want live supabase test.")
def test_create_solver_with_supabase(test_client):
    """
    Example of how you'd test real supabase usage, if you set env vars.
    """
    os.environ["SUPABASE_URL"] = "https://your-supabase-url"
    os.environ["SUPABASE_KEY"] = "your-service-key"

    payload = {
        "solver_id": "supabase-solver-1",
        "description": "Testing supabase solver",
        "entry_point": "my_module:MySupabaseSolver"
    }
    resp = test_client.post("/api/solvers", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["solver_id"] == "supabase-solver-1"
