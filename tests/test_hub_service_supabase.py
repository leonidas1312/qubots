# tests/test_hub_service_supabase.py

import os
import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from rastion_hub.main import app

load_dotenv()  # loads .env if present

@pytest.fixture(scope="module")
def supabase_test_client():
    """
    A pytest fixture that yields a FastAPI test client for supabase testing.
    We check if SUPABASE_URL & SUPABASE_KEY exist, otherwise skip.
    """
    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_KEY"):
        pytest.skip("No SUPABASE_URL/KEY set, skipping supabase tests.")
    client = TestClient(app)
    yield client

def test_create_solver_with_supabase(supabase_test_client):
    """
    Tests real supabase usage. Expects that SUPABASE_URL & SUPABASE_KEY
    are set in environment or .env
    """
    payload = {
        "solver_id": "supabase-solver-1",
        "description": "Testing supabase solver",
        "entry_point": "my_module:MySupabaseSolver"
    }
    resp = supabase_test_client.post("/api/solvers", json=payload)
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert data["solver_id"] == "supabase-solver-1"
    assert data["entry_point"] == "my_module:MySupabaseSolver"
    # Optionally verify the record in your supabase table if needed

def test_fetch_solver_with_supabase(supabase_test_client):
    # Now fetch it
    resp = supabase_test_client.get("/api/solvers/supabase-solver-1")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["solver_id"] == "supabase-solver-1"
    assert data["description"] == "Testing supabase solver"
