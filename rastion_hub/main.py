# rastion_hub/main.py

from fastapi import FastAPI, HTTPException, UploadFile, File
from .schemas import SolverCreate, SolverResponse, ProblemCreate, ProblemResponse
from .supabase_client import supabase, SOLVER_BUCKET, PROBLEM_BUCKET
import uuid

app = FastAPI(title="Rastion Hub")

# Fallback in-memory store if Supabase not configured
SOLVER_REGISTRY = {}
PROBLEM_REGISTRY = {}


#
# SOLVERS
#
@app.post("/api/solvers", response_model=SolverResponse)
def create_solver(payload: SolverCreate):
    """
    Create a solver metadata entry.
    If using Supabase, inserts into table "solvers".
    Otherwise, store in a local dictionary.
    """
    if supabase is None:
        # local memory approach
        if payload.solver_id in SOLVER_REGISTRY:
            raise HTTPException(status_code=400, detail="Solver ID already exists in local store.")
        record_id = str(uuid.uuid4())
        record = {
            "id": record_id,
            "solver_id": payload.solver_id,
            "description": payload.description,
            "entry_point": payload.entry_point,
            "default_params": payload.default_params,
            "code_url": None
        }
        SOLVER_REGISTRY[payload.solver_id] = record
        return SolverResponse(**record)
    else:
        # Insert into Supabase "solvers" table
        insert_data = {
            "solver_id": payload.solver_id,
            "description": payload.description,
            "entry_point": payload.entry_point,
            "default_params": payload.default_params
        }
        resp = supabase.table("solvers").insert(insert_data).execute()
        if not resp.data:
            raise HTTPException(status_code=500, detail="Error inserting solver data into Supabase.")
        record = resp.data[0]
        return SolverResponse(
            id=str(record["id"]),
            solver_id=record["solver_id"],
            description=record.get("description"),
            entry_point=record["entry_point"],
            default_params=record.get("default_params"),
            code_url=record.get("code_url")
        )

@app.post("/api/solvers/{solver_id}/upload", response_model=SolverResponse)
async def upload_solver_code(solver_id: str, file: UploadFile = File(...)):
    """
    Upload a zip file containing solver code for the given solver_id.
    """
    if supabase is None:
        # local approach: store code in memory or local disk
        if solver_id not in SOLVER_REGISTRY:
            raise HTTPException(status_code=404, detail="Solver not found in local store.")
        record = SOLVER_REGISTRY[solver_id]
        # For simplicity, write file to a local directory. We'll skip full code:
        unique_name = f"{solver_id}-{uuid.uuid4()}.zip"
        with open(unique_name, "wb") as f:
            f.write(await file.read())
        # We can't generate a real "public URL," but store a path
        record["code_url"] = unique_name
        return SolverResponse(**record)

    else:
        # Supabase approach
        find_resp = supabase.table("solvers").select("*").eq("solver_id", solver_id).execute()
        if not find_resp.data:
            raise HTTPException(status_code=404, detail="Solver not found in Supabase.")

        solver_record = find_resp.data[0]
        solver_uuid = str(solver_record["id"])

        file_bytes = await file.read()
        unique_name = f"{solver_id}-{uuid.uuid4()}.zip"
        path_in_bucket = f"{solver_id}/{unique_name}"

        res = supabase.storage.from_(SOLVER_BUCKET).upload(path_in_bucket, file_bytes)
        if "error" in res:
            raise HTTPException(status_code=500, detail=str(res["error"]))

        public_url = supabase.storage.from_(SOLVER_BUCKET).get_public_url(path_in_bucket)

        update_resp = supabase.table("solvers").update({"code_url": public_url}).match({"id": solver_uuid}).execute()
        updated = update_resp.data[0]
        return SolverResponse(
            id=str(updated["id"]),
            solver_id=updated["solver_id"],
            description=updated.get("description"),
            entry_point=updated["entry_point"],
            default_params=updated.get("default_params"),
            code_url=updated.get("code_url")
        )

@app.get("/api/solvers/{solver_id}", response_model=SolverResponse)
def get_solver(solver_id: str):
    """
    Retrieve solver metadata by solver_id.
    """
    if supabase is None:
        if solver_id not in SOLVER_REGISTRY:
            raise HTTPException(status_code=404, detail="Solver not found in local store.")
        record = SOLVER_REGISTRY[solver_id]
        return SolverResponse(**record)
    else:
        resp = supabase.table("solvers").select("*").eq("solver_id", solver_id).execute()
        if not resp.data:
            raise HTTPException(status_code=404, detail="Solver not found in Supabase.")
        record = resp.data[0]
        return SolverResponse(
            id=str(record["id"]),
            solver_id=record["solver_id"],
            description=record.get("description"),
            entry_point=record["entry_point"],
            default_params=record.get("default_params"),
            code_url=record.get("code_url")
        )


#
# PROBLEMS
#
@app.post("/api/problems", response_model=ProblemResponse)
def create_problem(payload: ProblemCreate):
    """
    Create a problem metadata entry (like TSP, QUBO, etc.).
    """
    if supabase is None:
        if payload.problem_id in PROBLEM_REGISTRY:
            raise HTTPException(status_code=400, detail="Problem ID already exists in local store.")
        record_id = str(uuid.uuid4())
        record = {
            "id": record_id,
            "problem_id": payload.problem_id,
            "description": payload.description,
            "code_url": None
        }
        PROBLEM_REGISTRY[payload.problem_id] = record
        return ProblemResponse(**record)
    else:
        insert_data = {
            "problem_id": payload.problem_id,
            "description": payload.description
        }
        resp = supabase.table("problems").insert(insert_data).execute()
        if not resp.data:
            raise HTTPException(status_code=500, detail="Error inserting problem data.")
        record = resp.data[0]
        return ProblemResponse(
            id=str(record["id"]),
            problem_id=record["problem_id"],
            description=record.get("description"),
            code_url=record.get("code_url")
        )

@app.post("/api/problems/{problem_id}/upload", response_model=ProblemResponse)
async def upload_problem_code(problem_id: str, file: UploadFile = File(...)):
    """
    Upload a zip file containing problem code for the given problem_id.
    E.g., a custom TSP problem or a specialized class with constraints.
    """
    if supabase is None:
        if problem_id not in PROBLEM_REGISTRY:
            raise HTTPException(status_code=404, detail="Problem not found in local store.")
        record = PROBLEM_REGISTRY[problem_id]
        unique_name = f"{problem_id}-{uuid.uuid4()}.zip"
        with open(unique_name, "wb") as f:
            f.write(await file.read())
        record["code_url"] = unique_name
        return ProblemResponse(**record)
    else:
        find_resp = supabase.table("problems").select("*").eq("problem_id", problem_id).execute()
        if not find_resp.data:
            raise HTTPException(status_code=404, detail="Problem not found in Supabase.")
        record = find_resp.data[0]
        record_uuid = str(record["id"])

        file_bytes = await file.read()
        unique_name = f"{problem_id}-{uuid.uuid4()}.zip"
        path_in_bucket = f"{problem_id}/{unique_name}"

        res = supabase.storage.from_(PROBLEM_BUCKET).upload(path_in_bucket, file_bytes)
        if "error" in res:
            raise HTTPException(status_code=500, detail=str(res["error"]))

        public_url = supabase.storage.from_(PROBLEM_BUCKET).get_public_url(path_in_bucket)

        update_resp = supabase.table("problems").update({"code_url": public_url}).match({"id": record_uuid}).execute()
        updated = update_resp.data[0]
        return ProblemResponse(
            id=str(updated["id"]),
            problem_id=updated["problem_id"],
            description=updated.get("description"),
            code_url=updated.get("code_url")
        )

@app.get("/api/problems/{problem_id}", response_model=ProblemResponse)
def get_problem(problem_id: str):
    """
    Retrieve problem metadata by problem_id.
    """
    if supabase is None:
        if problem_id not in PROBLEM_REGISTRY:
            raise HTTPException(status_code=404, detail="Problem not found in local store.")
        record = PROBLEM_REGISTRY[problem_id]
        return ProblemResponse(**record)
    else:
        resp = supabase.table("problems").select("*").eq("problem_id", problem_id).execute()
        if not resp.data:
            raise HTTPException(status_code=404, detail="Problem not found in Supabase.")
        record = resp.data[0]
        return ProblemResponse(
            id=str(record["id"]),
            problem_id=record["problem_id"],
            description=record.get("description"),
            code_url=record.get("code_url")
        )
