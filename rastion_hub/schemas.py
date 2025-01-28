# rastion_hub/schemas.py

from pydantic import BaseModel
from typing import Optional, Dict

class SolverCreate(BaseModel):
    solver_id: str
    description: Optional[str] = None
    entry_point: str
    default_params: Optional[Dict] = None

class SolverResponse(BaseModel):
    id: str
    solver_id: str
    description: Optional[str]
    entry_point: str
    default_params: Optional[Dict]
    code_url: Optional[str] = None

class ProblemCreate(BaseModel):
    problem_id: str
    description: Optional[str] = None

class ProblemResponse(BaseModel):
    id: str
    problem_id: str
    description: Optional[str]
    code_url: Optional[str] = None
