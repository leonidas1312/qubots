# rastion_hub/supabase_client.py

import os
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")  # e.g. "https://xyzcompany.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # your anon or service key

if not SUPABASE_URL or not SUPABASE_KEY:
    print("[WARNING] SUPABASE_URL or SUPABASE_KEY not defined. Using local memory store.")
    supabase = None
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Example bucket names
SOLVER_BUCKET = "solver-code"
PROBLEM_BUCKET = "problem-code"
