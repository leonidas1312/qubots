import numpy as np
from rastion_core.base_problem import BaseProblem

def setup_problem(num_flights,
                  num_gates,
                  transfer_probability,
                  min_tr_passengers,
                  max_tr_passengers,
                  min_flight_duration,
                  max_flight_duration,
                  min_onboard_passengers,
                  max_onboard_passengers,
                  min_departing_passengers,
                  max_departing_passengers):
    """
    Set up the flight gate assignment problem.
    Returns various parameters needed to build the QUBO.
    """
    np.random.seed(123)
    DAY = 1440  # minutes in a day

    # Generate flight and gate labels
    F = [f'flight{i}' for i in range(num_flights)]
    G = [f'gate{j}' for j in range(num_gates)]

    # Random passengers data
    n_dep = {f: np.random.randint(min_departing_passengers, max_departing_passengers) for f in F}
    n_arr = {f: np.random.randint(min_onboard_passengers, max_onboard_passengers) for f in F}

    # Transfer passengers between flights
    n = {}
    for f1 in F:
        for f2 in F:
            if f1 != f2 and np.random.rand() < transfer_probability:
                n[(f1, f2)] = np.random.randint(min_tr_passengers, max_tr_passengers)
            else:
                n[(f1, f2)] = 0

    # Transfer times for arrival and departure at gates
    t_arr = {g: np.random.randint(5, 20) for g in G}
    t_dep = {g: np.random.randint(5, 20) for g in G}
    t = {(g1, g2): np.random.randint(3, 10) for g1 in G for g2 in G if g1 != g2}

    # Flight times
    t_in = {f: np.random.randint(0, DAY) for f in F}
    t_out = {f: t_in[f] + np.random.randint(min_flight_duration, max_flight_duration) for f in F}
    t_buf = 20  # Buffer time between flights

    # x is used only for cost evaluation; we will generate Q later.
    x = {flight: {gate: 0 for gate in G} for flight in F}

    # Return all data needed for the QUBO
    return F, G, n_dep, n_arr, n, t_arr, t_dep, t, t_in, t_out, t_buf, x

def create_qubo_matrix(num_flights, num_gates, n, t, t_in, t_out, t_buf):
    size = num_flights * num_gates
    Q = np.zeros((size, size))

    def get_index(flight_idx, gate_idx):
        return flight_idx * num_gates + gate_idx

    # Objective function: transfer time cost terms
    for flight_i in range(num_flights):
        for flight_j in range(num_flights):
            for gate_i in range(num_gates):
                for gate_j in range(num_gates):
                    idx_i = get_index(flight_i, gate_i)
                    idx_j = get_index(flight_j, gate_j)
                    transfer_time = n.get((f'flight{flight_i}', f'flight{flight_j}'), 0) * \
                                    t.get((f'gate{gate_i}', f'gate{gate_j}'), 0)
                    Q[idx_i][idx_j] += transfer_time

    # Constraints with penalty
    penalty = max(Q.max(), -Q.min()) * 10

    # Assignment Constraint: each flight is assigned to one gate
    for flight in range(num_flights):
        for gate_i in range(num_gates):
            for gate_j in range(num_gates):
                idx_i = get_index(flight, gate_i)
                idx_j = get_index(flight, gate_j)
                if gate_i != gate_j:
                    Q[idx_i][idx_j] += penalty
                else:
                    Q[idx_i][idx_j] -= penalty

    # Occupancy Constraint: avoid overlapping assignments
    for flight_i in range(num_flights):
        for flight_j in range(num_flights):
            if (t_in[f'flight{flight_i}'] < t_in[f'flight{flight_j}'] < t_out[f'flight{flight_i}'] + t_buf) or \
               (t_in[f'flight{flight_j}'] < t_in[f'flight{flight_i}'] < t_out[f'flight{flight_j}'] + t_buf):
                for gate in range(num_gates):
                    idx_i = get_index(flight_i, gate)
                    idx_j = get_index(flight_j, gate)
                    Q[idx_i][idx_j] += penalty

    return Q

class GateAssignmentProblem(BaseProblem):
    """
    A QUBO formulation of the airline gate assignment problem.
    This problem sets up the instance using random data and then builds the QUBO matrix.
    """
    def __init__(self,
                 num_flights=5,
                 num_gates=3,
                 transfer_probability=0.3,
                 min_tr_passengers=10,
                 max_tr_passengers=50,
                 min_flight_duration=30,
                 max_flight_duration=180,
                 min_onboard_passengers=50,
                 max_onboard_passengers=200,
                 min_departing_passengers=50,
                 max_departing_passengers=200):
        # Set up the problem parameters.
        (self.F, self.G, self.n_dep, self.n_arr, self.n, self.t_arr, self.t_dep,
         self.t, self.t_in, self.t_out, self.t_buf, self.x) = setup_problem(
            num_flights,
            num_gates,
            transfer_probability,
            min_tr_passengers,
            max_tr_passengers,
            min_flight_duration,
            max_flight_duration,
            min_onboard_passengers,
            max_onboard_passengers,
            min_departing_passengers,
            max_departing_passengers
        )
        self.num_flights = num_flights
        self.num_gates = num_gates

        # Build the QUBO matrix for the problem.
        self.QUBO_matrix = create_qubo_matrix(num_flights, num_gates, self.n, self.t,
                                               self.t_in, self.t_out, self.t_buf)
        # In many QUBO formulations a constant term is also used.
        self.qubo_constant = 0

    def evaluate_solution(self, solution) -> float:
        # For illustration, you could evaluate a candidate bitstring using the QUBO:
        # (Usually, you compute solution^T Q solution.)
        sol = np.array(solution)
        return float(sol.T @ self.QUBO_matrix @ sol + self.qubo_constant)

    def random_solution(self):
        # Return a random bitstring of appropriate length.
        size = self.num_flights * self.num_gates
        return np.random.randint(0, 2, size).tolist()

    def get_qubo(self):
        """
        Return the QUBO matrix and the constant.
        """
        return self.QUBO_matrix, self.qubo_constant
