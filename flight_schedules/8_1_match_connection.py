import numpy as np
import pandas as pd
from tqdm import tqdm

def solve_assignment_problem(S: np.ndarray, airport: str, airline: str):
    import numpy as np
    import pulp

    # -------------------------------
    # 1. Define the Data
    # -------------------------------

    L_b = S.copy()
    L_b[np.isneginf(L_b)] = -1e6
    n = L_b.shape[0]
    m = L_b.shape[1]


    # The maximum number of matches is min(n, m)
    max_matches = min(n, m)
    lambda_penalty = 2000

    # -------------------------------
    # 2. Build the Optimization Model with PuLP
    # -------------------------------
    print(f'Building the optimization model with PuLP...')

    # Create the problem instance: we want to minimize the total cost.
    prob = pulp.LpProblem("StochasticAssignment", pulp.LpMaximize)

    # Create the decision variables:
    # x[i][j] == 1 if arrival i is matched to departure j, 0 otherwise.
    # We'll use a dictionary of PuLP binary variables.
    x = {}
    from tqdm import tqdm
    for i in tqdm(range(n), desc="Creating decision variables"):
        for j in range(m):
            x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", cat="Binary")

    # Create the slack variable s to penalize "lazy assignments"
    s = pulp.LpVariable("s", lowBound=0, cat="Continuous")

    # -------------------------------
    # 3. Set the Objective Function
    # -------------------------------
    print(f'Setting the objective function...')
    # Minimize total cost sum_{i,j} L_b(i,j) * x_{ij}
    prob += pulp.lpSum(L_b[i, j] * x[(i, j)] for i in range(n) for j in range(m)) - lambda_penalty * s, "TotalCost"

    # -------------------------------
    # 4. Add the Constraints
    # -------------------------------
    print(f'Adding the constraints...')
    # (a) Each arrival (row) is matched to at most one departure.
    for i in tqdm(range(n), desc="Adding arrival constraints"):
        prob += pulp.lpSum(x[(i, j)] for j in range(m)) <= 1, f"Arrival_{i}_Constraint"

    # (b) Each departure (column) is matched to at most one arrival.
    for j in tqdm(range(m), desc="Adding departure constraints"):
        prob += pulp.lpSum(x[(i, j)] for i in range(n)) <= 1, f"Departure_{j}_Constraint"

    # (c) Total number of matches must equal min(n, m).
    # prob += pulp.lpSum(x[(i, j)] for i in range(n) for j in range(m)) == max_matches, "TotalMatchesConstraint"

    # Constraint: Total number of assignments plus slack equals min(n, m)
    prob += pulp.lpSum(x[(i,j)] for i in range(n) for j in range(m)) + s == max_matches, "Lazy_Assignments"

    # -------------------------------
    # 5. Solve the Problem
    # -------------------------------
    import datetime
    print(f'Solution started at {datetime.datetime.now()}')

    solver = pulp.PULP_CBC_CMD(msg=True,  # Enable detailed messages
                            options=['printingOptions', 'all'])  # Show all printing options

    result_status = prob.solve(solver)

    print("Solver Status:", pulp.LpStatus[result_status])
    print("Objective Value:", pulp.value(prob.objective))
    # -------------------------------
    # 6. Extract the Results
    # -------------------------------
    assignments = []
    for i in tqdm(range(n), desc="Extracting results"):
        for j in range(m):
            # If the variable x[(i,j)] is 1, then arrival i is matched with departure j.
            if pulp.value(x[(i, j)]) > 0.5:
                assignments.append((i, j))
                # print(f"Arrival {i} is assigned to Departure {j} with cost {L_b[i, j]:.4f}")

    # Write the assignments to a file
    df_assignments = pd.DataFrame(assignments, columns=["arrival", "departure"])
    df_assignments.to_csv(f"flight_schedules/processed_data/assignments/{airport}_{airline}.csv", index=False)

    print("\nTotal number of assignments:", len(assignments))
    print(f'Maximum possible assignments: {max_matches}')

from glob import glob
import os

def go():
    # List all .npy files in the flight_schedules/processed_data/score_matrices directory
    score_matrices = glob("flight_schedules/processed_data/score_matrices/*.npy")
    # Remove all files that start with ._
    score_matrices = [f for f in score_matrices if not f.startswith("._")]
    # Create a folder for assignments
    os.makedirs("flight_schedules/processed_data/assignments", exist_ok=True)

    for matrix_number, score_matrix_path in enumerate(score_matrices):
        print(f"Processing matrix {matrix_number+1}/{len(score_matrices)}")
        filename = os.path.basename(score_matrix_path)
        airport = filename.split("_")[0]
        airline = filename.split("_")[1]
        print(f'Airport: {airport}, airline: {airline}')
        S = np.load(score_matrix_path)
        solve_assignment_problem(S, airport, airline)

if __name__ == "__main__":
    go()
