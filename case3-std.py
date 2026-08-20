import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

def _ensure_qce_repo_on_path() -> Path:
    start = Path.cwd()
    for d in [start, *start.parents]:
        if (d / "qce_cudaq_main.py").exists():
            sys.path.insert(0, str(d))
            return d
    sub = start / "cudaq-qce"
    if (sub / "qce_cudaq_main.py").exists():
        sys.path.insert(0, str(sub.resolve()))
        return sub.resolve()
    env = os.environ.get("QCE_REPO_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "qce_cudaq_main.py").exists():
            sys.path.insert(0, str(p))
            return p
    raise FileNotFoundError("qce_cudaq_main.py not found.")

_ensure_qce_repo_on_path()
from qce_cudaq_main import minimize_cudaq

# BIG 3

nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10, 11]
qubit_num = len(nodes)


edges = [
    [0,1],[0,6],[0,7],[0,9],[0,11],
    [1,6],[1,7],[1,8],[1,10],[1,11],
    [2,4],[2,5],[2,7],[2,9],
    [3,4],[3,5],[3,6],[3,8],[3,9],
    [4,5],[4,9],[4,11],
    [5,8],[5,10],[5,11],
    [6,9],[6,11],
    [7,8],[7,9],[7,10],
    [8,9]]

non_edges = [
    [u, v]
    for u in nodes
    for v in nodes
    if u < v and [u, v] not in edges
]

weights=[0.5244,0.6686,0.1453,0.6686,0.1453,0.6686,0.0504,0.2317,0.5244,0.6686,0.1453,0.6686]
penalty = 12.0


# QUBO

N = qubit_num
Q = np.zeros((N, N), dtype=np.float64)
const = 0.0

# Weights
for i, w in enumerate(weights):
    Q[i, i] += -w

# Penalty of Non-Edges
for u, v in non_edges:
    Q[u, v] += penalty
    Q[v, u] += penalty

# Classical solution

def qubo_energy(bitstring: str) -> float:
    x = np.array([int(b) for b in bitstring], dtype=float)
    return float(x @ Q @ x + const)

E_star = float("inf")
key_star = ""
for i in range(1 << N):
    key = format(i, f"0{N}b")
    e = qubo_energy(key)
    if e < E_star:
        E_star = e
        key_star = key

print("Classical optimal solution (E*):", E_star)
print("Classical optimal Bitstring (x*):", key_star, "\n")

# Quantum solution

num_runs = 10
generations = 1000
population = 48

history_matrix = []

print(f"Starting {num_runs} runs")

for run in range(num_runs):
    print(f"\n Execução {run + 1}/{num_runs}")
    
    result = minimize_cudaq(
        Q=Q,
        const=const,
        n_qubits=N,
        generations=generations,
        population=population,
        device="cpu",
        shots=4096,
        seed=42 + run, 
        verbose=True,
        ref_value=float("-inf"),
        tol=0.0,
    )
    
    run_values = result["values"]
    
    # Fill the matrix with the final value after convergence
    if len(run_values) < generations:
        last_val = run_values[-1]
        run_values += [last_val] * (generations - len(run_values))
        
    history_matrix.append(run_values)

history_matrix = np.array(history_matrix)

mean_evolution = np.mean(history_matrix, axis=0)
std_evolution = np.std(history_matrix, axis=0)

print(f"Classical optimal solution (E*): {E_star:.6f}")
print(f"Quantum cost:  {mean_evolution[-1]:.6f}")
print(f"Standard Deviation:         {std_evolution[-1]:.6f}")
print(f"Gap:             {mean_evolution[-1] - E_star:.6f}")

# Visualization

x_axis = np.arange(1, generations + 1)

plt.figure(figsize=(10, 6))

plt.axhline(y=E_star, color="red", linestyle="--", alpha=0.7, label=f"Optimal cost value (E = {E_star:.4f})")

plt.plot(x_axis, mean_evolution, color="#2ca02c", lw=2, label="Avarage cost")

plt.fill_between(
    x_axis,
    mean_evolution - std_evolution,
    mean_evolution + std_evolution,
    color="#2ca02c",
    alpha=0.2,
    label="Standard deviation"
)

plt.xlabel("Generation", fontsize=20)
plt.ylabel("Cost", fontsize=20)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=15, loc="upper right")

plt.tight_layout()

plt.savefig('case1-std.png', dpi=600)

plt.show()
