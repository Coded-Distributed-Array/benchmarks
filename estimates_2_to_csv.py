import math
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

size_map = dict({
    1: 608,
    8: 161,
    16: 130,
})

def binary_entropy(epsilon: float) -> float:
    """
    Calculates the binary entropy function h(ε) = -ε log₂(ε) - (1-ε)log₂(1-ε)

    Args:
        epsilon: A value between 0 and 1

    Returns:
        The binary entropy value
    """
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError("epsilon must be between 0 and 1")

    # Handle the case where epsilon is very close to 0 or 1 to avoid log(0)
    if epsilon < 1e-10:
        return -(1-epsilon) * math.log2(1-epsilon)
    if epsilon > 1 - 1e-10:
        return -epsilon * math.log2(epsilon)

    return -epsilon * math.log2(epsilon) - (1-epsilon) * math.log2(1-epsilon)

def calculate_rhs(
    delta_sd: float,
    T: int,
    delta_overlap: float,
    delta_overlap_min: float,
    k1: int,
    k2: int,
    epsilon: float,
    N: int
) -> float:
    """
    Calculates the right hand side of the inequality:
    δ ≤ δ_SD + ⌈(T+3)/(Δ_overlap - Δ_overlap,min + 1)⌉ · (k₁2^(h(ε)k₂)e^(-εN/k₁) + k₂e^(-N/k₂))

    Args:
        delta_sd: The δ_SD value
        T: Time parameter
        delta_overlap: Δ_overlap value
        delta_overlap_min: Δ_overlap,min value
        k1: k₁ parameter
        k2: k₂ parameter
        epsilon: ε parameter
        N: N parameter

    Returns:
        The calculated right hand side value
    """
    import math

    # Calculate the ceiling term
    ceiling_term = math.ceil((T + 2) / (delta_overlap - delta_overlap_min + 1))

    # Calculate h(ε)
    h_epsilon = binary_entropy(epsilon)

    # Calculate the exponential terms
    term1_base2 = pow(2, h_epsilon * k2)
    term1_exp_e = math.exp(-epsilon * N / k1)
    exp_term1 = k1 * term1_base2 * term1_exp_e

    exp_term2 = k2 * math.exp(-N / k2)

    # Put it all together
    return delta_sd + ceiling_term * (exp_term1 + exp_term2)

def find_max_k1(k2: int, epsilon: float, N: int, target_delta: float = 1e-9) -> int:
    """
    Finds the largest value of k1 that keeps the estimate below target_delta using binary search.

    Args:
        k2: k₂ parameter
        epsilon: ε parameter
        N: N parameter
        target_delta: Target upper bound (default: 10^-9)

    Returns:
        The largest valid k1 value
    """
    # Time parameters (fixed as per requirements)
    SECONDS_PER_ROUND = 4
    SECONDS_PER_HOUR = 3600
    SECONDS_PER_YEAR = 31557600

    T_rounds = int(10 * SECONDS_PER_YEAR / SECONDS_PER_ROUND)
    delta_overlap_rounds = int(6 * SECONDS_PER_HOUR / SECONDS_PER_ROUND)
    delta_overlap_min_rounds = 4

    # Binary search for k1
    left, right = 1, 10000  # Reasonable range for k1
    best_k1 = None

    while left <= right:
        mid = (left + right) // 2
        result = calculate_rhs(
            delta_sd=0.0,
            T=T_rounds,
            delta_overlap=delta_overlap_rounds,
            delta_overlap_min=delta_overlap_min_rounds,
            k1=mid,
            k2=k2,
            epsilon=epsilon,
            N=N
        )

        if result <= target_delta:
            # This k1 works, try to find a larger one
            best_k1 = mid
            left = mid + 1
        else:
            # This k1 is too large
            right = mid - 1

    return best_k1

def find_max_k2_with_k1_1(epsilon: float, N: int, target_delta: float = 1e-9) -> int:
    """
    Finds the largest value of k2 that keeps the estimate below target_delta when k1=1 using binary search.

    Args:
        epsilon: ε parameter
        N: N parameter
        target_delta: Target upper bound (default: 10^-9)

    Returns:
        The largest valid k2 value
    """
    # Time parameters (fixed as per requirements)
    SECONDS_PER_ROUND = 4
    SECONDS_PER_HOUR = 3600
    SECONDS_PER_YEAR = 31557600

    T_rounds = int(10 * SECONDS_PER_YEAR / SECONDS_PER_ROUND)
    delta_overlap_rounds = int(6 * SECONDS_PER_HOUR / SECONDS_PER_ROUND)
    delta_overlap_min_rounds = 30 * 60 / SECONDS_PER_ROUND # say roughly 15 minutes for sync

    # Binary search for k2
    left, right = 1, 2000  # Increased range for k2
    best_k2 = None

    while left <= right:
        mid = (left + right) // 2
        result = calculate_rhs(
            delta_sd=0.0,
            T=T_rounds,
            delta_overlap=delta_overlap_rounds,
            delta_overlap_min=delta_overlap_min_rounds,
            k1=1,  # Fixed to 1
            k2=mid,
            epsilon=epsilon,
            N=N
        )

        if result <= target_delta:
            # This k2 works, try to find a larger one
            best_k2 = mid
            left = mid + 1
        else:
            # This k2 is too large
            right = mid - 1

    return best_k2

def calculate_joining_complexity(k1: int, k2: int, n_max: int, n_bs: int = 100, t: int = 50, L_msg: int = 1) -> float:
    """
    Calculates the complexity of the Join operation.

    Args:
        k1: k₁ parameter
        k2: k₂ parameter
        n_max: maximum number of nodes
        n_bs: total number of nodes (default: 100)
        t: used number of bootstrap nodes (default: 50)
        L_msg: message length (default: 1)

    Returns:
        The Join operation complexity
    """
    term1 = 3 * t
    term2 = t * n_bs
    term3 = (t * n_max) / k1
    term4 = ((t + 4) * n_max * k2 - 2 * n_max + n_max * n_max) / (k2 * k2)

    return (term1 + term2 + term3 + term4) * L_msg

def calculate_get_complexity(k1: int, k2: int, n_max: int, n_hon_max: int, K: int, L_msg: int = 1) -> float:
    """
    Calculates the expected communication complexity of the GET operation (reduced form).

    Formula:
        ((n_max + n_hon_max*K)/(k1*k2)
         + 3*n_hon_max*(n_max + n_hon_max)/(k1*k2**2)) * L_msg
    """
    term1 = 0
    term2 = 0
    if K == 1:
        term1 = n_max / (k1 * k2)
        term2 = (n_hon_max) / (k1 * k2)
    else:
        ratio = size_map[K] / size_map[1]
        term1 = (n_max + n_hon_max * K * ratio) / (k1 * k2)
        term2 = (3 * n_hon_max * (n_max + n_hon_max * ratio)) / (k1 * (k2 ** 2))
    return (term1 + term2) * L_msg


def calculate_store_complexity(k1: int, k2: int, n_max: int, n_hon_max: int, K: int, L_msg: int = 1) -> float:
    """
    Calculates the expected communication complexity of the STORE operation.

    Formula:
        (n_max/(k1*k2) + 3*n_hon_max*n_max/(k1*k2**2)) * L_msg
    """
    ratio = size_map[K] / size_map[1]
    term1 = n_max / (k1 * k2)
    term2 = (3 * n_hon_max * n_max * ratio) / (k1 * (k2 ** 2))
    return (term1 + term2) * L_msg


def generate_rows(epsilon: float, N: int, K: int, target_delta: float = 1e-9):
    """
    Generates a list of tuples (k2, k1, data complexity, join complexity, get complexity, store complexity), such that
    (k1,k2) yield the target error probability delta and the given complexities.
    For each k2 from max possible down to 1, the function takes the maximum k1
    that yields the desired error.
    """

    # First find the maximum k2 possible with k1=1
    max_k2 = find_max_k2_with_k1_1(epsilon, N, target_delta)
    if max_k2 is None:
        return []

    n_max = 5 * N
    n_hon_max = 2 * N

    rows = []
    # For each k2 value from max down to 1
    for k2 in range(max_k2, 0, -1):
        k1 = find_max_k1(k2, epsilon, N, target_delta)
        if k1 is not None:
            # we have found a valid k1, now compute complexities
            data_complexity = 1.0 / k2
            join_complexity = calculate_joining_complexity(k1, k2, n_max)
            get_complexity = calculate_get_complexity(k1, k2, n_max, n_hon_max, K)
            store_complexity = calculate_store_complexity(k1, k2, n_max, n_hon_max, K)
            rows.append((k2, k1, data_complexity, join_complexity, get_complexity, store_complexity))

    return rows




def generate_rows(epsilon: float, N: int, K: int, target_delta: float = 1e-9):
    """
    Generates a list of tuples (k2, k1, data complexity, join complexity, get complexity, store complexity), such that
    (k1,k2) yield the target error probability delta and the given complexities.
    For each k2 from max possible down to 1, the function takes the maximum k1
    that yields the desired error.
    """

    # First find the maximum k2 possible with k1=1
    max_k2 = find_max_k2_with_k1_1(epsilon, N, target_delta)
    if max_k2 is None:
        return []

    n_max = 5 * N
    n_hon_max = 2 * N

    rows = []
    # For each k2 value from max down to 1
    for k2 in range(max_k2, 0, -1):
        k1 = find_max_k1(k2, epsilon, N, target_delta)
        if k1 is not None:
            # we have found a valid k1, now compute complexities
            data_complexity = 1.0 / k2
            join_complexity = calculate_joining_complexity(k1, k2, n_max)
            get_complexity = calculate_get_complexity(k1, k2, n_max, n_hon_max, K)
            store_complexity = calculate_store_complexity(k1, k2, n_max, n_hon_max, K)
            rows.append((k2, k1, data_complexity, join_complexity, get_complexity, store_complexity))

    return rows

# if __name__ == "__main__":
#     # Define the column headers for our csv file.
#     headers = [
#         "k2", "k1", "data_complexity", "join_complexity", "get_complexity", "store_complexity"
#     ]

#     # Parameters for analysis
#     N_values = [1000, 5000, 10000, 100000]
#     epsilon_nominator_values = [5, 10] # epsilon is this / 100
#     target_delta = 1e-9
#     K = 8

#     # iterate over a bunch of eps, N pairs
#     for eps_nom in epsilon_nominator_values:
#         for N in N_values:
#             # Generate the curve
#             eps = eps_nom / 100
#             rows = generate_rows(eps, N, K, target_delta)

#             if rows:
#                 filename = f"estimates_data_{eps_nom}_{N}.csv"
#                 # Write the data to the CSV file
#                 with open(filename, mode='w', newline='') as file:
#                     writer = csv.writer(file)
#                     writer.writerow(headers)  # Write the header row
#                     writer.writerows(rows)    # Write the data rows
#                 print(f"Data written to {filename}")



if __name__ == "__main__":
    N_values = [1000, 5000, 10000, 100000]
    epsilon_nominator_values = [5, 10] # epsilon is this / 100
    K_list = [1, 8, 16]

    outdir = Path(".")
    outdir.mkdir(parents=True, exist_ok=True)

    for eps_nom in epsilon_nominator_values:
        epsilon = eps_nom / 100.0
        for N in N_values:
            print(f"[Run] epsilon={epsilon:.2f}, N={N}")

            curves = {}
            for K in K_list:
                rows = generate_rows(epsilon, N, K, target_delta=1e-9)
                if not rows:
                    continue
                rows.sort(key=lambda x: x[0])
                curves[K] = dict(
                    k2=[r[0] for r in rows],
                    k1=[r[0] for r in rows],
                    join=[r[3] for r in rows],
                    get=[r[4] for r in rows],
                    store=[r[5] for r in rows],
                )

            if not curves:
                print(f"⚠️  No data for ε={epsilon:.2f}, N={N}")
                continue

            fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
            plt.subplots_adjust(wspace=0.3, top=0.83)
            markers = {1: "o", 8: "s", 16: "^"}
            labels = {1: "K=1", 8: "K=8", 16: "K=16"}

            # Max k1
            ax = axes[0]
            for K in K_list:
                if K not in curves:
                    continue
                ax.plot(curves[K]["k2"], curves[K]["k1"],
                        marker=markers[K], label=labels[K],
                        linewidth=1.8, markersize=4)
            ax.set_title("Max k1")
            ax.set_xlabel("k2")
            ax.set_ylabel("Max k1")
            ax.grid(True, alpha=0.3)
            ax.legend(title="K values")

            # Join
            ax = axes[1]
            for K in K_list:
                if K not in curves:
                    continue
                ax.plot(curves[K]["k2"], curves[K]["join"],
                        marker=markers[K], label=labels[K],
                        linewidth=1.8, markersize=4)
            ax.set_title("Join Complexity")
            ax.set_xlabel("k2")
            ax.set_ylabel("Join Complexity")
            ax.grid(True, alpha=0.3)
            ax.legend(title="K values")

            # Get
            ax = axes[2]
            for K in K_list:
                if K not in curves:
                    continue
                ax.plot(curves[K]["k2"], curves[K]["get"],
                        marker=markers[K], label=labels[K],
                        linewidth=1.8, markersize=4)
            ax.set_title("Get Complexity")
            ax.set_xlabel("k2")
            ax.set_ylabel("Get Complexity")
            ax.grid(True, alpha=0.3)
            # ax.set_yscale("log")

            # Store
            ax = axes[3]
            for K in K_list:
                if K not in curves:
                    continue
                ax.plot(curves[K]["k2"], curves[K]["store"],
                        marker=markers[K], label=labels[K],
                        linewidth=1.8, markersize=4)
            ax.set_title("Store Complexity")
            ax.set_xlabel("k2")
            ax.set_ylabel("Store Complexity")
            ax.grid(True, alpha=0.3)
            # ax.set_yscale("log")

            fig.suptitle(
                f"Complexity Comparison (epsilon={epsilon:.2f}, N={N})",
                fontsize=14, y=0.98, weight="bold"
            )

            out_name = outdir / f"complexity_raw_eps{eps_nom}_N{N}.png"
            fig.savefig(out_name.as_posix(), dpi=160, bbox_inches="tight")
            plt.close(fig)