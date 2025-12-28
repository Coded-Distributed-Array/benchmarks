import csv
import math
from typing import List, Optional, Sequence, Tuple


SECONDS_PER_ROUND = 4
SECONDS_PER_HOUR = 3600
SECONDS_PER_YEAR = 31557600

T_ROUNDS = int(10 * SECONDS_PER_YEAR / SECONDS_PER_ROUND)
DELTA_OVERLAP_ROUNDS = int(6 * SECONDS_PER_HOUR / SECONDS_PER_ROUND)

def binary_entropy(epsilon: float) -> float:
    """
    Calculates the binary entropy function h(ε) = -ε log₂(ε) - (1-ε)log₂(1-ε)
    """
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError("epsilon must be between 0 and 1")

    if epsilon < 1e-10:
        return -(1 - epsilon) * math.log2(1 - epsilon)
    if epsilon > 1 - 1e-10:
        return -epsilon * math.log2(epsilon)
    return -epsilon * math.log2(epsilon) - (1 - epsilon) * math.log2(1 - epsilon)


def _alpha_and_decay(N: int, k2: int, k_chunks: int) -> Optional[Tuple[float, float]]:
    """Returns (alpha, decay_factor) for the RLNC bound, or None if invalid."""
    if 1 - (k_chunks * k2 / N) <= 0 or 1 - (k_chunks * k2 / N) >= 1:
        return None

    alpha = N / (k2 * k_chunks) - 1
    if alpha <= 0:
        return None

    decay = (alpha * alpha) / (2 * (alpha + 1) * (alpha + 1))
    return alpha, decay


def calculate_rhs(
    delta_sd: float,
    T: int,
    delta_overlap: float,
    delta_overlap_min: float,
    k1: int,
    k2: int,
    epsilon: float,
    N: int,
    k_chunks: int,
) -> float:
    """
    Calculates:
    δ ≤ δ_SD + ⌈(T+2)/(Δ_overlap - Δ_overlap,min + 1)⌉ · (k₁2^(h(ε)k₂)e^(-εN/k₁) + k₂e^(-((α^2)/(2(α+1)^2))·N/k₂))
    with μ = (1+α)k and μ = N/k₂ ⇒ α determined by (N, k₂, k).
    """
    coeffs = _alpha_and_decay(N, k2, k_chunks)
    if coeffs is None:
        return math.inf

    _, decay = coeffs

    ceiling_term = math.ceil((T + 3) / (delta_overlap - delta_overlap_min + 1))

    h_epsilon = binary_entropy(epsilon)
    term1_base2 = pow(2, h_epsilon * k2)
    term1_exp_e = math.exp(-epsilon * N / k1)
    exp_term1 = k1 * term1_base2 * term1_exp_e

    exp_term2 = k2 * math.exp(-decay * N / k2)

    return delta_sd + ceiling_term * (exp_term1 + exp_term2)


def find_max_k1(
    k2: int,
    epsilon: float,
    N: int,
    k_chunks: int,
    target_delta: float = 1e-9,
) -> Optional[int]:
    """
    Finds the largest value of k1 that keeps the estimate below target_delta using binary search.
    """
    left, right = 1, 10000
    best_k1 = None
    delta_overlap_min_rounds = 4

    while left <= right:
        mid = (left + right) // 2
        result = calculate_rhs(
            delta_sd=0.0,
            T=T_ROUNDS,
            delta_overlap=DELTA_OVERLAP_ROUNDS,
            delta_overlap_min=delta_overlap_min_rounds,
            k1=mid,
            k2=k2,
            epsilon=epsilon,
            N=N,
            k_chunks=k_chunks,
        )

        if result <= target_delta:
            best_k1 = mid
            left = mid + 1
        else:
            right = mid - 1

    return best_k1


def find_max_k2_with_k1_1(
    epsilon: float,
    N: int,
    k_chunks: int,
    target_delta: float = 1e-9,
) -> Optional[int]:
    """
    Finds the largest k2 satisfying the bound when k1=1 for the supplied k_chunks.
    """
    right = 2000
    left = 1
    best_k2 = None
    delta_overlap_min_rounds = 30 * 60 / SECONDS_PER_ROUND 

    while left <= right:
        mid = (left + right) // 2
        result = calculate_rhs(
            delta_sd=0.0,
            T=T_ROUNDS,
            delta_overlap=DELTA_OVERLAP_ROUNDS,
            delta_overlap_min=delta_overlap_min_rounds,
            k1=1,
            k2=mid,
            epsilon=epsilon,
            N=N,
            k_chunks=k_chunks,
        )

        if result <= target_delta:
            best_k2 = mid
            left = mid + 1
        else:
            right = mid - 1

    return best_k2

def calculate_store_complexity(k1: int, k2: int, n_max: int, n_hon_max: int, K: int, L_msg: int = 1) -> float:
    """
    Calculates the expected communication complexity of the STORE operation.

    Formula:
        (n_max/(k1*k2) * L_msg + 3*n_hon_max*n_max/(k1*k2**2)) * L_msg
    """
    commitment_size = 48 + 96
    term1 = n_max / (k1 * k2) * (L_msg + commitment_size)
    term2 = (3 * n_hon_max * n_max) / (k1 * (k2 ** 2)) * (L_msg / K + 48 + 96)
    return term1 + term2

def generate_rows(
    epsilon: float,
    N: int,
    k_chunks: int,
    target_delta: float = 1e-9,
) -> List[Tuple[int, float, int, int, int]]:
    """
    Returns tuples describing the (k1, k2) pairs for a given epsilon, N, and k.
    Each tuple is (k, epsilon, N, k2, k1).
    """
    max_k2 = find_max_k2_with_k1_1(epsilon, N, k_chunks, target_delta)
    if max_k2 is None:
        return []

    n_max = 2 * N
    n_hon_max = N

    rows = []
    for k2 in range(max_k2, 0, -1):
        k1 = find_max_k1(k2, epsilon, N, k_chunks, target_delta)
        if k1 is None:
            continue
        data_duplication = N / (k2 * k_chunks) 
        store_complexity = calculate_store_complexity(k1, k2, n_max, n_hon_max, k_chunks, L_msg=512)
        rows.append((k_chunks, epsilon, N, k2, k1, data_duplication, store_complexity))

    return rows


def write_rows(
    rows: Sequence[Tuple[int, float, int, int, int, float, float]],
    filename: str,
) -> None:
    headers = ["k", "epsilon", "N", "k2", "k1", "data_duplication", "store_complexity"]
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


if __name__ == "__main__":
    N_values = [2500, 5000, 10000, 100000]
    epsilon_nominator_values = [5, 10]
    target_delta = 1e-9
    k_values = [8, 16]

    for k_chunks in k_values:
        for eps_nom in epsilon_nominator_values:
            epsilon = eps_nom / 100
            for N in N_values:
                rows = generate_rows(epsilon, N, k_chunks, target_delta)
                if not rows:
                    continue

                filename = f"estimates_k_{k_chunks}_eps{eps_nom}_N{N}.csv"
                write_rows(rows, filename)
                print(f"Data written to {filename}")
