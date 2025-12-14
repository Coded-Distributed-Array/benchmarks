import math
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================================
#                               UTILITIES
# =====================================================================

size_map = dict({
    1: 608,   # no coding
    2: 258,   # your RLNC K=2 case
    4: 132,   # your RLNC K=4 case
    8: 161,
    16: 130,
})


def binary_entropy(epsilon: float) -> float:
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError("epsilon must be between 0 and 1")

    if epsilon < 1e-10:
        return -(1 - epsilon) * math.log2(1 - epsilon)
    if epsilon > 1 - 1e-10:
        return -epsilon * math.log2(epsilon)

    return -epsilon * math.log2(epsilon) - (1 - epsilon) * math.log2(1 - epsilon)


# =====================================================================
#                ERROR BOUND (THEOREM 1) — PATCHED FOR k HONEST/COLUMN
# =====================================================================

def calculate_rhs(
    delta_sd: float,
    T: int,
    delta_overlap: float,
    delta_overlap_min: float,
    k1: int,
    k2: int,
    epsilon: float,
    N: int,
    k_required: int = 1
) -> float:
    if delta_overlap < delta_overlap_min:
        return float("inf")

    # giữ đúng T+2 như bạn muốn
    ceil_term = math.ceil((T + 2) / (delta_overlap - delta_overlap_min + 1))

    # ----- ROW TERM: cùng công thức, nhưng tính trong log-space -----
    h_eps = binary_entropy(epsilon)

    # log(term_row) = log k1 + (h_eps * k2) * ln 2 - εN/k1
    log_term_row = (
        math.log(k1)
        + (h_eps * k2) * math.log(2.0)
        - epsilon * N / k1
    )

    # nếu log_term_row quá lớn thì bản thân term_row đã khổng lồ -> trả inf
    if log_term_row > 700:      # ~ ln(1e304), gần sát giới hạn float64
        return float("inf")

    # nếu rất âm thì exp sẽ underflow về 0, ta coi luôn là 0 cho ổn định
    if log_term_row < -700:
        term_row = 0.0
    else:
        term_row = math.exp(log_term_row)

    # ----- COLUMN TERM: Binomial tail như mình bàn lúc trước -----
    p_k = 0.0
    for j in range(k_required):
        p_k += (
            math.comb(N, j)
            * (1.0 / k2) ** j
            * (1.0 - 1.0 / k2) ** (N - j)
        )

    term_col = k2 * p_k

    return delta_sd + ceil_term * (term_row + term_col)


# =====================================================================
#                       BINARY SEARCH FOR k1
# =====================================================================

def find_max_k1(k2: int, epsilon: float, N: int, k_required: int, target_delta: float = 1e-9):

    SECONDS_PER_ROUND = 4
    SECONDS_PER_HOUR = 3600
    SECONDS_PER_YEAR = 31557600  # 10-year lifetime

    T_rounds = int(10 * SECONDS_PER_YEAR / SECONDS_PER_ROUND)
    delta_overlap_rounds = int(6 * SECONDS_PER_HOUR / SECONDS_PER_ROUND)
    delta_overlap_min_rounds = 4  # 4s

    left, right = 1, 20000
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
            N=N,
            k_required=k_required
        )

        if result <= target_delta:
            best_k1 = mid
            left = mid + 1
        else:
            right = mid - 1

    return best_k1


# =====================================================================
#               BINARY SEARCH FOR k2 WHEN k1 = 1
# =====================================================================

def find_max_k2_with_k1_1(epsilon: float, N: int, k_required: int, target_delta: float = 1e-9):

    SECONDS_PER_ROUND = 4
    SECONDS_PER_HOUR = 3600
    SECONDS_PER_YEAR = 31557600

    T_rounds = int(10 * SECONDS_PER_YEAR / SECONDS_PER_ROUND)
    delta_overlap_rounds = int(6 * SECONDS_PER_HOUR / SECONDS_PER_ROUND)
    delta_overlap_min_rounds = 30 * 60 / SECONDS_PER_ROUND  # ~15 min

    left, right = 1, 3000
    best_k2 = None

    while left <= right:
        mid = (left + right) // 2
        result = calculate_rhs(
            delta_sd=0.0,
            T=T_rounds,
            delta_overlap=delta_overlap_rounds,
            delta_overlap_min=delta_overlap_min_rounds,
            k1=1,
            k2=mid,
            epsilon=epsilon,
            N=N,
            k_required=k_required
        )

        if result <= target_delta:
            best_k2 = mid
            left = mid + 1
        else:
            right = mid - 1

    return best_k2


# =====================================================================
#                      COMPLEXITY FUNCTIONS
# =====================================================================

def calculate_joining_complexity(k1: int, k2: int, n_max: int, n_bs: int = 100, t: int = 50, L_msg: int = 1) -> float:
    term1 = 3 * t
    term2 = t * n_bs
    term3 = (t * n_max) / k1
    term4 = ((t + 4) * n_max * k2 - 2 * n_max + n_max * n_max) / (k2 * k2)
    return (term1 + term2 + term3 + term4) * L_msg


def calculate_get_complexity(k1: int, k2: int, n_max: int, n_hon_max: int, K: int, L_msg: int = 1) -> float:
    if K == 1:
        term1 = n_max / (k1 * k2)
        term2 = (n_hon_max) / (k1 * k2)
    else:
        ratio = size_map[K] / size_map[1]
        term1 = (n_max + n_hon_max * K * ratio) / (k1 * k2)
        term2 = (3 * n_hon_max * (n_max + n_hon_max * ratio)) / (k1 * (k2 ** 2))
    return (term1 + term2) * L_msg


def calculate_store_complexity(k1: int, k2: int, n_max: int, n_hon_max: int, K: int, L_msg: int = 1) -> float:
    ratio = size_map[K] / size_map[1]
    term1 = n_max / (k1 * k2)
    term2 = (3 * n_hon_max * n_max * ratio) / (k1 * (k2 ** 2))
    return (term1 + term2) * L_msg


# =====================================================================
#                     GENERATE CURVES FOR A GIVEN k
# =====================================================================

def generate_rows(epsilon: float, N: int, K: int, k_required: int, target_delta: float = 1e-9):

    max_k2 = find_max_k2_with_k1_1(epsilon, N, k_required, target_delta)
    if max_k2 is None:
        return []

    n_max = 5 * N
    n_hon_max = 2 * N

    rows = []
    for k2 in range(max_k2, 0, -1):
        k1 = find_max_k1(k2, epsilon, N, k_required, target_delta)
        if k1 is not None:
            rows.append((
                k2,
                k1,
                1.0 / k2,
                calculate_joining_complexity(k1, k2, n_max),
                calculate_get_complexity(k1, k2, n_max, n_hon_max, K),
                calculate_store_complexity(k1, k2, n_max, n_hon_max, K)
            ))

    return rows


# =====================================================================
#                               MAIN
# =====================================================================

if __name__ == "__main__":

    N_values = [1000, 5000, 10000, 100000]
    epsilon_nominator_values = [5, 10]      # epsilon = 0.05, 0.10
    K_list = [2, 4, 8, 16]                  # <-- K = number of honest required per column

    outdir = Path(".")
    outdir.mkdir(parents=True, exist_ok=True)

    for eps_nom in epsilon_nominator_values:
        epsilon = eps_nom / 100.0

        for N in N_values:
            print(f"\n[Run] ε={epsilon:.2f}, N={N}")

            curves = {}
            for K in K_list:   # K is k_required = honest per column
                rows = generate_rows(epsilon, N, K, k_required=K, target_delta=1e-9)
                if not rows:
                    continue

                rows.sort(key=lambda x: x[0])
                curves[K] = dict(
                    k2=[r[0] for r in rows],
                    k1=[r[1] for r in rows],
                    join=[r[3] for r in rows],
                    get=[r[4] for r in rows],
                    store=[r[5] for r in rows],
                )

            if not curves:
                print(f"⚠️  No data for ε={epsilon:.2f}, N={N}")
                continue

            fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
            plt.subplots_adjust(wspace=0.3, top=0.83)
            markers = {2: "o", 4: "s", 8: "^", 16: "d"}
            labels = {2: "k=2", 4: "k=4", 8: "k=8", 16: "k=16"}

            # k1 curve
            ax = axes[0]
            for K in K_list:
                if K not in curves:
                    continue
                ax.plot(curves[K]["k2"], curves[K]["k1"], marker=markers[K], label=labels[K], linewidth=1.8, markersize=4)
            ax.set_title("Max k1")
            ax.set_xlabel("k2")
            ax.set_ylabel("k1")
            ax.grid(True, alpha=0.3)
            ax.legend(title="k required per column")

            # Join complexity
            ax = axes[1]
            for K in K_list:
                if K not in curves:
                    continue
                ax.plot(curves[K]["k2"], curves[K]["join"], marker=markers[K], label=labels[K], linewidth=1.8, markersize=4)
            ax.set_title("Join Complexity")
            ax.set_xlabel("k2")
            ax.set_ylabel("Join Complexity")
            ax.grid(True, alpha=0.3)

            # GET complexity
            ax = axes[2]
            for K in K_list:
                if K not in curves:
                    continue
                ax.plot(curves[K]["k2"], curves[K]["get"], marker=markers[K], label=labels[K], linewidth=1.8, markersize=4)
            ax.set_title("GET Complexity")
            ax.set_xlabel("k2")
            ax.set_ylabel("GET Complexity")
            ax.grid(True, alpha=0.3)

            # STORE complexity
            ax = axes[3]
            for K in K_list:
                if K not in curves:
                    continue
                ax.plot(curves[K]["k2"], curves[K]["store"], marker=markers[K], label=labels[K], linewidth=1.8, markersize=4)
            ax.set_title("STORE Complexity")
            ax.set_xlabel("k2")
            ax.set_ylabel("STORE Complexity")
            ax.grid(True, alpha=0.3)

            fig.suptitle(f"Complexity Comparison (ε={epsilon:.2f}, N={N})", fontsize=14, y=0.98, weight="bold")

            out_name = outdir / f"complexity_eps{eps_nom}_N{N}.png"
            fig.savefig(out_name.as_posix(), dpi=160, bbox_inches="tight")
            plt.close(fig)