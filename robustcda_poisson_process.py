import random
import math
from typing import Dict
import matplotlib.pyplot as plt
from grid_core import PartyGrid


def exp_sample(rng: random.Random, rate: float) -> float:
    return -math.log(1.0 - rng.random()) / rate


class Node:
    def __init__(self):
        self.busy_until = 0.0

    def request(self, now: float, mu: float, deadline: float, rng: random.Random) -> bool:
        start = max(now, self.busy_until)
        finish = start + exp_sample(rng, mu)
        self.busy_until = finish
        return finish <= now + deadline


def sample_one_cell(
    sampler_id: int,
    grid: PartyGrid,
    nodes: Dict[int, Node],
    k: int,
    mu_cell: float,
    mu_chunk: float,
    d_cell: float,
    d_chunk: float,
    now: float,
    rng: random.Random,
) -> bool:
    row_self, _ = grid.party_to_cell[sampler_id]

    blob_x = rng.randint(0, 511)
    col = blob_x % grid.k2

    # ---- Phase A ----
    gateways = grid.cell_nodes(row_self, col)
    if not gateways:
        return False

    B = rng.choice(gateways)
    if not nodes[B].request(now, mu_cell, d_cell, rng):
        return False

    # ---- Phase B ----
    piece_nodes = grid.column_nodes(col)
    total_success = 0
    for C in piece_nodes:
        if nodes[C].request(now, mu_chunk, d_chunk, rng):
            total_success += 1
    
    if total_success < k:
        return False

    return True


# ============================================================
# BENCHMARK — sweep M
# ============================================================

def bench_parallel_sampling():
    # ---- protocol ----
    N = 5000
    k1, k2 = 5, 50
    k = 16

    # ---- timing ----
    block_time = 12.0
    samples_per_node = 75
    total_samples = N * samples_per_node

    Ms = [2000, 2500, 5000, 10000, 20000, 100000]

    # ---- service + deadline ----
    mu_cell = 100
    mu_chunk = 2000
    d_cell = 0.5
    d_chunk = 0.05

    # ---- grid ----
    grid = PartyGrid(k1, k2, seed=42)
    for i in range(N):
        grid.add_party(i)

    rng = random.Random(0)

    results = []

    for M in Ms:
        # recompute dt for this M
        dt = block_time / (samples_per_node * N / M)

        # reset node states
        nodes = {i: Node() for i in range(N)}

        now = 0.0
        success = 0
        trials = 0

        while trials < total_samples:
            for _ in range(M):
                sampler_id = rng.randint(0, N - 1)
                ok = sample_one_cell(
                    sampler_id,
                    grid,
                    nodes,
                    k,
                    mu_cell,
                    mu_chunk,
                    d_cell,
                    d_chunk,
                    now,
                    rng,
                )
                if ok:
                    success += 1
                trials += 1
            now += dt

        rate = success / trials
        results.append((M, rate))
        print(f"M={M:4d} | success rate = {rate:.4f}")

    # ---- plot ----
    xs, ys = zip(*results)
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Number of concurrent samplers (M)")
    plt.ylabel("Single-sample success probability")
    plt.title("Sampling success vs concurrency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    bench_parallel_sampling()