import math
import matplotlib.pyplot as plt

CELL_SIZE = 512          # bytes per cell
COMMITMENT_UNIT_SIZE = 48  # bytes per commitment

n_values = list(range(8, 513, 8))
chunk_sizes = [2, 4, 8, 16]

plt.figure()

for chunk_size in chunk_sizes:
    ratios = []
    for n in n_values:
        block_size = (n * n) * CELL_SIZE
        commitment_size = n * chunk_size * COMMITMENT_UNIT_SIZE
        ratio = commitment_size / block_size
        ratios.append(ratio)
    plt.plot(n_values, ratios, label=f"chunk_size={chunk_size}")

plt.xlabel("n (matrix dimension, block = n x n cells)")
plt.ylabel("Commitment Size / Block Size")
plt.title("Commitment Size Ratio vs Block Size (varying n and chunk_size)")
plt.legend()
plt.show()