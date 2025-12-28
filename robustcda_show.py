from grid_core import *
import matplotlib.pyplot as plt



if __name__ == "__main__":
    # ================== Simulation Parameters ==================
    n_total = 2500
    n_init = 20
    n_warmup = n_total - n_init
    steps = 5000
    churn = 50
    lifetime_per_party = n_total // churn

    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

    # ================== Create 2x2 Figure ==================
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f'Protocol Simulation ({n_total} honest parties, lifetime {lifetime_per_party} steps)',
        fontsize=14
    )

    # ================== TOP-LEFT & BOTTOM-LEFT: k2 = 50 ==================
    k2 = 50
    rows_list = []

    for i, k1 in enumerate(rows_list):
        params = ProtocolParameters(k1=k1, k2=k2, delta_sub=1, m=100)
        schedule = generate_schedule(
            n_init=n_init,
            n_warmup=n_warmup,
            churn=churn,
            steps=steps
        )
        stats = simulate_protocol_run(schedule, params)

        # Corruption (relative %)
        corruption_pct = [100 * x / k2 for x in stats.corruption_graph]
        axes[0, 0].plot(
            corruption_pct,
            label=f'k1 = {k1}',
            marker=markers[i],
            color=colors[i],
            markevery=500
        )

        # Connections
        axes[1, 0].plot(
            stats.connections_graph,
            label=f'k1 = {k1}',
            marker=markers[i],
            color=colors[i],
            markevery=500
        )

    # ================== TOP-RIGHT & BOTTOM-RIGHT: k2 = 100 ==================
    k2 = 33
    rows_list = [5]

    for i, k1 in enumerate(rows_list):
        params = ProtocolParameters(k1=k1, k2=k2, delta_sub=1, m=100)
        schedule = generate_schedule(
            n_init=n_init,
            n_warmup=n_warmup,
            churn=churn,
            steps=steps
        )
        stats = simulate_protocol_run(schedule, params)

        # Corruption (relative %)
        corruption_pct = [100 * x / k2 for x in stats.corruption_graph]
        axes[0, 1].plot(
            corruption_pct,
            label=f'k1 = {k1}',
            marker=markers[i],
            color=colors[i],
            markevery=500
        )

        # Connections
        axes[1, 1].plot(
            stats.connections_graph,
            label=f'k1 = {k1}',
            marker=markers[i],
            color=colors[i],
            markevery=500
        )

    # ================== Styling & Labels ==================
    # --- Corruption plots
    axes[0, 0].set_title("Our Corruption Graphs (k2 = 50)")
    axes[0, 1].set_title("RDA Corruption Graphs (k2 = 100)")
    i = 0
    for ax in [axes[0, 0], axes[0, 1]]:
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Max Fraction of Corrupted Symbols (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True)
        i += 1

    # --- Connection plots
    axes[1, 0].set_title("Our Connection Graphs (k2 = 50)")
    axes[1, 1].set_title("RDA Connection Graphs (k2 = 100)")
    i = 0
    for ax in [axes[1, 0], axes[1, 1]]:
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Max Number of Peers")
        ax.legend()        
        ax.grid(True)
        i += 1

    # ================== Layout Fix ==================
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()