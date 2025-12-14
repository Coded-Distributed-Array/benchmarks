import csv
from simulation_core import *


def save_csv(filename, row_range, statistics_list, step_interval=10):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Time_Step"] + [f'Corruption_Rows_{r}' for r in row_range] + [f'Connections_Rows_{r}' for r in row_range])
        for t in range(0, len(statistics_list[0].corruption_graph), step_interval):
            row = [t]
            for stats in statistics_list:
                row.append(100 * stats.corruption_graph[t] / cols)  # Relative corruption in %
            for stats in statistics_list:
                row.append(stats.connections_graph[t])
            writer.writerow(row)


if __name__ == "__main__":
    cols = 100
    n_total = 2500
    n_init = 20
    n_warmup = n_total - n_init
    steps = 5000
    churn = 50
    lifetime_per_party = n_total // churn
    row_range = [1, 5, 10, 25]

    statistics_list = []

    for rows in row_range:
        params = ProtocolParameters(k1=rows, k2=cols, delta_sub=1, m=100)
        schedule = generate_schedule(n_init=n_init, n_warmup=n_warmup, churn=churn, steps=steps)
        statistics = simulate_protocol_run(schedule, params)
        statistics_list.append(statistics)

    csv_filename = "simulation_data.csv"
    save_csv(csv_filename, row_range, statistics_list, step_interval=10)  # Sample every 10 steps
    print(f"CSV file saved as {csv_filename}")
