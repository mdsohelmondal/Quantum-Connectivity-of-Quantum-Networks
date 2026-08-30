import random
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import logging
from matplotlib.ticker import FormatStrFormatter

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "cm",
})


def random_network(num_nodes, total_edges, c, seed=None):
    G = nx.gnm_random_graph(num_nodes, total_edges, seed=seed)
    for edge in G.edges():
        G.edges[edge]["conc"] = c
    return G


def sample_shortest_path_lengths(G, num_samples=10000, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    nodes = list(G.nodes())
    num_nodes = len(nodes)
    lengths = []

    sources = rng.integers(0, num_nodes, size=num_samples)
    targets = rng.integers(0, num_nodes, size=num_samples)

    for u, v in zip(sources, targets):
        if u == v:
            continue
        try:
            length = nx.shortest_path_length(G, source=int(u), target=int(v))
            lengths.append(length)
        except nx.NetworkXNoPath:
            continue

    return lengths


# Parameters
num_nodes = 10000
num_edges = 50000
c = 0.6
num_realizations = 10
samples_per_realization = 10000
base_seed = 42

realization_counts = []
max_observed_length = 0

# Collect distributions across 10 realizations
for r in range(num_realizations):
    graph_seed = base_seed + r
    rng = np.random.default_rng(seed=1000 + r)

    G = random_network(num_nodes, num_edges, c=c, seed=graph_seed)
    lengths = sample_shortest_path_lengths(
        G, num_samples=samples_per_realization, rng=rng
    )

    counts = Counter(lengths)
    realization_counts.append(counts)

    if lengths:
        max_observed_length = max(max_observed_length, max(lengths))

# Align all realizations onto a unified path-length grid
all_lengths = list(range(1, max_observed_length + 1))
fractions_matrix = []

for counts in realization_counts:
    total_paths = sum(counts.values())
    fractions_matrix.append(
        [counts.get(l, 0) / total_paths for l in all_lengths]
    )

fractions_matrix = np.array(fractions_matrix)  # shape: (10, len(all_lengths))

# Compute mean PMF and standard deviation
mean_fractions = np.mean(fractions_matrix, axis=0)
std_fractions = np.std(fractions_matrix, axis=0)

# Plot averaged distribution
fig, ax = plt.subplots(figsize=(7, 4))
ax.grid(True, linestyle="--", alpha=0.6)

# Bar chart of the averaged PMF with error bars
ax.bar(
    all_lengths,
    mean_fractions,
    yerr=std_fractions,
    capsize=4,
    width=0.6,
    color="green",
    align="center",
    edgecolor="black",
    linewidth=0.5,
)

ax.set_xlabel(r"$\ell_0$", fontsize=24)
ax.set_ylabel(r"$q(\ell_0)$", fontsize=24)
ax.set_xticks(all_lengths)
ax.tick_params(axis="both", labelsize=18)
ax.set_ylim(0, 0.6)

plt.tight_layout()
plt.show()
