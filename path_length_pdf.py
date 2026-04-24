import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

def random_network(num_nodes, total_edges, c, seed=42):
    G = nx.gnm_random_graph(num_nodes, total_edges, seed=seed)
    for edge in G.edges():
        G.edges[edge]['conc'] = c
    return G

def all_pairs(lst):
    pairs = []
    for i in range(len(lst)):
        for j in range(len(lst)):
            if i != j:
                pairs.append((lst[i], lst[j]))
    return pairs

def shortest_path_length_distribution(G, set_nodes, num_samples=200):
    node_pairs = random.sample(all_pairs(set_nodes), num_samples)
    lengths = []
    for node1, node2 in node_pairs:
        if nx.has_path(G, node1, node2):
            length = nx.shortest_path_length(G, source=node1, target=node2)
            lengths.append(length)
    return lengths

# Build network and collect shortest path lengths
G = random_network(10000, 50000, c=0.6, seed=42)
lengths = shortest_path_length_distribution(G, list(G.nodes()), num_samples=10000)

# Compute fraction distribution
unique_lengths = sorted(set(lengths))
min_l, max_l = min(unique_lengths), max(unique_lengths)
all_lengths = list(range(1, max_l + 1))

total = len(lengths)
fractions = [lengths.count(l) / total for l in all_lengths]

# Plot
fig, ax = plt.subplots(figsize=(7, 4))
ax.grid('True')
ax.bar(all_lengths, fractions, width=0.6, color='green', align='center')

ax.set_xlabel(r'$\ell_0$', fontsize=24)
ax.set_ylabel(r'$q(\ell_0)$', fontsize=24)
ax.set_xticks(all_lengths)
ax.tick_params(axis='both', labelsize=18)#, length=6, width=1.5)
ax.set_ylim(0, 0.6)
plt.show()