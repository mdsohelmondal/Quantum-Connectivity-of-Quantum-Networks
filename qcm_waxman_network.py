#Figure 3 in the manuscript

import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from math import pi


def path_edges_parameter(G, path):
    adj_matrix = nx.to_numpy_array(G, weight='conc')
    result = []
    for i in range(len(path)):
        if i + 1 < len(path):
            result.append(adj_matrix[path[i], path[i + 1]])
    return result


def unique_paths(G, S, D, N):
    def remove_edges(G, path):
        edges_to_remove = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        G.remove_edges_from(edges_to_remove)

    unique_paths = []
    G_copy = G.copy()
    k = 1

    while k < N + 1:
        try:
            shortest_path1 = nx.shortest_path(G, source=S, target=D)
            unique_paths.append(shortest_path1)
            remove_edges(G, shortest_path1)
        except nx.NetworkXNoPath:
            pass
        k += 1

    G.clear()
    G.add_nodes_from(G_copy.nodes(data=True))
    G.add_edges_from((u, v, d) for u, v, d in G_copy.edges(data=True))

    return unique_paths


def all_pairs(lst):
    pairs = []
    for i in range(len(lst)):
        for j in range(len(lst)):
            if i != j:
                pairs.append((lst[i], lst[j]))
    return pairs


def quantum_conn_coeff(G, set_nodes, k):
    node_pairs = all_pairs(set_nodes)
    s = len(node_pairs)

    if s == 0:
        return 0

    qcc_single_pair = []

    for node1, node2 in node_pairs:
        if nx.has_path(G, node1, node2):

            paths = unique_paths(G, node1, node2, k)
            paths_weight = []

            for path in paths:
                path_weight = 1
                weights = path_edges_parameter(G, path)

                for weight in weights:
                    path_weight *= weight

                paths_weight.append(path_weight)

            if max(paths_weight) >= 0.3:
                qcc_single_pair.append(max(paths_weight))
            else:
                qcc_single_pair.append(0)

    return sum(qcc_single_pair) * (1 / s)


def quantum_clus_coeff(G, node):
    set_nodes = list(G.neighbors(node))
    return quantum_conn_coeff(G, set_nodes, 1)


def waxman_network(
    N=500,
    R=1000,
    alphaL=226,
    beta=1.0,
    gamma=0.2,
    np_photons=1000,
    seed=42
):

    rng = np.random.default_rng(seed)

    theta = 2 * np.pi * rng.random(N)
    r = R * np.sqrt(rng.random(N))

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    positions = np.column_stack((x, y))

    dist_mat = distance_matrix(positions, positions)
    np.fill_diagonal(dist_mat, np.inf)

    waxman_prob = beta * np.exp(-dist_mat / alphaL)

    waxman_graph = nx.Graph()
    waxman_graph.add_nodes_from(range(N))

    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < waxman_prob[i, j]:
                waxman_graph.add_edge(i, j, length=dist_mat[i, j])

    quantum_graph = nx.Graph()
    quantum_graph.add_nodes_from(range(N))

    for i, j, attr in waxman_graph.edges(data=True):

        dij = attr['length']
        pij = 10 ** (-gamma * dij / 10)

        Pij = 1 - (1 - pij) ** np_photons

        if rng.random() < Pij:
            quantum_graph.add_edge(i, j, length=dij)

    for edge in quantum_graph.edges():
        quantum_graph.edges[edge]['conc'] = 0.6

    return quantum_graph, positions


def find_local_nodes(center_location, pos_dict, radius):

    pos_array = np.array([pos_dict[i] for i in range(len(pos_dict))])

    distances = np.linalg.norm(
        pos_array - np.array(center_location), axis=1
    )

    return list(np.where(distances <= radius)[0])


fig, ax = plt.subplots(figsize=(10, 8))

ax.set_xlim(-1100, 1100)
ax.set_ylim(-1100, 1100)

x_ticks = np.linspace(-1000, 1000, 5)
y_ticks = np.linspace(-1000, 1000, 5)

ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)

ax.set_xticklabels([f'{int(x)}' for x in x_ticks], fontsize=28)
ax.set_yticklabels([f'{int(y)}' for y in y_ticks], fontsize=28)

# Force non-italic tick labels
for label in ax.get_xticklabels():
    label.set_fontstyle('normal')

for label in ax.get_yticklabels():
    label.set_fontstyle('normal')

ax.tick_params(axis='both', labelsize=28)

ax.set_xlabel('X', fontsize=28)
ax.set_ylabel('Y', fontsize=28)

ax.grid(True, alpha=0.3)

G, pos = waxman_network()

lcc = []
x = []
y = []

all_positions = np.array([pos[i] for i in range(len(pos))])

thetas1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
thetas2 = np.linspace(0, 2 * np.pi, 12, endpoint=False)

circle1 = [(400 * np.cos(t), 400 * np.sin(t)) for t in thetas1]
circle2 = [(800 * np.cos(t), 800 * np.sin(t)) for t in thetas2]

centers = [(0, 0)] + circle1 + circle2

for center in centers:

    x1, y1 = center
    x.append(x1)
    y.append(y1)

    local_nodes = find_local_nodes(center, pos, radius=200)

    lcc.append(quantum_conn_coeff(G, local_nodes, 1))

for edge in G.edges():

    x1, y1 = pos[edge[0]]
    x2, y2 = pos[edge[1]]

    ax.plot([x1, x2], [y1, y2], 'gray', linewidth=0.5, alpha=0.7)

node_x = [pos[node][0] for node in G.nodes()]
node_y = [pos[node][1] for node in G.nodes()]

ax.scatter(node_x, node_y, s=20, c='blue', alpha=0.7)

norm = mcolors.Normalize(vmin=min(lcc), vmax=max(lcc))
cmap = cm.get_cmap('viridis')

for i in range(len(x)):

    color = cmap(norm(lcc[i]))

    circle = Circle(
        (x[i], y[i]),
        radius=200,
        color=color,
        alpha=0.6,
        zorder=10
    )

    ax.add_patch(circle)

sc = ax.scatter(x, y, c=lcc, cmap='viridis', s=0)

cbar = plt.colorbar(sc, ax=ax)

cbar.set_label(r'$\overline{\mathcal{Q}}^{(G)}$', fontsize=28)

cbar.ax.tick_params(labelsize=20)

for label in cbar.ax.get_yticklabels():
    label.set_fontstyle('normal')

ax.set_aspect('equal')

plt.tight_layout()
plt.show()
