#Figure 2 in the manuscript

import numpy as np
import matplotlib as mpl
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import gammainc, gamma
from scipy.special import comb as sp_comb
import math
import random
import networkx as nx
from collections import Counter

mpl.rcParams.update({
    'font.family'         : 'Liberation Serif',
    'font.size'           : 11,
    'axes.labelsize'      : 12,
    'xtick.labelsize'     : 18,
    'ytick.labelsize'     : 18,
    'legend.fontsize'     : 11,
    'axes.titlesize'      : 11,
    'mathtext.fontset'    : 'stix',
    'axes.linewidth'      : 0.8,
    'xtick.direction'     : 'in',
    'ytick.direction'     : 'in',
    'xtick.major.size'    : 4,
    'ytick.major.size'    : 4,
    'xtick.minor.size'    : 2,
    'ytick.minor.size'    : 2,
    'xtick.minor.visible' : True,
    'ytick.minor.visible' : True,
    'figure.dpi'          : 150,
    'savefig.dpi'         : 300,
    'savefig.bbox'        : 'tight',
    'pdf.fonttype'        : 42,
})

def parameters(bar_c, variance):
    delta_c = np.sqrt(12.0 * variance)
    cmin = bar_c - delta_c / 2.0
    cmax = bar_c + delta_c / 2.0
    if cmin <= 0:
        delta_c = bar_c
        cmin    = bar_c - delta_c / 2.0
        cmax    = bar_c + delta_c / 2.0
    elif cmax >= 1:
        delta_c = 1.0 - bar_c
        cmin    = bar_c - delta_c / 2.0
        cmax    = bar_c + delta_c / 2.0
    return delta_c, cmin, cmax

def path_integral(l0, a, b, B, Delta_c, U, lam):
    Tmax = min(l0 * b, U)
    if Tmax <= l0 * a:
        return 0.0
    M_l0  = int(np.floor((Tmax - l0 * a) / B))
    inner = 0.0
    for m in range(M_l0 + 1):
        T_lm = Tmax - (l0 * a + m * B)
        if T_lm <= 0:
            continue
        gval  = gamma(l0) * gammainc(l0, lam * T_lm)
        inner += ((-1)**m) * sp_comb(l0, m, exact=True) \
                 * np.exp(-lam * (l0 * a + m * B)) * gval
    return inner / ((lam * Delta_c)**l0 * math.factorial(l0 - 1))

def calc_uniform(bar_c, variance, d, prob_l, epsilon, lam):
    delta_c, cmin, cmax = parameters(bar_c, variance)
    if cmin <= 0 or cmax >= 1 or cmin >= cmax:
        return np.nan
    a = -np.log(cmax);  b = -np.log(cmin);  B = b - a
    U = -np.log(epsilon)
    return sum(prob_l(l0) * path_integral(l0, a, b, B, delta_c, U, lam)
               for l0 in range(1, d + 1))

def calc_delta(c0, d, prob_l, epsilon, lam):
    if c0 <= 0.0: return 0.0
    if c0 >= 1.0: return 1.0
    l_star = max(0, min(int(np.floor(np.log(epsilon) / np.log(c0))), d))
    if l_star == 0: return 0.0
    if lam == 1:
        return sum(prob_l(l) for l in range(1, l_star + 1))
    else:
        return sum(prob_l(l) * c0**l for l in range(1, l_star + 1))

def build_random_network_pmf(N, k, seed=42, n_samples=1000):
    G = nx.gnm_random_graph(N, int(k * N / 2), seed=seed)
    nodes   = list(G.nodes())
    lengths = []
    sampled_paths = []

    valid_samples = 0
    while valid_samples < n_samples:
        s, t = random.sample(nodes, 2)
        try:
            path = nx.shortest_path(G, s, t)
            lengths.append(len(path) - 1)
            path_edges = [tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1)]
            sampled_paths.append(path_edges)
            valid_samples += 1
        except nx.NetworkXNoPath:
            pass

    counts = Counter(lengths)
    total  = len(lengths)
    ls     = sorted(counts)
    ps     = [counts[l] / total for l in ls]
    return G, ls, ps, sampled_paths

def simulate_network_metrics(G, sampled_paths, bc_arr, variance, epsilon):
    fq_sim, qcm_sim = [], []
    fq_err, qcm_err = [], []
    num_samples = len(sampled_paths)

    for bc in bc_arr:
        delta_c, cmin, cmax = parameters(bc, variance)
        if cmin <= 0 or cmax >= 1 or cmin >= cmax:
            fq_sim.append(np.nan)
            qcm_sim.append(np.nan)
            fq_err.append(np.nan)
            qcm_err.append(np.nan)
            continue

        edge_weights = {tuple(sorted((u, v))): random.uniform(cmin, cmax) for u, v in G.edges()}

        fq_vals = []
        qcm_vals = []

        for path_edges in sampled_paths:
            c_path = 1.0
            for e in path_edges:
                c_path *= edge_weights[e]

            if c_path >= epsilon:
                fq_vals.append(1.0)
                qcm_vals.append(c_path)
            else:
                fq_vals.append(0.0)
                qcm_vals.append(0.0)

        fq_vals = np.array(fq_vals)
        qcm_vals = np.array(qcm_vals)

        fq_sim.append(np.mean(fq_vals))
        qcm_sim.append(np.mean(qcm_vals))
        
        # Standard Error of the Mean (SEM) = std / sqrt(N)
        fq_err.append(np.std(fq_vals, ddof=1) / np.sqrt(num_samples))
        qcm_err.append(np.std(qcm_vals, ddof=1) / np.sqrt(num_samples))

    return np.array(fq_sim), np.array(qcm_sim), np.array(fq_err), np.array(qcm_err)


EPSILON  = 0.3         #Threshold
VARIANCE = 0.005       #Variance of the uniform distribution

d_fc = 1
def prob_l_fc(l): return 1.0 if l == 1 else 0.0

print("Building random network and extracting paths ...")
N_rn, k_rn = 10000, 10
G_rn, lengths_rn, probs_rn, paths_rn = build_random_network_pmf(N_rn, k_rn)
d_rn = max(lengths_rn)
print(f"  PMF lengths : {lengths_rn}")
print(f"  PMF probs   : {[round(p,4) for p in probs_rn]}")

def prob_l_rn(l):
    try:    return probs_rn[lengths_rn.index(l)]
    except: return 0.0


c0_arr = np.linspace(0, 1, 500)
bc_arr = np.linspace(0, 1, 500)

ls_fc = '-'   # Fully Connected = Solid
ls_rn = '--'  # Random Network  = Dashed

col_fq_hom  = 'blue'   # Blue   : QCF Homo
col_qcm_hom = 'black'  # Black  : QCM Homo
col_fq_inh  = 'green'  # Green  : QCF Inhomo
col_qcm_inh = 'red'    # Red    : QCM Inhomo

# Analytic curves for Fully Connected
fq_fc_hom  = np.array([calc_delta(c,    d_fc, prob_l_fc, EPSILON, lam=1) for c  in c0_arr])
qcm_fc_hom = np.array([calc_delta(c,    d_fc, prob_l_fc, EPSILON, lam=2) for c  in c0_arr])
fq_fc_inh  = np.array([calc_uniform(bc, VARIANCE, d_fc, prob_l_fc, EPSILON, lam=1) for bc in bc_arr])
qcm_fc_inh = np.array([calc_uniform(bc, VARIANCE, d_fc, prob_l_fc, EPSILON, lam=2) for bc in bc_arr])

# Analytic curves for Random Network
fq_rn_hom  = np.array([calc_delta(c,    d_rn, prob_l_rn, EPSILON, lam=1) for c  in c0_arr])
qcm_rn_hom = np.array([calc_delta(c,    d_rn, prob_l_rn, EPSILON, lam=2) for c  in c0_arr])
fq_rn_inh  = np.array([calc_uniform(bc, VARIANCE, d_rn, prob_l_rn, EPSILON, lam=1) for bc in bc_arr])
qcm_rn_inh = np.array([calc_uniform(bc, VARIANCE, d_rn, prob_l_rn, EPSILON, lam=2) for bc in bc_arr])


#Numerical simulation for inhomogeneous random network
bc_arr_sim = np.linspace(0.05, 0.95, 20)
fq_rn_sim, qcm_rn_sim, fq_rn_err, qcm_rn_err = simulate_network_metrics(G_rn, paths_rn, bc_arr_sim, VARIANCE, EPSILON)

fig, ax = plt.subplots(figsize=(8.1, 5.5))

# Plot Fully Connected (Solid line '-')
ax.plot(c0_arr, fq_fc_hom,  color=col_fq_hom,  lw=2, linestyle=ls_fc)
ax.plot(c0_arr, qcm_fc_hom, color=col_qcm_hom, lw=2, linestyle=ls_fc)
ax.plot(bc_arr, fq_fc_inh,  color=col_fq_inh,  lw=2, linestyle=ls_fc)
ax.plot(bc_arr, qcm_fc_inh, color=col_qcm_inh, lw=2, linestyle=ls_fc)

# Plot Random Network (Dashed line '--')
ax.plot(c0_arr, fq_rn_hom,  color=col_fq_hom,  lw=2, linestyle=ls_rn)
ax.plot(c0_arr, qcm_rn_hom, color=col_qcm_hom, lw=2, linestyle=ls_rn)
ax.plot(bc_arr, fq_rn_inh,  color=col_fq_inh,  lw=2, linestyle=ls_rn)
ax.plot(bc_arr, qcm_rn_inh, color=col_qcm_inh, lw=2, linestyle=ls_rn)

# Plot Random Network Simulations with Error Bars
ax.errorbar(bc_arr_sim, fq_rn_sim, yerr=fq_rn_err, fmt='o', markersize=4,
            color=col_fq_inh, mfc=col_fq_inh, mec=col_fq_inh, ecolor=col_fq_inh,
            elinewidth=1.0, capsize=2, capthick=1.0, alpha=1)

ax.errorbar(bc_arr_sim, qcm_rn_sim, yerr=qcm_rn_err, fmt='o', markersize=4,
            color=col_qcm_inh, mfc=col_qcm_inh, mec=col_qcm_inh, ecolor=col_qcm_inh,
            elinewidth=1.0, capsize=2, capthick=1.0, alpha=1)

ax.set_xlim(0, 1)
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel(r'$\overline{c}$', size=24)
ax.grid(alpha=0.35, lw=0.5)

ax.set_ylabel(r'$\overline{\mathcal{F}}_Q^{(G)}$', size=20)

ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.tick_params(axis='y', direction='in', labelsize=18, length=4, which='major')
ax2.tick_params(axis='y', direction='in', length=2, which='minor')
ax2.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
ax2.set_ylabel(r'$\overline{\mathcal{Q}}^{(G)}$', size=20)

legend_fc = [
    Line2D([0], [0], color=col_fq_hom,  lw=1.5, ls=ls_fc, label=r'$\overline{\mathcal{F}}_Q^{(G)},\;\mathrm{Homo.}$'),
    Line2D([0], [0], color=col_qcm_hom, lw=1.5, ls=ls_fc, label=r'$\overline{\mathcal{Q}}^{(G)},\;\mathrm{Homo.}$'),
    Line2D([0], [0], color=col_fq_inh,  lw=1.5, ls=ls_fc, label=r'$\overline{\mathcal{F}}_Q^{(G)},\;\mathrm{Inhomo.}$'),
    Line2D([0], [0], color=col_qcm_inh, lw=1.5, ls=ls_fc, label=r'$\overline{\mathcal{Q}}^{(G)},\;\mathrm{Inhomo.}$'),
]

legend_rn = [
    Line2D([0], [0], color=col_fq_hom,  lw=1.5, ls=ls_rn, label=r'$\overline{\mathcal{F}}_Q^{(G)},\;\mathrm{Homo.}$'),
    Line2D([0], [0], color=col_qcm_hom, lw=1.5, ls=ls_rn, label=r'$\overline{\mathcal{Q}}^{(G)},\;\mathrm{Homo.}$'),
    Line2D([0], [0], color=col_fq_inh,  lw=1.5, ls=ls_rn, label=r'$\overline{\mathcal{F}}_Q^{(G)},\;\mathrm{Inhomo.}$'),
    Line2D([0], [0], color=col_qcm_inh, lw=1.5, ls=ls_rn, label=r'$\overline{\mathcal{Q}}^{(G)},\;\mathrm{Inhomo.}$'),
    Line2D([0], [0], marker='o', color=col_fq_inh, markerfacecolor=col_fq_inh, markersize=4, linestyle='None',
           label=r'$\mathrm{Sim.\ }\overline{\mathcal{F}}_Q\mathrm{\ (Inhomo.)}$'),
    Line2D([0], [0], marker='o', color=col_qcm_inh, markerfacecolor=col_qcm_inh, markersize=4, linestyle='None',
           label=r'$\mathrm{Sim.\ }\overline{\mathcal{Q}}\mathrm{\ (Inhomo.)}$'),
]

leg1 = ax.legend(handles=legend_fc, loc='upper left',
                 bbox_to_anchor=(0.0, 1.0),
                 bbox_transform=ax.transAxes,
                 title='Fully Conn. (—)', title_fontsize=12,
                 framealpha=0.85, handlelength=1.4,
                 labelspacing=0.2, handletextpad=0.5,
                 borderpad=0.4, borderaxespad=0.3)

fig.canvas.draw()
leg1_bbox = leg1.get_window_extent()
ax_bbox   = ax.get_window_extent()
leg1_bottom_axes = (leg1_bbox.y0 - ax_bbox.y0) / ax_bbox.height

leg2 = ax.legend(handles=legend_rn, loc='upper left',
                 bbox_to_anchor=(0.0, leg1_bottom_axes),
                 bbox_transform=ax.transAxes,
                 title='Random N/W (- -)', title_fontsize=12,
                 framealpha=0.85, handlelength=1.4,
                 labelspacing=0.2, handletextpad=0.5,
                 borderpad=0.4, borderaxespad=0.3)

ax.add_artist(leg1)

fig.tight_layout()
plt.show()
