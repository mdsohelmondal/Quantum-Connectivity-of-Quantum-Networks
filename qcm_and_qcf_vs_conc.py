import numpy as np
import matplotlib as mpl
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    'legend.fontsize'     : 13,
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
    """Adjusts limits near 0 and 1 to prevent out-of-bounds bounds."""
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

def numeric_integral(l0, cmin, cmax, epsilon, is_qcm, n_mc=100000):
    """
    Numerically integrates over the uniform distribution p(c) using Monte Carlo.
    n_mc: Number of Monte Carlo samples for robust numerical accuracy.
    """
    # Sample edge concurrences for a path of length l0
    c_samples = np.random.uniform(cmin, cmax, size=(n_mc, l0))
    # Calculate path connection strength (product of concurrences)
    path_str = np.prod(c_samples, axis=1)
    # Check if the path meets the functional threshold
    valid = path_str >= epsilon
    
    if is_qcm:
        # Expected value of connection strength for valid paths
        return np.mean(path_str * valid)
    else:
        # Fraction of valid paths (QCF)
        return np.mean(valid)

def calc_uniform(bar_c, variance, d, prob_l, epsilon, is_qcm, n_mc=100000):
    delta_c, cmin, cmax = parameters(bar_c, variance)
    # Safety check
    if cmin <= 0 or cmax >= 1 or cmin >= cmax:
        return np.nan
    
    expected_value = 0.0
    for l0 in range(1, d + 1):
        pl = prob_l(l0)
        if pl > 0:
            expected_value += pl * numeric_integral(l0, cmin, cmax, epsilon, is_qcm, n_mc)
    return expected_value

def calc_delta(c0, d, prob_l, epsilon, is_qcm):
    if c0 <= 0.0: return 0.0
    if c0 >= 1.0: return 1.0
    l_star = max(0, min(int(np.floor(np.log(epsilon) / np.log(c0))), d))
    if l_star == 0: return 0.0
    
    if not is_qcm:
        return sum(prob_l(l) for l in range(1, l_star + 1))
    else:
        return sum(prob_l(l) * c0**l for l in range(1, l_star + 1))

def build_random_network_pmf(N, k, seed=42, n_samples=10000):
    G = nx.gnm_random_graph(N, int(k * N / 2), seed=seed)
    nodes   = list(G.nodes())
    lengths = []
    for _ in range(n_samples):
        s, t = random.sample(nodes, 2)
        if nx.has_path(G, s, t):
            try:
                lengths.append(nx.shortest_path_length(G, s, t))
            except nx.NetworkXNoPath:
                pass
    counts = Counter(lengths)
    total  = len(lengths)
    ls     = sorted(counts)
    ps     = [counts[l] / total for l in ls]
    return ls, ps

# ─────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────
EPSILON  = 0.3
VARIANCE = 0.005

d_fc = 1
def prob_l_fc(l): return 1.0 if l == 1 else 0.0

print("Building random network ...")
N_rn, k_rn = 10000, 10
lengths_rn, probs_rn = build_random_network_pmf(N_rn, k_rn)
d_rn = max(lengths_rn)
print(f"  PMF lengths : {lengths_rn}")
print(f"  PMF probs   : {[round(p,4) for p in probs_rn]}")

def prob_l_rn(l):
    try:    return probs_rn[lengths_rn.index(l)]
    except: return 0.0

# ─────────────────────────────────────────────────────────────
# Evaluate curves
# ─────────────────────────────────────────────────────────────
c0_arr = np.linspace(0.001, 0.999, 150)
bc_arr = np.linspace(0.001, 0.999, 200)

col_fc = 'magenta'
col_rn = 'green'

# Calculate QCF and QCM arrays (Notice is_qcm flag replacing lam)
fq_fc_hom  = np.array([calc_delta(c,    d_fc, prob_l_fc, EPSILON, is_qcm=False) for c  in c0_arr])
fq_fc_inh  = np.array([calc_uniform(bc, VARIANCE, d_fc, prob_l_fc, EPSILON, is_qcm=False) for bc in bc_arr])
qcm_fc_inh = np.array([calc_uniform(bc, VARIANCE, d_fc, prob_l_fc, EPSILON, is_qcm=True) for bc in bc_arr])

fq_rn_hom  = np.array([calc_delta(c,    d_rn, prob_l_rn, EPSILON, is_qcm=False) for c  in c0_arr])
fq_rn_inh  = np.array([calc_uniform(bc, VARIANCE, d_rn, prob_l_rn, EPSILON, is_qcm=False) for bc in bc_arr])
qcm_rn_inh = np.array([calc_uniform(bc, VARIANCE, d_rn, prob_l_rn, EPSILON, is_qcm=True) for bc in bc_arr])

# ─────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.1, 5))

ax.plot(c0_arr, fq_fc_hom,  color=col_fc, lw=2, linestyle='-')
ax.plot(bc_arr, fq_fc_inh,  color=col_fc, lw=2, linestyle='--')
ax.plot(bc_arr, qcm_fc_inh, color=col_fc, lw=2, linestyle=':')

ax.plot(c0_arr, fq_rn_hom,  color=col_rn, lw=2, linestyle='-')
ax.plot(bc_arr, fq_rn_inh,  color=col_rn, lw=2, linestyle='--')
ax.plot(bc_arr, qcm_rn_inh, color=col_rn, lw=2, linestyle=':')

ax.set_xlim(0, 1)
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel(r'$\overline{c}$', size=24)
ax.grid(alpha=0.35, lw=0.5)

ax.set_ylabel(r'$\overline{\mathcal{F}}^{(G)}$', size=20)

ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.tick_params(axis='y', direction='in',
                labelsize=18, length=4, which='major')
ax2.tick_params(axis='y', direction='in',
                length=2, which='minor')
ax2.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
ax2.set_ylabel(r'$\overline{\mathcal{Q}}^{(G)}$', size=20)

legend_fc = [
    Line2D([0], [0], color=col_fc, lw=1.4, ls='-',
           label=r'$\overline{\mathcal{F}}^{(G)},\;\mathrm{Homo.}$'),
    Line2D([0], [0], color=col_fc, lw=1.4, ls='--',
           label=r'$\overline{\mathcal{F}}^{(G)},\;\mathrm{Inhomo.}$'),
    Line2D([0], [0], color=col_fc, lw=1.4, ls=':',
           label=r'$\overline{\mathcal{Q}}^{(G)},\;\mathrm{Inhomo.}$'),
]

legend_rn = [
    Line2D([0], [0], color=col_rn, lw=1.4, ls='-',
           label=r'$\overline{\mathcal{F}}^{(G)},\;\mathrm{Homo.}$'),
    Line2D([0], [0], color=col_rn, lw=1.4, ls='--',
           label=r'$\overline{\mathcal{F}}^{(G)},\;\mathrm{Inhomo.}$'),
    Line2D([0], [0], color=col_rn, lw=1.4, ls=':',
           label=r'$\overline{\mathcal{Q}}^{(G)},\;\mathrm{Inhomo.}$'),
]

leg1 = ax.legend(handles=legend_fc, loc='upper left',
                 bbox_to_anchor=(0.0, 1.0),
                 bbox_transform=ax.transAxes,
                 title='Fully Conn.', title_fontsize=13,
                 framealpha=0.8, handlelength=1.1,
                 labelspacing=0.15, handletextpad=0.5,
                 borderpad=0.55, borderaxespad=0.4)

fig.canvas.draw()
leg1_bbox = leg1.get_window_extent()               
ax_bbox   = ax.get_window_extent()                 
leg1_bottom_axes = (leg1_bbox.y0 - ax_bbox.y0) / ax_bbox.height

leg2 = ax.legend(handles=legend_rn,
                 loc='upper left',
                 bbox_to_anchor=(0.0, leg1_bottom_axes),
                 bbox_transform=ax.transAxes,
                 title='Random N/W', title_fontsize=13,
                 framealpha=0.8, handlelength=1.1,
                 labelspacing=0.15, handletextpad=0.5,
                 borderpad=0.55, borderaxespad=0.4)

ax.add_artist(leg1)   

fig.tight_layout()
plt.show()