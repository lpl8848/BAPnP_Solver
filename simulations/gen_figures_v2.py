"""
Generate publication-quality figures for BAPnP paper.
- Noise/density line plots: one figure per metric (full column width)
- Boxplots: y-axis truncated to focus on working range
- Planar: y-axis truncated to show detail
"""
import scipy.io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

outdir = r'D:\研究生学习\pnp\pnp\论文写作 - 副本\final'

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.labelsize': 11, 'legend.fontsize': 7,
    'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03
})

# Colors
C = {
    'BAPnP':      '#D55E00', 'BAPnP-GN': '#CC0000', 'EPnP-GN': '#1f77b4',
    'OPnP':       '#2ca02c', 'RPnP':     '#9467bd', 'SRPnP-GN':'#8c564b',
    'MLPnP':      '#e377c2', 'CPnP-GN':  '#7f7f7f', 'SQPnP':   '#bcbd22',
    'oDLT-GN':    '#17becf', 'EPnP-GN-Greedy': '#1f77b4',
}
S = {  # styles
    'BAPnP':'--', 'BAPnP-GN':'-', 'EPnP-GN':'-.',
    'OPnP':':', 'RPnP':'--', 'SRPnP-GN':'-.',
    'MLPnP':':', 'CPnP-GN':'--', 'SQPnP':'-',
    'oDLT-GN':'-.', 'EPnP-GN-Greedy':':',
}
M = {  # markers
    'BAPnP':'o', 'BAPnP-GN':'s', 'EPnP-GN':'^',
    'OPnP':'D', 'RPnP':'v', 'SRPnP-GN':'<',
    'MLPnP':'>', 'CPnP-GN':'p', 'SQPnP':'*',
    'oDLT-GN':'h', 'EPnP-GN-Greedy':'X',
}

def gs(name):
    c = C.get(name, 'black')
    ls = S.get(name, '-')
    m = M.get(name, 'o')
    lw = 2.2 if name == 'BAPnP-GN' else 1.3
    ms = 5 if name == 'BAPnP-GN' else 3.5
    return c, ls, m, lw, ms

# Load data
print("Loading...")
e1 = scipy.io.loadmat(r'D:\R2020b\bin\pnp\exp1_results.mat')
e2 = scipy.io.loadmat(r'D:\R2020b\bin\pnp\exp2_results.mat')
pl = scipy.io.loadmat(r'D:\R2020b\bin\pnp\planar_results.mat')

e1n = [str(x[0][0]).strip() for x in e1['algo_names']]
e2n = [str(x[0][0]).strip() for x in e2['algo_names']]
pln = [str(x[0][0]).strip() for x in pl['algo_names']]

noise = e1['noise_levels'].flatten()
npts = e2['n_points_list'].flatten()
zlev = pl['z_spread_levels'].flatten()

# Figure size: single column ~3.5in, double column ~7.2in
SW = 3.5   # single column width
DW = 7.2   # double column width

# ============================================================
# NOISE SENSITIVITY - One figure per metric (single column)
# ============================================================
print("Noise sensitivity figures...")
noise_order = ['BAPnP-GN','BAPnP','EPnP-GN','OPnP','RPnP','SRPnP-GN','MLPnP','CPnP-GN','SQPnP','oDLT-GN']

for metric, data, ylbl, fname in [
    ('rot', e1['median_rot_err_per_level'], 'Median Rotation Error (deg)', 'noise_R'),
    ('trans', e1['median_trans_err_per_level'], 'Median Translation Error (%)', 'noise_T'),
    ('reprj', e1['median_repr_err_per_level'], 'Median Norm. Reprojection Error', 'noise_P'),
]:
    fig, ax = plt.subplots(figsize=(SW*1.15, SW*0.85))
    for name in noise_order:
        if name not in e1n: continue
        j = e1n.index(name)
        c, ls, m, lw, ms = gs(name)
        ax.plot(noise, data[:, j], color=c, linestyle=ls, marker=m,
                markersize=ms, linewidth=lw, label=name)
    ax.set_xlabel('Gaussian Noise σ (pixels)')
    ax.set_ylabel(ylbl)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(noise))
    ax.legend(ncol=2, fontsize=6, loc='upper left')
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, fname + '.pdf'))
    fig.savefig(os.path.join(outdir, fname + '.eps'))
    plt.close(fig)
    print(f'  {fname} saved')

# ============================================================
# NOISE BOXPLOTS - y-axis truncated
# ============================================================
print("Noise boxplots...")
raw_rot = e1['raw_rot_data']
box_sigmas = [1.0, 3.0, 5.0]
box_idx = [list(noise).index(s) for s in box_sigmas]

for sigma, idx in zip(box_sigmas, box_idx):
    rot_data = raw_rot[idx, 0]
    fig, ax = plt.subplots(figsize=(SW*1.2, SW*0.7))
    box_data, box_labels = [], []
    for name in noise_order:
        if name not in e1n: continue
        j = e1n.index(name)
        vals = rot_data[:, j]
        vals = vals[~np.isnan(vals)]
        box_data.append(vals)
        box_labels.append(name)
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                    sym='', showfliers=False)
    for patch, name in zip(bp['boxes'], box_labels):
        patch.set_facecolor(C.get(name, 'white'))
        patch.set_alpha(0.5)
    ax.set_title(f'σ = {sigma:.1f} px')
    ax.set_ylabel('Rotation Error (deg)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    # Truncate y-axis based on sigma
    ymax_vals = {1.0: 0.5, 3.0: 2.0, 5.0: 2.5}
    ax.set_ylim(0, ymax_vals.get(sigma, 3.0))
    plt.tight_layout()
    tag = {1.0: '', 3.0: '2', 5.0: '3'}.get(sigma, '')
    fname = f'noise_R_Box{tag}'
    fig.savefig(os.path.join(outdir, fname + '.pdf'))
    fig.savefig(os.path.join(outdir, fname + '.eps'))
    plt.close(fig)
    print(f'  {fname} saved')

# ============================================================
# POINT DENSITY - One figure per metric (single column)
# ============================================================
print("Point density figures...")
dens_order = noise_order  # same

for metric, data, ylbl, fname in [
    ('rot', e2['median_rot_err_per_level'], 'Median Rotation Error (deg)', 'nub_R'),
    ('trans', e2['median_trans_err_per_level'], 'Median Translation Error (%)', 'nub_T'),
    ('reprj', e2['median_repr_err_per_level'], 'Median Norm. Reprojection Error', 'nub_P'),
]:
    fig, ax = plt.subplots(figsize=(SW*1.15, SW*0.85))
    for name in dens_order:
        if name not in e2n: continue
        j = e2n.index(name)
        c, ls, m, lw, ms = gs(name)
        ax.semilogy(npts, data[:, j], color=c, linestyle=ls, marker=m,
                    markersize=ms, linewidth=lw, label=name)
    ax.set_xlabel('Number of Points (N)')
    ax.set_ylabel(ylbl)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=6, loc='upper right')
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, fname + '.pdf'))
    fig.savefig(os.path.join(outdir, fname + '.eps'))
    plt.close(fig)
    print(f'  {fname} saved')

# ============================================================
# DENSITY BOXPLOTS - y-axis truncated
# ============================================================
print("Density boxplots...")
raw_rot_e2 = e2['raw_rot_data']
box_ns = [6, 10, 100]
box_idx2 = [list(npts).index(n) for n in box_ns]
ylims = {6: 4.0, 10: 1.5, 100: 0.5}

for nn, idx in zip(box_ns, box_idx2):
    rot_data = raw_rot_e2[idx, 0]
    fig, ax = plt.subplots(figsize=(SW*1.2, SW*0.7))
    box_data, box_labels = [], []
    for name in dens_order:
        if name not in e2n: continue
        j = e2n.index(name)
        vals = rot_data[:, j]
        vals = vals[~np.isnan(vals)]
        box_data.append(vals)
        box_labels.append(name)
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                    sym='', showfliers=False)
    for patch, name in zip(bp['boxes'], box_labels):
        patch.set_facecolor(C.get(name, 'white'))
        patch.set_alpha(0.5)
    ax.set_title(f'N = {nn}')
    ax.set_ylabel('Rotation Error (deg)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, ylims.get(nn, 3.0))
    plt.tight_layout()
    tag = {6: '', 10: '2', 100: '3'}.get(nn, '')
    fname = f'nub_R_Box{tag}'
    fig.savefig(os.path.join(outdir, fname + '.pdf'))
    fig.savefig(os.path.join(outdir, fname + '.eps'))
    plt.close(fig)
    print(f'  {fname} saved')

# ============================================================
# PLANAR - y-axis truncated
# ============================================================
print("Planar figures...")
z_labels = [f'$10^{{{int(np.log10(z))}}}$' if z > 0 else '$0$' for z in zlev]
x_idx = np.arange(len(zlev))
planar_order = ['BAPnP-GN','BAPnP','EPnP-GN','OPnP','RPnP','SRPnP-GN','MLPnP','CPnP-GN','SQPnP','oDLT-GN','EPnP-GN-Greedy']

# Success Rate
fig, ax = plt.subplots(figsize=(SW*1.2, SW*0.8))
for name in planar_order:
    if name not in pln: continue
    j = pln.index(name)
    c, ls, m, lw, ms = gs(name)
    ax.plot(x_idx, pl['SuccessRate'][:, j], color=c, linestyle=ls, marker=m,
            markersize=ms, linewidth=lw, label=name)
ax.set_xticks(x_idx)
ax.set_xticklabels(z_labels, fontsize=8)
ax.set_xlabel('Degree of Coplanarity γ')
ax.set_ylabel('Success Rate (%)')
ax.set_ylim(-2, 105)
ax.legend(ncol=3, fontsize=5.5, loc='lower left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(outdir, 'exp4_plane_S.pdf'))
fig.savefig(os.path.join(outdir, 'exp4_plane_S.eps'))
plt.close(fig)

# Rotation Error (truncated at 5 deg to show detail)
fig, ax = plt.subplots(figsize=(SW*1.2, SW*0.8))
for name in planar_order:
    if name not in pln: continue
    j = pln.index(name)
    c, ls, m, lw, ms = gs(name)
    ax.plot(x_idx, np.minimum(pl['MedianRot'][:, j], 5.0), color=c, linestyle=ls,
            marker=m, markersize=ms, linewidth=lw, label=name)
ax.set_xticks(x_idx)
ax.set_xticklabels(z_labels, fontsize=8)
ax.set_xlabel('Degree of Coplanarity γ')
ax.set_ylabel('Median Rotation Error (deg)')
ax.set_ylim(0, 5.5)
ax.legend(ncol=3, fontsize=5.5, loc='upper left')
ax.grid(True, alpha=0.3)
# Mark truncated lines with a note
ax.text(0.02, 0.98, 'truncated at 5°', transform=ax.transAxes, fontsize=7,
        va='top', fontstyle='italic', color='gray')
plt.tight_layout()
fig.savefig(os.path.join(outdir, 'exp4_plane_R.pdf'))
fig.savefig(os.path.join(outdir, 'exp4_plane_R.eps'))
plt.close(fig)

# Translation Error (truncated at 10%)
fig, ax = plt.subplots(figsize=(SW*1.2, SW*0.8))
for name in planar_order:
    if name not in pln: continue
    j = pln.index(name)
    c, ls, m, lw, ms = gs(name)
    ax.plot(x_idx, np.minimum(pl['MedianTrans'][:, j], 10.0), color=c, linestyle=ls,
            marker=m, markersize=ms, linewidth=lw, label=name)
ax.set_xticks(x_idx)
ax.set_xticklabels(z_labels, fontsize=8)
ax.set_xlabel('Degree of Coplanarity γ')
ax.set_ylabel('Median Translation Error (%)')
ax.set_ylim(0, 10.5)
ax.legend(ncol=3, fontsize=5.5, loc='upper left')
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.98, 'truncated at 10%', transform=ax.transAxes, fontsize=7,
        va='top', fontstyle='italic', color='gray')
plt.tight_layout()
fig.savefig(os.path.join(outdir, 'exp4_plane_T.pdf'))
fig.savefig(os.path.join(outdir, 'exp4_plane_T.eps'))
plt.close(fig)

print("  Planar figures saved.")

# ============================================================
# ABLATION: separate selection & formulation into clear figures
# These already exist (compare_rot, ablation_rot etc.) - skip
# ============================================================

print("\nAll figures generated!")
