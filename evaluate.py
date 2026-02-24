"""
Evaluation script with filtering for test_priors parameter range.

Author: Victoria Lloyd (2025)
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
import time

import emulmps_w0wa as pk_emu
import train_utils_pk_emulator as utils

start_batch = 100
num_pcs = 25
num_pcs_z = 15
n_batches = 1
cosmo_type = "lcdm"
nl_type = "lin"
prior_type = "constrained"

testSet = utils.COLASet(target_z=utils.z_mps, cosmo_type=cosmo_type, prior_type=prior_type, nl_type=nl_type, start_batch=start_batch)

iz = 0
z = 0

true_pks = testSet.pks_lin[:, iz, :]
syren_pks = [pk_emu.get_pks(params, cosmo_type=cosmo_type, prior_type=prior_type, nl_type=nl_type, use_approximation_only=True)[2][iz] for params in testSet.lhs]
pred_pks = [pk_emu.get_pks(params, cosmo_type=cosmo_type, prior_type=prior_type, nl_type=nl_type, use_approximation_only=False)[2][iz] for params in testSet.lhs]

# Remove NaN rows
rows_with_nan = np.isnan(syren_pks).any(axis=1)
pred_pks = np.asarray(pred_pks)
pred_pks = pred_pks[~rows_with_nan]
syren_pks = np.asarray(syren_pks)
syren_pks = syren_pks[~rows_with_nan]
true_pks = true_pks[~rows_with_nan]
lhs_clean = testSet.lhs[~rows_with_nan]

errors = pred_pks / true_pks - 1
errors_syren = syren_pks / true_pks - 1

print("Full dataset:")
print("Shape of errors:", errors.shape)
print("Min/Max of errors:", np.min(errors), np.max(errors))

# ========== DEFINE TEST PRIORS AND FILTER ==========
test_priors = {
    "ob_h2": (0.006, 0.038),
    "oc_h2": (0.04, 0.23),
    "H0": (60, 80),
    "lnAs": (1.7, 3.5),
    "ns": (0.8, 1.2),
    "w0": (-2.2, 0.5),
    "wa": (-4.0, 1.0),
}

# LHS format: [10^9 A_s, ns, H0, Ob, Om, w0, wa]
# Need to convert to match test_priors format

As_1e9 = lhs_clean[:, 0]
ns = lhs_clean[:, 1]
H0 = lhs_clean[:, 2]
Ob = lhs_clean[:, 3]  # Omega_b
Om = lhs_clean[:, 4]  # Omega_m
w0 = lhs_clean[:, 5]
wa = lhs_clean[:, 6]

# Convert to test_priors format
h = H0 / 100.0
mnu=0.06
mnu_contrib = (mnu * (3.046 / 3) ** 0.75) / 94.0708
om_h2 = Om * h**2
ob_h2 = Ob * h**2
oc_h2 = om_h2 - ob_h2 - mnu_contrib
lnAs = np.log(10 * As_1e9)  # Convert 10^9 A_s to ln(A_s)

# Create mask for each parameter
mask_ob_h2 = (ob_h2 >= test_priors["ob_h2"][0]) & (ob_h2 <= test_priors["ob_h2"][1])
mask_oc_h2 = (oc_h2 >= test_priors["oc_h2"][0]) & (oc_h2 <= test_priors["oc_h2"][1])
mask_H0 = (H0 >= test_priors["H0"][0]) & (H0 <= test_priors["H0"][1])
mask_lnAs = (lnAs >= test_priors["lnAs"][0]) & (lnAs <= test_priors["lnAs"][1])
mask_ns = (ns >= test_priors["ns"][0]) & (ns <= test_priors["ns"][1])
mask_w0 = (w0 >= test_priors["w0"][0]) & (w0 <= test_priors["w0"][1])
mask_wa = (wa >= test_priors["wa"][0]) & (wa <= test_priors["wa"][1])

# Combine all masks
full_mask = mask_ob_h2 & mask_oc_h2 & mask_H0 & mask_lnAs & mask_ns & mask_w0 & mask_wa

print(f"\nFiltering cosmologies to test_priors range:")
print(f"Total cosmologies: {len(lhs_clean)}")
print(f"Cosmologies within test_priors: {np.sum(full_mask)}")
print(f"Cosmologies outside test_priors: {np.sum(~full_mask)}")

# Print individual parameter filtering statistics
print(f"\nParameter-by-parameter filtering:")
print(f"  ob_h2: {np.sum(mask_ob_h2)}/{len(mask_ob_h2)} in range [{test_priors['ob_h2'][0]}, {test_priors['ob_h2'][1]}]")
print(f"  oc_h2: {np.sum(mask_oc_h2)}/{len(mask_oc_h2)} in range [{test_priors['oc_h2'][0]}, {test_priors['oc_h2'][1]}]")
print(f"  H0: {np.sum(mask_H0)}/{len(mask_H0)} in range [{test_priors['H0'][0]}, {test_priors['H0'][1]}]")
print(f"  lnAs: {np.sum(mask_lnAs)}/{len(mask_lnAs)} in range [{test_priors['lnAs'][0]}, {test_priors['lnAs'][1]}]")
print(f"  ns: {np.sum(mask_ns)}/{len(mask_ns)} in range [{test_priors['ns'][0]}, {test_priors['ns'][1]}]")
print(f"  w0: {np.sum(mask_w0)}/{len(mask_w0)} in range [{test_priors['w0'][0]}, {test_priors['w0'][1]}]")
print(f"  wa: {np.sum(mask_wa)}/{len(mask_wa)} in range [{test_priors['wa'][0]}, {test_priors['wa'][1]}]")

# Print parameter ranges
print(f"\nParameter ranges (full dataset):")
print(f"  ob_h2: [{np.min(ob_h2):.4f}, {np.max(ob_h2):.4f}]")
print(f"  oc_h2: [{np.min(oc_h2):.4f}, {np.max(oc_h2):.4f}]")
print(f"  H0: [{np.min(H0):.2f}, {np.max(H0):.2f}]")
print(f"  lnAs: [{np.min(lnAs):.3f}, {np.max(lnAs):.3f}]")
print(f"  ns: [{np.min(ns):.3f}, {np.max(ns):.3f}]")
print(f"  w0: [{np.min(w0):.3f}, {np.max(w0):.3f}]")
print(f"  wa: [{np.min(wa):.3f}, {np.max(wa):.3f}]")

if np.sum(full_mask) > 0:
    print(f"\nParameter ranges (filtered dataset):")
    print(f"  ob_h2: [{np.min(ob_h2[full_mask]):.4f}, {np.max(ob_h2[full_mask]):.4f}]")
    print(f"  oc_h2: [{np.min(oc_h2[full_mask]):.4f}, {np.max(oc_h2[full_mask]):.4f}]")
    print(f"  H0: [{np.min(H0[full_mask]):.2f}, {np.max(H0[full_mask]):.2f}]")
    print(f"  lnAs: [{np.min(lnAs[full_mask]):.3f}, {np.max(lnAs[full_mask]):.3f}]")
    print(f"  ns: [{np.min(ns[full_mask]):.3f}, {np.max(ns[full_mask]):.3f}]")
    print(f"  w0: [{np.min(w0[full_mask]):.3f}, {np.max(w0[full_mask]):.3f}]")
    print(f"  wa: [{np.min(wa[full_mask]):.3f}, {np.max(wa[full_mask]):.3f}]")

# Apply filter
errors_filtered = errors[full_mask]
errors_syren_filtered = errors_syren[full_mask]
lhs_filtered = lhs_clean[full_mask]

if np.sum(full_mask) > 0:
    print(f"\nFiltered dataset:")
    print(f"Shape of errors: {errors_filtered.shape}")
    print(f"Min/Max of errors: {np.min(errors_filtered):.6f}, {np.max(errors_filtered):.6f}")
else:
    print("\nWARNING: No cosmologies found within test_priors range!")

# ========== PLOTTING SETUP ==========
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = "STIXGeneral"

color50 = "#473C8A"
color90 = "#C45858"
color100 = "lightgray"
axes_label_fontsize = 20

line100 = mpl.lines.Line2D([], [], color=color100, label=r"$98\%$")
hatch50 = mpl.patches.Patch(facecolor=color50, hatch="\\", edgecolor="lightgray", label=r"$50\%$")
hatch90 = mpl.patches.Patch(facecolor=color90, label=r"$90\%$")

# ========== PLOT INDIVIDUAL SYREN FRACTIONAL DIFFERENCES (FILTERED) ==========
if np.sum(full_mask) > 0:
    plt.figure(figsize=(8, 5))
    n_plot = min(np.sum(full_mask), 200)  # Plot up to 200 cosmologies
    for i in range(n_plot):
        plt.semilogx(testSet.ks, errors_syren_filtered[i, :])
    plt.xlabel(r"$k \; (h/\mathrm{Mpc})$", fontsize=axes_label_fontsize)
    plt.ylabel(r"$P_\mathrm{CAMB}/P_\mathrm{Syren} - 1$", fontsize=axes_label_fontsize)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.text(0.05, 0.95, fr'$z={z}$ (Within original priors, N={np.sum(full_mask)})', 
             transform=plt.gca().transAxes,
             horizontalalignment='left', verticalalignment='top', fontsize=16)
    plt.savefig(f"mps_emu/validation_figs/val_fracs_{cosmo_type}_z{z}_testpriors.pdf", bbox_inches="tight")
    plt.close()

# ========== PLOT FULL DATASET ==========
emup2 = np.percentile(errors, 2, axis=0)
emup10 = np.percentile(errors, 10, axis=0)
emup25 = np.percentile(errors, 25, axis=0)
emup75 = np.percentile(errors, 75, axis=0)
emup90 = np.percentile(errors, 90, axis=0)
emup98 = np.percentile(errors, 98, axis=0)

fig, axs = plt.subplots(1, 1, figsize=(8, 6))
fig.subplots_adjust(right=1, left=0.05)

axs.semilogx(testSet.ks, emup2, color=color100)
axs.fill_between(testSet.ks, emup10, emup90, color=color90)
axs.fill_between(testSet.ks, emup25, emup75, color=color50, hatch="\\", edgecolor="lightgray")
axs.semilogx(testSet.ks, emup98, color=color100)
axs.set_ylabel(r"$P(k)_\mathrm{EmulMPS}/P(k)_\mathrm{CAMB} - 1$", fontsize=axes_label_fontsize)
axs.set_xscale("log")
axs.set_xlabel(r"$k \; (h/\mathrm{Mpc})$", fontsize=axes_label_fontsize)
axs.grid(alpha=0.4)
axs.set_xlim([testSet.ks[0], testSet.ks[-1]])
axs.tick_params(axis='both', which='major', labelsize=17)

plt.text(0.05, 0.95, fr'$z={z}$ (All cosmologies)', transform=plt.gca().transAxes,
         horizontalalignment='left', verticalalignment='top', fontsize=axes_label_fontsize)
axs.legend(handles=[hatch50, hatch90, line100], fontsize=axes_label_fontsize, loc="lower right")

plt.savefig(f"mps_emu/validation_figs/emul_errors_z{z:.3f}_{cosmo_type}_{prior_type}.pdf", bbox_inches="tight")
plt.close()

# ========== PLOT FILTERED DATASET (EMULATOR) ==========
if np.sum(full_mask) > 0:
    emup2_filt = np.percentile(errors_filtered, 2, axis=0)
    emup10_filt = np.percentile(errors_filtered, 10, axis=0)
    emup25_filt = np.percentile(errors_filtered, 25, axis=0)
    emup75_filt = np.percentile(errors_filtered, 75, axis=0)
    emup90_filt = np.percentile(errors_filtered, 90, axis=0)
    emup98_filt = np.percentile(errors_filtered, 98, axis=0)

    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    fig.subplots_adjust(right=1, left=0.05)

    axs.semilogx(testSet.ks, emup2_filt, color=color100)
    axs.fill_between(testSet.ks, emup10_filt, emup90_filt, color=color90)
    axs.fill_between(testSet.ks, emup25_filt, emup75_filt, color=color50, hatch="\\", edgecolor="lightgray")
    axs.semilogx(testSet.ks, emup98_filt, color=color100)
    axs.set_ylabel(r"$P(k)_\mathrm{EmulMPS}/P(k)_\mathrm{CAMB} - 1$", fontsize=axes_label_fontsize)
    axs.set_xscale("log")
    axs.set_xlabel(r"$k \; (h/\mathrm{Mpc})$", fontsize=axes_label_fontsize)
    axs.grid(alpha=0.4)
    axs.set_xlim([testSet.ks[0], testSet.ks[-1]])
    axs.tick_params(axis='both', which='major', labelsize=17)

    plt.text(0.05, 0.95, fr'$z={z}$ (Within original priors)', transform=plt.gca().transAxes,
             horizontalalignment='left', verticalalignment='top', fontsize=axes_label_fontsize)
    axs.legend(handles=[hatch50, hatch90, line100], fontsize=axes_label_fontsize, loc="lower right")

    plt.savefig(f"mps_emu/validation_figs/emul_errors_z{z:.3f}_{cosmo_type}_{prior_type}_testpriors.pdf", bbox_inches="tight")
    plt.close()

    # ========== PLOT FILTERED DATASET (SYREN) ==========
    emup2_syren_filt = np.percentile(errors_syren_filtered, 2, axis=0)
    emup10_syren_filt = np.percentile(errors_syren_filtered, 10, axis=0)
    emup25_syren_filt = np.percentile(errors_syren_filtered, 25, axis=0)
    emup75_syren_filt = np.percentile(errors_syren_filtered, 75, axis=0)
    emup90_syren_filt = np.percentile(errors_syren_filtered, 90, axis=0)
    emup98_syren_filt = np.percentile(errors_syren_filtered, 98, axis=0)

    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    fig.subplots_adjust(right=1, left=0.05)

    axs.semilogx(testSet.ks, emup2_syren_filt, color=color100)
    axs.fill_between(testSet.ks, emup10_syren_filt, emup90_syren_filt, color=color90)
    axs.fill_between(testSet.ks, emup25_syren_filt, emup75_syren_filt, color=color50, hatch="\\", edgecolor="lightgray")
    axs.semilogx(testSet.ks, emup98_syren_filt, color=color100)
    axs.set_ylabel(r"$P(k)_\mathrm{Syren}/P(k)_\mathrm{CAMB} - 1$", fontsize=axes_label_fontsize)
    axs.set_xscale("log")
    axs.set_xlabel(r"$k \; (h/\mathrm{Mpc})$", fontsize=axes_label_fontsize)
    axs.grid(alpha=0.4)
    axs.set_xlim([testSet.ks[0], testSet.ks[-1]])
    axs.tick_params(axis='both', which='major', labelsize=17)

    plt.text(0.05, 0.95, fr'$z={z}$ (Within original priors)', transform=plt.gca().transAxes,
             horizontalalignment='left', verticalalignment='top', fontsize=axes_label_fontsize)
    axs.legend(handles=[hatch50, hatch90, line100], fontsize=axes_label_fontsize, loc="lower right")

    plt.savefig(f"mps_emu/validation_figs/syren_errors_z{z:.3f}_{cosmo_type}_{prior_type}_testpriors.pdf", bbox_inches="tight")
    plt.close()

    # ========== COMPARISON PLOT (EMULATOR) ==========
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # Full dataset
    ax1.semilogx(testSet.ks, emup2, color=color100)
    ax1.fill_between(testSet.ks, emup10, emup90, color=color90, alpha=0.7)
    ax1.fill_between(testSet.ks, emup25, emup75, color=color50, hatch="\\", edgecolor="lightgray", alpha=0.7)
    ax1.semilogx(testSet.ks, emup98, color=color100)
    ax1.set_ylabel(r"$P(k)_\mathrm{EmulMPS}/P(k)_\mathrm{CAMB} - 1$", fontsize=axes_label_fontsize)
    ax1.grid(alpha=0.4)
    ax1.tick_params(axis='both', which='major', labelsize=17)
    ax1.text(0.05, 0.95, 'All cosmologies', transform=ax1.transAxes,
             horizontalalignment='left', verticalalignment='top', fontsize=axes_label_fontsize)

    # Filtered dataset
    ax2.semilogx(testSet.ks, emup2_filt, color=color100)
    ax2.fill_between(testSet.ks, emup10_filt, emup90_filt, color=color90, alpha=0.7)
    ax2.fill_between(testSet.ks, emup25_filt, emup75_filt, color=color50, hatch="\\", edgecolor="lightgray", alpha=0.7)
    ax2.semilogx(testSet.ks, emup98_filt, color=color100)
    ax2.set_ylabel(r"$P(k)_\mathrm{EmulMPS}/P(k)_\mathrm{CAMB} - 1$", fontsize=axes_label_fontsize)
    ax2.set_xlabel(r"$k \; (h/\mathrm{Mpc})$", fontsize=axes_label_fontsize)
    ax2.grid(alpha=0.4)
    ax2.set_xlim([testSet.ks[0], testSet.ks[-1]])
    ax2.tick_params(axis='both', which='major', labelsize=17)
    ax2.text(0.05, 0.95, r'Within original priors', transform=ax2.transAxes,
             horizontalalignment='left', verticalalignment='top', fontsize=axes_label_fontsize)

    ax2.legend(handles=[hatch50, hatch90, line100], fontsize=axes_label_fontsize, loc="lower right")

    plt.tight_layout()
    plt.savefig(f"mps_emu/validation_figs/emul_errors_z{z:.3f}_{cosmo_type}_{prior_type}_comparison.pdf", bbox_inches="tight")
    plt.close()

    # ========== COMPARISON PLOT (SYREN) ==========
    emup2_syren = np.percentile(errors_syren, 2, axis=0)
    emup10_syren = np.percentile(errors_syren, 10, axis=0)
    emup25_syren = np.percentile(errors_syren, 25, axis=0)
    emup75_syren = np.percentile(errors_syren, 75, axis=0)
    emup90_syren = np.percentile(errors_syren, 90, axis=0)
    emup98_syren = np.percentile(errors_syren, 98, axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # Full dataset
    ax1.semilogx(testSet.ks, emup2_syren, color=color100)
    ax1.fill_between(testSet.ks, emup10_syren, emup90_syren, color=color90, alpha=0.7)
    ax1.fill_between(testSet.ks, emup25_syren, emup75_syren, color=color50, hatch="\\", edgecolor="lightgray", alpha=0.7)
    ax1.semilogx(testSet.ks, emup98_syren, color=color100)
    ax1.set_ylabel(r"$P(k)_\mathrm{Syren}/P(k)_\mathrm{CAMB} - 1$", fontsize=axes_label_fontsize)
    ax1.grid(alpha=0.4)
    ax1.tick_params(axis='both', which='major', labelsize=17)
    ax1.text(0.05, 0.95, 'All cosmologies', transform=ax1.transAxes,
             horizontalalignment='left', verticalalignment='top', fontsize=axes_label_fontsize)

    # Filtered dataset
    ax2.semilogx(testSet.ks, emup2_syren_filt, color=color100)
    ax2.fill_between(testSet.ks, emup10_syren_filt, emup90_syren_filt, color=color90, alpha=0.7)
    ax2.fill_between(testSet.ks, emup25_syren_filt, emup75_syren_filt, color=color50, hatch="\\", edgecolor="lightgray", alpha=0.7)
    ax2.semilogx(testSet.ks, emup98_syren_filt, color=color100)
    ax2.set_ylabel(r"$P(k)_\mathrm{Syren}/P(k)_\mathrm{CAMB} - 1$", fontsize=axes_label_fontsize)
    ax2.set_xlabel(r"$k \; (h/\mathrm{Mpc})$", fontsize=axes_label_fontsize)
    ax2.grid(alpha=0.4)
    ax2.set_xlim([testSet.ks[0], testSet.ks[-1]])
    ax2.tick_params(axis='both', which='major', labelsize=17)
    ax2.text(0.05, 0.95, r'Within original priors', transform=ax2.transAxes,
             horizontalalignment='left', verticalalignment='top', fontsize=axes_label_fontsize)

    ax2.legend(handles=[hatch50, hatch90, line100], fontsize=axes_label_fontsize, loc="lower right")

    plt.tight_layout()
    plt.savefig(f"mps_emu/validation_figs/syren_errors_z{z:.3f}_{cosmo_type}_{prior_type}_comparison.pdf", bbox_inches="tight")
    plt.close()

    # ========== PRINT STATISTICS ==========
    print("\n=== ERROR STATISTICS ===")
    print("\nFull dataset:")
    print(f"  EmulMPS - Mean max |error|: {np.mean(np.max(np.abs(errors), axis=1)):.6f}")
    print(f"  EmulMPS - Median max |error|: {np.median(np.max(np.abs(errors), axis=1)):.6f}")
    print(f"  EmulMPS - 95th percentile max |error|: {np.percentile(np.max(np.abs(errors), axis=1), 95):.6f}")
    print(f"  EmulMPS - 99th percentile max |error|: {np.percentile(np.max(np.abs(errors), axis=1), 99):.6f}")
    
    print(f"\n  Syren - Mean max |error|: {np.mean(np.max(np.abs(errors_syren), axis=1)):.6f}")
    print(f"  Syren - Median max |error|: {np.median(np.max(np.abs(errors_syren), axis=1)):.6f}")
    print(f"  Syren - 95th percentile max |error|: {np.percentile(np.max(np.abs(errors_syren), axis=1), 95):.6f}")
    print(f"  Syren - 99th percentile max |error|: {np.percentile(np.max(np.abs(errors_syren), axis=1), 99):.6f}")

    print(f"\nFiltered dataset (within test_priors):")
    print(f"  EmulMPS - Mean max |error|: {np.mean(np.max(np.abs(errors_filtered), axis=1)):.6f}")
    print(f"  EmulMPS - Median max |error|: {np.median(np.max(np.abs(errors_filtered), axis=1)):.6f}")
    print(f"  EmulMPS - 95th percentile max |error|: {np.percentile(np.max(np.abs(errors_filtered), axis=1), 95):.6f}")
    print(f"  EmulMPS - 99th percentile max |error|: {np.percentile(np.max(np.abs(errors_filtered), axis=1), 99):.6f}")
    
    print(f"\n  Syren - Mean max |error|: {np.mean(np.max(np.abs(errors_syren_filtered), axis=1)):.6f}")
    print(f"  Syren - Median max |error|: {np.median(np.max(np.abs(errors_syren_filtered), axis=1)):.6f}")
    print(f"  Syren - 95th percentile max |error|: {np.percentile(np.max(np.abs(errors_syren_filtered), axis=1), 95):.6f}")
    print(f"  Syren - 99th percentile max |error|: {np.percentile(np.max(np.abs(errors_syren_filtered), axis=1), 99):.6f}")

    # ========== HISTOGRAM OF MAX ERRORS ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    max_errors_full = np.max(np.abs(errors), axis=1)
    max_errors_filt = np.max(np.abs(errors_filtered), axis=1)
    max_errors_syren_full = np.max(np.abs(errors_syren), axis=1)
    max_errors_syren_filt = np.max(np.abs(errors_syren_filtered), axis=1)

    # EmulMPS - Full
    axes[0, 0].hist(max_errors_full, bins=50, color='#1565c0', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel(r'Max $|P_\mathrm{EmulMPS}/P_\mathrm{CAMB} - 1|$', fontsize=16)
    axes[0, 0].set_ylabel('Count', fontsize=16)
    axes[0, 0].set_title('EmulMPS - All cosmologies', fontsize=18)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].tick_params(labelsize=14)

    # EmulMPS - Filtered
    axes[0, 1].hist(max_errors_filt, bins=50, color='#C45858', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel(r'Max $|P_\mathrm{EmulMPS}/P_\mathrm{CAMB} - 1|$', fontsize=16)
    axes[0, 1].set_ylabel('Count', fontsize=16)
    axes[0, 1].set_title(r'EmulMPS - Within original priors', fontsize=18)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].tick_params(labelsize=14)

    # Syren - Full
    axes[1, 0].hist(max_errors_syren_full, bins=50, color='#1565c0', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel(r'Max $|P_\mathrm{Syren}/P_\mathrm{CAMB} - 1|$', fontsize=16)
    axes[1, 0].set_ylabel('Count', fontsize=16)
    axes[1, 0].set_title('Syren - All cosmologies', fontsize=18)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].tick_params(labelsize=14)

    # Syren - Filtered
    axes[1, 1].hist(max_errors_syren_filt, bins=50, color='#C45858', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel(r'Max $|P_\mathrm{Syren}/P_\mathrm{CAMB} - 1|$', fontsize=16)
    axes[1, 1].set_ylabel('Count', fontsize=16)
    axes[1, 1].set_title(r'Syren - Within original priors', fontsize=18)
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].tick_params(labelsize=14)

    plt.tight_layout()
    plt.savefig(f"mps_emu/validation_figs/max_error_histogram_{cosmo_type}_{prior_type}.pdf", bbox_inches="tight")
    plt.close()

    print("\nPlots saved successfully!")
else:
    print("\nSkipping plots - no cosmologies found within test_priors range.")