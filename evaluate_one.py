"""
evaluate_one.py — Single-Model Emulator Evaluation Script

Evaluates a single trained emulator model against CAMB ground-truth power
spectra on the test set and produces:

  1. Per-cosmology Syren (EH approximation) error lines (filtered subset)
  2. Two-panel comparison plots (full vs. filtered) for the emulator and Syren
  3. Histograms of the maximum absolute error across cosmologies
  4. EH log-fraction diagnostic, coloured by w0

The filtering threshold (W0_THRESHOLD) isolates the w0 <= -2 region, which
exhibits systematically higher emulator errors and is treated separately.

Usage (standalone):
    python ./mps_emu/evaluate_one.py

Author: Victoria Lloyd (2025)
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import emulmps_w0wa as pk_emu
import train_utils_pk_emulator as utils


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

START_BATCH  = 100       # batch index used as the test/validation set
N_TRAIN      = 5         # number of training batches for the model being evaluated
COSMO_TYPE   = "w0wacdm"
NL_TYPE      = "lin"
PRIOR_TYPE   = "constrained"
IZ           = 0         # redshift index (0 → z=0)
Z_VAL        = 0

# Cosmologies with w0 <= W0_THRESHOLD are flagged as a high-error subset
W0_THRESHOLD = -2
W0_COL       = 5         # column index of w0 in the LHS parameter array

FIG_DIR      = "mps_emu/validation_figs"

# ---------------------------------------------------------------------------
# Plotting constants
# ---------------------------------------------------------------------------

COLOR_50   = "#473C8A"
COLOR_90   = "#C45858"
COLOR_100  = "lightgray"
COLOR_GOOD = "#2166ac"   # blue  — w0 > W0_THRESHOLD
COLOR_BAD  = "#d6604d"   # red   — w0 <= W0_THRESHOLD

AXES_FS   = 20
TICK_FS   = 17
LEGEND_FS = 17

HANDLE_98  = mlines.Line2D([], [], color=COLOR_100, label=r"$98\%$")
HANDLE_90  = mpatches.Patch(facecolor=COLOR_90, label=r"$90\%$")
HANDLE_50  = mpatches.Patch(facecolor=COLOR_50, hatch="\\", edgecolor="lightgray", label=r"$50\%$")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_test_set():
    """Load and return the COLASet test set for the configured run."""
    return utils.COLASet(
        target_z=utils.z_mps,
        cosmo_type=COSMO_TYPE,
        prior_type=PRIOR_TYPE,
        nl_type=NL_TYPE,
        start_batch=START_BATCH,
    )


def compute_predictions(test_set):
    """
    Run the emulator and EH approximation for every cosmology in test_set.

    Returns
    -------
    pred_pks   : (N_valid, N_k) ndarray — emulator P(k)
    syren_pks  : (N_valid, N_k) ndarray — EH-only approximation P(k)
    true_pks   : (N_valid, N_k) ndarray — CAMB ground truth P(k)
    lhs_clean  : (N_valid, N_params) ndarray — parameter array with NaN rows removed
    """
    true_pks_all = test_set.pks_lin[:, IZ, :]

    pred_list  = []
    syren_list = []

    for params in test_set.lhs:
        _, _, pk_full   = pk_emu.get_pks(params, cosmo_type=COSMO_TYPE,
                                          prior_type=PRIOR_TYPE, nl_type=NL_TYPE,
                                          num_batches=N_TRAIN,
                                          use_approximation_only=False)
        _, _, pk_approx = pk_emu.get_pks(params, cosmo_type=COSMO_TYPE,
                                          prior_type=PRIOR_TYPE, nl_type=NL_TYPE,
                                          num_batches=N_TRAIN,
                                          use_approximation_only=True)
        pred_list.append(pk_full[IZ])
        syren_list.append(pk_approx[IZ])

    pred_pks  = np.asarray(pred_list)
    syren_pks = np.asarray(syren_list)

    nan_mask = np.isnan(syren_pks).any(axis=1) | np.isnan(pred_pks).any(axis=1)
    if nan_mask.any():
        print(f"  WARNING: dropping {nan_mask.sum()} cosmologies with NaN output.")

    valid = ~nan_mask
    return pred_pks[valid], syren_pks[valid], true_pks_all[valid], test_set.lhs[valid]


def apply_w0_filter(errors, errors_syren, lhs_clean):
    """
    Split errors into full and w0-filtered subsets.

    Parameters
    ----------
    errors, errors_syren : (N, N_k) ndarray — fractional errors for emulator / Syren
    lhs_clean            : (N, N_params) ndarray — cosmological parameter array

    Returns
    -------
    errors_filt, errors_syren_filt : (N_filt, N_k) ndarrays
    w0_mask                        : (N,) bool array — True where w0 > W0_THRESHOLD
    """
    w0      = lhs_clean[:, W0_COL]
    w0_mask = w0 > W0_THRESHOLD
    return errors[w0_mask], errors_syren[w0_mask], w0_mask


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def percentiles(err):
    """Return (p2, p10, p25, p75, p90, p98) of err over axis=0 (cosmologies)."""
    ps = np.percentile(err, [2, 10, 25, 75, 90, 98], axis=0)
    return ps[0], ps[1], ps[2], ps[3], ps[4], ps[5]


def print_statistics(errors, errors_syren, errors_filt, errors_syren_filt):
    """Print mean, median, 95th, and 99th percentile of the max |error|."""
    subsets = [
        ("Full dataset",         errors,     errors_syren),
        (f"w0 > {W0_THRESHOLD}", errors_filt, errors_syren_filt),
    ]
    print("\n=== ERROR STATISTICS ===")
    for label, e_emu, e_syr in subsets:
        print(f"\n{label}:")
        for name, e in [("EmulMPS", e_emu), ("Syren", e_syr)]:
            max_e = np.max(np.abs(e), axis=1)
            print(f"  {name} - Mean max |error|:     {np.mean(max_e):.6f}")
            print(f"  {name} - Median max |error|:   {np.median(max_e):.6f}")
            print(f"  {name} - 95th pct max |error|: {np.percentile(max_e, 95):.6f}")
            print(f"  {name} - 99th pct max |error|: {np.percentile(max_e, 99):.6f}")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _fill_ax(ax, ks, p2, p10, p25, p75, p90, p98, ylabel, label_text):
    """Draw percentile bands on ax."""
    ax.semilogx(ks, p2,  color=COLOR_100)
    ax.semilogx(ks, p98, color=COLOR_100)
    ax.fill_between(ks, p10, p90, color=COLOR_90, alpha=0.7)
    ax.fill_between(ks, p25, p75, color=COLOR_50, hatch="\\",
                    edgecolor="lightgray", alpha=0.7)
    ax.set_ylabel(ylabel, fontsize=AXES_FS)
    ax.set_xscale("log")
    ax.grid(alpha=0.4)
    ax.set_xlim([ks[0], ks[-1]])
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    ax.text(0.05, 0.95, label_text, transform=ax.transAxes,
            ha='left', va='top', fontsize=AXES_FS)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_syren_lines_filtered(errors_syren_filt, ks, w0_mask):
    """
    Plot individual Syren error curves for each cosmology in the filtered
    (w0 > W0_THRESHOLD) subset.
    """
    n_plot = min(w0_mask.sum(), 200)
    fig, ax = plt.subplots(figsize=(8, 5))

    for i in range(n_plot):
        ax.semilogx(ks, errors_syren_filt[i, :])

    ax.set_xlabel(r"$k \; (h/\mathrm{Mpc})$", fontsize=AXES_FS)
    ax.set_ylabel(r"$P_\mathrm{Syren}/P_\mathrm{CAMB} - 1$", fontsize=AXES_FS)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    ax.text(0.05, 0.95,
            fr'$z={Z_VAL}$,  $w_0 > {W0_THRESHOLD}$  (N={w0_mask.sum()})',
            transform=ax.transAxes, ha='left', va='top', fontsize=LEGEND_FS)

    fname = (f"{FIG_DIR}/val_fracs_{COSMO_TYPE}_z{Z_VAL}"
             f"_w0gt{W0_THRESHOLD}_nTrain{N_TRAIN}.pdf")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_comparison_bands(errors, errors_filt, errors_syren, errors_syren_filt, ks):
    """
    Save two-panel (full vs. filtered) percentile-band figures for the
    emulator and Syren approximation.
    """
    filter_label = fr"$w_0 > {W0_THRESHOLD}$"

    plot_specs = [
        (errors, errors_filt,
         r"$P(k)_\mathrm{EmulMPS}/P(k)_\mathrm{CAMB} - 1$",
         "emul"),
        (errors_syren, errors_syren_filt,
         r"$P(k)_\mathrm{Syren}/P(k)_\mathrm{CAMB} - 1$",
         "syren"),
    ]

    for e_full, e_filt, ylabel, name in plot_specs:
        p_full = percentiles(e_full)
        p_filt = percentiles(e_filt)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
        _fill_ax(ax1, ks, *p_full, ylabel=ylabel,
                 label_text=f'$z={Z_VAL}$  (All cosmologies)')
        _fill_ax(ax2, ks, *p_filt, ylabel=ylabel,
                 label_text=fr'$z={Z_VAL}$,  {filter_label}')
        ax2.set_xlabel(r"$k \; (h/\mathrm{Mpc})$", fontsize=AXES_FS)
        ax2.legend(handles=[HANDLE_50, HANDLE_90, HANDLE_98],
                   fontsize=LEGEND_FS, loc="lower right")
        plt.tight_layout()

        fname = (f"{FIG_DIR}/{name}_errors_z{Z_VAL:.3f}_{COSMO_TYPE}"
                 f"_{PRIOR_TYPE}_{NL_TYPE}_w0gt{W0_THRESHOLD}_nTrain{N_TRAIN}"
                 f"_comparison.pdf")
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")


def plot_max_error_histograms(errors, errors_filt, errors_syren, errors_syren_filt):
    """
    Save a 2×2 grid of histograms showing the distribution of the maximum
    absolute fractional error per cosmology, for all four subsets.
    """
    filter_label = fr"$w_0 > {W0_THRESHOLD}$"

    max_err = {
        "emu_full":  np.max(np.abs(errors),               axis=1),
        "emu_filt":  np.max(np.abs(errors_filt),           axis=1),
        "syr_full":  np.max(np.abs(errors_syren),          axis=1),
        "syr_filt":  np.max(np.abs(errors_syren_filt),     axis=1),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    hist_specs = [
        (axes[0, 0], "emu_full", '#1565c0',
         r'Max $|P_\mathrm{EmulMPS}/P_\mathrm{CAMB} - 1|$',
         'EmulMPS — All cosmologies'),
        (axes[0, 1], "emu_filt", '#C45858',
         r'Max $|P_\mathrm{EmulMPS}/P_\mathrm{CAMB} - 1|$',
         fr'EmulMPS — {filter_label}'),
        (axes[1, 0], "syr_full", '#1565c0',
         r'Max $|P_\mathrm{Syren}/P_\mathrm{CAMB} - 1|$',
         'Syren — All cosmologies'),
        (axes[1, 1], "syr_filt", '#C45858',
         r'Max $|P_\mathrm{Syren}/P_\mathrm{CAMB} - 1|$',
         fr'Syren — {filter_label}'),
    ]

    for ax, key, color, xlabel, title in hist_specs:
        ax.hist(max_err[key], bins=50, color=color, alpha=0.7, edgecolor='black')
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel('Count', fontsize=16)
        ax.set_title(title, fontsize=18)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=14)

    plt.tight_layout()
    fname = (f"{FIG_DIR}/max_error_histogram_{COSMO_TYPE}_{PRIOR_TYPE}"
             f"_{NL_TYPE}_w0gt{W0_THRESHOLD}_nTrain{N_TRAIN}.pdf")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_eh_logfracs(test_set):
    """
    Plot EH log-fraction diagnostics (log(P_CAMB / P_EH)) coloured by w0.

    Cosmologies with w0 > W0_THRESHOLD are shown in blue; those at or below
    the threshold are shown in red. Bold median lines are drawn for each group.
    Individual lines are rasterized to keep PDF file sizes manageable.
    """
    w0        = test_set.lhs[:, W0_COL]
    mask_good = w0 >  W0_THRESHOLD
    mask_bad  = w0 <= W0_THRESHOLD

    logfracs_z0 = test_set.logfracs[:, 0, :]   # (N, N_k) at z=0
    ks          = test_set.ks

    fig, ax = plt.subplots(figsize=(9, 5))

    for idx in np.where(mask_good)[0]:
        ax.semilogx(ks, logfracs_z0[idx],
                    color=COLOR_GOOD, alpha=0.08, lw=0.5, rasterized=True)
    for idx in np.where(mask_bad)[0]:
        ax.semilogx(ks, logfracs_z0[idx],
                    color=COLOR_BAD, alpha=0.08, lw=0.5, rasterized=True)

    if mask_good.sum() > 0:
        ax.semilogx(ks, np.median(logfracs_z0[mask_good], axis=0),
                    color=COLOR_GOOD, lw=2.2,
                    label=fr"$w_0 > {W0_THRESHOLD}$ (n={mask_good.sum()})")
    if mask_bad.sum() > 0:
        ax.semilogx(ks, np.median(logfracs_z0[mask_bad], axis=0),
                    color=COLOR_BAD, lw=2.2,
                    label=fr"$w_0 \leq {W0_THRESHOLD}$ (n={mask_bad.sum()})")

    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"$k \; (h/\mathrm{Mpc})$", fontsize=AXES_FS)
    ax.set_ylabel(r"$\log\!\left(P_\mathrm{CAMB} / P_\mathrm{EH}\right)$",
                  fontsize=AXES_FS)
    ax.set_title(fr"$z={Z_VAL}$   |   {COSMO_TYPE}, {PRIOR_TYPE}, {NL_TYPE}",
                 fontsize=15)
    ax.legend(fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    fname = (f"{FIG_DIR}/eh_logfracs_z{Z_VAL}_{COSMO_TYPE}_{PRIOR_TYPE}"
             f"_{NL_TYPE}_nTrain{N_TRAIN}.pdf")
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family']      = 'STIXGeneral'

    print("=" * 60)
    print("[INFO] Single-Model Emulator Evaluation")
    print(f"       cosmo_type  = {COSMO_TYPE}")
    print(f"       prior_type  = {PRIOR_TYPE}")
    print(f"       nl_type     = {NL_TYPE}")
    print(f"       n_train     = {N_TRAIN}")
    print(f"       test_batch  = {START_BATCH}")
    print(f"       eval at     = z={Z_VAL}  (iz={IZ})")
    print(f"       w0 filter   = w0 > {W0_THRESHOLD}")
    print("=" * 60)

    print("\n[INFO] Loading test set...")
    test_set = load_test_set()

    print("[INFO] Computing predictions...")
    pred_pks, syren_pks, true_pks, lhs_clean = compute_predictions(test_set)

    errors       = pred_pks  / true_pks - 1
    errors_syren = syren_pks / true_pks - 1

    print("[INFO] Applying w0 filter...")
    errors_filt, errors_syren_filt, w0_mask = apply_w0_filter(
        errors, errors_syren, lhs_clean
    )

    w0 = lhs_clean[:, W0_COL]
    print(f"  Total cosmologies : {len(lhs_clean)}")
    print(f"  w0 > {W0_THRESHOLD}          : {w0_mask.sum()}")
    print(f"  w0 range (full)   : [{w0.min():.3f}, {w0.max():.3f}]")
    print(f"  w0 range (filt.)  : [{w0[w0_mask].min():.3f}, {w0[w0_mask].max():.3f}]")

    print_statistics(errors, errors_syren, errors_filt, errors_syren_filt)

    print("\n[INFO] Saving figures...")
    ks = test_set.ks
    plot_syren_lines_filtered(errors_syren_filt, ks, w0_mask)
    plot_comparison_bands(errors, errors_filt, errors_syren, errors_syren_filt, ks)
    plot_max_error_histograms(errors, errors_filt, errors_syren, errors_syren_filt)
    plot_eh_logfracs(test_set)

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()