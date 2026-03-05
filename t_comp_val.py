"""
t_comp_val.py — PCA and tPCA Reconstruction Validation Script

Validates that the two-stage dimensionality reduction (per-redshift PCA of
log-fractions, followed by a temporal PCA across redshifts) introduces only
small reconstruction errors.  For each stage the script produces a spaghetti
plot of fractional errors P_reconstructed / P_true - 1 across the test set.

Outputs (saved to FIG_DIR):
  1. pca_errors_<tag>.pdf   — per-cosmology PCA reconstruction errors at z=0
  2. tpca_errors_<tag>.pdf  — per-cosmology tPCA reconstruction errors at z=0
  3. pca_errors_z3_<tag>.pdf  — same as (1) but at z=3
  4. tpca_errors_z3_<tag>.pdf — same as (2) but at z=3

Additionally prints data-quality diagnostics (NaN/Inf counts, bad-cosmology
summary) before training-set preparation begins.

Usage (standalone):
    python ./mps_emu/t_comp_val.py

Author: Victoria Lloyd (2026)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import train_utils_pk_emulator as utils


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_BATCHES   = 20
START_BATCH = 0
TEST_BATCH  = 100
NUM_PCS     = 25       # number of spatial PCA components per redshift
NUM_PCS_Z   = 15       # number of temporal PCA components across redshifts
COSMO_TYPE  = "w0wacdm"
PRIOR_TYPE  = "expanded"
NL_TYPE     = "halofit"

Z_IDX_0     = 0        # redshift index for z≈0
Z_IDX_3     = 33       # redshift index for z≈3

FIG_DIR     = "mps_emu/validation_figs"

# ---------------------------------------------------------------------------
# Plotting constants
# ---------------------------------------------------------------------------

AXES_FS = 14
TICK_FS = 12


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_data_diagnostics(train_set):
    """
    Print shape, NaN/Inf counts, and a summary of bad cosmologies for all
    key arrays in train_set.  Intended as a sanity check before calling
    train_set.prepare().
    """
    print("=== BASIC SHAPES ===")
    print("lhs:          ", train_set.lhs.shape)
    print("mps_approxes: ", train_set.mps_approxes.shape)
    print("pks_lin:      ", train_set.pks_lin.shape)
    print("frac_pks:     ", train_set.frac_pks.shape)
    print("logfracs:     ", train_set.logfracs.shape)

    print("\n=== NaN / Inf SUMMARY ===")
    for name, arr in [
        ("mps_approxes", train_set.mps_approxes),
        ("pks_lin",      train_set.pks_lin),
        ("frac_pks",     train_set.frac_pks),
        ("logfracs",     train_set.logfracs),
    ]:
        print(f"  {name}: "
              f"NaNs={np.isnan(arr).sum()}, "
              f"infs={np.isinf(arr).sum()}, "
              f"min={np.nanmin(arr):.4g}, "
              f"max={np.nanmax(arr):.4g}")

    # Cosmologies with any NaN or Inf in logfracs (collapsed over z and k)
    bad_mask    = np.isnan(train_set.logfracs).any(axis=(1, 2)) \
                | np.isinf(train_set.logfracs).any(axis=(1, 2))
    bad_indices = np.where(bad_mask)[0]

    print(f"\n=== BAD COSMOLOGIES ===")
    print(f"  {bad_mask.sum()} / {train_set.lhs.shape[0]} cosmologies are bad")
    if bad_indices.size > 0:
        print("  First 10 bad indices:", bad_indices[:10])
        i = bad_indices[0]
        print(f"\n  Example (index {i}):")
        print(f"    LHS params:      {train_set.lhs[i]}")
        print(f"    mps_approxes:    "
              f"min={np.nanmin(train_set.mps_approxes[i]):.4g}, "
              f"max={np.nanmax(train_set.mps_approxes[i]):.4g}")
        print(f"    pks_lin:         "
              f"min={np.nanmin(train_set.pks_lin[i]):.4g}, "
              f"max={np.nanmax(train_set.pks_lin[i]):.4g}")
        print(f"    zeros in mps_approxes: "
              f"{(train_set.mps_approxes[i] == 0).sum()}")
        print(f"    non-positive frac_pks: "
              f"{(train_set.frac_pks[i] <= 0).sum()}")


# ---------------------------------------------------------------------------
# PCA reconstruction
# ---------------------------------------------------------------------------

def pca_reconstruction_errors(train_set, test_set, iz):
    """
    Compute per-cosmology PCA reconstruction fractional errors at redshift
    index iz:  exp(inverse_transform(PCA(logfracs))) / frac_pks - 1.

    Parameters
    ----------
    train_set : COLASet — fitted scalers and PCAs live here
    test_set  : COLASet — cosmologies to evaluate
    iz        : int     — redshift index

    Returns
    -------
    errors : (N_cosmo, N_k) ndarray
    """
    scaler = train_set.frac_pks_scalers[iz]
    pca    = train_set.pcas[iz]

    normed       = scaler.transform(test_set.logfracs[:, iz, :])
    pcs          = pca.transform(normed)
    reconstructed = scaler.inverse_transform(pca.inverse_transform(pcs))
    return np.exp(reconstructed) / test_set.frac_pks[:, iz, :] - 1


# ---------------------------------------------------------------------------
# tPCA reconstruction
# ---------------------------------------------------------------------------

def tpca_stacks(train_set, test_set):
    """
    Encode all test cosmologies with the per-z PCAs, compress with tPCA,
    then reconstruct back to log-fraction space.

    Returns
    -------
    stacks : (N_cosmo, 1, N_z, N_k) ndarray of reconstructed log-fractions
    """
    n_cosmo = len(test_set.lhs)
    n_z     = len(utils.z_mps)

    # --- encode: (N_cosmo, N_z, 1, NUM_PCS) ---
    all_pcs = []
    for i in range(n_cosmo):
        cosmos_pcs = []
        for iz in range(n_z):
            normed = train_set.frac_pks_scalers[iz].transform(
                [test_set.logfracs[i, iz, :]]
            )
            pcs = train_set.pcas[iz].transform(normed)
            cosmos_pcs.append([pcs[0]])
        all_pcs.append(cosmos_pcs)

    all_pcs  = np.transpose(np.array(all_pcs), (0, 2, 1, 3))  # (N, 1, N_z, NUM_PCS)
    pcs_flat = all_pcs.reshape(n_cosmo, n_z * NUM_PCS)

    # --- tPCA round-trip ---
    t_comps   = train_set.tpca.transform(pcs_flat)
    pcs_recon = train_set.tpca.inverse_transform(t_comps)
    pcs_per_z = pcs_recon.reshape(n_cosmo, n_z, NUM_PCS)

    # --- decode back to log-fraction space ---
    stacks = []
    for pcs_z_stack in pcs_per_z:
        reconstructed_fracs = [
            train_set.frac_pks_scalers[iz].inverse_transform(
                train_set.pcas[iz].inverse_transform(pcs_z.reshape(1, -1))
            )[0]
            for iz, pcs_z in enumerate(pcs_z_stack)
        ]
        stacks.append([np.stack(reconstructed_fracs)])

    return np.array(stacks)   # (N_cosmo, 1, N_z, N_k)


def tpca_reconstruction_errors(stacks, test_set, iz):
    """
    Compute tPCA fractional errors at redshift index iz.

    Parameters
    ----------
    stacks   : (N_cosmo, 1, N_z, N_k) ndarray — output of tpca_stacks()
    test_set : COLASet
    iz       : int — redshift index

    Returns
    -------
    errors : (N_cosmo, N_k) ndarray
    """
    return np.exp(stacks[:, 0, iz, :]) / test_set.frac_pks[:, iz, :] - 1


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _tag():
    """Shared filename tag built from the run configuration."""
    return f"{COSMO_TYPE}_{NL_TYPE}_{PRIOR_TYPE}_n{NUM_PCS}_z{NUM_PCS_Z}"


def _spaghetti_ax(ax, ks, errors, ylabel, title):
    """Draw one spaghetti error curve per cosmology on ax."""
    for error in errors:
        ax.semilogx(ks, error, lw=0.6, alpha=0.7)
    ax.set_xlabel(r"$k \; (h/\mathrm{Mpc})$", fontsize=AXES_FS)
    ax.set_ylabel(ylabel, fontsize=AXES_FS)
    ax.set_title(title, fontsize=AXES_FS)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    ax.grid(alpha=0.3)


def plot_pca_errors(errors, ks, iz_label):
    """Save PCA reconstruction error spaghetti plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    _spaghetti_ax(
        ax, ks, errors,
        ylabel=r"$P_\mathrm{PCA-reconstructed}/P - 1$",
        title=(r"PCA Reconstruction Errors"
               + fr" at $z\approx{iz_label}$, $N_\mathrm{{PC}}={NUM_PCS}$"),
    )
    plt.tight_layout()
    fname = f"{FIG_DIR}/pca_errors_z{iz_label}_{_tag()}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_tpca_errors(errors, ks, iz_label):
    """Save tPCA reconstruction error spaghetti plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    _spaghetti_ax(
        ax, ks, errors,
        ylabel=r"$P_\mathrm{tPCA-reconstructed}/P - 1$",
        title=(r"tPCA Reconstruction Errors"
               + fr" at $z\approx{iz_label}$,"
               + fr" $N_\mathrm{{PC}}={NUM_PCS}$, $N_\mathrm{{tcomp}}={NUM_PCS_Z}$"),
    )
    plt.tight_layout()
    fname = f"{FIG_DIR}/tpca_errors_z{iz_label}_{_tag()}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family']      = 'STIXGeneral'

    print("=" * 60)
    print("[INFO] PCA / tPCA Reconstruction Validation")
    print(f"       cosmo_type  = {COSMO_TYPE}")
    print(f"       prior_type  = {PRIOR_TYPE}")
    print(f"       nl_type     = {NL_TYPE}")
    print(f"       n_batches   = {N_BATCHES}  (starting at {START_BATCH})")
    print(f"       test_batch  = {TEST_BATCH}")
    print(f"       num_pcs     = {NUM_PCS}")
    print(f"       num_pcs_z   = {NUM_PCS_Z}")
    print("=" * 60)

    # --- Load data ---
    print("\n[INFO] Loading training set...")
    train_set = utils.COLASet(
        target_z=utils.z_mps,
        cosmo_type=COSMO_TYPE,
        prior_type=PRIOR_TYPE,
        nl_type=NL_TYPE,
        n_batches=N_BATCHES,
        start_batch=START_BATCH,
    )

    print_data_diagnostics(train_set)

    print("\n[INFO] Preparing training set (PCA + tPCA fit)...")
    train_set.prepare(num_pcs=NUM_PCS, num_pcs_z=NUM_PCS_Z)

    print("\n[INFO] Loading test set...")
    test_set = utils.COLASet(
        target_z=utils.z_mps,
        cosmo_type=COSMO_TYPE,
        nl_type=NL_TYPE,
        start_batch=TEST_BATCH,
    )

    ks = train_set.ks

    # --- PCA errors ---
    print("\n[INFO] Computing PCA reconstruction errors...")
    pca_err_z0 = pca_reconstruction_errors(train_set, test_set, Z_IDX_0)
    pca_err_z3 = pca_reconstruction_errors(train_set, test_set, Z_IDX_3)

    # --- tPCA errors ---
    print("[INFO] Computing tPCA reconstruction errors...")
    stacks       = tpca_stacks(train_set, test_set)
    tpca_err_z0  = tpca_reconstruction_errors(stacks, test_set, Z_IDX_0)
    tpca_err_z3  = tpca_reconstruction_errors(stacks, test_set, Z_IDX_3)

    # --- Figures ---
    print("\n[INFO] Saving figures...")
    plot_pca_errors(pca_err_z0,  ks, iz_label=0)
    plot_tpca_errors(tpca_err_z0, ks, iz_label=0)
    plot_pca_errors(pca_err_z3,  ks, iz_label=3)
    plot_tpca_errors(tpca_err_z3, ks, iz_label=3)

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()