import os
import pickle
from importlib import reload
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import torch
import keras

import train_utils_pk_emulator as utils

torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

rcParams["savefig.dpi"] = 100
rcParams["figure.dpi"] = 150
rcParams["font.size"] = 10

reload(utils)
# Configuration
n_batches = 20
start_batch = 0
test_batch = 100
target_z = 0.0
num_pcs = 25
num_pcs_z = 15
cosmo_type = "w0wacdm"
prior_type = "expanded"
nl_type = "halofit"

# Prepare training and test sets
trainSet = utils.COLASet(target_z=utils.z_mps, cosmo_type=cosmo_type, prior_type=prior_type, nl_type=nl_type, n_batches=n_batches, start_batch=start_batch)

import numpy as np

print("=== BASIC SHAPES ===")
print("lhs:", trainSet.lhs.shape)                    # (Ncosmo, Nparams)
print("mps_approxes:", trainSet.mps_approxes.shape)  # (Ncosmo, Nz, Nk)
print("pks_lin:", trainSet.pks_lin.shape)
print("frac_pks:", trainSet.frac_pks.shape)
print("logfracs:", trainSet.logfracs.shape)

print("\n=== SANITY CHECKS ===")
for name, arr in [
    ("mps_approxes", trainSet.mps_approxes),
    ("pks_lin", trainSet.pks_lin),
    ("frac_pks", trainSet.frac_pks),
    ("logfracs", trainSet.logfracs),
]:
    print(
        f"{name}: "
        f"NaNs={np.isnan(arr).sum()}, "
        f"infs={np.isinf(arr).sum()}, "
        f"min={np.nanmin(arr)}, "
        f"max={np.nanmax(arr)}"
    )

print("\n=== WHICH COSMOLOGIES ARE BAD? ===")

# Collapse over (z, k) to find cosmologies with *any* NaNs / infs
bad_nan = np.isnan(trainSet.logfracs).any(axis=(1, 2))
bad_inf = np.isinf(trainSet.logfracs).any(axis=(1, 2))
bad_any = bad_nan | bad_inf

bad_indices = np.where(bad_any)[0]

print(f"Number of bad cosmologies: {bad_any.sum()} / {trainSet.lhs.shape[0]}")
print("First 10 bad indices:", bad_indices[:10])

if bad_indices.size > 0:
    print("\n=== EXAMPLE BAD COSMOLOGY ===")
    i = bad_indices[0]
    print("Index:", i)
    print("LHS params:", trainSet.lhs[i])

    print(
        "mps_approxes stats:",
        np.nanmin(trainSet.mps_approxes[i]),
        np.nanmax(trainSet.mps_approxes[i])
    )
    print(
        "pks_lin stats:",
        np.nanmin(trainSet.pks_lin[i]),
        np.nanmax(trainSet.pks_lin[i])
    )

    # Explicit failure modes
    zero_mask = trainSet.mps_approxes[i] == 0
    neg_mask  = trainSet.frac_pks[i] <= 0

    print("Zeros in mps_approxes:", zero_mask.sum())
    print("Non-positive frac_pks:", neg_mask.sum())

trainSet.prepare(num_pcs=num_pcs, num_pcs_z=num_pcs_z)

# for i in [223,  424,  458,  527,  558,  901, 1271, 1454, 1618, 1733]: 
#     plt.loglog(trainSet.ks, trainSet.pks_lin[i,0,:])
#     plt.loglog(trainSet.ks, trainSet.mps_approxes[i,0,:], linestyle="--")
#     print("mps approxes bad len: ", np.count_nonzero(~np.isnan(trainSet.mps_approxes[i,0,:])))
# # plt.ylim(-0.02, 0.02)
# plt.xlabel("k")
# plt.ylabel(r"$P_\mathrm{tPCA-reconstructed}/P - 1$")
# plt.title(fr"PCA Reconstruction Errors, $N_\mathrm{{PC}} = ${num_pcs} $N_{{t comps}} = ${num_pcs_z}" )
# plt.savefig("mps_emu/validation_figs/syren_bad_cosmos.pdf")

testSet = utils.COLASet(target_z=utils.z_mps, cosmo_type=cosmo_type, nl_type=nl_type, start_batch=test_batch)
testLogFracsNorm = trainSet.frac_pks_scalers[0].transform(testSet.logfracs[:, 0, :])
testPcs = trainSet.pcas[0].transform(testLogFracsNorm)
testLogFracsNormReconstructed = trainSet.pcas[0].inverse_transform(testPcs)


errors = np.exp(trainSet.frac_pks_scalers[0].inverse_transform(testLogFracsNormReconstructed[:]))/testSet.frac_pks[:,0,:] - 1 #np.exp(trainSet.frac_pks_scaler.inverse_transform(testLogFracsNormReconstructed))/testSet.frac_pks - 1
for error in errors: plt.semilogx(trainSet.ks, error)
# plt.ylim(-0.05, 0.05)
plt.xlabel("k")
plt.ylabel(r"$P_\mathrm{PCA-reconstructed}/P - 1$")
plt.title(r"PCA Reconstruction Errors, $N_\mathrm{PC} = $" + f"{num_pcs}")
plt.savefig(f"mps_emu/validation_figs/pca_errors_{cosmo_type}_{nl_type}_{prior_type}_n{num_pcs}_z{num_pcs_z}.pdf")

all_pcs = []
for i in range(len(testSet.lhs)):
    cosmos_pcs = []
    for iz, z in enumerate(utils.z_mps):
        testLogFracsNorm = trainSet.frac_pks_scalers[iz].transform([testSet.logfracs[i, iz, :]])
        testPcs = trainSet.pcas[iz].transform(testLogFracsNorm)
        cosmos_pcs.append([testPcs[0]])
    all_pcs.append(cosmos_pcs)
    
print(np.array(all_pcs).shape)
all_pcs = np.transpose(np.array(all_pcs), (0, 2, 1, 3))
print(all_pcs.shape)
pcs_flat = all_pcs.reshape(len(testSet.lhs), len(utils.z_mps) * num_pcs)
t_comps = trainSet.tpca.transform(pcs_flat)

pcs_flatReconstructed = trainSet.tpca.inverse_transform(t_comps)
pcs_pred_z_stack = pcs_flat.reshape(len(testSet.lhs), len(utils.z_mps), num_pcs)
stacks = []
for pcs_pred_z_stack_i in pcs_pred_z_stack:
    reconstructed_fracs = [
        trainSet.frac_pks_scalers[iz].inverse_transform(
            trainSet.pcas[iz].inverse_transform(
                pcs_z.reshape(1, -1)
            )
        )[0] # Extract the 1D result from the (1, 2000) array
        for iz, pcs_z in zip(range(len(utils.z_mps)), pcs_pred_z_stack_i)
    ]
    stack = np.stack(reconstructed_fracs)
    stacks.append([stack])

stacks = np.array(stacks)

errors = np.exp(stacks[:,0,0,:])/testSet.frac_pks[:,0,:] - 1 #np.exp(trainSet.frac_pks_scaler.inverse_transform(testLogFracsNormReconstructed))/testSet.frac_pks - 1
for error in errors: plt.semilogx(trainSet.ks, error)
# plt.ylim(-0.02, 0.02)
plt.xlabel("k")
plt.ylabel(r"$P_\mathrm{tPCA-reconstructed}/P - 1$")
plt.title(fr"PCA Reconstruction Errors, $N_\mathrm{{PC}} = ${num_pcs} $N_{{t comps}} = ${num_pcs_z}" )
plt.savefig(f"mps_emu/validation_figs/tpca_errors_{cosmo_type}_{nl_type}__{prior_type}_n{num_pcs}_z{num_pcs_z}.pdf")

# After the existing z=0 plots, add z=3 analysis

# Find z=3 index
z_idx_3 = 33
print(f"\n=== Analysis for z=3 (index {z_idx_3}) ===")

# PCA reconstruction errors for z=3
plt.figure()
testLogFracsNorm_z3 = trainSet.frac_pks_scalers[z_idx_3].transform(testSet.logfracs[:, z_idx_3, :])
testPcs_z3 = trainSet.pcas[z_idx_3].transform(testLogFracsNorm_z3)
testLogFracsNormReconstructed_z3 = trainSet.pcas[z_idx_3].inverse_transform(testPcs_z3)

errors_z3 = np.exp(trainSet.frac_pks_scalers[z_idx_3].inverse_transform(testLogFracsNormReconstructed_z3[:]))/testSet.frac_pks[:,z_idx_3,:] - 1
for error in errors_z3: plt.semilogx(trainSet.ks, error)
plt.xlabel("k")
plt.ylabel(r"$P_\mathrm{PCA-reconstructed}/P - 1$")
plt.title(r"PCA Reconstruction Errors at z=3, $N_\mathrm{PC} = $" + f"{num_pcs}")
plt.savefig(f"mps_emu/validation_figs/pca_errors_z3_{cosmo_type}_{nl_type}_{prior_type}_n{num_pcs}_z{num_pcs_z}.pdf")

# tPCA reconstruction errors for z=3
plt.figure()
errors_tpca_z3 = np.exp(stacks[:,0,z_idx_3,:])/testSet.frac_pks[:,z_idx_3,:] - 1
for error in errors_tpca_z3: plt.semilogx(trainSet.ks, error)
plt.xlabel("k")
plt.ylabel(r"$P_\mathrm{tPCA-reconstructed}/P - 1$")
plt.title(fr"tPCA Reconstruction Errors at z=3, $N_\mathrm{{PC}} = ${num_pcs} $N_{{t comps}} = ${num_pcs_z}")
plt.savefig(f"mps_emu/validation_figs/tpca_errors_z3_{cosmo_type}_{nl_type}_{prior_type}_n{num_pcs}_z{num_pcs_z}.pdf")