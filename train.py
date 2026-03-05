"""
train.py — MPS Emulator Training Script

Usage examples:
    # Minimal (required flags only):
    python ./mps_emu/train.py --cosmo_type w0wacdm --prior_type expanded --nl_type lin

    # Full override:
    python ./mps_emu/train.py \
        --cosmo_type w0wacdm \
        --prior_type expanded \
        --nl_type lin \
        --n_batches 20 \
        --num_epochs 3000 \
        --num_layers 4 \
        --num_neurons 1024 \
        --num_pcs 25 \
        --num_pcs_z 15 \
        --start_batch 0 \
        --test_batch 100
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import train_utils_pk_emulator as utils

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the MPS power-spectrum emulator neural network.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required / core configuration ---
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--cosmo_type",
        required=True,
        choices=["lcdm", "w0wacdm"],
        help="Cosmological model type.",
    )
    required.add_argument(
        "--prior_type",
        required=True,
        choices=["constrained", "expanded"],
        help="Prior range to use for training.",
    )
    required.add_argument(
        "--nl_type",
        required=True,
        choices=["lin", "halofit", "mead2020", "mead2020_feedback"],
        help="Linear ('lin'), or non-linear matter power spectrum with halofit ('halofit'), " \
        "\n HM Code ('mead2020'), HM Code with baryonic feedback fixed to logT_AGN=7.8 " \
        "\n('mead2020_feedback'), and with logT_AGN free ('mead2020_feedback_Tfree').",
    )

    # --- Batch / data configuration ---
    data_group = parser.add_argument_group("data arguments")
    data_group.add_argument(
        "--start_batch",
        type=int,
        default=0,
        help="Index of the first training batch to load.",
    )
    data_group.add_argument(
        "--n_batches",
        type=int,
        default=20,
        help="Number of training batches to load.",
    )
    data_group.add_argument(
        "--test_batch",
        type=int,
        default=100,
        help="Batch index to use for the test/validation set.",
    )

    # --- PCA configuration ---
    pca_group = parser.add_argument_group("PCA arguments")
    pca_group.add_argument(
        "--num_pcs",
        type=int,
        default=25,
        help="Number of principal components for the power-spectrum axis.",
    )
    pca_group.add_argument(
        "--num_pcs_z",
        type=int,
        default=15,
        help="Number of principal components for the redshift axis.",
    )

    # --- Network / training configuration ---
    nn_group = parser.add_argument_group("neural network arguments")
    nn_group.add_argument(
        "--num_epochs",
        type=int,
        default=3000,
        help="Number of training epochs.",
    )
    nn_group.add_argument(
        "--num_layers",
        type=int,
        default=4,
        help="Number of hidden layers in the network.",
    )
    nn_group.add_argument(
        "--num_neurons",
        type=int,
        default=1024,
        help="Number of neurons per hidden layer.",
    )
    nn_group.add_argument(
        "--decay_every",
        type=int,
        default=80,
        help="Decay the learning rate every this many epochs.",
    )
    nn_group.add_argument(
        "--decay_rate",
        type=float,
        default=1.25,
        help="Learning-rate decay rate.",
    )
    nn_group.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Neural network training batch size.",
    )

    # --- Output configuration ---
    out_group = parser.add_argument_group("output arguments")
    out_group.add_argument(
        "--model_dir",
        default="/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/mps_emu/models",
        help="Directory where the trained model will be saved.",
    )
    out_group.add_argument(
        "--fig_dir",
        default="/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/mps_emu/validation_figs",
        help="Directory where validation figures will be saved.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Print run configuration ---
    print("=" * 60)
    print("[INFO] MPS Emulator Training Configuration")
    print("=" * 60)
    for key, val in vars(args).items():
        print(f"  {key:<20s} = {val}")
    print("=" * 60)

    # --- Directory setup ---
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    # --- Load and prepare training data ---
    start = time.perf_counter()
    print("\n[INFO] Loading training set...")

    target_zs = utils.z_mps

    train_set = utils.COLASet(
        target_z=target_zs,
        cosmo_type=args.cosmo_type,
        prior_type=args.prior_type,
        nl_type=args.nl_type,
        start_batch=args.start_batch,
        n_batches=args.n_batches,
    )
    train_set.prepare(num_pcs=args.num_pcs, num_pcs_z=args.num_pcs_z)

    elapsed = time.perf_counter() - start
    print(f"[INFO] Data loaded and PCA prepared in {elapsed / 60:.2f} minutes.")
    print(f"[INFO] Training redshifts: {train_set.z}")

    # --- Build and train the network ---
    print("\n[INFO] Building neural network...")
    nn_keras = utils.COLA_NN_Keras(
        train_set,
        num_layers=args.num_layers,
        num_neurons=args.num_neurons,
    )

    print("[INFO] Starting training...")
    nn_keras.fit_t_componets(
        train_set,
        num_epochs=args.num_epochs,
        decayevery=args.decay_every,
        decayrate=args.decay_rate,
        batch_size=args.batch_size
    )

    # --- Save model ---
    model = nn_keras.models["t-component"]
    model_name = f"emulator_{args.cosmo_type}_{args.prior_type}_{args.nl_type}_nTrain{args.n_batches}.h5"
    model_path = os.path.join(args.model_dir, model_name)
    model.save(model_path)

    total_elapsed = time.perf_counter() - start
    print(f"\n Model trained and saved to: {model_path}")
    print(f"   Total runtime: {total_elapsed / 60:.2f} minutes.")


if __name__ == "__main__":
    main()