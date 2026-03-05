#!/bin/bash
# =============================================================================
# train_all_parallel.sh — MPS Emulator N-train Scaling: All Jobs in One Node
#
# Runs 8 training processes (4 n_batches x 2 cosmo_types) in parallel on a
# single hbm-1tb-long-96core node, pinning each process to its own CPU slice
# to avoid contention. Once all training finishes, runs the evaluation script
# automatically.
#
# Usage:
#   sbatch ./mps_emu/train_all_parallel.sh
# =============================================================================

#SBATCH --job-name=mps_ntrain_scaling
#SBATCH --output=/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/mps_emu/out/ntrain_scaling_%j.txt
#SBATCH --time=48:00:00
#SBATCH --partition=hbm-1tb-long-96core
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=victoria.lloyd@stonybrook.edu

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

echo "============================================================"
echo " Job name        : $SLURM_JOB_NAME"
echo " Job ID          : $SLURM_JOBID"
echo " Running on      : $(hostname)"
echo " Start time      : $(date)"
echo " Total CPUs      : $SLURM_JOB_CPUS_PER_NODE"
echo " Tasks           : $SLURM_NTASKS"
echo " CPUs per task   : $SLURM_CPUS_PER_TASK"
echo "============================================================"

module purge > /dev/null 2>&1
module load slurm

source ~/.bashrc
source /gpfs/projects/MirandaGroup/victoria/miniconda/etc/profile.d/conda.sh
conda activate vic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE="/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa"
TRAIN_SCRIPT="${BASE}/mps_emu/train.py"
EVAL_SCRIPT="${BASE}/mps_emu/eval_ntrain_scaling.py"
LOG_DIR="${BASE}/mps_emu/out/parallel_logs"
mkdir -p "${LOG_DIR}"

PRIOR_TYPE="constrained"
NL_TYPE="halofit"
N_BATCHES_LIST=(30 40 50) # 5 10 15 20 with 14, 14, 6 respectively
COSMO_TYPES=("w0wacdm" "lcdm")

# Threads per training process: floor(96 / 8) = 12
THREADS_PER_JOB=${SLURM_CPUS_PER_TASK}

echo ""
echo "[INFO] Configuration:"
echo "  PRIOR_TYPE      = ${PRIOR_TYPE}"
echo "  NL_TYPE         = ${NL_TYPE}"
echo "  N_BATCHES_LIST  = ${N_BATCHES_LIST[*]}"
echo "  COSMO_TYPES     = ${COSMO_TYPES[*]}"
echo "  THREADS_PER_JOB = ${THREADS_PER_JOB}"
echo ""

# ---------------------------------------------------------------------------
# Launch all 8 training processes in parallel
#
# Each process is:
#   - pinned to its own contiguous block of THREADS_PER_JOB cores via taskset
#     to prevent TF thread pools from competing across processes
#   - given its own log file under LOG_DIR
#   - launched in the background with &
#
# CPU layout (0-indexed, 96 cores total, 12 per job):
#   job 0 → cores  0 –  11
#   job 1 → cores 12 –  23
#   job 2 → cores 24 –  35
#   job 3 → cores 36 –  47
#   job 4 → cores 48 –  59
#   job 5 → cores 60 –  71
#   job 6 → cores 72 –  83
#   job 7 → cores 84 –  95
#   ...
#   job 13 → cores 78 – 83
#   (cores 84–95 unused)
# ---------------------------------------------------------------------------

PIDS=()
JOB_LABELS=()
JOB_INDEX=0

for COSMO in "${COSMO_TYPES[@]}"; do
    for N_BATCHES in "${N_BATCHES_LIST[@]}"; do

        LABEL="${COSMO}_nb${N_BATCHES}"
        LOG_FILE="${LOG_DIR}/train_${LABEL}_${SLURM_JOBID}.txt"

        # Compute CPU range for this job slot
        CPU_START=$(( JOB_INDEX * THREADS_PER_JOB ))
        CPU_END=$(( CPU_START + THREADS_PER_JOB - 1 ))
        CPU_RANGE="${CPU_START}-${CPU_END}"

        echo "[INFO] Launching job ${JOB_INDEX}: ${LABEL}"
        echo "       CPUs: ${CPU_RANGE}  →  log: ${LOG_FILE}"

        # Set per-process thread counts and launch
        (
            export OMP_NUM_THREADS=${THREADS_PER_JOB}
            export TF_NUM_INTRAOP_THREADS=${THREADS_PER_JOB}
            export TF_NUM_INTEROP_THREADS=1
            export OMP_PROC_BIND=close
            export OMP_PLACES=cores

            taskset --cpu-list ${CPU_RANGE} python "${TRAIN_SCRIPT}" \
                --cosmo_type  "${COSMO}"      \
                --prior_type  "${PRIOR_TYPE}" \
                --nl_type     "${NL_TYPE}"    \
                --n_batches   "${N_BATCHES}"  \
                2>&1 | tee "${LOG_FILE}"
        ) &

        PIDS+=($!)
        JOB_LABELS+=("${LABEL}")

        JOB_INDEX=$(( JOB_INDEX + 1 ))
    done
done

echo ""
echo "[INFO] All 8 training processes launched. PIDs: ${PIDS[*]}"
echo "[INFO] Waiting for all training jobs to complete..."
echo ""

# ---------------------------------------------------------------------------
# Wait for all background processes and check exit codes
# ---------------------------------------------------------------------------

ALL_OK=true

for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    LABEL=${JOB_LABELS[$i]}

    wait "${PID}"
    EXIT_CODE=$?

    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "[  OK  ] ${LABEL} finished successfully (PID ${PID})"
    else
        echo "[ FAIL ] ${LABEL} exited with code ${EXIT_CODE} (PID ${PID})"
        echo "         Check log: ${LOG_DIR}/train_${LABEL}_${SLURM_JOBID}.txt"
        ALL_OK=false
    fi
done

echo ""

# ---------------------------------------------------------------------------
# Run evaluation — only if all training jobs succeeded
# ---------------------------------------------------------------------------

if [ "${ALL_OK}" = true ]; then
    echo "============================================================"
    echo "[INFO] All training jobs succeeded. Starting evaluation..."
    echo "       Start time: $(date)"
    echo "============================================================"
    echo ""

    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

    python "${EVAL_SCRIPT}" \
        --prior_type "${PRIOR_TYPE}" \
        --nl_type    "${NL_TYPE}"

    EVAL_EXIT=$?

    if [ ${EVAL_EXIT} -eq 0 ]; then
        echo ""
        echo "[INFO] Evaluation completed successfully."
    else
        echo ""
        echo "[ERROR] Evaluation script exited with code ${EVAL_EXIT}."
        exit ${EVAL_EXIT}
    fi

else
    echo "============================================================"
    echo "[ERROR] One or more training jobs FAILED. Skipping evaluation."
    echo "        Check individual logs in: ${LOG_DIR}/"
    echo "============================================================"
    exit 1
fi

echo ""
echo "============================================================"
echo " All done. Finished: $(date)"
echo "============================================================"