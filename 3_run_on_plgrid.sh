#!/bin/bash
#SBATCH --job-name=burgers_ALL
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --time=01:30:00
#SBATCH --account=plgar2025-cpu
#SBATCH --qos=normal
#SBATCH --partition=plgrid
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

################################################################################
# OPTIMIZED BURGERS EQUATION PARALLEL SCALING EXPERIMENTS
#
# This script runs BOTH weak and strong scaling tests with parameters
# optimized to complete in reasonable time while demonstrating good scaling.
#
# Key optimizations:
#   - Reduced viscosity: nu=0.2 (instead of 0.1)  50% fewer timesteps
#   - Reduced t_final: 0.2 (instead of 0.5)  60% fewer timesteps
#   - Appropriate grid sizes: ensures completion in ~1 hour
#
# Expected results:
#   - Weak scaling efficiency: 70-85%
#   - Strong scaling efficiency: 75-90% for appropriate P
#   - Total runtime: ~60-90 minutes
################################################################################

module load python/3.11
module load openmpi/4.1
module load scipy-bundle/2021.10-intel-2021b

mkdir -p plgrid_results
mkdir -p logs

# Optimized parameters for fast runtime (< 2 hours, max P=24)
T_FINAL=0.1      # Reduced from 0.2  50% fewer timesteps
NU=0.3           # Increased from 0.2  fewer timesteps
CFL=0.3
IC_TYPE="sine"
SNAPSHOTS=3      # Reduced from 5  faster I/O

echo "=============================================================================="
echo "BURGERS EQUATION PARALLEL SCALING EXPERIMENTS"
echo "=============================================================================="
echo "Parameters: t_final=$T_FINAL, nu=$NU, CFL=$CFL, IC=$IC_TYPE"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "=============================================================================="
echo ""

################################################################################
# QUICK VALIDATION TEST (catch errors early!)
################################################################################

echo ""
echo "=============================================================================="
echo "QUICK VALIDATION TEST"
echo "=============================================================================="
echo "Running quick test to ensure fixed code works correctly..."
echo ""

mpiexec -n 1 python 2_parallel_rusanov.py \
    --nx 300 --t-final 0.01 --cfl $CFL --nu $NU \
    --ic $IC_TYPE --snapshots 2 \
    --save plgrid_results/validation_test.npz

if [ $? -eq 0 ]; then
    echo "[OK] Validation test passed - proceeding with full experiments"
    echo ""
else
    echo "[FAILED] Validation test failed - aborting!"
    echo "Please check the code before resubmitting."
    exit 1
fi

################################################################################
# PART 1: WEAK SCALING (300 cells/processor)
################################################################################

echo ""
echo "=============================================================================="
echo "PART 1: WEAK SCALING EXPERIMENTS"
echo "=============================================================================="
echo "Strategy: Keep 300 cells per processor constant"
echo "Expected: Demonstrates weak scaling behavior"
echo ""

CELLS_PER_PROC=300
WEAK_PROCS="1 4 8 16 24"  # Reduced test points for faster runtime

for P in $WEAK_PROCS; do
    N=$((P * CELLS_PER_PROC))
    echo "----------------------------------------------------------------------"
    echo "Weak Scaling: P=$P, nx=$N ($CELLS_PER_PROC cells/proc)"
    echo "----------------------------------------------------------------------"

    OUTPUT_FILE="plgrid_results/burgers_weak_nx${N}_P${P}.npz"

    time mpiexec -n $P python 2_parallel_rusanov.py \
        --nx $N \
        --t-final $T_FINAL \
        --cfl $CFL \
        --nu $NU \
        --ic $IC_TYPE \
        --snapshots $SNAPSHOTS \
        --save $OUTPUT_FILE

    if [ $? -eq 0 ]; then
        echo "[OK] Success: $OUTPUT_FILE"
    else
        echo "[FAILED] $OUTPUT_FILE"
    fi
    echo ""
done

################################################################################
# PART 2: STRONG SCALING (fixed problem sizes)
################################################################################

echo ""
echo "=============================================================================="
echo "PART 2: STRONG SCALING EXPERIMENTS"
echo "=============================================================================="
echo "Strategy: Fixed grid sizes, increasing processors"
echo "Focus on larger grids for better scaling demonstration"
echo ""

# Grid 1: nx=1200 (medium)
echo "--- Grid Size: nx=1200 ---"
for P in 1 2 4 8; do
    echo "  P=$P: $((1200/P)) cells/proc"
    OUTPUT_FILE="plgrid_results/burgers_strong_nx1200_P${P}.npz"

    mpiexec -n $P python 2_parallel_rusanov.py \
        --nx 1200 --t-final $T_FINAL --cfl $CFL --nu $NU \
        --ic $IC_TYPE --snapshots $SNAPSHOTS --save $OUTPUT_FILE

    [ $? -eq 0 ] && echo "    [OK] $OUTPUT_FILE" || echo "    [FAILED]"
done
echo ""

# Grid 2: nx=2400 (large)
echo "--- Grid Size: nx=2400 ---"
for P in 1 4 8 16; do
    echo "  P=$P: $((2400/P)) cells/proc"
    OUTPUT_FILE="plgrid_results/burgers_strong_nx2400_P${P}.npz"

    mpiexec -n $P python 2_parallel_rusanov.py \
        --nx 2400 --t-final $T_FINAL --cfl $CFL --nu $NU \
        --ic $IC_TYPE --snapshots $SNAPSHOTS --save $OUTPUT_FILE

    [ $? -eq 0 ] && echo "    [OK] $OUTPUT_FILE" || echo "    [FAILED]"
done
echo ""

# Grid 3: nx=4800 (extra large)
echo "--- Grid Size: nx=4800 ---"
for P in 1 4 8 16 24; do
    echo "  P=$P: $((4800/P)) cells/proc"
    OUTPUT_FILE="plgrid_results/burgers_strong_nx4800_P${P}.npz"

    mpiexec -n $P python 2_parallel_rusanov.py \
        --nx 4800 --t-final $T_FINAL --cfl $CFL --nu $NU \
        --ic $IC_TYPE --snapshots $SNAPSHOTS --save $OUTPUT_FILE

    [ $? -eq 0 ] && echo "    [OK] $OUTPUT_FILE" || echo "    [FAILED]"
done
echo ""

################################################################################
# PART 3: SHOCK WAVE DEMONSTRATION (sequential only, for visualization)
################################################################################

echo ""
echo "=============================================================================="
echo "PART 3: SHOCK WAVE DEMONSTRATION"
echo "=============================================================================="
echo "Running one shock demo for visualization"
echo ""

# Low viscosity for shock formation
echo "--- Low Viscosity (nu=0.01) with Step IC ---"
mpiexec -n 1 python 2_parallel_rusanov.py \
    --nx 1200 --t-final 0.1 --cfl 0.2 --nu 0.01 \
    --ic step --snapshots 5 \
    --save plgrid_results/burgers_shock_step_nu001.npz
echo ""

################################################################################
# SUMMARY
################################################################################

echo ""
echo "=============================================================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "=============================================================================="
echo "End time: $(date)"
echo ""
echo "Results summary:"
echo "  Weak scaling: 5 runs (P=1,4,8,16,24)"
echo "  Strong scaling: 3 grid sizes × multiple P = ~17 runs"
echo "  Shock wave demo: 1 run"
echo "  Total experiments: ~23 runs"
echo ""
echo "Total result files: $(ls plgrid_results/*.npz 2>/dev/null | wc -l)"
echo ""
echo "Parameters used:"
echo "  t_final = $T_FINAL (reduced for speed)"
echo "  nu = $NU (increased for fewer timesteps)"
echo "  snapshots = $SNAPSHOTS (reduced I/O)"
echo ""
echo "Next steps:"
echo "  1. Download results: scp -r user@ares:/path/plgrid_results ."
echo "  2. Run analysis: jupyter notebook 4_analysis.ipynb"
echo "=============================================================================="
