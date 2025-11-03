#!/bin/bash
#SBATCH --job-name=burgers_mpi
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --time=02:00:00
#SBATCH --account=plgar2025-cpu
#SBATCH --qos=normal
#SBATCH --partition=plgrid
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load required modules
module load python/3.11
module load openmpi/4.1
module load scipy-bundle/2021.10-intel-2021b

# Ensure output directories exist
mkdir -p results
mkdir -p logs
mkdir -p plots

# Grid sizes to test (for strong scaling: same problem, different process counts)
GRIDS="300 600 1200"

# Process counts to test
PROCS="1 2 4 8 16 24 32 48"

# Simulation parameters
T_FINAL=0.5
CFL=0.3
NU=0.1
IC_TYPE="sine"

echo "Starting Burgers equation parallel experiments..."
echo "Grid sizes: $GRIDS"
echo "Process counts: $PROCS"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "======================================"

# Run experiments
for N in $GRIDS; do
    for P in $PROCS; do
        echo ""
        echo "Running nx=$N, P=$P processes..."
        OUTPUT_FILE="results/burgers_nx${N}_P${P}.npz"

        # Run with mpiexec
        mpiexec -n $P python 2_parallel_rusanov.py \
            --nx $N \
            --t-final $T_FINAL \
            --cfl $CFL \
            --nu $NU \
            --ic $IC_TYPE \
            --snapshots 10 \
            --save $OUTPUT_FILE

        if [ $? -eq 0 ]; then
            echo "  => Success ($OUTPUT_FILE)"
        else
            echo "  X Failed ($OUTPUT_FILE)"
        fi
    done
done

echo ""
echo "======================================"
echo "All experiments completed!"
echo "End time: $(date)"
echo "Results saved in results/"
echo "======================================"

# Create visualizations for selected results
echo ""
echo "Creating visualizations..."

# Visualize one representative result
for N in $GRIDS; do
    echo "Creating visualization for nx=$N, P=1 (sequential)..."
    python 3_visualization.py results/burgers_nx${N}_P1.npz --type shock
done

echo "Visualizations complete!"
