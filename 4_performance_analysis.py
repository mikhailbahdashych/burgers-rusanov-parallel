#!/usr/bin/env python3
"""
Performance analysis module for parallel Burgers equation solver.

Calculates and visualizes:
- Speedup: S(P) = T(1) / T(P)
- Efficiency: E(P) = S(P) / P × 100%
- Karp-Flatt metric: e = (1/S(P) - 1/P) / (1 - 1/P)
- Strong scaling analysis
- Communication overhead analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


class PerformanceAnalyzer:
    """Analyze parallel performance metrics."""

    def __init__(self, results_file: Optional[str] = None):
        """
        Initialize performance analyzer.

        Args:
            results_file: JSON file with timing results
        """
        self.timing_data: Dict[int, float] = {}
        self.problem_size: Optional[int] = None

        if results_file:
            self.load_results(results_file)

    def add_timing(self, n_procs: int, elapsed_time: float):
        """
        Add timing result for a given number of processes.

        Args:
            n_procs: Number of MPI processes
            elapsed_time: Elapsed time in seconds
        """
        self.timing_data[n_procs] = elapsed_time

    def load_results(self, filename: str):
        """
        Load timing results from JSON file.

        Args:
            filename: Path to JSON file
        """
        with open(filename, 'r') as f:
            data = json.load(f)
            self.timing_data = {int(k): v for k, v in data['timings'].items()}
            self.problem_size = data.get('problem_size', None)

        print(f"Loaded timing data for {len(self.timing_data)} process counts")

    def save_results(self, filename: str):
        """
        Save timing results to JSON file.

        Args:
            filename: Output file path
        """
        data = {
            'timings': self.timing_data,
            'problem_size': self.problem_size
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved timing results to {filename}")

    def compute_speedup(self) -> Dict[int, float]:
        """
        Compute speedup: S(P) = T(1) / T(P).

        Returns:
            Dictionary mapping number of processes to speedup
        """
        if 1 not in self.timing_data:
            raise ValueError("Sequential timing (P=1) required for speedup calculation")

        t_sequential = self.timing_data[1]
        speedup = {}

        for n_procs, t_parallel in self.timing_data.items():
            speedup[n_procs] = t_sequential / t_parallel

        return speedup

    def compute_efficiency(self) -> Dict[int, float]:
        """
        Compute parallel efficiency: E(P) = S(P) / P × 100%.

        Returns:
            Dictionary mapping number of processes to efficiency (%)
        """
        speedup = self.compute_speedup()
        efficiency = {}

        for n_procs, s in speedup.items():
            efficiency[n_procs] = (s / n_procs) * 100.0

        return efficiency

    def compute_karp_flatt(self) -> Dict[int, float]:
        """
        Compute Karp-Flatt metric: e = (1/S(P) - 1/P) / (1 - 1/P).

        The Karp-Flatt metric estimates the serial fraction of the code.
        Lower values indicate better parallelization.

        Returns:
            Dictionary mapping number of processes to Karp-Flatt metric
        """
        speedup = self.compute_speedup()
        karp_flatt = {}

        for n_procs, s in speedup.items():
            if n_procs == 1:
                continue  # Not defined for P=1

            numerator = (1.0 / s) - (1.0 / n_procs)
            denominator = 1.0 - (1.0 / n_procs)
            karp_flatt[n_procs] = numerator / denominator

        return karp_flatt

    def print_summary(self):
        """Print performance summary table."""
        speedup = self.compute_speedup()
        efficiency = self.compute_efficiency()
        karp_flatt = self.compute_karp_flatt()

        # Sort by number of processes
        n_procs_sorted = sorted(self.timing_data.keys())

        print("\n" + "="*80)
        print("PERFORMANCE ANALYSIS SUMMARY")
        print("="*80)

        if self.problem_size:
            print(f"Problem size: {self.problem_size} grid points")

        print(f"\n{'Procs':>6} {'Time (s)':>12} {'Speedup':>10} {'Efficiency':>12} {'Karp-Flatt':>12}")
        print("-"*80)

        for n_procs in n_procs_sorted:
            t = self.timing_data[n_procs]
            s = speedup[n_procs]
            e = efficiency[n_procs]
            kf = karp_flatt.get(n_procs, 0.0)

            kf_str = f"{kf:.6f}" if n_procs > 1 else "N/A"

            print(f"{n_procs:6d} {t:12.6f} {s:10.4f} {e:11.2f}% {kf_str:>12}")

        print("="*80)

        # Analyze results
        print("\nANALYSIS:")

        # Best speedup
        best_n_procs = max(speedup.keys(), key=lambda k: speedup[k])
        best_speedup = speedup[best_n_procs]
        print(f"  • Best speedup: {best_speedup:.4f}x with {best_n_procs} processes")

        # Efficiency at maximum processes
        max_procs = max(n_procs_sorted)
        max_efficiency = efficiency[max_procs]
        print(f"  • Efficiency at P={max_procs}: {max_efficiency:.2f}%")

        # Karp-Flatt analysis
        if len(karp_flatt) > 0:
            avg_kf = np.mean(list(karp_flatt.values()))
            print(f"  • Average Karp-Flatt metric: {avg_kf:.6f}")
            print(f"    (estimated serial fraction: {avg_kf*100:.4f}%)")

        # Scaling behavior
        if len(n_procs_sorted) >= 3:
            # Check if speedup is close to ideal
            last_three = n_procs_sorted[-3:]
            speedups_last = [speedup[p] / p for p in last_three]
            avg_scaled_speedup = np.mean(speedups_last)

            if avg_scaled_speedup > 0.8:
                scaling = "Excellent"
            elif avg_scaled_speedup > 0.6:
                scaling = "Good"
            elif avg_scaled_speedup > 0.4:
                scaling = "Moderate"
            else:
                scaling = "Poor"

            print(f"  • Scaling behavior: {scaling}")

        print()

    def plot_scaling_analysis(self, output_file: Optional[str] = None):
        """
        Create comprehensive scaling analysis plots.

        Args:
            output_file: Output file path
        """
        speedup = self.compute_speedup()
        efficiency = self.compute_efficiency()
        karp_flatt = self.compute_karp_flatt()

        n_procs = sorted(speedup.keys())
        speedup_vals = [speedup[p] for p in n_procs]
        efficiency_vals = [efficiency[p] for p in n_procs]

        # Ideal speedup for comparison
        ideal_speedup = n_procs

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Speedup (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(n_procs, speedup_vals, 'bo-', linewidth=2, markersize=8,
                label='Actual Speedup')
        ax1.plot(n_procs, ideal_speedup, 'r--', linewidth=2, alpha=0.7,
                label='Ideal Speedup')
        ax1.set_xlabel('Number of Processes (P)', fontsize=12)
        ax1.set_ylabel('Speedup S(P)', fontsize=12)
        ax1.set_title('Strong Scaling: Speedup', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=0)

        # 2. Efficiency (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(n_procs, efficiency_vals, 'go-', linewidth=2, markersize=8)
        ax2.axhline(y=100, color='r', linestyle='--', linewidth=2, alpha=0.7,
                   label='Ideal Efficiency')
        ax2.axhline(y=80, color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
                   label='80% Threshold')
        ax2.set_xlabel('Number of Processes (P)', fontsize=12)
        ax2.set_ylabel('Efficiency E(P) [%]', fontsize=12)
        ax2.set_title('Parallel Efficiency', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(left=0)
        ax2.set_ylim(bottom=0, top=110)

        # 3. Karp-Flatt metric (top right)
        if len(karp_flatt) > 0:
            ax3 = fig.add_subplot(gs[0, 2])
            kf_procs = sorted(karp_flatt.keys())
            kf_vals = [karp_flatt[p] for p in kf_procs]
            ax3.plot(kf_procs, kf_vals, 'mo-', linewidth=2, markersize=8)
            ax3.set_xlabel('Number of Processes (P)', fontsize=12)
            ax3.set_ylabel('Karp-Flatt Metric e', fontsize=12)
            ax3.set_title('Karp-Flatt: Serial Fraction Estimate', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(left=0)

            # Add interpretation
            avg_kf = np.mean(kf_vals)
            ax3.axhline(y=avg_kf, color='red', linestyle='--', alpha=0.5,
                       label=f'Average: {avg_kf:.6f}')
            ax3.legend(fontsize=10)

        # 4. Execution time (bottom left)
        ax4 = fig.add_subplot(gs[1, 0])
        times = [self.timing_data[p] for p in n_procs]
        ax4.plot(n_procs, times, 'ro-', linewidth=2, markersize=8)
        ax4.set_xlabel('Number of Processes (P)', fontsize=12)
        ax4.set_ylabel('Execution Time [s]', fontsize=12)
        ax4.set_title('Execution Time vs. Process Count', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(left=0)
        ax4.set_yscale('log')

        # 5. Speedup on log-log scale (bottom middle)
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.loglog(n_procs, speedup_vals, 'bo-', linewidth=2, markersize=8,
                  label='Actual Speedup')
        ax5.loglog(n_procs, ideal_speedup, 'r--', linewidth=2, alpha=0.7,
                  label='Ideal (slope=1)')
        ax5.set_xlabel('Number of Processes (P)', fontsize=12)
        ax5.set_ylabel('Speedup S(P)', fontsize=12)
        ax5.set_title('Log-Log Scaling Plot', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3, which='both')

        # 6. Performance table (bottom right)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')

        # Create table data
        table_data = [['Procs', 'Time[s]', 'Speedup', 'Eff[%]']]
        for p in n_procs:
            row = [
                f"{p}",
                f"{self.timing_data[p]:.4f}",
                f"{speedup[p]:.2f}",
                f"{efficiency[p]:.1f}"
            ]
            table_data.append(row)

        table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(table_data)):
            color = '#f0f0f0' if i % 2 == 0 else 'white'
            for j in range(4):
                table[(i, j)].set_facecolor(color)

        ax6.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)

        # Overall title
        title = 'Strong Scaling Analysis: Burgers Equation (Rusanov Method)'
        if self.problem_size:
            title += f'\nProblem Size: {self.problem_size} grid points'
        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.995)

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"\nSaved scaling analysis to {output_file}")
        else:
            plt.show()

        plt.close()


def run_scaling_study(nx: int, t_final: float, max_procs: int,
                     output_dir: str = 'scaling_results'):
    """
    Run a full scaling study by executing the parallel solver multiple times.

    Args:
        nx: Global grid size
        t_final: Final simulation time
        max_procs: Maximum number of processes to test
        output_dir: Directory for results
    """
    import subprocess
    import time

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\n{'='*80}")
    print(f"RUNNING STRONG SCALING STUDY")
    print(f"{'='*80}")
    print(f"Problem size: {nx} grid points")
    print(f"Final time: {t_final}")
    print(f"Testing process counts: 1 to {max_procs}")
    print(f"Results directory: {output_dir}\n")

    analyzer = PerformanceAnalyzer()
    analyzer.problem_size = nx

    # Test different process counts
    process_counts = [2**i for i in range(int(np.log2(max_procs)) + 1) if 2**i <= max_procs]
    if 1 not in process_counts:
        process_counts = [1] + process_counts

    for n_procs in process_counts:
        print(f"\nRunning with {n_procs} process(es)...")

        result_file = output_path / f'result_p{n_procs:04d}.npz'

        if n_procs == 1:
            cmd = [
                'python3', '1_sequential_rusanov.py',
                '--nx', str(nx),
                '--t-final', str(t_final),
                '--save', str(result_file)
            ]
        else:
            cmd = [
                'mpiexec', '-n', str(n_procs),
                'python3', '2_parallel_rusanov.py',
                '--nx', str(nx),
                '--t-final', str(t_final),
                '--save', str(result_file)
            ]

        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"  => Completed in {elapsed:.4f} seconds")
            analyzer.add_timing(n_procs, elapsed)
        else:
            print(f"  X Failed with return code {result.returncode}")
            print(f"  Error: {result.stderr}")

    # Save timing results
    timing_file = output_path / 'timing_results.json'
    analyzer.save_results(str(timing_file))

    # Print summary
    analyzer.print_summary()

    # Create plots
    plot_file = output_path / 'scaling_analysis.png'
    analyzer.plot_scaling_analysis(str(plot_file))

    print(f"\n{'='*80}")
    print(f"Scaling study complete! Results saved to {output_dir}/")
    print(f"{'='*80}\n")


def main():
    """Main function for performance analysis."""
    parser = argparse.ArgumentParser(description='Performance analysis for parallel solver')
    parser.add_argument('--timing-file', type=str,
                       help='JSON file with timing results')
    parser.add_argument('--output', type=str, default='scaling_analysis.png',
                       help='Output file for plots')
    parser.add_argument('--run-study', action='store_true',
                       help='Run a full scaling study')
    parser.add_argument('--nx', type=int, default=1000,
                       help='Grid size for scaling study')
    parser.add_argument('--t-final', type=float, default=0.2,
                       help='Final time for scaling study')
    parser.add_argument('--max-procs', type=int, default=8,
                       help='Maximum number of processes to test')
    parser.add_argument('--output-dir', type=str, default='scaling_results',
                       help='Output directory for scaling study')

    args = parser.parse_args()

    if args.run_study:
        run_scaling_study(args.nx, args.t_final, args.max_procs, args.output_dir)
    elif args.timing_file:
        analyzer = PerformanceAnalyzer(args.timing_file)
        analyzer.print_summary()
        analyzer.plot_scaling_analysis(args.output)
    else:
        parser.print_help()
        print("\nError: Either --timing-file or --run-study must be specified")


if __name__ == "__main__":
    main()
