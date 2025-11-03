#!/usr/bin/env python3
"""
Visualization module for Burgers equation results.

Creates plots showing:
- Solution evolution over time
- Shock wave formation and propagation
- Gradient analysis to identify discontinuities
- Comparison between different time steps
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Optional


def plot_solution_evolution(x: np.ndarray, snapshots: np.ndarray,
                            times: np.ndarray, output_file: Optional[str] = None):
    """
    Plot solution evolution over time showing multiple snapshots.

    Args:
        x: Spatial coordinates
        snapshots: Solution snapshots (n_snapshots, nx)
        times: Time values for each snapshot
        output_file: Output file path (if None, display interactively)
    """
    import os
    os.makedirs('plots', exist_ok=True)

    n_snapshots = len(snapshots)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Use colormap for time progression
    colors = plt.cm.viridis(np.linspace(0, 1, n_snapshots))

    for i, (snapshot, t) in enumerate(zip(snapshots, times)):
        ax.plot(x, snapshot, color=colors[i], linewidth=2,
               label=f't = {t:.4f}', alpha=0.7)

    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('u(x, t)', fontsize=14)
    ax.set_title('Burgers Equation Solution Evolution\n(Rusanov Method)', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10, ncol=2)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved evolution plot to {output_file}")
    else:
        plt.show()

    plt.close()


def plot_shock_analysis(x: np.ndarray, snapshots: np.ndarray,
                        times: np.ndarray, output_file: Optional[str] = None):
    """
    Create detailed shock wave analysis plots.

    Args:
        x: Spatial coordinates
        snapshots: Solution snapshots
        times: Time values
        output_file: Output file path
    """
    import os
    os.makedirs('plots', exist_ok=True)

    n_snapshots = len(snapshots)

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Solution evolution (upper left)
    ax1 = fig.add_subplot(gs[0, :])
    colors = plt.cm.viridis(np.linspace(0, 1, n_snapshots))
    for i, (snapshot, t) in enumerate(zip(snapshots, times)):
        ax1.plot(x, snapshot, color=colors[i], linewidth=2, alpha=0.7)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('u(x, t)', fontsize=12)
    ax1.set_title('Solution Evolution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. Initial vs Final comparison (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x, snapshots[0], 'b-', linewidth=2, label=f't = {times[0]:.4f}')
    ax2.plot(x, snapshots[-1], 'r-', linewidth=2, label=f't = {times[-1]:.4f}')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('u(x, t)', fontsize=12)
    ax2.set_title('Initial vs Final State', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Gradient analysis for shock detection (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    for i in [0, n_snapshots//2, n_snapshots-1]:
        gradient = np.gradient(snapshots[i], x)
        ax3.plot(x, np.abs(gradient), linewidth=2,
                label=f't = {times[i]:.4f}', alpha=0.7)
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('|du/dx|', fontsize=12)
    ax3.set_title('Gradient Magnitude (Shock Indicator)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # 4. Heatmap of solution evolution (lower left)
    ax4 = fig.add_subplot(gs[2, 0])
    T, X = np.meshgrid(times, x)
    contour = ax4.contourf(X, T, snapshots.T, levels=20, cmap='RdYlBu_r')
    plt.colorbar(contour, ax=ax4, label='u(x, t)')
    ax4.set_xlabel('x', fontsize=12)
    ax4.set_ylabel('t', fontsize=12)
    ax4.set_title('Space-Time Evolution (Heatmap)', fontsize=14, fontweight='bold')

    # 5. Maximum gradient over time (shock strength evolution)
    ax5 = fig.add_subplot(gs[2, 1])
    max_gradients = []
    for snapshot in snapshots:
        gradient = np.abs(np.gradient(snapshot, x))
        max_gradients.append(np.max(gradient))
    ax5.plot(times, max_gradients, 'ko-', linewidth=2, markersize=6)
    ax5.set_xlabel('Time', fontsize=12)
    ax5.set_ylabel('Max |du/dx|', fontsize=12)
    ax5.set_title('Shock Strength Evolution', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Add shock detection annotation
    final_max_gradient = max_gradients[-1]
    if final_max_gradient > 10.0:
        shock_text = "Strong Shock Waves Detected!"
        color = 'red'
    elif final_max_gradient > 2.0:
        shock_text = "Moderate Discontinuities"
        color = 'orange'
    else:
        shock_text = "Smooth Solution"
        color = 'green'

    fig.text(0.5, 0.02, shock_text, ha='center', fontsize=14,
            fontweight='bold', color=color)

    plt.suptitle('Burgers Equation: Shock Wave Analysis (Rusanov Method)',
                fontsize=18, fontweight='bold', y=0.995)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved shock analysis to {output_file}")
    else:
        plt.show()

    plt.close()


def create_animation_frames(x: np.ndarray, snapshots: np.ndarray,
                           times: np.ndarray, output_dir: str = 'frames'):
    """
    Create individual frames for animation.

    Args:
        x: Spatial coordinates
        snapshots: Solution snapshots
        times: Time values
        output_dir: Directory to save frames
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Determine y-axis limits
    y_min = np.min(snapshots) - 0.1
    y_max = np.max(snapshots) + 0.1

    for i, (snapshot, t) in enumerate(zip(snapshots, times)):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(x, snapshot, 'b-', linewidth=2)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('u(x, t)', fontsize=14)
        ax.set_title(f'Burgers Equation Solution\nt = {t:.6f}',
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add gradient indicator
        gradient = np.gradient(snapshot, x)
        max_grad = np.max(np.abs(gradient))
        ax.text(0.02, 0.98, f'Max |du/dx| = {max_grad:.2f}',
               transform=ax.transAxes, fontsize=12,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        frame_file = output_path / f'frame_{i:04d}.png'
        plt.savefig(frame_file, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nCreated {len(snapshots)} animation frames in {output_dir}/")
    print(f"To create video, run:")
    print(f"  ffmpeg -framerate 10 -pattern_type glob -i '{output_dir}/frame_*.png' "
          f"-c:v libx264 -pix_fmt yuv420p burgers_evolution.mp4")


def compare_sequential_parallel(seq_file: str, par_file: str,
                                output_file: Optional[str] = None):
    """
    Compare sequential and parallel results.

    Args:
        seq_file: Sequential results file
        par_file: Parallel results file
        output_file: Output file path
    """
    # Load data
    seq_data = np.load(seq_file)
    par_data = np.load(par_file)

    x_seq = seq_data['x']
    u_seq = seq_data['u_final']

    x_par = par_data['x']
    u_par = par_data['u_final']

    # Interpolate if grids are different
    if len(x_seq) != len(x_par):
        u_par_interp = np.interp(x_seq, x_par, u_par)
    else:
        u_par_interp = u_par

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Solution comparison
    ax1.plot(x_seq, u_seq, 'b-', linewidth=2, label='Sequential')
    ax1.plot(x_par, u_par, 'r--', linewidth=2, label='Parallel', alpha=0.7)
    ax1.set_xlabel('x', fontsize=14)
    ax1.set_ylabel('u(x, t)', fontsize=14)
    ax1.set_title('Sequential vs Parallel Solution', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Error analysis
    error = np.abs(u_seq - u_par_interp)
    ax2.plot(x_seq, error, 'g-', linewidth=2)
    ax2.set_xlabel('x', fontsize=14)
    ax2.set_ylabel('|Error|', fontsize=14)
    ax2.set_title('Absolute Error', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Statistics
    max_error = np.max(error)
    mean_error = np.mean(error)
    fig.text(0.5, 0.02,
            f'Max Error: {max_error:.2e}  |  Mean Error: {mean_error:.2e}',
            ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {output_file}")
    else:
        plt.show()

    plt.close()


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Visualize Burgers equation results')
    parser.add_argument('input', type=str, help='Input results file (.npz)')
    parser.add_argument('--output', type=str, help='Output file for plots')
    parser.add_argument('--type', type=str, default='shock',
                       choices=['evolution', 'shock', 'frames', 'compare'],
                       help='Type of visualization')
    parser.add_argument('--compare-with', type=str,
                       help='File to compare with (for --type compare)')
    parser.add_argument('--frames-dir', type=str, default='frames',
                       help='Directory for animation frames')

    args = parser.parse_args()

    # Load data
    data = np.load(args.input)
    x = data['x']
    snapshots = data['snapshots']
    times = data['times']

    print(f"\nLoaded results from {args.input}")
    print(f"Grid points: {len(x)}")
    print(f"Number of snapshots: {len(snapshots)}")
    print(f"Time range: [{times[0]:.6f}, {times[-1]:.6f}]")

    # Generate visualization based on type
    if args.type == 'evolution':
        output = args.output or 'plots/solution_evolution_parallel.png'
        plot_solution_evolution(x, snapshots, times, output)

    elif args.type == 'shock':
        output = args.output or 'plots/shock_analysis_parallel.png'
        plot_shock_analysis(x, snapshots, times, output)

    elif args.type == 'frames':
        create_animation_frames(x, snapshots, times, args.frames_dir)

    elif args.type == 'compare':
        if not args.compare_with:
            print("Error: --compare-with required for comparison")
            return
        output = args.output or 'comparison.png'
        compare_sequential_parallel(args.input, args.compare_with, output)

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
