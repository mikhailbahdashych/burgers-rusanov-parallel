# %% [markdown]
# # Sequential Rusanov Method for Burgers Equation
# 
# This notebook implements a sequential solver for the 1D Burgers equation using the **Rusanov method**.
# 
# ## The Burgers Equation
# 
# **Inviscid form:**
# $$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = 0$$
# 
# **Viscous form:**
# $$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$
# 
# where:
# - $u(x,t)$ is the solution (velocity field)
# - $\nu$ is the viscosity coefficient
# 
# ## The Rusanov Method
# 
# The Rusanov method is a first-order finite volume scheme with numerical flux:
# 
# $$F_{i+1/2} = \frac{1}{2}\left(F(u_i) + F(u_{i+1})\right) - \frac{1}{2}\alpha (u_{i+1} - u_i)$$
# 
# where:
# - $F(u) = \frac{u^2}{2}$ is the flux function for Burgers equation
# - $\alpha = \max(|u_i|, |u_{i+1}|)$ is the maximum wave speed (local Lax-Friedrichs)
# 
# **Update formula:**
# $$u_i^{n+1} = u_i^n - \frac{\Delta t}{\Delta x}\left(F_{i+1/2} - F_{i-1/2}\right) + \text{viscous term}$$
# 
# **CFL Stability Condition:**
# $$\Delta t \leq \text{CFL} \cdot \frac{\Delta x}{\max(|u|)}$$
# 
# where typically $\text{CFL} = 0.5$ for stability.
# 
# **Reference:** Section 2.1.1 of "THE SOLUTION OF A BURGERS EQUATION.pdf"

# %% [markdown]
# ## Import Dependencies

# %%
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Tuple
import os

# Configure matplotlib for better plots
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create output directory for plots
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# %% [markdown]
# ## Solver Implementation
# 
# The `BurgersRusanovSolver` class implements the complete sequential solver.

# %%
class BurgersRusanovSolver:
    """Sequential Rusanov solver for the 1D Burgers equation."""

    def __init__(self, nx: int, domain: Tuple[float, float],
                 t_final: float, cfl: float = 0.3, nu: float = 0.1):
        """
        Initialize the Burgers equation solver.

        Args:
            nx: Number of spatial grid points
            domain: Spatial domain (x_min, x_max)
            t_final: Final simulation time
            cfl: CFL number for stability (typically < 1.0)
            nu: Viscosity coefficient (set to 0 for inviscid)
        """
        self.nx = nx
        self.x_min, self.x_max = domain
        self.t_final = t_final
        self.cfl = cfl
        self.nu = nu

        # Create spatial grid for periodic domain
        self.dx = (self.x_max - self.x_min) / nx
        self.x = np.linspace(self.x_min, self.x_max, nx, endpoint=False)

        # Solution array
        self.u = np.zeros(nx)

        # Time tracking
        self.t = 0.0
        self.dt = 0.0
        self.n_steps = 0

        # Storage for snapshots
        self.snapshots = []
        self.snapshot_times = []

    def set_initial_condition(self, ic_type: str = 'sine'):
        """
        Set initial condition for the solution.

        Args:
            ic_type: Type of initial condition
                - 'sine': Smooth sine wave that develops shock
                - 'step': Step function
                - 'rarefaction': Rarefaction wave test
        """
        if ic_type == 'sine':
            # Smooth sine wave - will develop shock
            self.u = 0.5 + 0.5 * np.sin(2 * np.pi * self.x)
        elif ic_type == 'step':
            # Step function - immediate shock
            self.u = np.where(self.x < 0.5 * (self.x_min + self.x_max), 1.0, 0.0)
        elif ic_type == 'rarefaction':
            # Rarefaction wave
            self.u = np.where(self.x < 0.5 * (self.x_min + self.x_max), 0.0, 1.0)
        else:
            raise ValueError(f"Unknown initial condition type: {ic_type}")

        # Store initial state
        self.snapshots = [self.u.copy()]
        self.snapshot_times = [0.0]

    def flux(self, u: np.ndarray) -> np.ndarray:
        """
        Compute the flux function F(u) = u^2/2 for Burgers equation.

        Args:
            u: Solution values

        Returns:
            Flux values
        """
        return 0.5 * u**2

    def compute_dt(self) -> float:
        """
        Compute time step based on CFL condition.

        For Burgers equation: dt <= CFL * dx / max(|u|)
        For viscous term: dt <= dx^2 / (4*nu) (conservative)

        Returns:
            Time step size
        """
        # Convection CFL condition
        max_speed = np.max(np.abs(self.u))
        max_speed = max(max_speed, 1e-10)
        dt_convection = self.cfl * self.dx / max_speed

        # Diffusion stability condition (conservative)
        dt_diffusion = 0.25 * self.dx**2 / self.nu

        return min(dt_convection, dt_diffusion)

    def step(self) -> None:
        """Perform one time step using the Rusanov method with periodic boundaries."""
        # Compute time step
        self.dt = self.compute_dt()

        # Don't overshoot final time
        if self.t + self.dt > self.t_final:
            self.dt = self.t_final - self.t

        # Periodic boundary using roll
        u_plus = np.roll(self.u, -1)   # u[i+1]
        u_minus = np.roll(self.u, 1)   # u[i-1]

        # Rusanov flux at i+1/2 interface
        alpha_right = np.maximum(np.abs(self.u), np.abs(u_plus))
        f_right = 0.5 * (self.flux(self.u) + self.flux(u_plus)) - 0.5 * alpha_right * (u_plus - self.u)

        # Rusanov flux at i-1/2 interface
        alpha_left = np.maximum(np.abs(u_minus), np.abs(self.u))
        f_left = 0.5 * (self.flux(u_minus) + self.flux(self.u)) - 0.5 * alpha_left * (self.u - u_minus)

        # Update with convective term
        u_new = self.u - (self.dt / self.dx) * (f_right - f_left)

        # Add viscous term (central difference)
        u_new += self.nu * (self.dt / self.dx**2) * (u_plus - 2*self.u + u_minus)

        self.u = u_new

        # Update time
        self.t += self.dt
        self.n_steps += 1

    def solve(self, n_snapshots: int = 10, verbose: bool = True) -> Tuple[np.ndarray, list, list]:
        """
        Solve the Burgers equation until t_final.

        Args:
            n_snapshots: Number of solution snapshots to save
            verbose: Print progress information

        Returns:
            Tuple of (final solution, list of snapshots, list of snapshot times)
        """
        # Determine snapshot interval
        snapshot_interval = self.t_final / n_snapshots
        next_snapshot_time = snapshot_interval

        if verbose:
            print(f"Starting sequential Rusanov solver...")
            print(f"Grid points: {self.nx}")
            print(f"Domain: [{self.x_min}, {self.x_max}]")
            print(f"Grid spacing: {self.dx:.6f}")
            print(f"Final time: {self.t_final}")
            print(f"CFL number: {self.cfl}")
            print(f"Viscosity: {self.nu}")
            print()

        start_time = time.time()

        # Time integration loop
        while self.t < self.t_final:
            self.step()

            # Save snapshots
            if self.t >= next_snapshot_time or abs(self.t - self.t_final) < 1e-10:
                self.snapshots.append(self.u.copy())
                self.snapshot_times.append(self.t)
                next_snapshot_time += snapshot_interval
                if verbose:
                    print(f"Step {self.n_steps}: t = {self.t:.6f}, dt = {self.dt:.6e}, "
                          f"max(u) = {np.max(self.u):.6f}, min(u) = {np.min(self.u):.6f}")

        self.elapsed_time = time.time() - start_time

        if verbose:
            print()
            print(f"Simulation complete!")
            print(f"Total time steps: {self.n_steps}")
            print(f"Elapsed time: {self.elapsed_time:.6f} seconds")
            print(f"Average time per step: {self.elapsed_time/self.n_steps:.6e} seconds")

        return self.u, self.snapshots, self.snapshot_times

# %% [markdown]
# ## Problem Setup
# 
# Configure the simulation parameters:
# - **Grid points** (`nx`): Number of spatial discretization points
# - **Domain**: Spatial extent $[x_{\min}, x_{\max}]$
# - **Final time** (`t_final`): Simulation duration
# - **CFL number**: Controls timestep stability
# - **Viscosity** ($\nu$): Diffusion coefficient
# - **Initial condition**: Starting solution profile

# %%
# Configuration: Multiple grid sizes for scaling study
GRID_SIZES = [1200, 2400, 4800]  # Grid sizes to test

# Simulation parameters (shared across all runs)
domain = (0.0, 1.0)        # Spatial domain [x_min, x_max]
t_final = 0.5              # Final simulation time
cfl = 0.3                  # CFL number (reduced for stability)
nu = 0.1                   # Viscosity coefficient (balanced for stability and shock formation)
ic_type = 'sine'           # Initial condition: 'sine', 'step', or 'rarefaction'
n_snapshots = 10           # Number of snapshots to save

print(f"Configuration:")
print(f"  Grid sizes: {GRID_SIZES}")
print(f"  Domain: {domain}")
print(f"  Final time: {t_final}")
print(f"  CFL: {cfl}")
print(f"  Viscosity: {nu}")
print(f"  Initial condition: {ic_type}")
print(f"\nExpected sequential times (based on scaling):")
print(f"  nx=1200:  ~7.5 seconds")
print(f"  nx=2400:  ~30 seconds")
print(f"  nx=4800:  ~2 minutes")

# %% [markdown]
# ## Run Experiments for All Grid Sizes
# 
# We'll run the sequential solver for multiple grid sizes to establish baseline timings.

# %%
# Dictionary to store results for all grid sizes
sequential_results = {}

print(f"\nStarting experiments at {time.strftime('%H:%M:%S')}")
print(f"Total grid sizes to test: {len(GRID_SIZES)}\n")

for idx, nx in enumerate(GRID_SIZES, 1):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {idx}/{len(GRID_SIZES)}: GRID SIZE nx = {nx}")
    print(f"{'='*60}")
    
    # Create solver
    solver = BurgersRusanovSolver(
        nx=nx,
        domain=domain,
        t_final=t_final,
        cfl=cfl,
        nu=nu
    )
    
    # Set initial condition
    solver.set_initial_condition(ic_type)
    
    # Solve
    u_final, snapshots, snapshot_times = solver.solve(
        n_snapshots=n_snapshots,
        verbose=True
    )
    
    # Calculate gradient for shock detection
    gradients = np.abs(np.gradient(u_final, solver.x))
    max_gradient = np.max(gradients)
    
    # Store result
    sequential_results[nx] = {
        'solver': solver,
        'u_final': u_final,
        'snapshots': snapshots,
        'snapshot_times': snapshot_times,
        'max_gradient': max_gradient
    }
    
    print(f"\nShock Wave Analysis:")
    print(f"Maximum gradient: {max_gradient:.6f}")
    if max_gradient > 10.0:
        print("=> Strong discontinuities (shock waves) detected!")
    elif max_gradient > 2.0:
        print("=> Moderate discontinuities detected")
    else:
        print("=> Solution appears smooth")
    
    # Save solution for this grid size
    output_file = f'results/results_sequential_nx{nx}.npz'
    np.savez(output_file,
             x=solver.x,
             u_final=u_final,
             snapshots=np.array(snapshots),
             times=np.array(snapshot_times),
             nx=nx,
             nu=nu,
             t_final=t_final,
             elapsed_time=solver.elapsed_time)
    
    print(f"Saved: {output_file}")

print(f"\n{'='*60}")
print(f"All experiments completed at {time.strftime('%H:%M:%S')}")
print(f"{'='*60}")

# %% [markdown]
# ## Visualization: Example from One Grid Size
# 
# Let's visualize the solution from one grid size (smallest for faster plotting).

# %%
# Visualize results from the smallest grid size
nx_vis = GRID_SIZES[0]
result = sequential_results[nx_vis]
solver = result['solver']
snapshots = result['snapshots']
snapshot_times = result['snapshot_times']

# Plot solution evolution
fig, ax = plt.subplots(figsize=(12, 8))

colors = plt.cm.viridis(np.linspace(0, 1, len(snapshots)))

for i, (snapshot, t) in enumerate(zip(snapshots, snapshot_times)):
    ax.plot(solver.x, snapshot, color=colors[i], linewidth=2,
           label=f't = {t:.4f}', alpha=0.7)

ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('u(x, t)', fontsize=14)
ax.set_title(f'Burgers Equation Solution Evolution (nx={nx_vis})', 
            fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10, ncol=2)
plt.tight_layout()
output_file = os.path.join(output_dir, f'solution_evolution_nx{nx_vis}.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")
plt.show()

# %% [markdown]
# ## Save Timing Data to JSON
# 
# Save all timing results in the format expected by the performance analysis notebook.

# %%
# Save timing data to JSON (following AR lab 3 format)
import json

timing_data = {
    'method': 'rusanov_sequential',
    'tolerance': None,  # Burgers doesn't use tolerance, it runs to t_final
    't_final': float(t_final),
    'cfl': float(cfl),
    'nu': float(nu),
    'initial_condition': ic_type,
    'vectorized': True,  # Marks this as optimized version
    'results': {}
}

# Add all grid sizes
for nx, result in sequential_results.items():
    timing_data['results'][str(nx)] = {
        'n_steps': int(result['solver'].n_steps),
        'time': float(result['solver'].elapsed_time),
        'converged': True,  # Always "converges" by reaching t_final
        'final_max_gradient': float(result['max_gradient'])
    }

# Ensure results directory exists
os.makedirs('results', exist_ok=True)

# Save to JSON
json_file = 'results/sequential_timing.json'
with open(json_file, 'w') as f:
    json.dump(timing_data, f, indent=2)

print(f"Timing data saved to {json_file}")
print(f"\nGrid sizes in timing data: {list(timing_data['results'].keys())}")

# %% [markdown]
# ## Performance Summary
# 
# Display a summary table of sequential performance across all grid sizes.

# %%
print("\n" + "="*95)
print("SEQUENTIAL RUSANOV SOLVER - PERFORMANCE SUMMARY")
print("="*95)
print(f"{'Grid Size':>12} | {'Steps':>12} | {'Time (s)':>12} | {'Time/Step (ms)':>16} | {'Max Gradient':>14}")
print("-"*95)

for nx in sorted(sequential_results.keys()):
    result = sequential_results[nx]
    solver = result['solver']
    time_per_step = solver.elapsed_time / solver.n_steps * 1000
    max_grad = result['max_gradient']
    print(f"{nx:>12d} | {solver.n_steps:>12d} | {solver.elapsed_time:>12.4f} | {time_per_step:>16.4f} | {max_grad:>14.6f}")

print("="*95)
print("\n[OK] All experiments completed successfully!")

# %% [markdown]
# ## Summary
# 
# This notebook successfully implements the Rusanov method for solving the 1D Burgers equation with **multiple grid sizes** for performance analysis.
# 
# **Key Results:**
# 1. **Numerical Method**: First-order Rusanov scheme with explicit time integration
# 2. **Stability**: CFL condition ensures numerical stability
# 3. **Shock Capturing**: Method successfully captures shock wave formation
# 4. **Boundary Conditions**: Periodic boundaries implemented
# 5. **Scaling Study**: Tested multiple grid sizes for parallel comparison
# 
# **Next Steps:**
# - Run parallel experiments on PLGrid with matching grid sizes
# - Performance analysis: Calculate S(P), E(P), f(P) metrics
# - Compare different initial conditions


