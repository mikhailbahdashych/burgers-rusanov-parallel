#!/usr/bin/env python3
"""
Parallel MPI implementation of the Rusanov method for solving the 1D Burgers equation.

=== MATHEMATICAL BACKGROUND ===

The 1D viscous Burgers equation:
    du/dt + u * du/dx = nu * d^2u/dx^2

where:
    u(x,t) - velocity field (solution variable)
    nu - viscosity coefficient (controls diffusion strength)
    x - spatial coordinate
    t - time

Physical interpretation:
    - Advection term (u * du/dx): Nonlinear wave propagation, causes shock formation
    - Diffusion term (nu * d^2u/dx^2): Viscous smoothing, prevents discontinuities

=== RUSANOV METHOD (Local Lax-Friedrichs) ===

The Rusanov method is a first-order finite volume scheme for hyperbolic conservation laws.

1. Conservative form: du/dt + dF(u)/dx = nu * d^2u/dx^2
   where F(u) = u^2/2 is the flux function (integral of u * du/dx)

2. Rusanov numerical flux at cell interface i+1/2:
   F_i+1/2 = (F(u_i) + F(u_i+1))/2 - alpha_i+1/2/2 * (u_i+1 - u_i)

   where alpha_i+1/2 = max(|u_i|, |u_i+1|) is the local wave speed

   This flux averages left and right states and adds numerical dissipation
   for stability (the alpha term stabilizes shocks).

3. Finite volume update (explicit time stepping):
   u_i^(n+1) = u_i^n - (dt/dx) * (F_i+1/2 - F_i-1/2)
                     + nu * (dt/dx^2) * (u_i+1 - 2*u_i + u_i-1)

   First term: flux difference (advection)
   Second term: centered difference for diffusion

=== STABILITY CONDITIONS ===

For explicit schemes, the timestep dt must satisfy TWO conditions:

1. CFL condition (Courant-Friedrichs-Lewy) for convection:
   dt <= CFL * dx / max(|u|)

   Physical meaning: Information cannot propagate more than one cell per timestep
   Typical CFL = 0.3 for safety (stability requires CFL <= 1.0)

2. Diffusion stability condition (von Neumann analysis):
   dt <= 0.25 * dx^2 / nu

   Physical meaning: Diffusion operator has dt proportional to dx^2
   This is why explicit diffusion has poor scaling (timestep shrinks quadratically)

The actual timestep is: dt = min(CFL condition, diffusion condition)

=== PARALLELIZATION STRATEGY ===

Domain decomposition approach for distributed memory (MPI):

1. Spatial grid division:
   - Global grid of nx points split among P processors
   - Each processor owns nx_local = nx / P cells (approximately)
   - Remainder cells distributed among first processes

2. Ghost/halo cells:
   - Each process maintains 2 extra cells (1 left, 1 right)
   - Ghost cells store boundary values from neighboring processes
   - Required for computing fluxes at subdomain boundaries

3. Communication pattern:
   - Halo exchange: Non-blocking point-to-point (MPI_Isend/Irecv)
   - Global reduction: Collective operation for max(|u|) (MPI_Allreduce)
   - Gather: Collective operation for assembling solution (MPI_Gatherv)

4. Periodic boundary conditions:
   - Rank 0 exchanges with rank P-1 for periodicity
   - Ensures circular domain topology

5. Smart timestep caching:
   - Recompute dt only when needed (every 10 steps or 5% change in max(|u|))
   - Reduces expensive MPI_Allreduce calls by 10x
   - Balances correctness (CFL safety) with performance

Reference: Section 2.1.1 of "THE SOLUTION OF A BURGERS EQUATION.pdf"
"""

import numpy as np
import time
import argparse
from typing import Tuple, Optional
from mpi4py import MPI


class BurgersRusanovParallel:
    """Parallel MPI Rusanov solver for the 1D Burgers equation."""

    def __init__(self, nx_global: int, domain: Tuple[float, float],
                 t_final: float, cfl: float = 0.3, nu: float = 0.1,
                 comm: Optional[MPI.Comm] = None):
        """
        Initialize the parallel Burgers equation solver.

        Args:
            nx_global: Total number of spatial grid points (global)
            domain: Spatial domain (x_min, x_max)
            t_final: Final simulation time
            cfl: CFL number for stability (typically 0.3)
            nu: Viscosity coefficient (controls diffusion strength)
            comm: MPI communicator (defaults to MPI.COMM_WORLD)
        """
        # MPI setup: Get communicator, rank, and total number of processes
        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()  # This process's ID (0 to size-1)
        self.size = self.comm.Get_size()  # Total number of MPI processes

        # Problem parameters
        self.nx_global = nx_global
        self.x_min, self.x_max = domain
        self.t_final = t_final
        self.cfl = cfl
        self.nu = nu

        # Set up domain decomposition (split grid among processors)
        self._setup_domain_decomposition()

        # Time integration state
        self.t = 0.0        # Current simulation time
        self.dt = 0.0       # Current timestep
        self.n_steps = 0    # Total number of timesteps taken

        # Smart timestep caching for performance optimization
        # Avoid recomputing dt every step (reduces MPI_Allreduce calls)
        self.cached_dt = None           # Cached timestep value
        self.cached_max_speed = None    # Previous max speed for change detection
        self.dt_recompute_interval = 10  # Recompute every 10 steps (periodic safety)

        # Storage for solution snapshots (only rank 0 stores full solution)
        if self.rank == 0:
            self.snapshots = []
            self.snapshot_times = []
            # Global spatial grid (endpoint=False for periodic domain)
            self.x_global = np.linspace(self.x_min, self.x_max, nx_global, endpoint=False)

    def _setup_domain_decomposition(self):
        """
        Set up 1D domain decomposition across MPI processes.

        Divides the global spatial grid into local subdomains, one per process.
        Each process owns nx_local consecutive cells plus 2 ghost cells.

        Grid layout example (P=3 processes, nx_global=10):
            Rank 0: [ghost][0][1][2][3][ghost]
            Rank 1: [ghost][4][5][6][ghost]
            Rank 2: [ghost][7][8][9][ghost]
        """
        # Calculate base number of cells per process (integer division)
        self.nx_local = self.nx_global // self.size
        remainder = self.nx_global % self.size

        # Distribute remainder cells: first 'remainder' processes get one extra cell
        # Example: nx=100, P=3 -> ranks get 34, 33, 33 cells
        if self.rank < remainder:
            self.nx_local += 1
            self.i_start = self.rank * self.nx_local
        else:
            self.i_start = self.rank * self.nx_local + remainder

        # Global indices: [i_start, i_end) owned by this process
        self.i_end = self.i_start + self.nx_local

        # Create local array with ghost/halo cells for boundary exchange
        # Array structure: [left_ghost, cell_0, cell_1, ..., cell_nx_local-1, right_ghost]
        self.nx_with_ghosts = self.nx_local + 2

        # Grid spacing (uniform for all processes)
        # For periodic domain: dx = L / nx (NOT L / (nx-1))
        dx = (self.x_max - self.x_min) / self.nx_global
        self.dx = dx

        # Local spatial coordinates (interior cells only, no ghosts)
        x_start = self.x_min + self.i_start * dx
        self.x_local = x_start + np.arange(self.nx_local) * dx

        # Allocate solution array with ghost cells
        # Indices: 0 (left ghost), 1 to nx_local (interior), nx_local+1 (right ghost)
        self.u = np.zeros(self.nx_with_ghosts)

        # Identify neighboring processes for halo exchange
        # MPI.PROC_NULL means no neighbor (boundary process)
        self.left_neighbor = self.rank - 1 if self.rank > 0 else MPI.PROC_NULL
        self.right_neighbor = self.rank + 1 if self.rank < self.size - 1 else MPI.PROC_NULL

        # Print decomposition info (only from rank 0)
        if self.rank == 0:
            print(f"Domain decomposition:")
            print(f"  Total processes: {self.size}")
            print(f"  Global grid points: {self.nx_global}")
            print(f"  Local grid points per process: ~{self.nx_global // self.size}")

    def set_initial_condition(self, ic_type: str = 'sine'):
        """
        Set initial condition for the Burgers equation and distribute to all processes.

        This function implements a scatter operation to distribute the initial condition:
            1. Rank 0 creates the global initial condition array
            2. MPI_Scatterv distributes chunks to all processes
            3. Each process fills its interior cells (u[1:-1])
            4. Halo exchange fills ghost cells from neighbors
            5. Rank 0 stores the initial state for output

        Available initial conditions:

        1. 'sine': Smooth periodic wave
            u(x,0) = 0.5 + 0.5 * sin(2*pi*x)
            - Smooth everywhere (no shocks initially)
            - Develops shock due to nonlinear steepening
            - Good for testing convergence and accuracy

        2. 'step': Riemann problem (shock formation)
            u(x,0) = 1.0 for x < 0.5, 0.0 for x >= 0.5
            - Discontinuous initial condition
            - Tests shock-capturing ability of Rusanov method
            - High resolution needed to resolve sharp gradient

        3. 'rarefaction': Expansion wave
            u(x,0) = 0.0 for x < 0.5, 1.0 for x >= 0.5
            - Opposite of step (expanding instead of compressing)
            - Smooth rarefaction wave (no shock formation)
            - Easier to resolve than shock case

        Periodic boundary conditions:
            - Uses endpoint=False in linspace (avoids duplicating x=0 and x=L)
            - Ensures grid spacing dx = L / nx (not L / (nx-1))

        Args:
            ic_type: Type of initial condition ('sine', 'step', or 'rarefaction')

        Raises:
            ValueError: If ic_type is not one of the valid choices
        """
        # STEP 1: Create global initial condition on root process
        if self.rank == 0:
            # For periodic domain: endpoint=False avoids duplicating boundary point
            # This ensures dx = (x_max - x_min) / nx_global
            x_global = np.linspace(self.x_min, self.x_max, self.nx_global, endpoint=False)

            # Generate initial condition based on type
            if ic_type == 'sine':
                # Smooth sine wave: u in [0, 1]
                u_global = 0.5 + 0.5 * np.sin(2 * np.pi * x_global)
            elif ic_type == 'step':
                # Shock formation: left half u=1, right half u=0
                u_global = np.where(x_global < 0.5 * (self.x_min + self.x_max), 1.0, 0.0)
            elif ic_type == 'rarefaction':
                # Rarefaction wave: left half u=0, right half u=1
                u_global = np.where(x_global < 0.5 * (self.x_min + self.x_max), 0.0, 1.0)
            else:
                raise ValueError(f"Unknown initial condition type: {ic_type}")
        else:
            # Non-root processes don't create global array (save memory)
            u_global = None

        # STEP 2: Scatter initial condition to all processes using MPI_Scatterv
        # Prepare send counts and displacements for variable-size scatter
        sendcounts = np.zeros(self.size, dtype=int)
        displs = np.zeros(self.size, dtype=int)

        # Replicate domain decomposition logic to compute send counts
        for i in range(self.size):
            nx_local_i = self.nx_global // self.size
            remainder = self.nx_global % self.size
            if i < remainder:
                nx_local_i += 1  # First 'remainder' processes get one extra cell
            sendcounts[i] = nx_local_i

            # Displacement: cumulative sum of previous send counts
            if i > 0:
                displs[i] = displs[i-1] + sendcounts[i-1]

        # Allocate receive buffer for local portion (no ghost cells yet)
        u_local = np.zeros(self.nx_local)

        # MPI_Scatterv: Rank 0 sends chunks to all processes
        # Each process receives its portion into u_local
        self.comm.Scatterv([u_global, sendcounts, displs, MPI.DOUBLE],
                          u_local, root=0)

        # STEP 3: Place received data in interior cells (skip ghost cells at indices 0 and -1)
        self.u[1:-1] = u_local

        # STEP 4: Exchange ghost cells to fill boundary values
        # This ensures ghost cells have correct values from neighbors
        self._exchange_halos()

        # STEP 5: Store initial state on root for output
        if self.rank == 0:
            self.snapshots = [u_global.copy()]
            self.snapshot_times = [0.0]

    def _exchange_halos(self):
        """
        Exchange ghost/halo cells with neighboring processes.

        This is the CRITICAL communication step for domain decomposition.
        Must be called BEFORE computing fluxes to ensure current boundary values.

        Communication pattern (non-blocking for performance):
            Rank i sends u[nx_local] to rank i+1, receives into u[0] from rank i-1
            Rank i sends u[1] to rank i-1, receives into u[nx_local+1] from rank i+1

        Example (3 processes):
            Rank 0: send u[1] to 0's left ghost <- comes from rank 2 (periodic)
                    send u[nx_local] -> rank 1's left ghost
            Rank 1: send u[1] -> rank 0's right ghost
                    send u[nx_local] -> rank 2's left ghost
            Rank 2: send u[1] -> rank 1's right ghost
                    send u[nx_local] to 2's right ghost <- comes from rank 0 (periodic)
        """
        # Use non-blocking communication for better performance
        # Allows computation-communication overlap and prevents deadlocks
        requests = []

        # ========== Interior neighbor exchanges (ranks 1 to P-2) ==========

        # Send my rightmost interior cell to right neighbor's left ghost
        # Receive from left neighbor's rightmost interior into my left ghost
        if self.right_neighbor != MPI.PROC_NULL:
            req = self.comm.Isend(self.u[-2:-1], dest=self.right_neighbor, tag=0)
            requests.append(req)

        if self.left_neighbor != MPI.PROC_NULL:
            req = self.comm.Irecv(self.u[0:1], source=self.left_neighbor, tag=0)
            requests.append(req)

        # Send my leftmost interior cell to left neighbor's right ghost
        # Receive from right neighbor's leftmost interior into my right ghost
        if self.left_neighbor != MPI.PROC_NULL:
            req = self.comm.Isend(self.u[1:2], dest=self.left_neighbor, tag=1)
            requests.append(req)

        if self.right_neighbor != MPI.PROC_NULL:
            req = self.comm.Irecv(self.u[-1:], source=self.right_neighbor, tag=1)
            requests.append(req)

        # Wait for all non-blocking operations to complete
        MPI.Request.Waitall(requests)

        # ========== Periodic boundary conditions (rank 0 <-> rank P-1) ==========

        # Rank 0's left ghost comes from rank P-1's right boundary (periodicity)
        if self.rank == 0 and self.left_neighbor == MPI.PROC_NULL:
            if self.size > 1:
                # Multiple processes: exchange with last process
                self.comm.Sendrecv(self.u[1:2], dest=self.size-1, sendtag=2,
                                  recvbuf=self.u[0:1], source=self.size-1, recvtag=3)
            else:
                # Single process: periodic boundary on itself
                self.u[0] = self.u[-2]

        # Rank P-1's right ghost comes from rank 0's left boundary (periodicity)
        if self.rank == self.size - 1 and self.right_neighbor == MPI.PROC_NULL:
            if self.size > 1:
                # Multiple processes: exchange with first process
                self.comm.Sendrecv(self.u[-2:-1], dest=0, sendtag=3,
                                  recvbuf=self.u[-1:], source=0, recvtag=2)
            else:
                # Single process: periodic boundary on itself
                self.u[-1] = self.u[1]

    def flux(self, u: np.ndarray) -> np.ndarray:
        """
        Compute the physical flux function F(u) = u^2/2 for the Burgers equation.

        The Burgers equation in conservative form is:
            du/dt + dF(u)/dx = nu * d^2u/dx^2

        where F(u) is the flux function. For the inviscid Burgers equation,
        the advection term u * du/dx can be written as d(u^2/2)/dx.

        This flux function represents the nonlinear wave propagation:
            - When u > 0: wave travels to the right
            - When u < 0: wave travels to the left
            - Speed of propagation is |u| (characteristic speed)

        Args:
            u: Solution values (velocity field)

        Returns:
            Flux values F(u) = u^2/2
        """
        return 0.5 * u**2

    def rusanov_flux(self, u_left: np.ndarray, u_right: np.ndarray) -> np.ndarray:
        """
        Compute Rusanov (Local Lax-Friedrichs) numerical flux at cell interfaces.

        The Rusanov flux is a simple and robust upwind scheme for hyperbolic conservation laws.
        At each cell interface i+1/2, it computes:

            F_i+1/2 = (F(u_i) + F(u_i+1))/2 - alpha_i+1/2/2 * (u_i+1 - u_i)

        where:
            - (F(u_i) + F(u_i+1))/2 is the centered average (unstable alone)
            - alpha_i+1/2/2 * (u_i+1 - u_i) is the numerical dissipation term
            - alpha_i+1/2 = max(|u_i|, |u_i+1|) is the local maximum wave speed

        Why add dissipation?
            - The centered flux alone is unstable for hyperbolic problems
            - Numerical dissipation stabilizes shocks and discontinuities
            - The alpha term acts like artificial viscosity
            - Larger alpha means more dissipation (more stable but less accurate)

        Physical interpretation:
            - When u_left = u_right: no dissipation needed, flux = F(u)
            - When u_left != u_right: dissipation proportional to jump size
            - At shocks (large jumps): strong dissipation prevents oscillations

        Trade-offs:
            - First-order accuracy: smears discontinuities over 2-3 cells
            - Very robust: handles strong shocks without oscillations
            - Simple: no characteristic decomposition needed

        Args:
            u_left: Solution values on left side of interfaces
            u_right: Solution values on right side of interfaces

        Returns:
            Numerical flux values at cell interfaces
        """
        # Compute physical flux on both sides
        f_left = self.flux(u_left)
        f_right = self.flux(u_right)

        # Local maximum wave speed (characteristic speed for Burgers: |u|)
        alpha = np.maximum(np.abs(u_left), np.abs(u_right))

        # Rusanov flux: centered flux - numerical dissipation
        return 0.5 * (f_left + f_right) - 0.5 * alpha * (u_right - u_left)

    def compute_dt(self) -> float:
        """
        Compute time step with smart caching for performance.

        Recomputes dt when:
        1. First call (no cache)
        2. Every dt_recompute_interval steps (periodic safety)
        3. max(|u|) changes significantly (>5% - adaptive safety)

        For Burgers equation: dt <= CFL * dx / max(|u|)
        For viscous term: dt <= 0.25 * dx^2 / nu

        Returns:
            Time step satisfying both stability conditions
        """
        # Local maximum speed (always compute - cheap)
        max_speed_local = np.max(np.abs(self.u[1:-1]))

        # Decide if we need global reduction (expensive)
        needs_recompute = (
            self.cached_dt is None or  # First call
            self.n_steps % self.dt_recompute_interval == 0 or  # Periodic safety
            (self.cached_max_speed is not None and
             abs(max_speed_local - self.cached_max_speed) / max(self.cached_max_speed, 1e-10) > 0.05)  # 5% change
        )

        if needs_recompute:
            # Global maximum speed across all processes
            max_speed = self.comm.allreduce(max_speed_local, op=MPI.MAX)

            # Ensure non-zero denominator
            max_speed = max(max_speed, 1e-10)

            # CFL condition for convection
            dt_convection = self.cfl * self.dx / max_speed

            # Stability condition for diffusion
            if self.nu > 0:
                dt_diffusion = 0.25 * self.dx**2 / self.nu
                self.cached_dt = min(dt_convection, dt_diffusion)
            else:
                self.cached_dt = dt_convection

            # Store max_speed for next iteration's change detection
            self.cached_max_speed = max_speed_local

        return self.cached_dt

    def step(self):
        """
        Perform one time step using the Rusanov method with operator splitting.

        This implements the complete time integration algorithm for the Burgers equation:
            du/dt + dF(u)/dx = nu * d^2u/dx^2

        The algorithm consists of 5 critical steps (ORDER MATTERS):

        STEP 1: Exchange ghost/halo cells with neighbors
            - MUST be done FIRST before any computation
            - Ensures boundary values are current for flux calculation
            - Without this: fluxes at subdomain boundaries would be stale/wrong

        STEP 2: Compute timestep dt
            - Uses smart caching to reduce MPI_Allreduce overhead
            - Satisfies both CFL and diffusion stability conditions
            - Synchronized across all processes (all use same dt)

        STEP 3: Compute Rusanov fluxes at all cell interfaces
            - Uses current ghost cells from step 1
            - Applies Rusanov dissipation for shock stability
            - Computes F_i+1/2 for all interfaces including boundaries

        STEP 4: Update solution using finite volume formula
            - Advection update: u_new = u_old - (dt/dx) * (F_i+1/2 - F_i-1/2)
            - This is the discrete divergence of the flux
            - Conservative: total mass is preserved

        STEP 5: Add viscous diffusion (if nu > 0)
            - Centered difference: nu * (dt/dx^2) * (u_i+1 - 2*u_i + u_i-1)
            - Explicit time integration (forward Euler)
            - Requires dt <= 0.25 * dx^2 / nu for stability

        Array indexing:
            - u[0]: left ghost cell
            - u[1:-1]: interior cells (nx_local points)
            - u[-1]: right ghost cell
            - f_interfaces has nx_local+1 points (one flux per interface)

        Example (nx_local=3):
            u = [ghost_L, u0, u1, u2, ghost_R]
            f_interfaces = [f_-1/2, f_0+1/2, f_1+1/2, f_2+1/2]
            u_new[1] (u0) uses f_0+1/2 and f_-1/2
            u_new[2] (u1) uses f_1+1/2 and f_0+1/2
            u_new[3] (u2) uses f_2+1/2 and f_1+1/2
        """
        # STEP 1: Exchange ghost cells FIRST to get current boundary values
        # CRITICAL: Must be before flux computation, not after!
        self._exchange_halos()

        # STEP 2: Compute time step (synchronized across all processes)
        # Smart caching reduces MPI_Allreduce overhead by 10x
        self.dt = self.compute_dt()

        # Adjust final timestep to hit t_final exactly
        if self.t + self.dt > self.t_final:
            self.dt = self.t_final - self.t

        # STEP 3: Compute Rusanov fluxes at all cell interfaces
        # u_left[i] and u_right[i] are values on either side of interface i+1/2
        u_left = self.u[:-1]   # [ghost_L, u0, u1, ..., u_nx-1]
        u_right = self.u[1:]   # [u0, u1, ..., u_nx-1, ghost_R]
        f_interfaces = self.rusanov_flux(u_left, u_right)

        # STEP 4: Update interior points using finite volume formula
        # Advection term: -dt/dx * (F_i+1/2 - F_i-1/2)
        u_new = self.u.copy()
        u_new[1:-1] = self.u[1:-1] - (self.dt / self.dx) * (
            f_interfaces[1:] - f_interfaces[:-1]
        )

        # STEP 5: Add viscous diffusion term (operator splitting)
        # Centered difference for d^2u/dx^2
        if self.nu > 0:
            u_new[1:-1] += self.nu * (self.dt / self.dx**2) * (
                self.u[2:] - 2*self.u[1:-1] + self.u[:-2]
            )

        # Replace old solution with new solution
        self.u = u_new

        # Update simulation time and step counter
        self.t += self.dt
        self.n_steps += 1

    def gather_solution(self) -> Optional[np.ndarray]:
        """
        Gather the distributed solution from all processes to root process.

        This is a collective MPI operation that assembles the full global solution
        from local subdomains. Uses MPI_Gatherv (variable-size gather) because
        different processes may own different numbers of cells.

        Communication pattern:
            - Each process sends its interior cells (no ghost cells)
            - Root process (rank 0) receives and assembles into global array
            - Non-root processes receive nothing

        Why Gatherv instead of Gather?
            - Load balancing: If nx_global is not divisible by P, some processes
              have more cells than others (e.g., nx=100, P=3 -> 34, 33, 33)
            - MPI_Gatherv handles variable send counts via sendcounts array
            - MPI_Gather would require all processes to send same amount

        Memory efficiency:
            - Only rank 0 allocates the full global array
            - Other processes use None to save memory
            - Important for large-scale simulations

        Example (P=3, nx_global=10):
            Rank 0: sends u[1:5] (4 cells)  -> u_global[0:4]
            Rank 1: sends u[1:4] (3 cells)  -> u_global[4:7]
            Rank 2: sends u[1:4] (3 cells)  -> u_global[7:10]

        Returns:
            Global solution array (nx_global points) on root process
            None on all other processes
        """
        # Prepare send buffer: interior cells only (exclude ghost cells)
        # Each process sends u[1:-1] which has nx_local elements
        u_local = self.u[1:-1].copy()

        # Prepare receive buffer (only on root process)
        if self.rank == 0:
            u_global = np.zeros(self.nx_global)
        else:
            u_global = None  # Save memory on non-root processes

        # Compute send counts and displacements for MPI_Gatherv
        # sendcounts[i] = number of cells process i will send
        # displs[i] = starting index in u_global where process i's data goes
        sendcounts = np.zeros(self.size, dtype=int)
        displs = np.zeros(self.size, dtype=int)

        for i in range(self.size):
            # Replicate domain decomposition logic to compute nx_local for each process
            nx_local_i = self.nx_global // self.size
            remainder = self.nx_global % self.size
            if i < remainder:
                nx_local_i += 1
            sendcounts[i] = nx_local_i

            # Displacement: cumulative sum of previous send counts
            if i > 0:
                displs[i] = displs[i-1] + sendcounts[i-1]

        # MPI_Gatherv: variable-size gather operation
        # All processes participate (collective operation)
        self.comm.Gatherv(u_local, [u_global, sendcounts, displs, MPI.DOUBLE], root=0)

        return u_global

    def solve(self, n_snapshots: int = 10) -> Optional[Tuple[np.ndarray, list, list]]:
        """
        Solve the Burgers equation until t_final using parallel time integration.

        This is the main driver function that orchestrates the entire simulation:
            1. Prints simulation parameters (rank 0 only)
            2. Synchronizes all processes at start (MPI_Barrier)
            3. Runs time integration loop (all processes in lockstep)
            4. Periodically gathers and saves solution snapshots
            5. Returns final solution and timing data

        Time integration loop:
            - Adaptive timestep: dt changes based on max(|u|) to satisfy CFL
            - All processes execute same number of steps (synchronized)
            - Loop continues until t >= t_final

        Snapshot mechanism:
            - Saves solution at regular time intervals (not step intervals!)
            - Uses gather operation: expensive, so don't do every step
            - Only rank 0 stores snapshots (saves memory)
            - Final snapshot is always at t = t_final

        Performance measurement:
            - Uses MPI_Wtime() for high-precision wall-clock time
            - Synchronized via MPI_Barrier before/after timing
            - Reports total time, number of steps, time per step

        SPMD pattern (Single Program Multiple Data):
            - All processes run the same code
            - Data distribution handled internally
            - Rank 0 has special role: printing and storage

        Args:
            n_snapshots: Number of solution snapshots to save (default: 10)

        Returns:
            On rank 0: Tuple of (final_solution, snapshots, times, elapsed_time)
                - final_solution: Global solution at t_final (nx_global points)
                - snapshots: List of global solution arrays at snapshot times
                - times: List of snapshot times
                - elapsed_time: Total wall-clock time in seconds
            On other ranks: None
        """
        # Compute snapshot interval in time (not steps, because dt varies)
        snapshot_interval = self.t_final / n_snapshots
        next_snapshot_time = snapshot_interval

        # Print simulation info (only rank 0 prints to avoid clutter)
        if self.rank == 0:
            print(f"\nStarting parallel Rusanov solver...")
            print(f"MPI processes: {self.size}")
            print(f"Global grid points: {self.nx_global}")
            print(f"Local grid points: {self.nx_local}")
            print(f"Domain: [{self.x_min}, {self.x_max}]")
            print(f"Final time: {self.t_final}")
            print(f"CFL number: {self.cfl}")
            print(f"Viscosity: {self.nu}\n")

        # Barrier to synchronize all processes before timing
        # Ensures all processes start time integration simultaneously
        self.comm.Barrier()
        start_time = MPI.Wtime()  # High-precision wall-clock time

        # ==== MAIN TIME INTEGRATION LOOP ====
        # All processes execute in lockstep (same number of steps)
        while self.t < self.t_final:
            # Advance solution by one timestep (step method does everything)
            self.step()

            # Check if it's time to save a snapshot
            # Use >= to avoid missing snapshots due to floating point errors
            if self.t >= next_snapshot_time or abs(self.t - self.t_final) < 1e-10:
                # Gather distributed solution to rank 0
                u_global = self.gather_solution()

                # Store snapshot and print progress (rank 0 only)
                if self.rank == 0:
                    self.snapshots.append(u_global.copy())
                    self.snapshot_times.append(self.t)
                    max_u = np.max(np.abs(u_global))
                    print(f"Step {self.n_steps}: t = {self.t:.6f}, dt = {self.dt:.6e}, "
                          f"max(|u|) = {max_u:.6f}")

                # Advance to next snapshot time
                next_snapshot_time += snapshot_interval

        # Final gather to get solution at t_final
        u_final = self.gather_solution()

        # Barrier to synchronize all processes after time integration
        self.comm.Barrier()
        elapsed_time = MPI.Wtime() - start_time

        # Print summary and return results (rank 0 only)
        if self.rank == 0:
            print(f"\nSimulation complete!")
            print(f"Total time steps: {self.n_steps}")
            print(f"Elapsed time: {elapsed_time:.6f} seconds")
            print(f"Average time per step: {elapsed_time/self.n_steps:.6e} seconds")

            return u_final, self.snapshots, self.snapshot_times, elapsed_time
        else:
            # Non-root processes return None (don't need the data)
            return None


def main():
    """
    Main function for parallel Rusanov solver - command-line interface.

    This function provides a complete command-line interface for running
    parallel simulations of the Burgers equation using MPI.

    Workflow:
        1. Parse command-line arguments (all processes parse, results identical)
        2. Create parallel solver instance with domain decomposition
        3. Set initial condition (scatter from rank 0 to all processes)
        4. Run time integration (all processes in parallel)
        5. Save results to .npz file (rank 0 only)
        6. Save timing data to .json file (rank 0 only)
        7. Analyze solution for shock detection (rank 0 only)

    Command-line arguments:
        --nx: Global grid points (total across all processes)
        --domain: Spatial domain [x_min, x_max] (default: [0, 1])
        --t-final: Final simulation time (default: 0.5)
        --cfl: CFL number for stability (default: 0.3)
        --nu: Viscosity coefficient (default: 0.1)
        --ic: Initial condition type (sine, step, or rarefaction)
        --snapshots: Number of snapshots to save (default: 10)
        --save: Output filename for results (default: results_parallel.npz)

    Example usage:
        # Single process (serial)
        python 2_parallel_rusanov.py --nx 300 --ic sine

        # Multiple processes (parallel)
        mpirun -np 4 python 2_parallel_rusanov.py --nx 1200 --ic step

        # High resolution with many processors
        mpirun -np 16 python 2_parallel_rusanov.py --nx 4800 --nu 0.01

    Output files:
        1. .npz file (NumPy compressed archive):
            - x: Spatial grid points
            - u_final: Final solution at t_final
            - snapshots: Solution at intermediate times
            - times: Snapshot times
            - nx, nu, t_final: Problem parameters
            - n_procs: Number of MPI processes
            - elapsed_time: Total wall-clock time
            - n_steps: Number of timesteps taken

        2. .json file (timing data for performance analysis):
            - method: 'rusanov_mpi'
            - nx, processes, n_steps: Problem size info
            - time: Elapsed wall-clock time
            - converged: Always True (Burgers runs until t_final)
            - t_final, cfl, nu, initial_condition: Parameters

    Shock detection:
        - Computes max(|du/dx|) to detect steep gradients
        - Warns if strong shocks are present (max gradient > 10)
        - Helps diagnose if resolution is sufficient
    """
    # Get MPI communicator and rank
    # All processes execute this code (SPMD pattern)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Parse command-line arguments
    # All processes parse (ensures identical configuration)
    parser = argparse.ArgumentParser(description='Parallel MPI Rusanov solver')
    parser.add_argument('--nx', type=int, default=300, help='Global grid points')
    parser.add_argument('--domain', type=float, nargs=2, default=[0.0, 1.0])
    parser.add_argument('--t-final', type=float, default=0.5)
    parser.add_argument('--cfl', type=float, default=0.3)
    parser.add_argument('--nu', type=float, default=0.1)
    parser.add_argument('--ic', type=str, default='sine',
                       choices=['sine', 'step', 'rarefaction'])
    parser.add_argument('--snapshots', type=int, default=10)
    parser.add_argument('--save', type=str, default='results_parallel.npz')

    args = parser.parse_args()

    # Create parallel solver instance
    # This sets up domain decomposition and allocates local arrays
    solver = BurgersRusanovParallel(
        nx_global=args.nx,
        domain=tuple(args.domain),
        t_final=args.t_final,
        cfl=args.cfl,
        nu=args.nu,
        comm=comm
    )

    # Set initial condition
    # Rank 0 creates global IC, then scatters to all processes
    solver.set_initial_condition(args.ic)

    # Run time integration (all processes participate)
    # Returns results on rank 0, None on other processes
    result = solver.solve(n_snapshots=args.snapshots)

    # Save results (only rank 0 executes this block)
    if rank == 0:
        # Unpack results from solve()
        u_final, snapshots, snapshot_times, elapsed_time = result

        # Save full simulation data to NumPy compressed archive
        np.savez(args.save,
                 x=solver.x_global,
                 u_final=u_final,
                 snapshots=np.array(snapshots),
                 times=np.array(snapshot_times),
                 nx=args.nx,
                 nu=args.nu,
                 t_final=args.t_final,
                 n_procs=comm.Get_size(),
                 elapsed_time=elapsed_time,
                 n_steps=solver.n_steps)

        print(f"\nResults saved to {args.save}")

        # Save timing data to JSON (following AR lab 3 format)
        # This allows automated performance analysis scripts
        import json
        from pathlib import Path

        # Generate JSON filename from .npz filename
        # Example: results_parallel.npz -> results_parallel.json
        npz_path = Path(args.save)
        json_filename = npz_path.stem + '.json'  # Replace .npz with .json
        json_path = npz_path.parent / json_filename

        # Create timing data dictionary
        timing_data = {
            'method': 'rusanov_mpi',
            'nx': int(args.nx),
            'processes': int(comm.Get_size()),
            'n_steps': int(solver.n_steps),
            'time': float(elapsed_time),
            'converged': True,  # Burgers always "converges" by reaching t_final
            't_final': float(args.t_final),
            'cfl': float(args.cfl),
            'nu': float(args.nu),
            'initial_condition': args.ic
        }

        # Write JSON file with indentation for readability
        with open(json_path, 'w') as f:
            json.dump(timing_data, f, indent=2)

        print(f"Timing data saved to {json_path}")

        # Shock detection: analyze final solution for steep gradients
        # High gradients indicate shocks or discontinuities
        gradients = np.abs(np.gradient(u_final, solver.x_global))
        max_gradient = np.max(gradients)
        print(f"\nMaximum gradient: {max_gradient:.6f}")

        # Warn if strong shocks are detected (may need higher resolution)
        if max_gradient > 10.0:
            print("Strong shock waves detected!")
        elif max_gradient > 2.0:
            print("Moderate discontinuities detected")


if __name__ == "__main__":
    main()
