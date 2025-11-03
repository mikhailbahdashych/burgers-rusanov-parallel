#!/usr/bin/env python3
"""
Parallel MPI implementation of the Rusanov method for solving the 1D Burgers equation.

Uses domain decomposition with halo/ghost cell exchange between neighboring processes.
Each process handles a local subdomain and communicates boundary values with neighbors.

Parallelization strategy:
- 1D domain decomposition along spatial dimension
- Halo exchange using MPI point-to-point communication
- Collective operations for global reductions and I/O

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
            cfl: CFL number for stability
            nu: Viscosity coefficient
            comm: MPI communicator (defaults to MPI.COMM_WORLD)
        """
        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.nx_global = nx_global
        self.x_min, self.x_max = domain
        self.t_final = t_final
        self.cfl = cfl
        self.nu = nu

        # Domain decomposition
        self._setup_domain_decomposition()

        # Time tracking
        self.t = 0.0
        self.dt = 0.0
        self.n_steps = 0

        # Storage for snapshots (only on root)
        if self.rank == 0:
            self.snapshots = []
            self.snapshot_times = []
            # For periodic domain: endpoint=False
            self.x_global = np.linspace(self.x_min, self.x_max, nx_global, endpoint=False)

    def _setup_domain_decomposition(self):
        """Set up domain decomposition across MPI processes."""
        # Divide spatial domain among processes
        self.nx_local = self.nx_global // self.size
        remainder = self.nx_global % self.size

        # Handle remainder by giving extra points to first processes
        if self.rank < remainder:
            self.nx_local += 1
            self.i_start = self.rank * self.nx_local
        else:
            self.i_start = self.rank * self.nx_local + remainder

        self.i_end = self.i_start + self.nx_local

        # Create local grid (including ghost/halo cells)
        # Ghost cells: one on each side for boundary exchange
        self.nx_with_ghosts = self.nx_local + 2
        # For periodic domain: dx = L / nx (NOT nx-1)
        dx = (self.x_max - self.x_min) / self.nx_global
        self.dx = dx

        # Local coordinates (without ghosts)
        x_start = self.x_min + self.i_start * dx
        self.x_local = x_start + np.arange(self.nx_local) * dx

        # Solution array with ghost cells: [ghost_left, interior, ghost_right]
        self.u = np.zeros(self.nx_with_ghosts)

        # Determine neighbors for communication
        self.left_neighbor = self.rank - 1 if self.rank > 0 else MPI.PROC_NULL
        self.right_neighbor = self.rank + 1 if self.rank < self.size - 1 else MPI.PROC_NULL

        if self.rank == 0:
            print(f"Domain decomposition:")
            print(f"  Total processes: {self.size}")
            print(f"  Global grid points: {self.nx_global}")
            print(f"  Local grid points per process: ~{self.nx_global // self.size}")

    def set_initial_condition(self, ic_type: str = 'sine'):
        """
        Set initial condition for the solution.

        Args:
            ic_type: Type of initial condition
        """
        # Create global initial condition on root
        if self.rank == 0:
            # For periodic domain: endpoint=False
            x_global = np.linspace(self.x_min, self.x_max, self.nx_global, endpoint=False)

            if ic_type == 'sine':
                u_global = 0.5 + 0.5 * np.sin(2 * np.pi * x_global)
            elif ic_type == 'step':
                u_global = np.where(x_global < 0.5 * (self.x_min + self.x_max), 1.0, 0.0)
            elif ic_type == 'rarefaction':
                u_global = np.where(x_global < 0.5 * (self.x_min + self.x_max), 0.0, 1.0)
            else:
                raise ValueError(f"Unknown initial condition type: {ic_type}")
        else:
            u_global = None

        # Scatter initial condition to all processes
        # Prepare send counts and displacements
        sendcounts = np.zeros(self.size, dtype=int)
        displs = np.zeros(self.size, dtype=int)

        for i in range(self.size):
            nx_local_i = self.nx_global // self.size
            remainder = self.nx_global % self.size
            if i < remainder:
                nx_local_i += 1
            sendcounts[i] = nx_local_i
            if i > 0:
                displs[i] = displs[i-1] + sendcounts[i-1]

        # Receive local portion (without ghosts)
        u_local = np.zeros(self.nx_local)
        self.comm.Scatterv([u_global, sendcounts, displs, MPI.DOUBLE],
                          u_local, root=0)

        # Place in interior cells (index 1:-1)
        self.u[1:-1] = u_local

        # Exchange ghost cells
        self._exchange_halos()

        # Store initial state on root
        if self.rank == 0:
            self.snapshots = [u_global.copy()]
            self.snapshot_times = [0.0]

    def _exchange_halos(self):
        """Exchange ghost/halo cells with neighboring processes."""
        # Send right boundary to right neighbor, receive into left ghost
        # Send left boundary to left neighbor, receive into right ghost

        # Non-blocking communication for better overlap
        requests = []

        # Send to right, receive from left
        if self.right_neighbor != MPI.PROC_NULL:
            req = self.comm.Isend(self.u[-2:-1], dest=self.right_neighbor, tag=0)
            requests.append(req)

        if self.left_neighbor != MPI.PROC_NULL:
            req = self.comm.Irecv(self.u[0:1], source=self.left_neighbor, tag=0)
            requests.append(req)

        # Send to left, receive from right
        if self.left_neighbor != MPI.PROC_NULL:
            req = self.comm.Isend(self.u[1:2], dest=self.left_neighbor, tag=1)
            requests.append(req)

        if self.right_neighbor != MPI.PROC_NULL:
            req = self.comm.Irecv(self.u[-1:], source=self.right_neighbor, tag=1)
            requests.append(req)

        # Wait for all communications to complete
        MPI.Request.Waitall(requests)

        # Handle periodic boundary conditions for first and last process
        if self.rank == 0 and self.left_neighbor == MPI.PROC_NULL:
            # Get value from last process
            if self.size > 1:
                self.comm.Sendrecv(self.u[1:2], dest=self.size-1, sendtag=2,
                                  recvbuf=self.u[0:1], source=self.size-1, recvtag=3)
            else:
                # Single process: periodic on itself
                self.u[0] = self.u[-2]

        if self.rank == self.size - 1 and self.right_neighbor == MPI.PROC_NULL:
            # Get value from first process
            if self.size > 1:
                self.comm.Sendrecv(self.u[-2:-1], dest=0, sendtag=3,
                                  recvbuf=self.u[-1:], source=0, recvtag=2)
            else:
                # Single process: periodic on itself
                self.u[-1] = self.u[1]

    def flux(self, u: np.ndarray) -> np.ndarray:
        """Compute flux F(u) = u²/2."""
        return 0.5 * u**2

    def rusanov_flux(self, u_left: np.ndarray, u_right: np.ndarray) -> np.ndarray:
        """Compute Rusanov numerical flux at cell interfaces."""
        f_left = self.flux(u_left)
        f_right = self.flux(u_right)
        alpha = np.maximum(np.abs(u_left), np.abs(u_right))
        return 0.5 * (f_left + f_right) - 0.5 * alpha * (u_right - u_left)

    def compute_dt(self) -> float:
        """Compute time step based on CFL condition (global minimum)."""
        # Local maximum speed
        max_speed_local = np.max(np.abs(self.u[1:-1]))

        # Global maximum speed
        max_speed = self.comm.allreduce(max_speed_local, op=MPI.MAX)

        if max_speed > 1e-10:
            dt_convection = self.cfl * self.dx / max_speed
        else:
            dt_convection = self.cfl * self.dx / 1e-10

        if self.nu > 0:
            # Conservative diffusion stability factor (0.25 instead of 0.5)
            dt_diffusion = 0.25 * self.dx**2 / self.nu
            return min(dt_convection, dt_diffusion)

        return dt_convection

    def step(self):
        """Perform one time step using the Rusanov method."""
        # Compute time step (synchronized across all processes)
        self.dt = self.compute_dt()

        if self.t + self.dt > self.t_final:
            self.dt = self.t_final - self.t

        # Compute fluxes at interfaces (using ghost cells)
        u_left = self.u[:-1]
        u_right = self.u[1:]
        f_interfaces = self.rusanov_flux(u_left, u_right)

        # Update interior points
        u_new = self.u.copy()
        u_new[1:-1] = self.u[1:-1] - (self.dt / self.dx) * (
            f_interfaces[1:] - f_interfaces[:-1]
        )

        # Add viscous term if present
        if self.nu > 0:
            u_new[1:-1] += self.nu * (self.dt / self.dx**2) * (
                self.u[2:] - 2*self.u[1:-1] + self.u[:-2]
            )

        self.u = u_new

        # Exchange ghost cells with neighbors
        self._exchange_halos()

        # Update time
        self.t += self.dt
        self.n_steps += 1

    def gather_solution(self) -> Optional[np.ndarray]:
        """
        Gather the distributed solution to root process.

        Returns:
            Global solution array on root, None on other processes
        """
        # Prepare send buffer (interior cells only)
        u_local = self.u[1:-1].copy()

        # Prepare receive buffer on root
        if self.rank == 0:
            u_global = np.zeros(self.nx_global)
        else:
            u_global = None

        # Gather counts and displacements
        sendcounts = np.zeros(self.size, dtype=int)
        displs = np.zeros(self.size, dtype=int)

        for i in range(self.size):
            nx_local_i = self.nx_global // self.size
            remainder = self.nx_global % self.size
            if i < remainder:
                nx_local_i += 1
            sendcounts[i] = nx_local_i
            if i > 0:
                displs[i] = displs[i-1] + sendcounts[i-1]

        self.comm.Gatherv(u_local, [u_global, sendcounts, displs, MPI.DOUBLE], root=0)

        return u_global

    def solve(self, n_snapshots: int = 10) -> Optional[Tuple[np.ndarray, list, list]]:
        """
        Solve the Burgers equation until t_final.

        Returns:
            Tuple of (final solution, snapshots, times) on root, None on others
        """
        snapshot_interval = self.t_final / n_snapshots
        next_snapshot_time = snapshot_interval

        if self.rank == 0:
            print(f"\nStarting parallel Rusanov solver...")
            print(f"MPI processes: {self.size}")
            print(f"Global grid points: {self.nx_global}")
            print(f"Local grid points: {self.nx_local}")
            print(f"Domain: [{self.x_min}, {self.x_max}]")
            print(f"Final time: {self.t_final}")
            print(f"CFL number: {self.cfl}")
            print(f"Viscosity: {self.nu}\n")

        # Barrier to synchronize start
        self.comm.Barrier()
        start_time = MPI.Wtime()

        # Time integration loop
        while self.t < self.t_final:
            self.step()

            # Save snapshots
            if self.t >= next_snapshot_time or abs(self.t - self.t_final) < 1e-10:
                u_global = self.gather_solution()

                if self.rank == 0:
                    self.snapshots.append(u_global.copy())
                    self.snapshot_times.append(self.t)
                    max_u = np.max(np.abs(u_global))
                    print(f"Step {self.n_steps}: t = {self.t:.6f}, dt = {self.dt:.6e}, "
                          f"max(|u|) = {max_u:.6f}")

                next_snapshot_time += snapshot_interval

        # Final gather
        u_final = self.gather_solution()

        self.comm.Barrier()
        elapsed_time = MPI.Wtime() - start_time

        if self.rank == 0:
            print(f"\nSimulation complete!")
            print(f"Total time steps: {self.n_steps}")
            print(f"Elapsed time: {elapsed_time:.6f} seconds")
            print(f"Average time per step: {elapsed_time/self.n_steps:.6e} seconds")

            return u_final, self.snapshots, self.snapshot_times
        else:
            return None


def main():
    """Main function for parallel Rusanov solver."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

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

    # Create solver
    solver = BurgersRusanovParallel(
        nx_global=args.nx,
        domain=tuple(args.domain),
        t_final=args.t_final,
        cfl=args.cfl,
        nu=args.nu,
        comm=comm
    )

    # Set initial condition
    solver.set_initial_condition(args.ic)

    # Solve
    result = solver.solve(n_snapshots=args.snapshots)

    # Save results (only on root)
    if rank == 0:
        u_final, snapshots, snapshot_times = result

        np.savez(args.save,
                 x=solver.x_global,
                 u_final=u_final,
                 snapshots=np.array(snapshots),
                 times=np.array(snapshot_times),
                 nx=args.nx,
                 nu=args.nu,
                 t_final=args.t_final,
                 n_procs=comm.Get_size())

        print(f"\nResults saved to {args.save}")

        # Check for shocks
        gradients = np.abs(np.gradient(u_final, solver.x_global))
        max_gradient = np.max(gradients)
        print(f"\nMaximum gradient: {max_gradient:.6f}")
        if max_gradient > 10.0:
            print("Strong shock waves detected!")
        elif max_gradient > 2.0:
            print("Moderate discontinuities detected")


if __name__ == "__main__":
    main()
