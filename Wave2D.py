import numpy as np
import sympy as sp
from scipy import sparse
import pytest
import sys

x, y, t = sp.symbols("x,y,t")


class Wave2D:
    """Class for solving the 2D wave equation"""


    def create_mesh(
        self, N: int, sparse: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return 2D mesh created using np.meshgrid

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        sparse : bool, optional
            Whether to create a sparse mesh or not. Default is False.
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh"""

        xi = np.linspace(0, 1, N + 1)
        return np.meshgrid(xi, xi, indexing="ij", sparse=sparse)
        
    def D2(self, N: int) -> sparse.lil_matrix:
        """Return second order differentiation matrix

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Returns
        -------
        D : scipy sparse LIL matrix
            The second order differentiation matrix
        """

        D = sparse.diags(
            [1.0, -2.0, 1.0], [-1, 0, 1],
            shape=(N + 1, N + 1), format="lil",
        )
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D
    
    @property
    def w(self):
        """Return the dispersion coefficient"""
        return self.c * np.pi * np.sqrt(self.mx**2 + self.my**2)

    
    def ue(self, mx: int, my: int) -> sp.Expr:
        """Return the exact standing wave

        Parameters
        ----------
        mx, my : int
            Parameters for the standing wave
        Returns
        -------
        ue : Sympy expression
            The exact solution as a Sympy expression in x, y and t
        """
        return sp.sin(mx * sp.pi * x) * sp.sin(my * sp.pi * y) * sp.cos(self.w * t)

    def initialize(
        self, N: int, mx: int, my: int) -> tuple[np.ndarray, np.ndarray]:
        
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """

        xij, yij = self.create_mesh(N)

        # U^0: initial displacement, including boundary conditions.
        Unm1 = np.zeros((N + 1, N + 1))
        ue0 = self.ue(mx, my).subs(t, 0)
        Unm1[:] = sp.lambdify((x, y), ue0, "numpy")(xij, yij)
        self.apply_bcs(Unm1)

        # U^1: special first step for zero initial velocity.
        D = self.D2(N).tocsr() / self.h**2
        Un = Unm1 + 0.5 * (self.c * self.dt)**2 * (
            D @ Unm1 + Unm1 @ D.T
        )
        self.apply_bcs(Un)

        return Unm1, Un

    @property
    def dt(self) -> float:
        """Return the time step"""

        return self.cfl * self.h / self.c

    def l2_error(self, u: np.ndarray, t0: float) -> float:
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        N = u.shape[0] - 1
        h = 1 / N
        xij, yij = self.create_mesh(N)

        ue_mesh = sp.lambdify(
            (x, y, t), self.ue(self.mx, self.my), "numpy")(xij, yij, t0)

        return float(np.sqrt(h**2 * np.sum((u - ue_mesh)**2)))

    
    def apply_bcs(self, u: np.ndarray):
        """Apply boundary conditions to the solution mesh function

        Parameters
        ----------
        u : array
            The solution mesh function
        """
        u[0, :] = 0
        u[-1, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0

    def __call__(
        self,
        N: int,
        Nt: int,
        cfl: float = 0.5,
        c: float = 1.0,
        mx: int = 3,
        my: int = 3,
        store_data: int = -1,
    ):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        if N < 3 or Nt < 1:
            raise ValueError("Require N >= 3 and Nt >= 1.")
        if c <= 0 or cfl <= 0:
            raise ValueError("Require c > 0 and cfl > 0.")
        if store_data != -1 and store_data <= 0:
            raise ValueError("store_data must be -1 or a positive integer.")

        self.h = 1 / N
        self.cfl = cfl
        self.c = c
        self.mx = mx
        self.my = my

        D = self.D2(N).tocsr() / self.h**2
        dt = self.dt
        Unm1, Un = self.initialize(N, mx, my)

        errors = []
        data = {}

        if store_data == -1:
            errors = [
                self.l2_error(Unm1, 0.0),
                self.l2_error(Un, dt),
            ]
        else:
            data[0] = Unm1.copy()
            if store_data == 1:
                data[1] = Un.copy()


        for n in range(1, Nt):
            Unp1 = 2 * Un - Unm1 + (self.c * dt)**2 * (
                D @ Un + Un @ D.T
            )
            self.apply_bcs(Unp1)

            Unm1, Un = Un, Unp1

            if store_data == -1:
                errors.append(self.l2_error(Un, (n + 1) * dt))
            elif (n + 1) % store_data == 0:
                data[n + 1] = Un.copy()

        if store_data == -1:
            return self.h, np.array(errors)

        return data

    
    def convergence_rates(
        self, m: int = 4, cfl: float = 0.1, Nt: int = 10, mx: int = 3, my: int = 3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [
            np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i])
            for i in range(1, m, 1)
        ]
        return np.array(r), np.array(E), np.array(h)


class Wave2D_Neumann(Wave2D):
    def D2(self, N: int) -> sparse.lil_matrix:

        D = super().D2(N)

        # Replace the one-sided boundary stencils with Neumann stencils.
        D[0, :4] = -2, 2, 0, 0
        D[-1, -4:] = 0, 0, 2, -2

        return D
    
    def ue(self, mx: int, my: int) -> sp.Expr:

        return (
            sp.cos(mx * sp.pi * x)
            * sp.cos(my * sp.pi * y)
            * sp.cos(self.w * t)
        )
    
    def apply_bcs(self, u: np.ndarray):
        # Neumann conditions are already enforced by the boundary rows of D2.
        pass

def test_convergence_wave2d():
    sol = Wave2D()
    r, _, _ = sol.convergence_rates(m=5, mx=2, my=3)
    assert abs(r[-1] - 2) < 1e-2, r


def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, _, _ = solN.convergence_rates(mx=3, my=3)
    assert abs(r[-1] - 2) < 0.05


def test_exact_wave2d():
    for Solver in (Wave2D, Wave2D_Neumann):
        sol = Solver()

        _, errors = sol(
            N=32,
            Nt=100,
            cfl=1 / np.sqrt(2),
            mx=2,
            my=2,
            store_data=-1,
        )

        # Check every computed time level, not only the final time.
        max_error = np.max(errors)
        assert max_error < 1e-12, (Solver.__name__, max_error)


def create_movie():
    from pathlib import Path
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    sol = Wave2D_Neumann()
    data = sol(40, 40, cfl=1 / np.sqrt(2), mx=2, my=2, store_data=2)
    xij, yij = sol.create_mesh(40)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_zlim(-1, 1)

    frames = []
    for n, U in data.items():
        frame = ax.plot_wireframe(xij, yij, U, rstride=2, cstride=2)
        frames.append([frame])

    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)

    output = Path(__file__).resolve().parent / "report" / "neumannwave.gif"
    output.parent.mkdir(exist_ok=True)
    ani.save(str(output), writer="pillow", fps=10)
    plt.close(fig)

if __name__ == "__main__":
    #create_movie() #uncomment to create a movie of the solution
    sys.exit(pytest.main([__file__]))