import numpy as np
import sympy as sp
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from poisson import Poisson

x, y = sp.symbols("x,y")


# Below we create a solver that reuses some of the implementation from
# the 1D solver in poisson.py.


class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, L] x [0, L]

    with Dirichlet boundary conditions.
    """

    def __init__(self, L: float):
        self.p = Poisson(L)  # we can reuse some of the code from the 1D case

    def create_mesh(self, N: int) -> tuple[np.ndarray, np.ndarray]:
        """Return a 2D Cartesian mesh

        Parameters
        ----------
        N : int
            The number of uniform intervals in both x and y directions
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh
        """
        xi = self.p.create_mesh(N)
        xij, yij = np.meshgrid(xi, xi, indexing="ij", sparse=True)
        return xij, yij

    def laplace(self, N: int) -> sparse.lil_matrix:
        """Return a vectorized Laplace operator

        Parameters
        ----------
        N : int
            The number of uniform intervals in both x and y directions

        Returns
        -------
        A : scipy sparse LIL matrix
            The vectorized Laplace operator
        """

        # 1D second order differentiation matrix (already scaled by 1/dx^2)
        dx = self.p.L / N
        D2 = self.p.D2(N, dx)

        # Compute the 2D Laplace operator using the Kronecker product
        I = sparse.eye(N + 1, format="lil")
        A = sparse.kron(I, D2) + sparse.kron(D2, I)

        return A.tolil()
        

    def assemble(
        self, N: int, f: sp.Expr, ue: sp.Expr
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Return assembled coefficient matrix A and right hand side vector b

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in both x and y directions
        f : Sympy expression
            The right hand side as a Sympy expression in x and y
        ue : Sympy expression
            The exact solution as a Sympy expression in x and y

        Returns
        -------
        A : scipy sparse CSR matrix
            Coefficient matrix
        b : 1D array
            Right hand side vector

        Note
        ----
        Compute the Kronecker product of the 1D Laplace operator with itself
        to create the 2D Laplace operator. Then, assemble the right-hand side
        vector b by evaluating the function f at the mesh points and applying
        Dirichlet boundary conditions using the exact solution ue.

        """

        # Compute the 2D Laplace operator
        A = self.laplace(N)

        # Create the mesh
        xij, yij = self.create_mesh(N)

        # Evaluate the right-hand side function f at the mesh points
        b = self.meshfunction(f, xij, yij).ravel()

        # Apply Dirichlet boundary conditions using the exact solution ue
        boundary_indices = self.get_boundary_indices(N)

        ue_mesh = self.meshfunction(ue, xij, yij)
        b[boundary_indices] = ue_mesh.ravel()[boundary_indices]

        # Replace boundary rows by u_i = ue_i
        for i in boundary_indices:
            A.rows[i] = [i]
            A.data[i] = [1.0]

        return A.tocsr(), b

    
    def meshfunction(self, u: sp.Expr, xij: np.ndarray, yij: np.ndarray) -> np.ndarray:
        """Return Sympy function as mesh function

        Parameters
        ----------
        u : Sympy function

        Returns
        -------
        array - The input function as a mesh function
        """

        # Evaluate u at the mesh points (xij, yij). Adding to zeros broadcasts
        # scalars and single-coordinate results onto the full (N+1, N+1) grid.
        return np.zeros(np.broadcast(xij, yij).shape) + sp.lambdify((x, y), u)(xij, yij)


    def get_boundary_indices(self, N: int) -> np.ndarray:
        """Return indices of vectorized matrix that belongs to the boundary"""

        # Create a boolean array to mark boundary points
        boundary = np.zeros((N + 1, N + 1), dtype=bool)

        # Mark the boundary points as True
        boundary[0, :] = True
        boundary[-1, :] = True
        boundary[:, 0] = True
        boundary[:, -1] = True

        return np.flatnonzero(boundary.ravel())

    def l2_error(self, u: np.ndarray, ue: sp.Expr) -> float:
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy expression
            The exact solution

        Returns
        -------
        float - The l2-error

        """

        # Evaluate the exact solution on the same mesh as u
        xij, yij = self.create_mesh(u.shape[0] - 1)
        ue_mesh = self.meshfunction(ue, xij, yij)

        # l2-error: sqrt(h^2 * sum((u - ue)^2)) = h * ||u - ue||_2
        return np.linalg.norm(u - ue_mesh) * (self.p.L / (u.shape[0] - 1))  

    
    def __call__(self, N: int, ue: sp.Expr) -> np.ndarray:
        """Solve Poisson's equation with a given manufactured solution

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in both x and y directions
        ue : Sympy expression
            The exact solution

        Returns
        -------
        The solution as a Numpy array

        """
        A, b = self.assemble(N, sp.diff(ue, x, 2) + sp.diff(ue, y, 2), ue)
        return sparse_linalg.spsolve(A, b.ravel()).reshape((N + 1, N + 1))

    def convergence_rates(self, ue: sp.Expr, m: int = 6):
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            u = self(N0, ue)
            E.append(self.l2_error(u, ue))
            h.append(self.p.L / N0)
            N0 *= 2
        r = [np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i]) for i in range(1, m, 1)]
        return r, np.array(E), np.array(h)

    def eval(self, U: np.ndarray, x: float, y: float) -> float:
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """

        # Check if the coordinates are within the domain
        if x < 0 or x > self.p.L or y < 0 or y > self.p.L:
            raise ValueError("Coordinates (x, y) are outside the domain.")

        # Compute the mesh size
        N = U.shape[0] - 1
        h = self.p.L / N

        # Compute the indices of the grid cell containing (x, y).
        # Clamp to N-1 so that points on the upper boundary (x = L or y = L)
        # fall in the last cell and are interpolated like any other point.
        i = min(int(x / h), N - 1)
        j = min(int(y / h), N - 1)

        # Compute the local coordinates within the grid cell
        xi = (x - i * h) / h
        eta = (y - j * h) / h
        # Perform bilinear interpolation
        u00 = U[i, j]
        u10 = U[i + 1, j]
        u01 = U[i, j + 1]
        u11 = U[i + 1, j + 1]
        return (1 - xi) * (1 - eta) * u00 + xi * (1 - eta) * u10 + (1 - xi) * eta * u01 + xi * eta * u11


def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    sol = Poisson2D(1)
    r, _, _ = sol.convergence_rates(ue)
    assert abs(r[-1] - 2) < 1e-2


def test_interpolation():
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    sol = Poisson2D(1)
    N = 100
    U = sol(N, ue)
    h = sol.p.L / N
    assert abs(sol.eval(U, 0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(U, h / 2, 1 - h / 2) - ue.subs({x: h, y: 1 - h / 2}).n()) < 1e-3


if __name__ == "__main__":
    test_convergence_poisson2d()
    test_interpolation()
    print("All tests passed!")
