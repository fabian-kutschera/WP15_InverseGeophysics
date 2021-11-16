# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 1D FD Forward and Adjoint Wave Propagation
#
# This little notebook implements frequency-domain wave propagation in 1D using second-order finite differences in space. It includes the solution of the forward problem and the adjoint-based computation of a misfit gradient. Furthermore, the numerical accuracy of the gradients is checked using gradient tests, including the hockey-stick test.
#
# **copyright**: Andreas Fichtner (andreas.fichtner@erdw.ethz.ch), December 2020,
#
# **license**: BSD 3-Clause (\"BSD New\" or \"BSD Simplified\")

# ## 0. Python packages

# +
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams.update({"font.size": 70})
plt.rcParams["xtick.major.pad"] = "12"
plt.rcParams["ytick.major.pad"] = "12"
# -

# ## 1. Input
#
# Our simulations need only a small number of input parameters: the total number of grid points along the line (n), the spacing between these grid points (dx), the frequency of the monochromatic waves (f), the grid point indices where the sources are located (ns), and the grid point indices where we make the measurements (rix).

# Number of grid points.
n = 1000
# Space increment [m].
dx = 1000.0
# Frequencies [Hz].
# f=[0.15,0.20,0.25,0.30,0.35,0.40,0.45]
f = [0.25, 0.15]
# Indices of point source locations.
# ns=[100,200,300,400,500,600,700,800,900]
ns = [400]
# Measurement indices (receiver locations).
rix = [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]

# ## 2. Initialisations
#
# We need to initialise various field variables, namely the true velocity distribution that one may want to recover (c_obs), the initial velocity distribution (c), and the spatial distribution of the sources (s).

# +
# Types of models.
model_type = "boxes1"

# Make and plot velocity distribution [m/s].
x = np.arange(0.0, n * dx, dx)

if model_type == "spike":
    c = 3000.0 * np.ones(n, dtype=np.cfloat) + 1.0j * 40.0
    c_obs = 3000.0 * np.ones(n, dtype=np.cfloat) + 1.0j * 40.0
    c_obs[520] = 4000.0 + 1.0j * 30.0

elif model_type == "boxes1":
    c_obs = 3000.0 * np.ones(n, dtype=np.cfloat) + 1.0j * 40.0
    c_obs[30:100] = 2000.0 + 1.0j * 60.0
    c_obs[150:200] = 3500.0 + 1.0j * 10.0
    c_obs[200:400] = 2500.0 + 1.0j * 40.0
    c_obs[450:600] = 2200.0 + 1.0j * 100.0
    c_obs[700:850] = 4000.0 + 1.0j * 10.0

    c = 0.9 * c_obs.copy()
    for i in range(100):
        c[1 : n - 1] = (c[0 : n - 2] + c[2:n] + c[1 : n - 1]) / 3.0


# Plot models.
plt.subplots(1, figsize=(30, 10))
plt.plot(x / 1000.0, np.real(c_obs), "k", LineWidth=4)
plt.plot(x / 1000.0, np.real(c), "--", color=[0.5, 0.5, 0.5], LineWidth=5)
plt.xlabel(r"$x$ [km]")
plt.ylabel(r"$c^{re}$ [m/s]", labelpad=20)
plt.ylim(0.9 * np.min(np.real(c)), 1.2 * np.max(np.real(c)))
plt.xlim([x[0] / 1000.0, x[-1] / 1000.0])
plt.grid()
plt.title("real velocity distribution (black=true, red=initial)", pad=40)
plt.savefig("OUTPUT_forward/velocity_real.pdf", bbox_inches="tight", format="pdf")
plt.show()

plt.subplots(1, figsize=(30, 10))
plt.plot(x / 1000.0, np.imag(c_obs), "k", LineWidth=4)
plt.plot(x / 1000.0, np.imag(c), "--", color=[0.5, 0.5, 0.5], LineWidth=5)
plt.xlabel(r"$x$ [km]")
plt.ylabel(r"$c^{im}$ [m/s]", labelpad=20)
plt.ylim(0.3 * np.min(np.imag(c)), 1.2 * np.max(np.imag(c)))
plt.xlim([x[0] / 1000.0, x[-1] / 1000.0])
plt.grid()
plt.title("imaginary velocity distribution (black=true, red=initial)", pad=40)
plt.savefig("OUTPUT_forward/velocity_imag.pdf", bbox_inches="tight", format="pdf")
plt.show()

# Wavelength and number of grid points per wavelength.
lambda_min = np.min(np.real(c)) / np.max(f)
lambda_max = np.max(np.real(c)) / np.max(f)
gppmw = lambda_min / dx

print("minimum wavelength: %f m" % lambda_min)
print("maximum wavelength: %f m" % lambda_max)
print("grid points per minimum wavelength: %f" % gppmw)


# -

# ## 3. Forward problem solution
#
# We solve the forward problem using second-order central finite differences in space. Hence, the impedance matrix $\mathbf{L}$ is defined through its action on the discrete wavefield $\mathbf{u}$ as
# \begin{equation}
# L_{ij}u_j = \omega^2 u_i + \frac{c_i^2}{dx^2} [u_{i+1}-2u_i+u_{i-1}]\,.
# \end{equation}
# The complete discrete system is then
# \begin{equation}
# \mathbf{Lu}=-\mathbf{s}\,.
# \end{equation}
# Numerically, we solve the problem with a sparse LU decomposition. This allows us to quickly solve the problem for many different sources. $$ $$


def forward(n, dx, c, ns, f):
    """
    Forward problem solution via LU decomposition.
    :param n: number of grid points
    :param dx: spatial finite-difference increment
    :param c: velocity distribution of dimension n
    :param ns: number of sources
    :param f: frequency vector
    """

    # Initialise displacement vector.
    u = np.zeros((n, len(ns), len(f)), dtype=np.cfloat)

    # March through frequencies.
    for nf in range(len(f)):

        # Diagonal offsets.
        offsets = np.array([0, 1, -1])
        # Initialise (sub)diagonal entries.
        data = np.zeros((3, n), dtype=np.cfloat)
        data[0, :] = -2.0 * (c ** 2) / (dx ** 2) + (2.0 * np.pi * f[nf]) ** 2
        data[1, :] = np.roll(c ** 2, 1) / (dx ** 2)
        data[2, :] = np.roll(c ** 2, -1) / (dx ** 2)
        # Make impedance matrix.
        L = sp.dia_matrix((data, offsets), shape=(n, n), dtype=np.cfloat)

        # Make sparse LU decomposition.
        lu = sla.splu(L.tocsc())

        # March through sources.
        for i in range(len(ns)):
            # Make ith point source. Scale with large number to avoid underflow.
            s = np.zeros(n, dtype=np.cfloat)
            s[ns[i]] = 1.0 / dx
            # Solve linear system.
            u[:, i, nf] = lu.solve(-s)

    # Return.
    return u


# +
# Compute wavefields for true and for initial velocity distributions.
u = forward(n, dx, c, ns, f)
u_obs = forward(n, dx, c_obs, ns, f)

# Plot the wavefields.
for j in range(len(f)):
    for i in range(len(ns)):
        plt.subplots(1, figsize=(30, 10))
        plt.plot(
            x / 1000.0,
            1000.0 * np.real(u_obs[:, i, j].reshape(np.shape(x))),
            "k",
            LineWidth=4,
        )
        plt.plot(
            x / 1000.0,
            1000.0 * np.real(u[:, i, j].reshape(np.shape(x))),
            "--",
            color=[0.5, 0.5, 0.5],
            LineWidth=5,
        )
        plt.plot(
            x[ns] / 1000.0,
            np.zeros(len(ns)),
            "*",
            markerfacecolor=[0.2, 0.2, 0.2],
            markersize=30,
            markeredgecolor="k",
            markeredgewidth=2,
        )
        plt.plot(
            x[ns[i]] / 1000.0,
            0.0,
            "*",
            markerfacecolor=[0.2, 0.2, 0.2],
            markersize=60,
            markeredgecolor="k",
            markeredgewidth=2,
        )
        plt.plot(
            x[rix] / 1000.0,
            1000.0 * np.real(u_obs[rix, i, j]),
            "^",
            markerfacecolor=[0.85, 0.85, 0.85],
            markersize=30,
            markeredgecolor="k",
            markeredgewidth=2,
        )
        plt.xlim([x[0] / 1000.0, x[-1] / 1000.0])
        plt.xlabel(r"$x$ [km]")
        plt.ylabel(r"$u$ [mm$\cdot$s]")
        plt.grid()
        plt.title("wavefield for source %d and frequency %f Hz" % (i, f[j]), pad=40)
        fn = "OUTPUT_forward/wavefield_" + str(i) + "_" + str(j) + ".pdf"
        plt.savefig(fn, bbox_inches="tight", format="pdf")
        plt.show()
# -

# ## 4. Misfit and adjoint problem
#
# To measure the difference between the wavefields $u$ and $u^{obs}$ at the receiver locations, we define a simple $L_2$ misfit
# \begin{equation}
# \chi = \frac{1}{2} \sum_f \sum_r [u_r(f) - u_r^{obs}(f)]^2\,,
# \end{equation}
# where the sum is over all receiver indices and frequencies. The corresponding adjoint source has the non-zero entries
# \begin{equation}
# s_r^* = \frac{1}{2} (u_r^{obs} - u_r)\,.
# \end{equation}
# The impedance matrix of the adjoint problem is the Hermetian conjugate of $\mathbf{L}$.

# Compute L2 misfit.
sigma = 0.01 / 1000.0
chi = 0.5 * np.sum(np.abs(u[rix, :, :] - u_obs[rix, :, :]) ** 2) / sigma ** 2
print("misfit: %g" % chi)


def adjoint(n, dx, c, s, f):
    """
    Forward problem solution via LU decomposition.
    :param n: number of grid points
    :param dx: spatial finite-difference increment
    :param c: velocity distribution of dimension n
    :param s: adjoint source of dimension n
    :param f: frequency [Hz]
    """

    # Diagonal offsets.
    offsets = np.array([0, 1, -1])
    # Initialise (sub)diagonal entries.
    data = np.zeros((3, n), dtype=np.cfloat)
    data[0, :] = np.conj(-2.0 * (c ** 2) / (dx ** 2) + (2.0 * np.pi * f) ** 2)
    data[1, :] = np.conj(np.roll(c ** 2, 1) / (dx ** 2))
    data[2, :] = np.conj(np.roll(c ** 2, -1) / (dx ** 2))
    # Make impedance matrix.
    L = sp.dia_matrix((data, offsets), shape=(n, n), dtype=np.cfloat)

    # Solve via sparse LU decomposition.
    lu = sla.splu(L.transpose().tocsc())
    v = lu.solve(-s)

    # Return.
    return v


# ## 5. Compute gradient
#
# We finally compute the discrete gradient, the components of which are given as
# \begin{equation}
# \frac{\partial\chi}{\partial c_i^{(r)}} = 4 Re\, c_i v_i^* e_i\,,
# \end{equation}
# and
# \begin{equation}
# \frac{\partial\chi}{\partial c_i^{(i)}} = -4 Im\, c_i v_i^* e_i\,,
# \end{equation}
# where $e_i$ is the discrete second derivative of the forward wavefield
# \begin{equation}
# e_i = \frac{1}{dx^2} [u_{i+1} - 2u_i + u_{i-1}]\,.
# \end{equation}

# +
# Derivative with respect to real part of velocity c.
dchi_r = np.zeros(n, dtype=np.cfloat)
# Derivative with respect to imaginary part of velocity c.
dchi_i = np.zeros(n, dtype=np.cfloat)

# Accumulate gradient by marching through frequencies and sources.
for j in range(len(f)):
    for i in range(len(ns)):
        # Make adjoint source.
        sa = np.zeros(n, dtype=np.cfloat)
        sa[rix] = 0.5 * (u[rix, i, j] - u_obs[rix, i, j]) / sigma ** 2
        # Solve adjoint problem.
        v = adjoint(n, dx, c, sa, f[j])
        # Add to gradient.
        e = np.zeros(n, dtype=np.cfloat)
        e[1 : n - 1] = (
            u[0 : n - 2, i, j] - 2.0 * u[1 : n - 1, i, j] + u[2:n, i, j]
        ) / (dx ** 2)
        # Compute gradients.
        dchi_r += 4.0 * np.real(c * np.conj(v) * e)
        dchi_i -= 4.0 * np.imag(c * np.conj(v) * e)

# +
# Plot.
plt.subplots(1, figsize=(30, 10))
plt.plot(x / 1000.0, np.real(dchi_r), "k", LineWidth=4)
plt.xlabel(r"$x$ [km]")
plt.ylabel(r"$\partial\chi/\partial c_i^{re}$ [s/m]", labelpad=20)
plt.grid()
plt.title(r"derivative w.r.t. real part", pad=40)
plt.xlim([x[0] / 1000.0, x[-1] / 1000.0])
plt.savefig("OUTPUT_forward/derivative_real.pdf", bbox_inches="tight", format="pdf")
plt.show()

plt.subplots(1, figsize=(30, 10))
plt.plot(x / 1000.0, np.real(dchi_i), "k", LineWidth=4)
plt.xlabel(r"$x$ [km]")
plt.ylabel(r"$\partial\chi/\partial c_i^{im}$ [s/m]", labelpad=20)
plt.grid()
plt.title("derivative w.r.t. imaginary part", pad=40)
plt.savefig("OUTPUT_forward/derivative_imag.pdf", bbox_inches="tight", format="pdf")
plt.xlim([x[0] / 1000.0, x[-1] / 1000.0])
plt.show()
# -

# ## 7. Gradient tests

# ### 7.1. Hockey stick test
#
# The hockey stick test compares the adjoint-derived derivative with a finite-difference approximation of the derivative. In general, we expect this difference to be small. However, the difference will increase when the finite-difference increment is too large (poor finite-difference approximation) and when it is too small (floating point inaccuracy). This produces a characteristic hockey stick plot.

# #### 7.1.1. Real part of velocity

# +
# Index of the model parameter.
idx = 50
# Range of model perturbation.
dc = 10.0 ** np.arange(-5.0, 2.0, 0.1)

# Initialise arrays.
dchi_fd = np.zeros(len(dc))
c1 = np.zeros(n, dtype=np.cfloat)
c2 = np.zeros(n, dtype=np.cfloat)
c1[:] = c[:]
c2[:] = c[:]

# March through perturbation ranges.
for i in range(len(dc)):
    # Positive and negative model perturbations.
    c1[idx] = c[idx] + dc[i]
    c2[idx] = c[idx] - dc[i]
    # Solve forward problems.
    u1 = forward(n, dx, c1, ns, f)
    u2 = forward(n, dx, c2, ns, f)
    # Finite-difference approximation of derivative.
    dchi_fd[i] = 0.5 * (
        np.sum(np.abs(u1[rix, :, :] - u_obs[rix, :, :]) ** 2) / sigma ** 2
        - np.sum(np.abs(u2[rix, :, :] - u_obs[rix, :, :]) ** 2) / sigma ** 2
    )
    dchi_fd[i] = dchi_fd[i] / (2.0 * dc[i])
# -

plt.subplots(1, figsize=(20, 20))
plt.loglog(dc, np.abs((dchi_fd - dchi_r[idx]) / dchi_r[idx]), "k", LineWidth=3)
plt.loglog(dc, np.abs((dchi_fd - dchi_r[idx]) / dchi_r[idx]), "ko", MarkerSize=10)
plt.xlabel(r"increment $\Delta c_i^{re}$ [m/s]", labelpad=20)
plt.ylabel("relative derivative error", labelpad=20)
plt.title("hockey stick plot", pad=40)
plt.grid()
plt.savefig("OUTPUT_forward/hockey_stick_real.pdf", bbox_inches="tight", format="pdf")
plt.show()

# #### 7.1.2. Imaginary part of velocity

# +
# Index of the model parameter.
idx = 50
# Range of model perturbation.
dc = 10.0 ** np.arange(-5.0, 2.0, 0.1)

# Initialise arrays.
dchi_fd = np.zeros(len(dc))
c1 = np.zeros(n, dtype=np.cfloat)
c2 = np.zeros(n, dtype=np.cfloat)
c1[:] = c[:]
c2[:] = c[:]

# March through perturbation ranges.
for i in range(len(dc)):
    # Positive and negative model perturbations.
    c1[idx] = c[idx] + 1j * dc[i]
    c2[idx] = c[idx] - 1j * dc[i]
    # Solve forward problems.
    u1 = forward(n, dx, c1, ns, f)
    u2 = forward(n, dx, c2, ns, f)
    # Finite-difference approximation of derivative.
    dchi_fd[i] = 0.5 * (
        np.sum(np.abs(u1[rix, :, :] - u_obs[rix, :, :]) ** 2) / sigma ** 2
        - np.sum(np.abs(u2[rix, :, :] - u_obs[rix, :, :]) ** 2) / sigma ** 2
    )
    dchi_fd[i] = dchi_fd[i] / (2.0 * dc[i])
# -

plt.subplots(1, figsize=(20, 20))
plt.loglog(dc, np.abs((dchi_fd - dchi_i[idx]) / dchi_i[idx]), "k", LineWidth=3)
plt.loglog(dc, np.abs((dchi_fd - dchi_i[idx]) / dchi_i[idx]), "ko", MarkerSize=10)
plt.xlabel(r"increment $\Delta c_i^{im}$ [m/s]", labelpad=20)
plt.ylabel("relative derivative error", labelpad=20)
plt.title("hockey stick plot", pad=40)
plt.grid()
plt.savefig("OUTPUT_forward/hockey_stick_imag.pdf", bbox_inches="tight", format="pdf")
plt.show()

# ### 7.2. Space-dependent derivative
#
# For a fixed finite-difference increment, we may also compute a finite-difference approximation of the misfit derivative for all grid points and compare this to the adjoint-based derivative. This is precisely the brute-force approach that adjoint methods are supposed to avoid.

# #### 7.2.1. Real part of velocity

# +
# Fixed finite-difference increment.
dc = 1.0e-2

# Initialise arrays.
dchi_fd = np.zeros(n)
c1 = np.zeros(n, dtype=np.cfloat)
c2 = np.zeros(n, dtype=np.cfloat)

# March through grid points.
for i in range(n):
    # Positive and negative model perturbations.
    c1[:] = c[:]
    c2[:] = c[:]
    c1[i] = c[i] + dc
    c2[i] = c[i] - dc
    # Solve forward problems.
    u1 = forward(n, dx, c1, ns, f)
    u2 = forward(n, dx, c2, ns, f)
    # Finite-difference approximation of derivative.
    dchi_fd[i] = 0.5 * (
        np.sum(np.abs(u1[rix, :, :] - u_obs[rix, :, :]) ** 2) / sigma ** 2
        - np.sum(np.abs(u2[rix, :, :] - u_obs[rix, :, :]) ** 2) / sigma ** 2
    )
    dchi_fd[i] = dchi_fd[i] / (2.0 * dc)

# +
plt.subplots(1, figsize=(30, 10))
plt.plot(x / 1000.0, dchi_fd, "k", LineWidth=4)
plt.xlabel(r"$x$ [km]")
plt.ylabel(r"$\partial\chi/\partial c_i^{re}|_{FD}$ [s/m]", labelpad=20)
plt.grid()
plt.title(r"FD derivative w.r.t. real part", pad=40)
plt.xlim([x[0] / 1000.0, x[-1] / 1000.0])
plt.savefig("OUTPUT_forward/derivative_real_fd.pdf", bbox_inches="tight", format="pdf")
plt.show()

plt.subplots(1, figsize=(30, 10))
plt.plot(x / 1000.0, 1.0e10 * np.real(dchi_fd - dchi_r), "k", LineWidth=4)
plt.xlabel(r"$x$ [km]")
plt.ylabel(r"$\Delta \partial\chi/\partial c_i^{re}$ [$10^{-10}$ s/m]", labelpad=20)
plt.grid()
plt.title(r"FD derivative error w.r.t. real part", pad=40)
plt.xlim([x[0] / 1000.0, x[-1] / 1000.0])
plt.ylim([-2.0, 2.0])
plt.savefig(
    "OUTPUT_forward/derivative_real_error.pdf", bbox_inches="tight", format="pdf"
)
plt.show()

# -

# #### 7.1.2. Imaginary part of velocity

# +
# Fixed finite-difference increment.
dc = 1.0e-2

# Initialise arrays.
dchi_fd = np.zeros(n)
c1 = np.zeros(n, dtype=np.cfloat)
c2 = np.zeros(n, dtype=np.cfloat)

# March through grid points.
for i in range(n):
    # Positive and negative model perturbations.
    c1[:] = c[:]
    c2[:] = c[:]
    c1[i] = c[i] + 1j * dc
    c2[i] = c[i] - 1j * dc
    # Solve forward problems.
    u1 = forward(n, dx, c1, ns, f)
    u2 = forward(n, dx, c2, ns, f)
    # Finite-difference approximation of derivative.
    dchi_fd[i] = 0.5 * (
        np.sum(np.abs(u1[rix, :, :] - u_obs[rix, :, :]) ** 2)
        - np.sum(np.abs(u2[rix, :, :] - u_obs[rix, :, :]) ** 2)
    )
    dchi_fd[i] = dchi_fd[i] / (2.0 * dc)

# +
plt.plot(x, dchi_fd)
plt.xlabel("x [m]")
plt.ylabel("misfit gradient (FD)")
plt.title("FD approximation of misfit gradient")
plt.show()

plt.semilogy(x, np.abs(dchi_fd - dchi_i))
plt.xlabel("x [m]")
plt.ylabel("misfit gradient (FD)")
plt.title("error of FD approximation")
plt.show()
# -
