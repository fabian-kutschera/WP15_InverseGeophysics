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

# # Backus-Gilbert Theory
#
# Here comes some general description.

# # 0. Python packages
#
# We begin with the import of some Python packages and a few lines of code that help us to embellish figures.

# +
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams.update({"font.size": 30})
plt.rcParams["xtick.major.pad"] = "12"
plt.rcParams["ytick.major.pad"] = "12"
# -

# # 1. Love waves

# ## 1.1. General background
#
# The following is a notebook for the calculation of surface wave dispersion and sensitivity kernels for surface waves propagating in layered media. The developments closely follow the classical work of Takeuchi & Saito (1972) ["Seismic Surface Waves", Methods in Computational Physics: Advances in Research and Applications, 217 - 295]. For this, we consider elastic media with vertical symmetry axis, where the stress-strain relation is given by
#
# \begin{equation}
# \sigma_{xx} = A (\epsilon_{xx} + \epsilon_{yy}) - 2N \epsilon_{yy} + F \epsilon_{zz}\,, \\
# \sigma_{yy} = A (\epsilon_{xx} + \epsilon_{yy}) - 2N \epsilon_{xx} + F \epsilon_{zz}\,, \\
# \sigma_{zz} = F (\epsilon_{xx} + \epsilon_{yy}) + C \epsilon_{zz}\,, \\
# \sigma_{yz} = 2 L (\epsilon_{yz}\,, \\
# \sigma_{zx} = 2 L (\epsilon_{zx})\,, \\
# \sigma_{xy} = 2N \epsilon_{xy}\,,
# \end{equation}
#
# with the strain tensor components
#
# \begin{equation}
# \epsilon_{ij} = \frac{1}{2} ( \partial_i u_j + \partial_j u_i )\,.
# \end{equation}
#
# The $z$-axis is defined positive upwards, medium parameters are assumed to depend only on $z$, and all waves propagate in $x$-direction. We will generally consider freely propagating waves, meaning that no sources are active. For this case, the equations of motion in the frequency domain are
#
# \begin{equation}
# -\omega^2 \rho u_i - \partial_j \sigma_{ij} = 0\,.
# \end{equation}
#
# Love waves are SH-waves, and so we seek solutions to the equations of motion in the form
#
# \begin{equation}
# u_x = u_z = 0\,,\\
# u_y = y_1(z; \omega, k) \, e^{i(\omega t - kx)}\,.
# \end{equation}
#
# Inserting this ansatz into the stress-strain relation and the equations of motion, yields a system of two ordinary differential equations,
#
# \begin{equation}
# \partial_z y_2 = (k^2 N - \omega^2 \rho) y_1\,,\\
# \partial_z y_1 = y_2/L\,.
# \end{equation}
#
# These equations have the useful advantage that they do not contain derivatives of the material parameters, and that they are written in terms of continuous displacement ($y_1$) and stress ($y_2$) quantities. These have to satisfy the boundary conditions
#
# \begin{equation}
# y_1 \text{ and } y_2 \text{ are continuous}\,,\\
# y_2 = 0 \text{ at the surface}\,,\\
# y_1,y_2 \to 0 \text{ as } z\to -\infty\,.
# \end{equation}

# ## 1.2. Analytical solution for layer over a half-space

# For a homogeneous, isotropic layer ($0<z\leq H$, with medium properties $L_1=N_1=\mu_1$, $\rho_1$) over a homogeneous, isotropic half-space ($z<0$, with medium properties $L_2=N_2=\mu_2$, $\rho_2$) the Love wave equations can be solved analytically. In the upper layer, we find
#
# \begin{equation}
# y_1(z) = A \cos(\nu_1 z) + \frac{\mu_2\nu_2}{\mu_2\nu_1} A \sin(\nu_1 z)\,,\\
# y_2(z) = -A \mu_1\nu_1 \sin(\nu_1 z) + A \mu_2 \nu_2 \cos(\nu_1 z)\,,
# \end{equation}
#
# and in the half-space we have
#
# \begin{equation}
# y_1(z) = A e^{\nu_2 z}\,\\
# y_2(z) = \mu_2\nu_2 A e^{\nu_2 z}\,,
# \end{equation}
#
# with some arbitrary amplitude $A$. The positive scalars $\nu_1$ and $\nu_2$ are defined as
#
# \begin{equation}
# \nu_1^2 = \frac{\rho_1\omega^2}{\mu_1} - k^2 > 0\,,\\
# \nu_2^2 = k^2 - \frac{\rho_2\omega^2}{\mu_2} > 0\,.
# \end{equation}
#
# It follows immediately from the last two relations that a Love wave can only exist in the presence of a low-velocity layer, that is, when
#
# \begin{equation}
# \frac{\mu_1}{\rho_1} = \beta_1^2 < \frac{\omega^2}{k^2} < \beta_2^2 = \frac{\mu_2}{\rho_2}\,.
# \end{equation}
#
# Hence, the phase velocity $\omega/k$ of a Love wave is between the phase velocities of the layer (slower) and the half-space (faster). For a given frequency $\omega$, the wave number $k$ is so far undetermined. It follows, however, from the free-surface boundary condition, which yields the characteristic equation of Love waves:
#
# \begin{equation}
# \mu_2\nu_2\,\cos(\nu_1 H) = \mu_1\nu_1\,\sin(\nu_1 H)\,.
# \end{equation}

# ### 1.2.1. Medium properties
#
# First, we define some medium properties and plot the characteristic function.

# +
# velocity [m/s] and density [kg/m^3] of the layer
beta_1 = 2000.0
rho_1 = 2700.0

# velocity [m/s] and density [kg/m^3] of the half-space
beta_2 = 3000.0
rho_2 = 3100.0

# thickness of the layer [m]
H = 10000.0

# frequency [Hz]
f = 0.1

# +
# shear moduli
mu_1 = (beta_1 ** 2) * rho_1
mu_2 = (beta_2 ** 2) * rho_2

# circular frequency
omega = 2.0 * np.pi * f
# -

# ### 1.2.2. Characteristic function

# +
# march through phase velocities and plot left- versus right-hand side of the characteristic equation
eps = 1.0e-9
c = np.linspace(beta_1 + eps, beta_2 - eps, 10000)
k = omega / c

nu_1 = np.sqrt(rho_1 * omega ** 2 / mu_1 - k ** 2)
nu_2 = np.sqrt(k ** 2 - rho_2 * omega ** 2 / mu_2)

plt.subplots(1, figsize=(30, 10))
plt.plot(c, (mu_2 * nu_2) * np.cos(nu_1 * H), "--k", linewidth=2)
plt.plot(c, (mu_1 * nu_1) * np.sin(nu_1 * H), "k", linewidth=2)
plt.grid()
plt.xlim([beta_1, beta_2])
plt.xlabel("phase velocity, $c$ [m/s]", labelpad=20)
plt.ylabel(r"[N$/$m$^3$]", labelpad=20)
plt.tight_layout()
plt.savefig("characteristic.pdf", format="pdf")
plt.show()


# -

# Obviously, the characteristic equation can have more than one solution, depending on the frequency. In general, the number of solutions increases with increasing frequency.

# ### 1.2.3. Dispersion curves
#
# As a next step, we will march through frequency $f$ and determine the wave numbers $k$ (or, equivalently, phase velocities $c$) that solve the characteristic equation. Each solution is referred to as a mode. The mode with the lowest frequency is the fundamental mode. All others are higher modes or overtones.
#
# To make the solution of the characteristic equation easier, we define it as a separate function:


def cf(omega, c):

    k = omega / c
    nu_1 = np.sqrt(rho_1 * omega ** 2 / mu_1 - k ** 2)
    nu_2 = np.sqrt(k ** 2 - rho_2 * omega ** 2 / mu_2)

    return (mu_2 * nu_2) * np.cos(nu_1 * H) - (mu_1 * nu_1) * np.sin(nu_1 * H)


# Then we define some input parameters; the frequency range of interest, and the maximum number of modes we wish to find. Approximate solutions are then found by bisection. To find these solutions with reasonable accuracy and to avoid missing modes, the frequency increment $df$ needs to be sufficiently small.

# +
# frequency range [Hz]
f_min = 0.02
f_max = 1.0
df = 0.01

# maximum number of higher modes
n = 10
# -

# Then we march through the discrete frequency intervals.

# +
# test phase velocities [m/s]
b = np.linspace(beta_1 + eps, beta_2 - eps, 1000)

# march through frequency-phase velocity pairs
f = np.arange(f_min, f_max + df, df)
c = np.zeros((len(f), 10))

for i in range(len(f)):

    omega = 2.0 * np.pi * f[i]
    count = 0

    for j in range(len(b) - 1):

        if cf(omega, b[j]) * cf(omega, b[j + 1]) < 0.0:
            c[i, count] = 0.5 * (b[j] + b[j + 1])
            count += 1
# -

# Finally, we plot the results.

# +
plt.subplots(1, figsize=(25, 10))
for i in range(len(f)):
    for j in range(n):
        if c[i, j] > 0.0:
            plt.plot(f[i], c[i, j], "kx")

plt.xlabel("frequency [Hz]", labelpad=20)
plt.ylabel("phase velocity, $c$ [m/s]", labelpad=20)
plt.xlim([f_min, f_max])
plt.grid()
plt.tight_layout()
plt.savefig("dispersion.pdf", format="pdf")
plt.show()
# -

# ### 1.2.4. Displacement and stress function
#
# Based on the computed dispersion curves, we can plot the displacement function $y_1$ and the stress function $y_2$ as a function of depth.

# +
# frequency index
i = 58
# mode index
j = 0

print("frequency=%f Hz, mode=%d, phase velocity=%f m/s" % (f[i], j, c[i, j]))

# +
# compute nu_1 and nu_2
omega = 2.0 * np.pi * f[i]
k = omega / c[i, j]
nu_1 = np.sqrt(rho_1 * omega ** 2 / mu_1 - k ** 2)
nu_2 = np.sqrt(k ** 2 - rho_2 * omega ** 2 / mu_2)

# plot lower half space
z = np.linspace(-3.0 * H, 0.0, 100)
y_1 = np.exp(nu_2 * z)
y_2 = mu_2 * nu_2 * np.exp(nu_2 * z)

plt.subplots(figsize=(8, 12))
plt.plot(y_1, z, "k", linewidth=2)

# plot layer
z = np.linspace(0.0, H, 100)
y_1 = np.cos(nu_1 * z) + ((mu_2 * nu_2) / (mu_1 * nu_1)) * np.sin(nu_1 * z)
y_2 = -mu_1 * nu_1 * np.sin(nu_1 * z) + mu_2 * nu_2 * np.cos(nu_1 * z)

plt.plot(y_1, z, "--k", linewidth=2)
plt.grid()
plt.title("displacement $y_1$", pad=30)
plt.xlabel(r"$y_1$", labelpad=20)
plt.ylabel(r"$z$ [m]", labelpad=20)
plt.ylim([-2.0 * H, H])
plt.tight_layout()
plt.savefig("displacement_05Hz_m3.pdf", format="pdf")
plt.plot()
# -

# ## 1.3. Sensitivity kernel
#
# Based on the displacement and stress functions, we can compute the sensitivity kernel $K(z)$ that relates variations $\delta\mu(z)$ to fractional variations in the phase velocity $\delta c /c$ via
#
# \begin{equation}
# \frac{\delta c}{c} =  \int\limits_{-\infty}^H K(z) \delta\mu(z)\, dz\,.
# \end{equation}
#
# Explicitly, $K(z)$ is given by
#
# \begin{equation}
# K(z) = \frac{  k^2  y_1(z)^2 +  \frac{1}{\mu^2} y_2(z)^2 }{ 2 k^2  \int\limits_{-\infty}^H \mu(z) y_1(z)^2 \, dz  }\,.
# \end{equation}

# +
# frequency index
i = 58
# mode index
j = 4

print("frequency=%f Hz, mode=%d, phase velocity=%f m/s" % (f[i], j, c[i, j]))

# compute nu_1 and nu_2
omega = 2.0 * np.pi * f[i]
k = omega / c[i, j]
nu_1 = np.sqrt(rho_1 * omega ** 2 / mu_1 - k ** 2)
nu_2 = np.sqrt(k ** 2 - rho_2 * omega ** 2 / mu_2)

# +
## Compose mu, y_1 and y_2 over the complete depth range.
N = 200
z = np.linspace(-3.0 * H, H, N)
y_1 = np.zeros(len(z))
y_2 = np.zeros(len(z))
mu = np.zeros(len(z))
idx = np.int(np.where(np.abs(z) == np.min(np.abs(z)))[0])

mu[:idx] = mu_2
y_1[:idx] = np.exp(nu_2 * z[:idx])
y_2[:idx] = mu_2 * nu_2 * np.exp(nu_2 * z[:idx])

mu[idx:N] = mu_1
y_1[idx:N] = np.cos(nu_1 * z[idx:N]) + ((mu_2 * nu_2) / (mu_1 * nu_1)) * np.sin(
    nu_1 * z[idx:N]
)
y_2[idx:N] = -mu_1 * nu_1 * np.sin(nu_1 * z[idx:N]) + mu_2 * nu_2 * np.cos(
    nu_1 * z[idx:N]
)

# Compute sensitivity kernel.
dz = z[1] - z[0]
I = 2.0 * k ** 2 * np.sum(mu * y_1 ** 2) * dz
K = k ** 2 * y_1 ** 2 + y_2 ** 2 / mu ** 2
K = K / I

# Plot kernels.
plt.subplots(figsize=(8, 12))
plt.plot(K, z, "k", linewidth=2)
plt.xlabel(r"$K$ [m/N]", labelpad=20)
plt.ylabel(r"$z$ [m]", labelpad=20)
plt.grid()
plt.ylim([-3.0 * H, H])
plt.tight_layout()
plt.savefig("kernel_060Hz_m4.pdf", format="pdf")
plt.show()
# -

# # 2. Backus-Gilbert optimisation
#
# Having established the sensitivity kernels $K(z)$, which play the role of data kernels $G(z)$ in Backus-Gilbert theory, we can continue with the actual solution of the Backus-Gilbert optimisation. This is intended to find an averaging kernel $A(z)$ that is optimally localised around a certain depth $z_0$.

# ## 2.1. Input
#
# First, we provide some input, namely the target depth, and the frequency and mode indices of the Love wave modes that we wish to include.

# +
# target depth [m]
z_0 = -5000.0

# Make an array of [frequency index, mode index].
modes = [
    [0, 0],
    [10, 0],
    [20, 0],
    [30, 0],
    [40, 0],
    [50, 0],
    [60, 0],
    [70, 0],
    [80, 0],
    [90, 0],
    [20, 1],
    [30, 1],
    [40, 1],
    [50, 1],
    [60, 1],
    [70, 1],
    [80, 1],
    [90, 1],
    [30, 2],
    [40, 2],
    [50, 2],
    [60, 2],
    [70, 2],
    [80, 2],
    [90, 2],
    [50, 3],
    [60, 3],
    [70, 3],
    [80, 3],
    [90, 3],
]
Nm = len(modes)
# -

print(modes)

# ## 2.2. Setup of linear system
#
# Backus-Gilbert optimisation can be formulated as a linear system of equations. In the following we set up the system matrix and the right-hand side. First, for convenience, we collect the kernels for all modes into a Numpy array.

# +
# Accumulate kernels
N = 200
z = np.linspace(-3.0 * H, H, N)
dz = z[1] - z[0]
G = np.zeros((Nm, len(z)))
y_1 = np.zeros(len(z))
y_2 = np.zeros(len(z))

mu = np.zeros(len(z))
idx = np.int(np.where(np.abs(z) == np.min(np.abs(z)))[0])
mu[idx:N] = mu_1
mu[:idx] = mu_2

for i in range(Nm):
    # Compute nu_1 and nu_2.
    omega = 2.0 * np.pi * f[modes[i][0]]
    k = omega / c[modes[i][0], modes[i][1]]
    nu_1 = np.sqrt(rho_1 * omega ** 2 / mu_1 - k ** 2)
    nu_2 = np.sqrt(k ** 2 - rho_2 * omega ** 2 / mu_2)

    # Compute displacement and stress functions.
    y_1[idx:N] = np.cos(nu_1 * z[idx:N]) + ((mu_2 * nu_2) / (mu_1 * nu_1)) * np.sin(
        nu_1 * z[idx:N]
    )
    y_2[idx:N] = -mu_1 * nu_1 * np.sin(nu_1 * z[idx:N]) + mu_2 * nu_2 * np.cos(
        nu_1 * z[idx:N]
    )
    y_1[:idx] = np.exp(nu_2 * z[:idx])
    y_2[:idx] = mu_2 * nu_2 * np.exp(nu_2 * z[:idx])

    # Compute sensitivity kernel.
    I = 2.0 * k ** 2 * np.sum(mu * y_1 ** 2) * dz
    G[i, :] = (k ** 2 * y_1 ** 2 + y_2 ** 2 / mu ** 2) / I
# -

# With this we can compute the matrix $\mathbf{S}$ and the vector $\mathbf{u}$, and solve the linear system.

# +
# compute S matrix and u vector.
S = np.zeros((Nm, Nm))
u = np.zeros(Nm)

for i in range(Nm):
    u[i] = np.sum(G[i, :]) * dz
    for j in range(Nm):
        S[i, j] = 24.0 * np.sum((z - z_0) ** 2 * G[i, :] * G[j, :]) * dz

# Solve linear system.
a = np.dot(np.linalg.inv(S), u)

# Compute normalisation.
a = a / np.dot(a, u)
# -

# ## 2.3. Compute and visualise averaging kernel
#
# From the solution of the linear system, we can accumulate the averaging kernel $A(z)$ and compute the averaging length scale.

# +
# Assemble the averaging kernel.
A = np.zeros(len(z))
for i in range(Nm):
    A += a[i] * G[i, :]

# Compute averaging length scale.
s = 12.0 * np.sum((z - z_0) ** 2 * A ** 2) * dz
print("averaging length: %f m" % s)
# -

# Plot averaging kernel.
plt.subplots(figsize=(8, 12))
plt.plot(A, z, "k", linewidth=2)
plt.xlabel(r"$A$ [1/m]", labelpad=20)
plt.ylabel(r"$z$ [m]", labelpad=20)
plt.grid()
plt.ylim([-2.0 * H, H])
plt.tight_layout()
plt.savefig("A_all_m5000.pdf", format="pdf")
plt.show()

# # 2.4. Investigate influence of data errors

# +
# Make some data covariance matrix.
C = np.identity(len(modes))

# Choose some weight.
gamma = 1.0e-22

# Compute new averaging coefficients.
# Solve linear system.
ae = np.dot(np.linalg.inv(S + gamma * C), u)

# Compute normalisation.
ae = ae / np.dot(ae, u)

# +
# Assemble the averaging kernel.
Ae = np.zeros(len(z))
for i in range(Nm):
    Ae += ae[i] * G[i, :]

# Compute averaging length scale.
s = 0.5 * np.dot(ae, np.dot(S, ae))
print("averaging length: %f m" % s)

# Compute standard deviation of the average.
sigma = np.sqrt(np.dot(ae, np.dot(C, ae)))
print("standard deviation of average: %g N/m**2" % sigma)

# +
gamma_v = [
    1.0e-26,
    1.0e-25,
    1.0e-24,
    1.0e-23,
    1.0e-22,
    1.0e-21,
    1.0e-20,
    1.0e-19,
    1.0e-18,
]
sigma_v = [
    2.0e14,
    3.2e13,
    1.39e13,
    1.15e13,
    6.44e12,
    1.11e12,
    5.11e11,
    1.99e11,
    8.34e10,
]
s_v = [3619.4, 4035.8, 4118.0, 4244.4, 6027.9, 11133.9, 13622.7, 16944.2, 21547.9]

plt.subplots(figsize=(12, 12))
plt.semilogy(s_v, sigma_v, "k")
plt.semilogy(s_v, sigma_v, "ko")
plt.grid()
plt.xlabel("averaging length [m]", labelpad=20)
plt.ylabel("standard deviation of average [N/m$^2$]")
plt.tight_layout()
plt.savefig("L.pdf", format="pdf")
plt.show()
# -

# Plot averaging kernel.
plt.subplots(figsize=(8, 12))
plt.plot(A, z, "--", color=[0.5, 0.5, 0.5], linewidth=2)
plt.plot(Ae, z, "k", linewidth=2)
plt.xlabel(r"$A$ [1/m]", labelpad=20)
plt.ylabel(r"$z$ [m]", labelpad=20)
plt.grid()
plt.ylim([-2.0 * H, H])
plt.tight_layout()
plt.savefig("Ae_e22.pdf", format="pdf")
plt.show()
