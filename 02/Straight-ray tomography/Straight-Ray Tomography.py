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

# # Straight-ray tomography
#
# In the following notebook, we perform a simple straight-ray tomography in 2D. This serves to illustrate some of the basic concept of linear least-squares inversion, such as prior and posterior covariances, damping, smoothing, over- and under-fitting, and resolution.

# # 0. Import some Python packages
#
# We begin by importing some Python packages for matrix-vector operations, for plotting, and for computing the forward modelling matrix $\mathbf{G}$.

# +
# Some Python packages.

import numpy as np
import scipy
from scipy.sparse import linalg

import sys

sys.path.insert(0, "./utils")  # This contains functions to compute G.
from grid import *
from straight_ray_tracer import *

# Set some parameters to make plots nicer.

plt.rcParams["font.family"] = "serif"
plt.rcParams.update({"font.size": 35})
# -

# # 1. Basic input
#
# In the following few lines, we define the most basic geometric input, including the dimensions of the model domain, as well as the positions of sources and receivers. The initial setup mimics a cross-hole tomography where sources are on one side of the domain, and receivers are on the other one.

# +
# Define the numerical grid. ---------------------------------------------
dimension = 2  # Here we only consider 2D problems anyway.
x_min = 0.0  # Minimum x-coordinate
y_min = 0.0  # Minimum y-coordinate
dx = 2.5  # Grid spacing in x-direction
dy = 2.5  # Grid spacing in y-direction
Nx = 20.0  # Number of grid points in x-direction
Ny = 20.0  # Number of grid points in y-direction
g = grid(dimension, [x_min, y_min], [dx, dy], np.array([Nx, Ny]))

# Sources and receivers. -------------------------------------------------
src_locations = np.array([0.0 * np.ones((11,)), np.linspace(0, 50, 11)])
rec_locations = np.array([50.0 * np.ones((21,)), np.linspace(0, 50, 21)])

sources, receivers = get_all_to_all_locations(src_locations, rec_locations)
plot_rays(sources, receivers, g)
# -

# # 2. Compute forward matrix G
#
# Knowing source and receiver positions, and the setup of the domain, we can compute the forward modelling matrix **G** that connects a slowness model **m** to a synthetic data vector **d** via **d**=**Gm**. In addition to computing **G**, we also visualise the ray density and the entries of **G**. We will see that ray density is rather uneven, and that **G** is pretty sparse.

# +
# Compute G and measure how long that takes.
G = create_forward_operator(sources, receivers, g)

# Print some statistics of G.
print("Matrix shape:            ", G.shape)
print("Data points:             ", G.shape[0])
print("Unknowns in model space: ", G.shape[1])
print("Non-zero entries:        ", G.count_nonzero())
print(
    "Ratio of non-zeros: {:10.4f} %".format(
        100 * G.count_nonzero() / (G.shape[0] * G.shape[1])
    )
)

# Plot ray density and entries of G.
plot_ray_density(G, g)

# Plot non-zero matrix entries.
print("Sparsity pattern of the forward matrix:")
plt.figure(figsize=(15, 20))
plt.spy(G, markersize=2, color="k")
plt.gca().xaxis.tick_bottom()
plt.xlabel("model space index")
plt.ylabel("data space index")
plt.title(r"non-zero entries of $\mathbf{G}$")
plt.savefig("non-zeros.pdf", format="pdf")
plt.show()
# -

# # 3. Create target model
#
# Since we work with artifical and not with real traveltime data, we need to define a target model that we wish to reconstruct. In the basic setup, our target model is a simple chequerboard pattern of slowness values.

# +
# Input model setup (chequerboard). --------------------------------------
dvp = 100.0  # velcity variations in m/s.
dd = 4  # Width of the chequerboard cells in number of cells.

# Allocate velocity matrix. Homogeneous background model.
vp = 3000.0 * np.ones(g.npoints)

# Add some heterogeneities
s = 1.0
for i in range(0, g.npoints[0], dd):

    for j in range(0, g.npoints[1], dd):
        end_i = min(g.npoints[0], i + dd)
        end_j = min(g.npoints[1], j + dd)
        vp[i:end_i, j:end_j] += s * dvp
        s *= -1

m_true = (1 / vp).ravel()

clim = [1 / 3.1, 1 / 2.9]
plot_model(
    1000.0 * m_true, g, "true model [ms/m]", caxis=clim, savename="true_model.pdf"
)
# -

# # 4. Generate synthetic measurements
#
# Having defined the target model, it remains to compute artificial traveltime data that enter the observed data vector $\mathbf{d}^\text{obs}$. To be more realistic, we add random errors to the traveltime data. The amplitude of these errors are defined by the parameter *sigma_d*. Knowing *sigma_d*, we then compute the prior data covariance $\mathbf{C}_D$.
#
# In the real world, measurement errors will usually not do us the favour of being nicely normally distributed. Even if they were, it could be difficult to estimate the correct prior data covariance.

# +
# Create observed data ---------------------------------------------------
d_true = G * m_true

# Prior covariance parameters. -------------------------------------------
sigma_d = 0.2e-4  # Data standard deviation.
d_obs = d_true + sigma_d * np.random.randn(len(d_true))

# Data covariance matrix. ------------------------------------------------
Cd = sigma_d ** 2 * scipy.sparse.eye(len(d_obs))
Cd_inv = 1 / sigma_d ** 2 * scipy.sparse.eye(len(d_obs))

# Traveltimes.
plt.subplots(figsize=(15, 10))
plt.plot(1000.0 * d_obs, "k")
plt.ylabel("travel time [ms]")
plt.xlabel("ray path idx")
plt.show()

# Traveltime errors.
plt.subplots(figsize=(15, 10))
plt.plot(1000.0 * (d_obs - d_true), "k")
plt.ylabel("travel time errors [ms]")
plt.xlabel("ray path idx")
plt.show()
# -

# # 5. Compute prior model covariance
#
# The final step before solving the inverse problem is to define the prior mean model $\mathbf{m}^\text{prior}$, and the prior covariance in model space, $\mathbf{C}_M$. For the latter, we need to define the correlation length, contained in the parameter *corr_len*. To visualise the effect of $\mathbf{C}_M$, we can plot its action on the target model.

# +
# Prior model. -----------------------------------------------------------
m_prior = np.ones(m_true.shape) / 3000.0

# Prior covariance. ------------------------------------------------------
correlation_length = 3.0  # lambda
regularization_weight = 2.5e-5  # sigma_M

Cm = g.get_gaussian_prior(correlation_length)

plot_model(1000.0 * m_prior, g, "prior model [ms/m]", caxis=clim)
plot_model(1000.0 * Cm * m_true, g, "smoothed true model [ms/m]", caxis=clim)

Cm *= regularization_weight ** 2
Cm_inv = linalg.inv(Cm)
# -

# # 6. Solve inverse problem
#
# We are now equipped with all ingredients needed to solve the inverse problem. For this, we need to compute the inverse of the Hessian of the least-squares misfit functional, $\mathbf{C}_M^{-1}+\mathbf{G}^T \mathbf{C}_D^{-1} \mathbf{G}$, which is equal to the posterior covariance, $\mathbf{\tilde{C}}_M$.

# +
# Hessian ----------------------------------------------------------------
H = G.T * Cd_inv * G + Cm_inv

# Posterior covariance ---------------------------------------------------
Cm_post = scipy.sparse.linalg.inv(H)

# Posterior mean. --------------------------------------------------------
m_est = Cm_post * (G.T * Cd_inv * d_obs + Cm_inv * m_prior)
d_est = G * m_est
d_prior = G * m_prior

# Plot. ------------------------------------------------------------------
plot_model(
    1000.0 * m_est,
    g,
    "reconstructed slowness [ms/m]",
    caxis=clim,
    savename="reconstructed_model.pdf",
)
# -

# # 7. Data fit and posterior covariance
#
# To check for over- or under-fitting, we compute the rms misfit. For an assessment of model quality, we visualise the prior and posterior covariances.

# +
# Compute estimated data.
d_est = G * m_est

# Compute and print rms misfit.
N = len(d_est)
rms = np.sqrt(np.sum((d_est - d_obs) ** 2) / (N * sigma_d ** 2))
print("rms misfit %f" % rms)

# Plot traveltime residuals.
plt.figure(figsize=(15, 8))
plt.plot(1000.0 * (d_obs - d_est), "kx")
plt.plot([0.0, N], [1000.0 * sigma_d, 1000.0 * sigma_d], "--k")
plt.plot([0.0, N], [-1000.0 * sigma_d, -1000.0 * sigma_d], "--k")
plt.title("traveltime residuals [ms]", pad=20)
plt.xlim([0.0, N])
plt.savefig("residuals.pdf", format="pdf")
plt.show()

# Plot prior and posterior covariances.
Cmpd = Cm_post.todense()
Cmd = Cm.todense()

x = np.zeros(len(m_est))
xp = np.zeros(len(m_est))

for i in range(len(x)):
    x[i] = 1.0e9 * Cm[210, i]
    xp[i] = 1.0e9 * Cmpd[210, i]

plt.figure(figsize=(15, 8))
plt.plot(x, "--k")
plt.plot(xp, "k")
plt.xlim([150, 250])
plt.grid()
plt.title("covariance [ms**2/s**2]", pad=20)
plt.savefig("covariance.pdf", format="pdf")
# -

# # 8. Model resolution
#
# To formally assess the resolution of our model, we compute the resolution matrix and visualise some point-spread functions and averaging kernels.

# +
# Compute generalised inverse.
Ginv = Cm * G.T * scipy.sparse.linalg.inv(Cd + G * Cm * G.T)
# Compute model resolution matrix.
R = Ginv * G

# Plot model resolution matrix.
R = R.todense()
plt.figure(figsize=(25, 25))
plt.imshow(R, vmin=-0.6, vmax=0.6, cmap="Greys")
plt.colorbar(shrink=0.75)
plt.title("model resolution matrix", pad=20)
plt.tight_layout()
plt.savefig("resolution.pdf", format="pdf")
plt.show()

# Number of resolved parameters.
print("number of resolved parametrers: %f" % np.trace(R))

# +
# Extract and plot averaging kernels (rows of the resolution matrix).
row = 30
avk = R[row, :].reshape(int(Nx), int(Ny))

clim = [-0.25, 0.25]
plot_model(avk, g, "averaging kernel", caxis=clim, savename="avk.pdf")

# +
# Extract and plot point-spread functions (columns of the resolution matrix).
column = 315
psf = R[:, column].reshape(int(Nx), int(Ny))

clim = [-0.25, 0.25]
plot_model(psf, g, "point-spread function", caxis=clim, savename="psf.pdf")
# -

# # 9. Data resolution

# +
# Compute generalised inverse.
Ginv = Cm * G.T * scipy.sparse.linalg.inv(Cd + G * Cm * G.T)
# Compute model resolution matrix.
Rd = G * Ginv

# Plot data resolution matrix.
Rd = Rd.todense()
plt.figure(figsize=(25, 25))
plt.imshow(Rd, vmin=-0.6, vmax=0.6, cmap="Greys")
plt.colorbar(shrink=0.75)
plt.title("data resolution matrix", pad=20)
plt.tight_layout()
plt.savefig("resolution_data.pdf", format="pdf")
plt.show()

# Plot diagonal of data resolution matrix.
plt.figure(figsize=(15, 10))
plt.plot(np.diag(Rd), "k")
plt.title("diagonal of data resolution matrix")
plt.show()
# -

# # 10. Singular value analysis and nullspace

# Singular-value decomposition
# u,s,vt=scipy.sparse.linalg.svds(G,k=len(d_obs)-1)
u, s, vt = scipy.sparse.linalg.svds(G.T * G, k=399)


# Plot singular vectors.
k = 2
scale = np.max(np.abs(vt[k, :]))
clim = [-scale, scale]
plot_model(vt[k, :], g, "eigenmodel", caxis=clim, savename="eigenmodel.pdf")

# # Exercises
#
# **Exercise 1**: Plot the difference between (artificial) observed data and data estimated from the posterior mean model. Are the observations explained to within their uncertainties? Are the data over- or under-fit?
#
# **Exercise 2**: Plot the posterior covariance for a model parameter near the centre and a model parameter near the edge of the domain. How do the posterior variances and covariances compare? Is the result in accord with your intuitive expectations based on the ray coverage plot?
#
# **Exercise 3**: Compute and plot an L-curve for the smoothing parameter (correlation length). Choose an optimal smoothing based on the L-curve.
#
# **Exercise 4**: Compute and visualise the resolution matix.
