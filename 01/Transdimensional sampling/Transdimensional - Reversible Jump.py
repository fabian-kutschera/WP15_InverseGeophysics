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

# # Transdimensional sampling using the reversible-jump algorithm
#
# This notebook implements the birth-death variant of the reversible-jump algorithm. The goal is to estimate the coefficients of a polynomial of unknown degree.

# ## 0. Import some Python packages
#
# We begin by importing some Python packages for random numbers and for plotting.

# +
# Some Python packages.

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Set some parameters to make plots nicer.

plt.rcParams["font.family"] = "serif"
plt.rcParams.update({"font.size": 20})


# -

# ## 1. Artificial data
#
# We will solve a synthetic inverse problem where we try to find the coefficients of a polynomial. For this, we compute artificial data using a polynomial of some degree $N_m-1$, where $N_m$ is the dimension of the model space.
#
# In a first step, we define (also for later convenience) a forward model function. In the second step, this is used to actually compute some artificial data that are polluted by normally distributed random errors.

# +
# Forward problem function. -------------------------------------------------


def forward(m, x, Nm):
    """
    Definition of the forward problem, which is a polynomial of degree Nm-1.

       y= m[0] + m[1]*x + m[2]*x**2.0 + ... + m[Nm-1]*x**(Nm-1) .

    :param m: Coefficients of the polynomial. Numpy array of dimension Nm.
    :param x: Scalar argument of the polynomial.
    :param Nm: Model space dimension.
    :return: Value of the polynomial at x.
    """

    d = 0.0

    for k in range(Nm):
        d += m[k] * (x ** (k))

    return d


# Input parameters for computation of artificial data. ----------------------

# Measurement locations.
x = np.arange(0.0, 11.0, 1.0)

# Model parameters and model space dimension.
m = np.array([1.0, 1.0])
Nm = len(m)

# Standard deviation of the Gaussian errors.
sigma = 2.0

# Fixed random seed to make examples reproducible.
np.random.seed(3)

# Compute artificial data. --------------------------------------------------
d = forward(m, x, Nm) + sigma * np.random.randn(len(x))

# Plot data. ----------------------------------------------------------------

# Plot with errorbars.
plt.plot(x, d, "ko")
plt.errorbar(x, d, yerr=sigma, xerr=0.0, ecolor="k", ls="none")

# Superimpose regression polynomials up to some degree.
for n in range(5):
    z = np.polyfit(x, d, n)
    p = np.poly1d(z)
    d_fit = p(x)
    plt.plot(x, d_fit)

plt.xlabel("x")
plt.ylabel("d")
plt.show()
# -

# ## 2. Sampling with the birth-death algorithm
#
# The following lines implement the actual sampler. We start with some basic input, including the estimated data variance, the maximum dimension, the allowable range of model parameter variations, and the total number of samples.
#
# In the sampling loop, we first compute the dimension of the next proposed samples. This determines the ratio $d_i/b_j$ or $b_i/d_j$ in the Metropolis rule. Finally, the forward problem for the proposed model is solved, and the result is used to evaluate the Metropolis rule.

# +
# Input parameters. ---------------------------------------------------------

# Estimated standard deviation of the data errors.
sigma = 2.0

# Maximum allowable dimension of the model space.
N_max = 4

# Allowable range of the model parameters around 0.
m_range = 5.0

# Total number of samples in each fixed-dimensional sampler.
Nsamples = 400000


# Initialisation. -----------------------------------------------------------

# Allocate empty vectors to collect samples. The first component is reserved
# for the model space dimension, the remaining components take the model parameters.
samples = np.zeros((N_max + 1, Nsamples))

# Compute initial misfit.
N_current = round(np.random.rand() * (N_max - 1)) + 1
m_current = 2.0 * m_range * (np.random.rand(N_current) - 0.5)
d_current = forward(m_current, x, N_current)
x_current = np.sum(((d - d_current) ** 2.0) / (2.0 * sigma ** 2.0))

# Assign first sample.
x_min = x_current
samples[0, 0] = float(N_current)
samples[1 : N_current + 1, 0] = m_current


# Sampling. -----------------------------------------------------------------

for k in range(1, Nsamples):

    # Randomly generate a new model space dimension.
    if (N_current < N_max) & (N_current > 1):  # Either birth or death.
        N_test = N_current + 2 * np.random.randint(0, 2) - 1
    elif N_current == 1:  # Birth.
        N_test = N_current + 1
    elif N_current == N_max:  # Death.
        N_test = N_current - 1

    # Compute birth-death factor for the Metropolis rule.
    if N_test > N_current:
        if N_test == N_max:
            di = 1.0
            bj = 0.5
        else:
            di = 0.5
            bj = 0.5
        occam = di / bj
    else:
        if N_test == 1:
            bi = 1.0
            dj = 0.5
        else:
            bi = 0.5
            dj = 0.5
        occam = bi / dj

    # Test sample and misfit.
    m_test = 2.0 * m_range * (np.random.rand(N_test) - 0.5)
    d_test = forward(m_test, x, N_test)
    x_test = np.sum(((d - d_test) ** 2.0) / (2.0 * sigma ** 2.0))

    # Metropolis rule (in logarithmic form, to avoid exponential overflow).
    p = np.minimum(0.0, np.log(occam) + (-x_test + x_current))
    if p >= np.log(np.random.rand(1)):
        N_current = N_test
        m_current = m_test
        x_current = x_test

    samples[0, k] = float(N_current)
    samples[1 : N_current + 1, k] = m_current
# -

# ## 3. Plotting results
#
# In the next few lines, we plot posterior marginals for the model space dimension and for selected model parameters.

# +
# Plot marginal for selected model parameters.

dimension = 2
parameter = 1  # Must range between 1 and dimension.

idx = np.where(samples[0, :] == dimension)
plt.hist(samples[parameter, idx].flatten(), bins=10, color="k", density=True)
plt.xlabel("m" + str(dimension))
plt.ylabel("posterior marginal")
plt.show()

# Plot marginal for dimension.
plt.hist(samples[0, :], bins=15, color="k", density=True)
plt.yscale("log", nonposy="clip")
plt.xlabel("dimension")
plt.ylabel("posterior for dimension")
plt.show()
