import numpy as np
import matplotlib.pyplot as plt


class BiVariateGaussian:

    def __init__(self, mean, cov):
        self.mu = np.array(mean)
        self.chol = np.linalg.cholesky(np.array(cov))

    def sample(self, size):
        return np.dot(self.chol, np.random.randn(2)) + self.mu


def uni_normal_pdf(x, mu, var):
    return np.power(2.0 * np.pi * var, -0.5) * np.exp(-0.5 * ((x - mu) * (x - mu) / var))


rho = 0.5
unknown_mean = [10.0, 20.0]
known_cov = [[1.0, rho], [rho, 1.0]]
underlying_dist = BiVariateGaussian(mean=unknown_mean, cov=known_cov)
total_period = 1000
burn_in_period = 500
# Metropolis Algorithm
mu = np.zeros(shape=(total_period, 2), dtype=np.float)
for t in range(1, total_period):

    y = underlying_dist.sample(1)
    mu_0_candidate = np.random.randn(1)*np.sqrt(1.0 - rho*rho) + y[0] + rho * (mu[t-1][1] - y[1])
    r = uni_normal_pdf(mu_0_candidate, mu=y[0] + rho * (mu[t-1][1] - y[1]), var=np.sqrt(1.0 - rho * rho)) /\
        uni_normal_pdf(mu[t-1][0],     mu=y[0] + rho * (mu[t-1][1] - y[1]), var=np.sqrt(1.0 - rho * rho))
    if np.random.rand() < min(r, 1):
        mu[t][0] = mu_0_candidate
    else:
        mu[t][0] = mu[t-1][0]

    mu_1_candidate = np.random.randn(1)*np.sqrt(1.0 - rho*rho) + y[1] + rho * (mu[t][0] - y[0])
    r = uni_normal_pdf(mu_1_candidate, mu=y[1] + rho * (mu[t][0] - y[0]), var=np.sqrt(1.0 - rho * rho)) /\
        uni_normal_pdf(mu[t-1][1],     mu=y[1] + rho * (mu[t][0] - y[0]), var=np.sqrt(1.0 - rho * rho))
    if np.random.rand() < min(r, 1):
        mu[t][1] = mu_1_candidate
    else:
        mu[t][1] = mu[t-1][1]

plt.scatter(mu[:, 0], mu[:, 1], s=1)
print('mu_0 : ', mu[burn_in_period:total_period, 0].mean(), 'mu_1 : ', mu[burn_in_period:total_period, 1].mean())
plt.show()