import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import poisson, nbinom


# Choose one distribution for the Panjer recursion test:
#   "poisson" or "negative_binomial"
DISTRIBUTION = "poisson"


def panjer_pmf(distribution, n_max):
    """Compute pmf recursively using Panjer's recursion.

    For a counting distribution with probability mass function f_n,
    Panjer recursion is of the form:
        f_n = (a + b/n) f_{n-1},   n >= 1

    Parameters
    ----------
    distribution : str
        Either "poisson" or "negative_binomial".
    n_max : int
        Largest index n to compute.
    """
    if distribution == "poisson":
        lam = 3.0
        a = 0.0
        b = lam
        f0 = np.exp(-lam)
    elif distribution == "negative_binomial":
        r = 5
        p = 0.6
        a = 1.0 - p
        b = (r - 1.0) * (1.0 - p)
        f0 = p ** r
    else:
        raise ValueError("distribution must be 'poisson' or 'negative_binomial'")

    pmf = np.zeros(n_max + 1)
    pmf[0] = f0

    for n in range(1, n_max + 1):
        pmf[n] = (a + b / n) * pmf[n - 1]

    return pmf


def exact_pmf(distribution, n_max):
    if distribution == "poisson":
        lam = 3.0
        return poisson.pmf(np.arange(n_max + 1), lam)
    elif distribution == "negative_binomial":
        r = 5
        p = 0.6
        return nbinom.pmf(np.arange(n_max + 1), r, p)
    else:
        raise ValueError("distribution must be 'poisson' or 'negative_binomial'")


if __name__ == "__main__":
    n_max = 20
    recursive = panjer_pmf(DISTRIBUTION, n_max)
    exact = exact_pmf(DISTRIBUTION, n_max)
    x = np.arange(n_max + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(x, exact, "o-", label="Exact PMF", linewidth=2, markersize=6)
    plt.plot(x, recursive, "s--", label="Panjer recursion", linewidth=2, markersize=5)
    plt.xlabel("n")
    plt.ylabel("Probability")

    if DISTRIBUTION == "poisson":
        title = "Poisson distribution (λ = 3): exact PMF vs Panjer recursion"
    else:
        title = "Negative binomial distribution (r = 5, p = 0.6): exact PMF vs Panjer recursion"

    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("panjer_recursion_plot.png", dpi=200)
    plt.show()

    print(f"Distribution: {DISTRIBUTION}")
    print("n    Exact PMF      Panjer recursive PMF")
    for n in range(n_max + 1):
        print(f"{n:2d}  {exact[n]:.10f}  {recursive[n]:.10f}")

    max_abs_error = np.max(np.abs(exact - recursive))
    print(f"\nMaximum absolute error: {max_abs_error:.12e}")
