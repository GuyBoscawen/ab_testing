import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import norm, binom
from statsmodels.stats.weightstats import ttest_ind

#######################
## TUNING PARAMETERS ##
#######################

sample_mean = 12
sample_sd = 5
# significance
alpha = 0.05
target_power = 0.8
# The number of simulations
simulations = 500
alternative = "two-sided"

max_sample_size = 5000
sample_step_size = 10

#######################
## DATA ###############
#######################

MDEs = np.linspace(1.4, 1.03, 50)
# a list of our resulting sample sizes which achieve our 80% power target (for a given MDE)
optimal_sample_sizes = [None for _ in range(50)]

sample_sizes = range(0, max_sample_size, sample_step_size)
sample_data = norm.rvs(loc=sample_mean, scale=sample_sd, size=max_sample_size)

#######################
## SIMULATION #########
#######################

def find_sample_size(sample_sizes, MDE):
    for i in range(0, len(sample_sizes)):
        N = sample_sizes[i]

        control_data = sample_data[0:N]
        # Multiply the control data by the minimum detectable effect
        variant_data = control_data * MDE

        significance_results = []
        for j in range(0, simulations):
            # Randomly allocate the sample data to the control and variant
            rv = binom.rvs(1, 0.5, size=N)
            control_sample = control_data[rv == True]
            variant_sample = variant_data[rv == False]

            # Welch's t-test
            test_result = ttest_ind(
                control_sample,
                variant_sample,
                alternative=alternative,
                usevar='unequal'
            )
            # Test for significance
            significance_results.append(test_result[1] <= alpha)

        # The power is the number of times we have a significant result
        power = np.mean(significance_results)
        if power > target_power:
            return N

    # never reached power 0.8
    return sample_sizes[-1]


for i, MDE in enumerate(MDEs):
    N = find_sample_size(sample_sizes, MDE)
    optimal_sample_sizes[i] = N
    # create new sample size range starting at N
    sample_sizes = range(N, max_sample_size, sample_step_size)

#######################
## PLOTS ##############
#######################

plt.scatter(MDEs, optimal_sample_sizes)
plt.xlabel("MDE")
plt.ylabel("sample size")
plt.show()
