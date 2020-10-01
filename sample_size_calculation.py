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
# The number of simulations
simulations = 500
# The minimum detectable effect
MDE = 1.5
alternative = "two-sided"

max_sample_size = 1000
sample_step_size = 10

#######################
## DATA ###############
#######################

sample_data = norm.rvs(loc=sample_mean, scale=sample_sd, size=max_sample_size)
# range of sample sizes that we will use to run the simulations
sample_sizes = range(0, max_sample_size + 1, sample_step_size)

#######################
## SIMULATION #########
#######################

power_dist = np.empty((len(sample_sizes), 2))
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
    power_dist[i,] = [N, np.mean(significance_results)]

#######################
## PLOTS ##############
#######################

plt.scatter(power_dist[:, 0], power_dist[:, 1])
plt.hlines(0.8, 0, max_sample_size, linestyle='--')

plt.xlim((0, max_sample_size))
plt.ylim((0, 1))

plt.xlabel("Sample Size")
plt.ylabel("Power")
plt.show()
