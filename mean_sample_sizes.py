import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import norm, binom
from statsmodels.stats.weightstats import ttest_ind

#######################
## TUNING PARAMETERS ##
#######################

# significance
alpha = 0.05
target_power = 0.8
# The number of simulations
simulations = 500
# The minimum detectable effect
MDE = 1.2
alternative = "two-sided"

max_sample_size = 1000
sample_step_size = 10

#######################
## DATA ###############
#######################

sample_means = np.linspace(5, 50, 20)
sample_sds = np.linspace(2, 20, 20)
MEANS, SDS = np.meshgrid(sample_means, sample_sds)

# a grid of our resulting sample sizes which achieve our 80% power target (for a given mean and sd)
optimal_sample_sizes = [[None for _ in MEANS] for _ in MEANS[0]]

sample_sizes = range(0, max_sample_size + 1, sample_step_size)

#######################
## SIMULATION #########
#######################

def find_sample_size(sample_sizes, mean, sd):
    for i in range(0, len(sample_sizes)):
        N = sample_sizes[i]

        # create our control data from our normal distribution
        control_data = norm.rvs(loc=mean, scale=sd, size=N)
        # Multiply the control data by the relative effect
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


for row in range(len(MEANS)):
    for column in range(len(MEANS[0])):
        N = find_sample_size(sample_sizes, MEANS[row][column], SDS[row][column])
        optimal_sample_sizes[row][column] = N

#######################
## PLOTS ##############
#######################

plt.imshow(optimal_sample_sizes, extent = [5 , 50, 20 , 2])
plt.ylabel('$\sigma$')
plt.xlabel('$\mu$')
plt.show()
