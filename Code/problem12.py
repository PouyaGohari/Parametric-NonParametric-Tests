import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

x = np.zeros(shape=101)
y = np.zeros(shape=101)
z = np.zeros(shape=101)
for i in range(0, len(x)):
    y[i] = stats.binom.pmf(i, n=100, p=0.5)
    z[i] = stats.binom.pmf(i, n=100, p=0.7)
    x[i] = i
 
fig, axs = plt.subplots(2, 1, figsize=(14, 14))
axs[0].bar(x, y, color=['red' if (i < 40 or i > 60) else 'blue' for i in x])
axs[0].set_title('X ~ Binomial(100, 0.5)')
axs[0].set_xlabel('Number of successes')
axs[0].set_ylabel('Probability')
axs[1].bar(x, z, color=['green' if (40 <= i and i <= 60) else 'blue' for i in x])
axs[1].set_title('X ~ Binomial(100, 0.7)')
axs[1].set_xlabel('Number of successes')
axs[1].set_ylabel('Probability')
plt.show()

Likelyhood = stats.binom.cdf(59, n=100, p=0.5) -stats.binom.cdf(40, n=100, p=0.5)
type_two_error = stats.binom.cdf(60, n=100, p=0.7) 
power_test = 1 - type_two_error
print(f'Likelihood of adopting the null hypothesis: {Likelyhood}')
print(f'Power of test is: {power_test}')