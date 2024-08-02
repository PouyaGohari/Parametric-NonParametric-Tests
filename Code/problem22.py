import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

frequency = [7, 45, 181, 478, 829, 1112, 1343, 1033, 670, 286, 104, 24, 3]
frequency = np.array(frequency)

def calculate_expected(frequency):
    probabilities = np.zeros(shape=len(frequency))
    expected_freq = np.zeros(shape=len(frequency))
    for number_of_sons in range(len(frequency)):
        probabilities[number_of_sons] = stats.binom.pmf(number_of_sons, n=12, p=0.5)
        expected_freq[number_of_sons] = probabilities[number_of_sons] * np.sum(frequency)
    return expected_freq

def calculate_chi_square_score(frequency, expected_freq):
    return np.sum(((frequency-expected_freq)**2)/expected_freq)

expected_freq = calculate_expected(frequency)
print(f'The expected frequnecies: {expected_freq}')
chi_square_statistic, p_value = stats.chisquare(frequency, expected_freq)
print(f'From using built-in functions we have: {chi_square_statistic} for statistic, and p-value of chisquare test is: {p_value}')
chi_score =  calculate_chi_square_score(frequency, expected_freq)
ppf = stats.chi2.ppf(0.95, df=10)
print(f'From my code we calculated the statistic which is : {chi_score} and ppf is: {ppf}')
## part b

def simulations(number_simulations, expected_freq , number_families=6115, n=12, p=0.5):
    chi_statistics = np.zeros(number_simulations)
    for i in range(number_simulations):
        sample = np.random.binomial(n=n, p=p, size=number_families)
        simulated_freqs, _ = np.histogram(sample, bins=np.arange(-0.5, 13.5, 1), density=False)
        chi_statistics[i] = calculate_chi_square_score(simulated_freqs, expected_freq)
    return chi_statistics

chi_stats = simulations(5000, expected_freq)
sns.histplot(chi_stats, bins='auto', alpha=0.7)
plt.xlabel('Chi-squared statistics')
plt.ylabel('Frequency')
plt.title('Histogram of statistic')
# plt.show()

p_value = np.mean(chi_stats >= chi_score)
print(f'The p-value based on obtained distribution: {p_value}')