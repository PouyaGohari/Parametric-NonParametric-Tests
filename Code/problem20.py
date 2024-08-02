import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# np.random.seed(0)

def generating_sample_size(a, b, sample_size=50):
    return np.random.beta(a, b, sample_size)

def calculate_d_i(sample, hypothesis_median):
    negative, positive, zeros =0 ,0, 0
    for x in sample:
        if x < hypothesis_median:
            negative += 1
        elif x > hypothesis_median:
            positive += 1
        else:
            zeros += 1
    if zeros == 0:
        return positive, negative
    while(zeros != 0):
        if(positive > negative):
            negative += 1
        elif(positive < negative):
            positive += 1
        else:
            break
        zeros -= 1
    return positive, negative

def calculate_power(hypothesis_median, rejecting_alpha):
    number_simulations = 1000
    number_of_rejections = 0
    for _ in range(number_simulations):
        beta_dist = generating_sample_size(2,5)
        pos, _= calculate_d_i(beta_dist, hypothesis_median)
        p_values =stats.binomtest(pos, n=50, alternative='greater', p=0.5)
        if(p_values.pvalue < rejecting_alpha):
            number_of_rejections +=1
    
    return number_of_rejections/number_simulations
        


beta_dist = generating_sample_size(2, 5)
sns.histplot(beta_dist, alpha=0.7, bins= 20, kde=True)
plt.xlabel('sample')
plt.ylabel('probability')
plt.title('beta distributions')
plt.show()
pos, neg = calculate_d_i(beta_dist, 0.3)
print(np.median(beta_dist), pos)

p_value = stats.binomtest(pos, n=len(beta_dist), alternative='greater')
print(f'The p-value for a sample of size 50 is: {p_value}')

power = calculate_power(hypothesis_median=0.3, rejecting_alpha=0.05)
print(power)