import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2

# np.random.seed(10)
numbers = np.random.uniform(0, 100, 300)

group_dict = {
    "group1_1": 0,
    "group1_2": 0,
    "group1_3": 0,
    "group2_1": 0,
    "group2_2": 0,
    "group2_3": 0,
    "group3_1": 0,
    "group3_2": 0,
    "group3_3": 0
}

for x in numbers:
    if x < 15:
        group_dict['group1_1'] += 1
    elif x < 24 and x >= 15:
        group_dict['group1_2'] += 1
    elif x < 30 and x >= 24:
        group_dict["group1_3"] += 1
    elif x < 45 and x >= 30:
        group_dict["group2_1"] += 1
    elif x < 54 and x >= 45:
        group_dict["group2_2"] += 1
    elif x < 60 and x >= 54:
        group_dict["group2_3"] += 1
    elif x < 80 and x >= 60:
        group_dict["group3_1"] += 1
    elif x < 92 and x >= 80:
        group_dict["group3_2"] += 1
    else:
        group_dict["group3_3"] += 1

data = {
    'gp1': [49,22,17],
    'gp2': [51,25,26],
    'gp3': [52,42,16]
}

data = [data["gp1"],data["gp2"], data["gp3"]]
data = np.array(data)
expected = [[0.15, 0.09, 0.06],
            [0.15, 0.09, 0.06],
            [0.2, 0.12, 0.08]]
expected = np.array(expected)*300

print(f'The chi-score is : {(((data-expected)**2)/expected).sum()}')

dof = 4
print(f'The p-value is : {1-chi2.cdf((((data-expected)**2)/expected).sum(), dof)}')

## part 4 optional

def generating_nums():
    my_numb = [[0 for i in range(3)] for i in range(3)]
    numbers = np.random.uniform(0, 100, 300)
    for x in numbers:
        if x < 15:
            my_numb[0][0] += 1
        elif x < 24 and x >= 15:
            my_numb[0][1] += 1
        elif x < 30 and x >= 24:
            my_numb[0][2] += 1
        elif x < 45 and x >= 30:
            my_numb[1][0] += 1
        elif x < 54 and x >= 45:
            my_numb[1][1] += 1
        elif x < 60 and x >= 54:
            my_numb[1][2] += 1
        elif x < 80 and x >= 60:
            my_numb[2][0] += 1
        elif x < 92 and x >= 80:
            my_numb[2][1] += 1
        else:
            my_numb[2][2] += 1
    return my_numb

def statistic_of_chi(number_of_contingency):
    chi_stat = np.zeros(shape=number_of_contingency)
    expected = [[0.15, 0.09, 0.06],
                [0.15, 0.09, 0.06],
                [0.2, 0.12, 0.08]]
    expected = np.array(expected)*300
    for i in range(number_of_contingency):
        generated_numbers = np.array(generating_nums())
        chi_stat[i] = ((generated_numbers-expected)**2/expected).sum()
    return chi_stat

chi_stats = statistic_of_chi(10000)
sns.histplot(chi_stats, bins='auto', alpha=0.7, kde=True, stat='density')
plt.xlabel('chi-statistic')
plt.ylabel('probability')
plt.title('Histogram for chi-square obtained form each simulation')
plt.show()

chi_score_from_prev = 10.574074074074073
p_value = np.mean(chi_stats >= chi_score_from_prev)
print(f'The obtained p-value from 10000 simulation for calculated chi score in previous part is: {p_value}')