import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

mens_age = [52, 18, 27, 12, 24, 17, 68, 25, 12, 9, 51, 44,
42, 34, 44, 15, 21, 66, 61, 32, 31, 20, 6, 13, 34, 38, 45, 17,
16, 15, 36, 21, 29, 21, 29, 9, 33, 15, 37, 27, 31, 15, 57, 37,
27, 31, 38, 27, 60, 23]

womens_age = [36, 49, 20, 31, 51, 31, 15, 16, 39, 70, 52,
16, 39, 34, 18, 34, 30, 18, 26, 18, 25, 16, 39, 49, 22, 37, 39,
21, 16, 63, 45, 43, 17, 28, 29, 23, 42, 23, 28, 55, 41, 18, 23,
8, 13, 26, 13, 27, 28, 18]

mens_age = np.array(mens_age)
womens_age = np.array(womens_age)

print(len(mens_age), len(womens_age))

# part 1

fig_count = 0
plt.figure(fig_count)
fig_count += 1
sns.histplot(mens_age, kde=True, alpha=0.7, bins=20)
plt.xlabel('age of mens')
plt.ylabel('frequencies')

plt.figure(fig_count)
fig_count += 1
sns.histplot(womens_age, kde=True, alpha=0.7, bins=20)
plt.xlabel('age of womens_age')
plt.ylabel('frequencies')
# plt.show()


# part 2
alpha = 0.05
paramaters = 2
bins = 20
df = len(mens_age)-paramaters-1

men_counts, edges_mens = np.histogram(mens_age, bins=bins)
womens_counts, edges_womens = np.histogram(womens_age, bins=bins)

mean_of_men, std_of_men = mens_age.mean(), mens_age.std()
mean_of_wemon, std_of_wemon = womens_age.mean(), womens_age.std()

print(f'mean of mens and womens ages: {mean_of_men}, {mean_of_wemon}')
print(f'std of mens and womens ages: {std_of_men}, {std_of_wemon}')

def probability_calc(edges, mean, sigma, bins=20):
    probability = np.zeros(shape=bins)
    for i in range(bins):
        probability[i] = stats.norm.cdf(edges[i+1], mean, sigma) - stats.norm.cdf(edges[i], mean, sigma)
    return probability

def expected_mean(counts, probability, bins=20):
    means = np.zeros(shape=bins)
    for i in range(bins):
        means[i] = counts * probability[i]
    return means

def chi_square_calc(frequencies, expected_mean):
    return (((frequencies-expected_mean)**2)/expected_mean).sum()

probabilities_men = probability_calc(edges_mens, mean_of_men, std_of_men, bins)
probabilities_women = probability_calc(edges_womens, mean_of_wemon, std_of_wemon, bins)

means_men = expected_mean(men_counts.sum(), probabilities_men, bins)
means_women = expected_mean(womens_counts.sum(), probabilities_women, bins)

chi_square_men = chi_square_calc(men_counts, means_men)
chi_square_women = chi_square_calc(womens_counts, means_women)

print(f'calculated chi square for men and women respectively: {chi_square_men}, {chi_square_women}')
critical_value = stats.chi2.ppf(1-alpha, df)
print(critical_value)
z_score = stats.norm.ppf(1-alpha,0,1)
print(z_score)




## using shapiro test for normality
mens_age = [52, 18, 27, 12, 24, 17, 68, 25, 12, 9, 51, 44,
42, 34, 44, 15, 21, 66, 61, 32, 31, 20, 6, 13, 34, 38, 45, 17,
16, 15, 36, 21, 29, 21, 29, 9, 33, 15, 37, 27, 31, 15, 57, 37,
27, 31, 38, 27, 60, 23]

womens_age = [36, 49, 20, 31, 51, 31, 15, 16, 39, 70, 52,
16, 39, 34, 18, 34, 30, 18, 26, 18, 25, 16, 39, 49, 22, 37, 39,
21, 16, 63, 45, 43, 17, 28, 29, 23, 42, 23, 28, 55, 41, 18, 23,
8, 13, 26, 13, 27, 28, 18]

mens_age = np.array(mens_age)
womens_age = np.array(womens_age)

stat, p_value = stats.shapiro(mens_age)
print(f'statistic and p_value for men\'s age are {stat},{p_value}')
stat, p_value = stats.shapiro(womens_age)
print(f'statistic and p_value for women\'s age are {stat},{p_value}')

mens_log_data = np.log(mens_age)
mens_sqrt_data = np.sqrt(mens_age)
mens_boxcox_data, _ = stats.boxcox(mens_age)
mens_yeojohnson_data, _= stats.yeojohnson(mens_age)

data_dict = {
    'mens log data':mens_log_data,
    'mens sqrt data':mens_sqrt_data,
    'mens boxcox data':mens_boxcox_data,
    'mens yeojohnson data':mens_yeojohnson_data
}

fig_count = 2
for item,data in data_dict.items():
    stat, p_value = stats.shapiro(data)
    plt.figure(fig_count)
    sns.histplot(data, alpha=0.7, kde=True)
    plt.xlabel(f'mens ages')
    plt.ylabel(f'{item}')
    plt.title(f'histogram of {item}')
    print(f'for {item} the p-value is: {p_value}')
    fig_count += 1

womens_log_data = np.log(womens_age)
womens_sqrt_data = np.sqrt(womens_age)
womens_boxcox_data, _ = stats.boxcox(womens_age)
womens_yeojohnson_data, _= stats.yeojohnson(womens_age)

data_dict = {
    'womens log data':womens_log_data,
    'womens sqrt data':womens_sqrt_data,
    'womens boxcox data':womens_boxcox_data,
    'womens yeojohnson data':womens_yeojohnson_data
}

for item,data in data_dict.items():
    stat, p_value = stats.shapiro(data)
    plt.figure(fig_count)
    sns.histplot(data, alpha=0.7, kde=True)
    plt.xlabel(f'womens ages')
    plt.ylabel(f'{item}')
    plt.title(f'histogram of {item}')
    print(f'for {item} the p-value is: {p_value}')
    fig_count += 1

# plt.show()
    

mean_of_men, var_of_men = mens_boxcox_data.mean(), mens_boxcox_data.var()
mean_of_wemon, var_of_wemon = womens_boxcox_data.mean(), womens_boxcox_data.var()

print(f'mean of mens and womens ages(boxcox): {mean_of_men}, {mean_of_wemon}')
print(f'std of mens and womens ages(boxcox): {var_of_men}, {var_of_wemon}')

# part 6
mens_age = [52, 18, 27, 12, 24, 17, 68, 25, 12, 9, 51, 44,
42, 34, 44, 15, 21, 66, 61, 32, 31, 20, 6, 13, 34, 38, 45, 17,
16, 15, 36, 21, 29, 21, 29, 9, 33, 15, 37, 27, 31, 15, 57, 37,
27, 31, 38, 27, 60, 23]

womens_age = [36, 49, 20, 31, 51, 31, 15, 16, 39, 70, 52,
16, 39, 34, 18, 34, 30, 18, 26, 18, 25, 16, 39, 49, 22, 37, 39,
21, 16, 63, 45, 43, 17, 28, 29, 23, 42, 23, 28, 55, 41, 18, 23,
8, 13, 26, 13, 27, 28, 18]
class struct:
    def __init__(self, age, which_sample):
        self.age = age
        self.which_sample = which_sample
        self.rank = None
    def assign_rank(self, rank):
        self.rank = rank

merged_group = []
for i in range(50):
    merged_group.append(struct(mens_age[i],'men'))
for i in range(50):
    merged_group.append(struct(womens_age[i],'women'))

sorting_struct = sorted(merged_group, key=lambda y: y.age)
for i in range(len(sorting_struct)):
    sorting_struct[i].assign_rank(i+1)

ages = np.zeros(shape=100)

for i in range(100):
    ages[i] = sorting_struct[i].age

unique_values, counts= np.unique(ages, return_counts=True)

for i in range(len(unique_values)):
    if(counts[i] == 1):
        continue
    indicies = []
    for j in range(len(sorting_struct)):
        if(sorting_struct[j].age == unique_values[i]):
            indicies.append(j)
    r = 0
    for k in range(len(indicies)):
        r += sorting_struct[indicies[k]].rank
    for k in range(len(indicies)):
        sorting_struct[indicies[k]].assign_rank(r/len(indicies))

pos_w, neg_w = 0, 0
for i in range(100):
    if(sorting_struct[i].which_sample == 'men'):
        pos_w += sorting_struct[i].rank
    else:
        neg_w += sorting_struct[i].rank

print(pos_w, neg_w)