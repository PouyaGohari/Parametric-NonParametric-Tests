import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import math

sizes = 50
first_sample = np.random.normal(loc=0, scale=math.sqrt(3), size=sizes)
second_sample = np.random.normal(loc=0, scale=math.sqrt(10), size=sizes)
plt.figure(1)
sns.histplot(first_sample, bins=20, alpha=0.6, kde=True, stat='density')
plt.xlabel('sample 1')
plt.ylabel('probability')
plt.title('Generated 50 samples with variance 3')
plt.figure(2)
sns.histplot(second_sample, bins=20, alpha=0.6, kde=True, stat='density')
plt.xlabel('sample 2')
plt.ylabel('probability')
plt.title('Generated 50 samples with variance 10')
plt.show()

first_variance = np.var(first_sample, ddof=1)
second_variance = np.var(second_sample, ddof=1)

df1 = df2 = sizes-1

f_value = first_variance/second_variance

if first_variance > second_variance:
    p_value = 1 - stats.f.cdf(f_value, df1, df2)
else:
    p_value = stats.f.cdf(f_value,df2, df1)

print(df1, df2, f_value, p_value)

levene_stat, levene_pvalue = stats.levene(first_sample, second_sample)
print(f'for levene statistic: {levene_stat}, p_value : {levene_pvalue}')

bartlett_stat, bartlett_pvalue = stats.bartlett(first_sample, second_sample)
print(f'for bartlett statistic: {bartlett_stat}, p_value : {bartlett_pvalue}')