import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic
from scipy.stats import chi2_contingency, fisher_exact

## part 1 a)
path = "HW#3/titanic.csv"
df = pd.read_csv(path)

## part 1 b)
contingency_table = pd.crosstab(df['sex'], df['survive'], margins=True)

## part 1 c)
contingency_table.to_csv('Code/sex_survive_table.csv')

## part 1 d)
mosaic(df, ['sex', 'survive'], properties=lambda key: {'color': 'skyblue' if 'male' and 'yes' in key else 'salmon'}, labelizer=lambda k:'')
plt.title('mosaic plot of survive and sex')

## part 1 e)
print(contingency_table)

## part 1 f)
mosaic(df, ['sex', 'survive'])
plt.title('detailed mosiac plot')
plt.show()

## part 2 a)
chi2, p, dof, expected = chi2_contingency(pd.crosstab(df['sex'], df['survive']))
print(f'chi-square statistic:{chi2}')
print(f'P-value: {p}')
print(f'Degree of freedom: {dof}')
print(f'Expected frequencies: {expected}')

## part 2 c)
odds_ratio, p_value = fisher_exact(pd.crosstab(df['sex'], df['survive']))
print(f'Odd ratio is: {odds_ratio}')
print(f'P-value is: {p_value}')