import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 自定義檢定函數 ==========

def f_test_variances(x1, x2, sides, alpha):
    a1 = np.array(x1)
    a2 = np.array(x2)
    result = np.full((7, 2), None, dtype=float)

    result[0] = [np.mean(a1), np.mean(a2)]
    result[1] = [np.std(a1, ddof=1), np.std(a2, ddof=1)]
    result[2] = [a1.size, a2.size]
    dfn, dfd = a1.size - 1, a2.size - 1
    result[3] = [dfn, dfd]

    f_stat = np.var(a1, ddof=1) / np.var(a2, ddof=1)
    result[4, 0] = f_stat

    if sides > 1:
        result[5] = [stats.f.isf(alpha / 2, dfn, dfd), stats.f.ppf(alpha / 2, dfn, dfd)]
    else:
        result[5] = [stats.f.isf(alpha, dfn, dfd), stats.f.ppf(alpha, dfn, dfd)]

    if f_stat > 1:
        p_value = 1 - stats.f.cdf(f_stat, dfn, dfd)
    else:
        p_value = stats.f.cdf(f_stat, dfn, dfd)
    if sides > 1:
        p_value *= 2
    result[6, 0] = p_value

    labels = ['Mean', 'Std Dev', 'Size', 'Degrees of Freedom', 'F-statistic', 'F-critical', 'p-value']
    return pd.DataFrame(result, index=labels)


def t_test_two_means_unequal_variance(x1, x2, sides, alpha, H0_diff):
    a1 = np.array(x1)
    a2 = np.array(x2)
    result = np.full((8, 2), None, dtype=float)

    mean1, mean2 = np.mean(a1), np.mean(a2)
    var1, var2 = np.var(a1, ddof=1), np.var(a2, ddof=1)
    n1, n2 = a1.size, a2.size

    result[0] = [mean1, mean2]
    result[1] = [var1, var2]
    result[2] = [n1, n2]

    mean_diff = mean1 - mean2
    diff_var = var1 / n1 + var2 / n2
    result[3, 0] = H0_diff

    numerator = diff_var ** 2
    denominator = (var1**2 / (n1**2 * (n1 - 1))) + (var2**2 / (n2**2 * (n2 - 1)))
    df = numerator / denominator
    result[4, 0] = df

    t_stat = (mean_diff - H0_diff) / np.sqrt(diff_var)
    result[5, 0] = t_stat

    if sides > 1:
        result[6] = [stats.t.isf(alpha / 2, df), stats.t.ppf(alpha / 2, df)]
    else:
        result[6] = [stats.t.isf(alpha, df), stats.t.ppf(alpha, df)]

    if t_stat > 0:
        p_value = 1 - stats.t.cdf(t_stat, df)
    else:
        p_value = stats.t.cdf(t_stat, df)
    if sides > 1:
        p_value *= 2
    result[7, 0] = p_value

    labels = [
        'Mean', 'Variance', 'Size', 'H0 Diff',
        'Degrees of Freedom', 't-statistic', 't-critical', 'p-value'
    ]
    return pd.DataFrame(result, index=labels)


def run_t_test_summary(df, col1, col2, usevar='pooled', alpha=0.05):
    group1 = sms.DescrStatsW(df[col1].values)
    group2 = sms.DescrStatsW(df[col2].values)
    t_test = sms.CompareMeans(group1, group2)
    return t_test.summary(usevar=usevar, alpha=alpha)

# ========== 開始作檢定==========

df = pd.read_csv('movie_summary.csv')

# filter data
filtered_df = df[
    (df['production_year'] > 2000) &
    (df['production_budget'] < 20000000) &
    df['sequel'].notna() &
    df['domestic_box_office'].notna() &
    df['international_box_office'].notna()
].copy()
# 計算總票房（可重複確認）
filtered_df['total_box_office'] = (
    filtered_df['domestic_box_office'] + filtered_df['international_box_office']
)

print(filtered_df['sequel'].value_counts())

plt.figure(figsize=(12, 6))
sns.stripplot(data=filtered_df, x='sequel', y='production_budget', jitter=True)
plt.yscale('log')
plt.title("Production Budget by Sequel")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.stripplot(data=filtered_df, x='sequel', y='total_box_office', jitter=True)
plt.yscale('log')
plt.title("Total Box Office by Sequel")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 分組
x0 = filtered_df[filtered_df['sequel'] == 0]['total_box_office']
x1 = filtered_df[filtered_df['sequel'] == 1]['total_box_office']

# 顯示基本描述統計
print("Sequel = 0:")
print(x0.describe())
print("\nSequel = 1:")
print(x1.describe())

# === 1. F 檢定：變異數是否相等 ===
print("\n=== F Test: Equality of Variances ===")
f_result = f_test_variances(x0, x1, sides=2, alpha=0.05)
print(f_result.round(4))

# === 2. t 檢定（Welch's）：平均數是否相等 ===
print("\n=== Welch's t-test: Equality of Means ===")
t_result = t_test_two_means_unequal_variance(x0, x1, sides=2, alpha=0.05, H0_diff=0)
print(t_result.round(4))

# === 3. t 檢定摘要報告：使用 statsmodels ===
print("\n=== Summary Report using statsmodels ===")
group1 = sms.DescrStatsW(x0.values)
group2 = sms.DescrStatsW(x1.values)
t_test = sms.CompareMeans(group1, group2)
print(t_test.summary(usevar='unequal', alpha=0.05))