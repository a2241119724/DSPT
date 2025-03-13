from scipy import stats
import numpy as np

baseline_scores = np.array([72.36,96.51,96.75,73.67,83.2,90.61,36.55,78.32,69.34,83.29,81.54])
proposed_scores = np.array([73.67,96.58,96.76,74.1,83.26,91.48,37.09,80.22,67.9,86.68,82])

_, p_normal_baseline = stats.shapiro(baseline_scores)
_, p_normal_proposed = stats.shapiro(proposed_scores)

if p_normal_baseline > 0.05 and p_normal_proposed > 0.05:
    # 使用配对t检验
    t_stat, p_value = stats.ttest_rel(baseline_scores, proposed_scores)
else:
    # 使用非参数检验
    _, p_value = stats.wilcoxon(baseline_scores, proposed_scores)

# p < 0.05 表示差异显著
print(f"P-value: {p_value:.4f}")