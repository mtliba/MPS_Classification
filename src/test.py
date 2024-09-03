# src/test_anova.py
import numpy as np
from scipy.stats import f_oneway

# Load the metrics collected during training
metrics = np.load('results/metrics_for_anova.npy', allow_pickle=True).item()

# Prepare data for ANOVA
model_names = list(metrics.keys())
losses = [metrics[model_name]['loss'] for model_name in model_names]
accuracies = [metrics[model_name]['accuracy'] for model_name in model_names]
precisions = [metrics[model_name]['precision'] for model_name in model_names]
recalls = [metrics[model_name]['recall'] for model_name in model_names]
f1_scores = [metrics[model_name]['f1'] for model_name in model_names]

# Perform ANOVA for each metric
print("ANOVA Test Results:")
print("Validation Loss:")
f_stat, p_val = f_oneway(*losses)
print(f"F-statistic: {f_stat:.4f}, p-value: {p_val:.4f}")

print("Validation Accuracy:")
f_stat, p_val = f_oneway(*accuracies)
print(f"F-statistic: {f_stat:.4f}, p-value: {p_val:.4f}")

print("Validation Precision:")
f_stat, p_val = f_oneway(*precisions)
print(f"F-statistic: {f_stat:.4f}, p-value: {p_val:.4f}")

print("Validation Recall:")
f_stat, p_val = f_oneway(*recalls)
print(f"F-statistic: {f_stat:.4f}, p-value: {p_val:.4f}")

print("Validation F1 Score:")
f_stat, p_val = f_oneway(*f1_scores)
print(f"F-statistic: {f_stat:.4f}, p-value: {p_val:.4f}")
