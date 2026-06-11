"""
PulseAI - Phase 1 Results Visualization
Generate comprehensive visualization of all Phase 1 results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))
fig.suptitle('PulseAI Phase 1 - Complete Results Dashboard', fontsize=20, fontweight='bold', y=0.98)

# ========================
# 1. Progress Timeline
# ========================
ax1 = plt.subplot(2, 3, 1)
phases = ['Baseline\n(43%)', 'Augmentation\n(53.38%)', 'Traditional ML\n(52.26%)', 
          'Deep Learning\n(50.25%)', 'Ensemble\n(52.76%)', 'Optimized\n(51.76%)']
accuracies = [43.00, 53.38, 52.26, 50.25, 52.76, 51.76]
colors = ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#1f77b4', '#8c564b']

bars = ax1.bar(range(len(phases)), accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.axhline(y=85, color='red', linestyle='--', linewidth=2, label='Target (85%)')
ax1.axhline(y=52.76, color='blue', linestyle='--', linewidth=2, label='Best (52.76%)')

for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

ax1.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
ax1.set_title('Phase 1 Progress Timeline', fontweight='bold', fontsize=14)
ax1.set_xticks(range(len(phases)))
ax1.set_xticklabels(phases, rotation=0, ha='center', fontsize=9)
ax1.set_ylim(0, 95)
ax1.legend(loc='upper right')
ax1.grid(axis='y', alpha=0.3)

# ========================
# 2. Model Comparison
# ========================
ax2 = plt.subplot(2, 3, 2)
models = ['Extra Trees', 'LR', 'Stacking', 'LightGBM', 'RF', 'MLP', 'XGBoost', 'CatBoost']
model_accs = [52.76, 52.26, 52.26, 51.26, 51.26, 50.25, 50.25, 50.75]
model_colors = ['#1f77b4' if acc >= 52 else '#ff7f0e' if acc >= 51 else '#d62728' for acc in model_accs]

bars = ax2.barh(range(len(models)), model_accs, color=model_colors, alpha=0.8, edgecolor='black', linewidth=1)

for i, (bar, acc) in enumerate(zip(bars, model_accs)):
    width = bar.get_width()
    ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
             f'{acc:.2f}%', ha='left', va='center', fontweight='bold', fontsize=9)

ax2.set_xlabel('Test Accuracy (%)', fontweight='bold', fontsize=12)
ax2.set_title('Model Performance Comparison', fontweight='bold', fontsize=14)
ax2.set_yticks(range(len(models)))
ax2.set_yticklabels(models, fontsize=10)
ax2.set_xlim(0, 100)
ax2.axvline(x=52.76, color='blue', linestyle='--', linewidth=1.5, alpha=0.5, label='Best')
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

# ========================
# 3. Per-Class Performance
# ========================
ax3 = plt.subplot(2, 3, 3)
classes = ['Low Risk', 'Medium Risk', 'High Risk']
precision = [77.1, 42.7, 66.7]
recall = [39.1, 82.4, 35.5]
f1 = [51.9, 56.3, 46.3]

x = np.arange(len(classes))
width = 0.25

bars1 = ax3.bar(x - width, precision, width, label='Precision', color='#2ca02c', alpha=0.8, edgecolor='black')
bars2 = ax3.bar(x, recall, width, label='Recall', color='#ff7f0e', alpha=0.8, edgecolor='black')
bars3 = ax3.bar(x + width, f1, width, label='F1-Score', color='#1f77b4', alpha=0.8, edgecolor='black')

ax3.set_ylabel('Score (%)', fontweight='bold', fontsize=12)
ax3.set_title('Per-Class Performance (Best Model)', fontweight='bold', fontsize=14)
ax3.set_xticks(x)
ax3.set_xticklabels(classes, fontsize=10)
ax3.set_ylim(0, 100)
ax3.legend(loc='upper right')
ax3.grid(axis='y', alpha=0.3)

# Add values on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)

# ========================
# 4. Task Impact Analysis
# ========================
ax4 = plt.subplot(2, 3, 4)
tasks = ['Data\nAugmentation', 'Feature\nEngineering', 'Deep\nLearning', 
         'Advanced\nEnsemble', 'Hyper-param\nOptimization']
impacts = [10.38, 0.00, -2.01, 0.50, -1.00]
impact_colors = ['#2ca02c' if imp > 2 else '#ff7f0e' if imp > 0 else '#d62728' for imp in impacts]

bars = ax4.bar(range(len(tasks)), impacts, color=impact_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)

for i, (bar, imp) in enumerate(zip(bars, impacts)):
    height = bar.get_height()
    y_pos = height + 0.5 if height > 0 else height - 0.5
    ax4.text(bar.get_x() + bar.get_width()/2., y_pos,
             f'{imp:+.2f}%', ha='center', va='bottom' if height > 0 else 'top', 
             fontweight='bold', fontsize=9)

ax4.set_ylabel('Accuracy Impact (%)', fontweight='bold', fontsize=12)
ax4.set_title('Task Impact Analysis', fontweight='bold', fontsize=14)
ax4.set_xticks(range(len(tasks)))
ax4.set_xticklabels(tasks, fontsize=9)
ax4.set_ylim(-5, 15)
ax4.grid(axis='y', alpha=0.3)

# ========================
# 5. Confusion Matrix (Best Model)
# ========================
ax5 = plt.subplot(2, 3, 5)
cm = np.array([[27, 36, 6], [7, 56, 5], [1, 39, 22]])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
            xticklabels=['Low', 'Med', 'High'],
            yticklabels=['Low', 'Med', 'High'],
            linewidths=1, linecolor='black', ax=ax5,
            annot_kws={'fontsize': 12, 'fontweight': 'bold'})
ax5.set_xlabel('Predicted Class', fontweight='bold', fontsize=12)
ax5.set_ylabel('Actual Class', fontweight='bold', fontsize=12)
ax5.set_title('Confusion Matrix (Extra Trees)', fontweight='bold', fontsize=14)

# ========================
# 6. Key Metrics Summary
# ========================
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_text = f"""
Phase 1 Complete Summary
{'='*45}

📊 Final Results:
   • Best Model: Extra Trees (unoptimized)
   • Test Accuracy: 52.76%
   • Baseline: 43.00%
   • Improvement: +9.76%
   • Target: 85.00%
   • Gap: 32.24%

✅ What Worked:
   • Data augmentation: +10.38% (most effective)
   • Tree ensembles: consistent 51-53%
   • Proper evaluation: 199 test samples

❌ What Didn't Work:
   • Feature engineering: 0% impact
   • Deep learning: -2% (too few samples)
   • Hyperparameter tuning: -1% (ceiling reached)

📈 Dataset Growth:
   • Original: 150 real samples
   • Augmented: 663 samples (4.4x)
   • Features: 3 → 64 → 30 selected

🎯 Class Performance:
   • Medium Risk: 82% recall ✅ (best)
   • Low Risk: 39% recall ⚠️ (challenging)
   • High Risk: 36% recall ❌ (poorest)

💡 Root Cause:
   Insufficient real data (150 samples)
   Need 500-1000 samples for 70-80% accuracy
   Need 1000-2000 samples for 85% target

🚀 Recommendations:
   1. Collect more real-world data (priority #1)
   2. Consult domain experts
   3. Focus on Low/High risk classes
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(rect=[0, 0.02, 1, 0.96])

# Save figure
output_path = 'reports/PHASE1_COMPLETE_DASHBOARD.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Dashboard saved to: {output_path}")
print(f"\n🎉 Phase 1 visualization complete!")

plt.show()
