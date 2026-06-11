"""
Model Evaluation and Visualization Module for PulseAI
Generates comprehensive evaluation reports and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import logging
from typing import Dict, Any, List
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for professional plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ModelEvaluator:
    """
    Professional model evaluation and visualization
    """
    
    def __init__(self, output_dir: str = "../reports"):
        """
        Initialize evaluator
        
        Args:
            output_dir: Directory to save reports and visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_confusion_matrices(self, 
                                 results: Dict[str, Any], 
                                 y_test: pd.Series,
                                 class_names: List[str] = ['Low', 'Medium', 'High']):
        """
        Plot confusion matrices for all models
        
        Args:
            results: Dictionary of model results
            y_test: True labels
            class_names: Names of classes
        """
        logger.info("Generating confusion matrices...")
        
        n_models = len(results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (model_name, result) in enumerate(results.items()):
            cm = np.array(result['confusion_matrix'])
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, 
                       yticklabels=class_names,
                       ax=axes[idx],
                       cbar_kws={'label': 'Count'})
            
            axes[idx].set_title(f'{model_name}\nAccuracy: {result["metrics"]["accuracy"]:.4f}', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=10)
            axes[idx].set_xlabel('Predicted Label', fontsize=10)
        
        # Hide unused subplots
        for idx in range(len(results), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrices.png'), 
                   dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrices saved to {self.output_dir}/confusion_matrices.png")
        plt.close()
    
    def plot_model_comparison(self, results_df: pd.DataFrame):
        """
        Plot comparison of all models
        
        Args:
            results_df: DataFrame with model evaluation results
        """
        logger.info("Generating model comparison plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # Sort by metric
            sorted_df = results_df.sort_values(metric, ascending=True)
            
            # Create horizontal bar chart
            bars = ax.barh(sorted_df['Model'], sorted_df[metric], color=color, alpha=0.7)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{width:.4f}',
                       ha='left', va='center', fontsize=9, fontweight='bold')
            
            ax.set_xlabel(metric, fontsize=12, fontweight='bold')
            ax.set_title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 1.0)
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison saved to {self.output_dir}/model_comparison.png")
        plt.close()
    
    def plot_metrics_radar(self, results_df: pd.DataFrame):
        """
        Create radar chart for model comparison
        
        Args:
            results_df: DataFrame with model evaluation results
        """
        logger.info("Generating radar chart...")
        
        from math import pi
        
        # Get top 5 models
        top_models = results_df.head(5)
        
        categories = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_models)))
        
        for idx, (_, row) in enumerate(top_models.iterrows()):
            values = row[categories].values.tolist()
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10)
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        plt.title('Model Performance Radar Chart (Top 5 Models)', 
                 size=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metrics_radar.png'), 
                   dpi=300, bbox_inches='tight')
        logger.info(f"Radar chart saved to {self.output_dir}/metrics_radar.png")
        plt.close()
    
    def generate_classification_reports(self, 
                                        results: Dict[str, Any],
                                        class_names: List[str] = ['Low', 'Medium', 'High']):
        """
        Generate detailed classification reports
        
        Args:
            results: Dictionary of model results
            class_names: Names of classes
        """
        logger.info("Generating classification reports...")
        
        report_path = os.path.join(self.output_dir, 'classification_reports.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("PULSEAI - DETAILED CLASSIFICATION REPORTS\n")
            f.write("=" * 100 + "\n\n")
            
            for model_name, result in results.items():
                f.write(f"\n{'=' * 80}\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"{'=' * 80}\n\n")
                
                # Overall metrics
                metrics = result['metrics']
                f.write(f"Overall Metrics:\n")
                f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall:    {metrics['recall']:.4f}\n")
                f.write(f"  F1 Score:  {metrics['f1_score']:.4f}\n")
                f.write(f"  CV Score:  {metrics['cv_score']:.4f}\n\n")
                
                # Per-class metrics
                report = result['classification_report']
                f.write(f"Per-Class Metrics:\n\n")
                f.write(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}\n")
                f.write("-" * 60 + "\n")
                
                for class_idx, class_name in enumerate(class_names):
                    class_key = str(class_idx)
                    if class_key in report:
                        f.write(f"{class_name:<12} "
                               f"{report[class_key]['precision']:<12.4f} "
                               f"{report[class_key]['recall']:<12.4f} "
                               f"{report[class_key]['f1-score']:<12.4f} "
                               f"{report[class_key]['support']:<12}\n")
                
                f.write("\n" + "-" * 60 + "\n")
                
                # Confusion Matrix
                f.write(f"\nConfusion Matrix:\n")
                cm = np.array(result['confusion_matrix'])
                f.write(f"{'':>12} " + " ".join([f"{name:>10}" for name in class_names]) + "\n")
                for idx, row in enumerate(cm):
                    f.write(f"{class_names[idx]:>12} " + " ".join([f"{val:>10}" for val in row]) + "\n")
                
                f.write("\n")
        
        logger.info(f"Classification reports saved to {report_path}")
    
    def generate_summary_report(self, results_df: pd.DataFrame):
        """
        Generate executive summary report
        
        Args:
            results_df: DataFrame with model evaluation results
        """
        logger.info("Generating executive summary...")
        
        report_path = os.path.join(self.output_dir, 'EXECUTIVE_SUMMARY.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# PulseAI - Model Enhancement Report\n\n")
            f.write(f"**Date:** {pd.Timestamp.now().strftime('%B %d, %Y')}\n\n")
            f.write("---\n\n")
            
            # Best model
            best_model = results_df.iloc[0]
            f.write("## 🏆 Best Performing Model\n\n")
            f.write(f"**Model:** {best_model['Model']}\n\n")
            f.write(f"| Metric | Score |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Accuracy | {best_model['Accuracy']:.4f} ({best_model['Accuracy']*100:.2f}%) |\n")
            f.write(f"| Precision | {best_model['Precision']:.4f} |\n")
            f.write(f"| Recall | {best_model['Recall']:.4f} |\n")
            f.write(f"| F1 Score | {best_model['F1_Score']:.4f} |\n")
            f.write(f"| CV Score | {best_model['CV_Score']:.4f} |\n\n")
            
            # All models comparison
            f.write("## 📊 Model Comparison Table\n\n")
            f.write(results_df.to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n")
            
            # Key insights
            f.write("## 💡 Key Insights\n\n")
            f.write(f"- **Total Models Trained:** {len(results_df)}\n")
            f.write(f"- **Best Model:** {best_model['Model']}\n")
            f.write(f"- **Accuracy Range:** {results_df['Accuracy'].min():.4f} - {results_df['Accuracy'].max():.4f}\n")
            f.write(f"- **Average Accuracy:** {results_df['Accuracy'].mean():.4f}\n")
            f.write(f"- **Top 3 Models:** {', '.join(results_df.head(3)['Model'].tolist())}\n\n")
            
            # Recommendations
            f.write("## 🎯 Recommendations\n\n")
            if best_model['Accuracy'] >= 0.90:
                f.write("✅ **Model Performance:** Excellent! Ready for production deployment.\n\n")
            elif best_model['Accuracy'] >= 0.80:
                f.write("⚠️ **Model Performance:** Good, but consider collecting more training data.\n\n")
            else:
                f.write("❌ **Model Performance:** Needs improvement. Recommend data augmentation.\n\n")
            
            f.write("### Next Steps:\n")
            f.write("1. Deploy best model to production API\n")
            f.write("2. Set up monitoring and logging\n")
            f.write("3. Collect more real-world data\n")
            f.write("4. Implement A/B testing\n")
            f.write("5. Schedule periodic retraining\n\n")
            
            f.write("---\n\n")
            f.write("*Generated by PulseAI Model Enhancement Pipeline*\n")
        
        logger.info(f"Executive summary saved to {report_path}")


if __name__ == "__main__":
    logger.info("Model evaluator module loaded successfully!")
