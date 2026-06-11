"""
PulseAI - Data Analysis Module
Comprehensive dataset analysis and statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

class DataAnalyzer:
    """Analyzes dataset and provides comprehensive statistics"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.stats = {}
        
    def load_and_analyze(self):
        """Load data and perform comprehensive analysis"""
        print("="*70)
        print("PulseAI Dataset Analysis")
        print("="*70)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"\n✅ Dataset loaded: {self.data_path}")
        print(f"   Original shape: {self.df.shape}")
        
        # Basic info
        self._analyze_basic_info()
        
        # Check duplicates
        self._analyze_duplicates()
        
        # Analyze features
        self._analyze_features()
        
        # Analyze target
        self._analyze_target()
        
        # Statistical summary
        self._statistical_summary()
        
        # Correlation analysis
        self._correlation_analysis()
        
        # Data quality
        self._data_quality_check()
        
        # Save analysis report
        self._save_report()
        
        print("\n" + "="*70)
        print("Analysis Complete!")
        print("="*70)
        
        return self.stats
    
    def _analyze_basic_info(self):
        """Basic dataset information"""
        print(f"\n📊 Basic Information:")
        print(f"   - Total records: {len(self.df)}")
        print(f"   - Features: {len(self.df.columns)}")
        print(f"   - Memory usage: {self.df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        self.stats['total_records'] = len(self.df)
        self.stats['total_features'] = len(self.df.columns)
        self.stats['columns'] = list(self.df.columns)
    
    def _analyze_duplicates(self):
        """Check for duplicates"""
        duplicates = self.df.duplicated().sum()
        duplicate_rows = self.df[self.df.duplicated(keep=False)]
        
        print(f"\n🔍 Duplicate Analysis:")
        print(f"   - Duplicate rows: {duplicates}")
        print(f"   - Unique records: {len(self.df) - duplicates}")
        print(f"   - Duplicate percentage: {(duplicates/len(self.df)*100):.2f}%")
        
        self.stats['duplicates'] = int(duplicates)
        self.stats['unique_records'] = len(self.df) - duplicates
        self.stats['duplicate_percentage'] = float(duplicates/len(self.df)*100)
        
        # Show sample duplicates
        if duplicates > 0:
            print(f"\n   Sample duplicate patterns:")
            feature_cols = [col for col in self.df.columns if col != 'Patient ID']
            dup_sample = self.df[self.df.duplicated(subset=feature_cols, keep=False)].head(6)
            print(dup_sample.to_string(index=False))
    
    def _analyze_features(self):
        """Analyze feature distributions"""
        print(f"\n📈 Feature Analysis:")
        
        feature_cols = ['Temperature Data', 'ECG Data', 'Pressure Data']
        self.stats['features'] = {}
        
        for col in feature_cols:
            print(f"\n   {col}:")
            print(f"      Mean: {self.df[col].mean():.2f}")
            print(f"      Std:  {self.df[col].std():.2f}")
            print(f"      Min:  {self.df[col].min():.2f}")
            print(f"      Max:  {self.df[col].max():.2f}")
            print(f"      Range: {self.df[col].max() - self.df[col].min():.2f}")
            
            self.stats['features'][col] = {
                'mean': float(self.df[col].mean()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max()),
                'range': float(self.df[col].max() - self.df[col].min())
            }
    
    def _analyze_target(self):
        """Analyze target variable distribution"""
        print(f"\n🎯 Target Distribution:")
        
        target_counts = self.df['Target'].value_counts().sort_index()
        
        print(f"   Class 0 (Low):    {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(self.df)*100:.1f}%)")
        print(f"   Class 1 (Medium): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(self.df)*100:.1f}%)")
        print(f"   Class 2 (High):   {target_counts.get(2, 0)} ({target_counts.get(2, 0)/len(self.df)*100:.1f}%)")
        
        # Check balance
        min_class = target_counts.min()
        max_class = target_counts.max()
        imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
        
        print(f"\n   Imbalance ratio: {imbalance_ratio:.2f}:1")
        if imbalance_ratio <= 1.5:
            print(f"   ✅ Dataset is well-balanced")
        else:
            print(f"   ⚠️  Dataset has class imbalance")
        
        self.stats['target_distribution'] = {
            'class_0': int(target_counts.get(0, 0)),
            'class_1': int(target_counts.get(1, 0)),
            'class_2': int(target_counts.get(2, 0)),
            'imbalance_ratio': float(imbalance_ratio)
        }
    
    def _statistical_summary(self):
        """Statistical summary"""
        print(f"\n📊 Statistical Summary:")
        print(self.df.describe())
        
        # Skewness and kurtosis
        print(f"\n   Skewness and Kurtosis:")
        feature_cols = ['Temperature Data', 'ECG Data', 'Pressure Data']
        for col in feature_cols:
            skew = self.df[col].skew()
            kurt = self.df[col].kurtosis()
            print(f"      {col}: skew={skew:.3f}, kurtosis={kurt:.3f}")
    
    def _correlation_analysis(self):
        """Correlation analysis"""
        print(f"\n🔗 Correlation Analysis:")
        
        feature_cols = ['Temperature Data', 'ECG Data', 'Pressure Data', 'Target']
        corr_matrix = self.df[feature_cols].corr()
        
        print(f"\n   Correlation with Target:")
        target_corr = corr_matrix['Target'].drop('Target').sort_values(ascending=False)
        for feat, corr_val in target_corr.items():
            print(f"      {feat}: {corr_val:.3f}")
        
        self.stats['correlation'] = {
            feat: float(corr_val) for feat, corr_val in target_corr.items()
        }
    
    def _data_quality_check(self):
        """Check data quality"""
        print(f"\n✅ Data Quality Check:")
        
        # Missing values
        missing = self.df.isnull().sum()
        print(f"   - Missing values: {missing.sum()}")
        
        # Outliers (using IQR method)
        feature_cols = ['Temperature Data', 'ECG Data', 'Pressure Data']
        outliers = {}
        
        for col in feature_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            outliers[col] = outlier_count
            
        print(f"   - Outliers detected:")
        for col, count in outliers.items():
            print(f"      {col}: {count} ({count/len(self.df)*100:.1f}%)")
        
        self.stats['data_quality'] = {
            'missing_values': int(missing.sum()),
            'outliers': {col: int(count) for col, count in outliers.items()}
        }
    
    def _save_report(self):
        """Save analysis report"""
        report_path = Path('reports/data_analysis_report.json')
        report_path.parent.mkdir(exist_ok=True)
        
        # Convert numpy types to Python types
        stats_serializable = json.loads(json.dumps(self.stats, default=str))
        
        with open(report_path, 'w') as f:
            json.dump(stats_serializable, f, indent=4)
        
        print(f"\n💾 Report saved: {report_path}")
    
    def visualize_distributions(self):
        """Create visualizations of data distributions"""
        print(f"\n📊 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('PulseAI Dataset Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Feature distributions
        feature_cols = ['Temperature Data', 'ECG Data', 'Pressure Data']
        for idx, col in enumerate(feature_cols):
            row = idx // 2
            col_pos = idx % 2
            
            axes[row, col_pos].hist(self.df[col], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
            axes[row, col_pos].set_title(f'{col} Distribution')
            axes[row, col_pos].set_xlabel(col)
            axes[row, col_pos].set_ylabel('Frequency')
            axes[row, col_pos].grid(alpha=0.3)
        
        # 2. Target distribution
        target_counts = self.df['Target'].value_counts().sort_index()
        axes[1, 1].bar(['Low (0)', 'Medium (1)', 'High (2)'], target_counts.values, 
                      color=['green', 'orange', 'red'], edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('Target Class Distribution')
        axes[1, 1].set_xlabel('Risk Level')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path('reports/data_distributions.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {plot_path}")
        
        # Create correlation heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        feature_cols_with_target = ['Temperature Data', 'ECG Data', 'Pressure Data', 'Target']
        corr_matrix = self.df[feature_cols_with_target].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, ax=ax,
                   cbar_kws={"shrink": 0.8})
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save correlation plot
        corr_path = Path('reports/correlation_heatmap.png')
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {corr_path}")
        
        plt.close('all')


def main():
    """Main execution"""
    analyzer = DataAnalyzer('iot_dataset_expanded.csv')
    stats = analyzer.load_and_analyze()
    analyzer.visualize_distributions()
    
    print(f"\n🎯 Key Findings:")
    print(f"   - Dataset has {stats['unique_records']} unique records")
    print(f"   - {stats['duplicates']} duplicates found ({stats['duplicate_percentage']:.1f}%)")
    print(f"   - Classes are {'balanced' if stats['target_distribution']['imbalance_ratio'] <= 1.5 else 'imbalanced'}")
    print(f"   - No missing values detected" if stats['data_quality']['missing_values'] == 0 else f"   - ⚠️  {stats['data_quality']['missing_values']} missing values")
    
    print(f"\n💡 Recommendation:")
    if stats['unique_records'] < 100:
        print(f"   ⚠️  Dataset is small ({stats['unique_records']} samples)")
        print(f"   → Data augmentation is CRITICAL for model performance")
        print(f"   → Target: 500+ samples using SMOTE/ADASYN")
    
    return stats


if __name__ == "__main__":
    main()
