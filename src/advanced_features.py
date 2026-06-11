"""
PulseAI - Advanced Feature Engineering Module
Creates domain-specific medical features for improved predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_FILE = PROJECT_ROOT / "iot_dataset_expanded.csv"
OUTPUT_ENGINEERED = PROJECT_ROOT / "iot_dataset_engineered.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """Advanced medical feature engineering for IoT health monitoring"""
    
    def __init__(self):
        self.feature_names = []
        self.feature_stats = {}
        
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering techniques"""
        print("="*70)
        print("PulseAI Advanced Feature Engineering")
        print("="*70)
        
        print(f"\n📊 Input: {len(df)} samples, {len(df.columns)} features")
        
        # Create a copy
        df_engineered = df.copy()
        
        # Track original features
        original_features = ['Temperature Data', 'ECG Data', 'Pressure Data']
        
        # 1. Medical domain features
        df_engineered = self._create_medical_features(df_engineered)
        
        # 2. Statistical features
        df_engineered = self._create_statistical_features(df_engineered)
        
        # 3. Ratio features
        df_engineered = self._create_ratio_features(df_engineered)
        
        # 4. Polynomial features
        df_engineered = self._create_polynomial_features(df_engineered)
        
        # 5. Binary indicators
        df_engineered = self._create_binary_indicators(df_engineered)
        
        # 6. Anomaly scores
        df_engineered = self._create_anomaly_scores(df_engineered)
        
        # 7. Interaction features
        df_engineered = self._create_interaction_features(df_engineered)
        
        # 8. Aggregate features
        df_engineered = self._create_aggregate_features(df_engineered)
        
        # Get new feature list
        self.feature_names = [col for col in df_engineered.columns 
                             if col not in ['Patient ID', 'Target', 'Augmentation_Method']]
        
        print(f"\n✅ Output: {len(df_engineered)} samples, {len(self.feature_names)} features")
        print(f"   📈 Features created: {len(self.feature_names) - len(original_features)}")
        
        return df_engineered
    
    def _create_medical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create medical domain-specific features"""
        print(f"\n🏥 Creating Medical Domain Features...")
        
        # Temperature-based medical indicators
        df['Temp_Fever'] = (df['Temperature Data'] > 37.5).astype(int)  # Fever threshold
        df['Temp_Hypothermia'] = (df['Temperature Data'] < 35.0).astype(int)  # Hypothermia
        df['Temp_Severe_Fever'] = (df['Temperature Data'] > 39.0).astype(int)  # High fever
        df['Temp_Deviation_Normal'] = np.abs(df['Temperature Data'] - 37.0)  # Distance from normal (37°C)
        
        # ECG-based indicators
        df['ECG_Abnormal'] = (df['ECG Data'] > 20).astype(int)  # Abnormal ECG reading
        df['ECG_Critical'] = (df['ECG Data'] > 50).astype(int)  # Critical ECG
        df['ECG_Very_Low'] = (df['ECG Data'] < 5).astype(int)  # Very low ECG
        df['ECG_Severity_Score'] = np.clip(df['ECG Data'] / 10, 0, 10)  # Normalized severity
        
        # Pressure-based indicators (Normal BP: 75-77 mmHg assumed for this simplified model)
        df['Pressure_Low'] = (df['Pressure Data'] < 75.5).astype(int)  # Hypotension
        df['Pressure_High'] = (df['Pressure Data'] > 77.5).astype(int)  # Hypertension
        df['Pressure_Deviation'] = np.abs(df['Pressure Data'] - 76.5)  # Distance from normal
        df['Pressure_Normal_Range'] = ((df['Pressure Data'] >= 75.5) & 
                                       (df['Pressure Data'] <= 77.5)).astype(int)
        
        # Combined risk indicators
        df['Multiple_Abnormalities'] = (df['Temp_Fever'] + df['ECG_Abnormal'] + 
                                        (1 - df['Pressure_Normal_Range']))
        df['Critical_Vitals'] = ((df['Temp_Severe_Fever']) | 
                                 (df['ECG_Critical'])).astype(int)
        
        print(f"   ✅ Created 14 medical features")
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features"""
        print(f"\n📊 Creating Statistical Features...")
        
        # Standardized scores (z-scores)
        for col in ['Temperature Data', 'ECG Data', 'Pressure Data']:
            mean = df[col].mean()
            std = df[col].std()
            df[f'{col}_ZScore'] = (df[col] - mean) / (std + 1e-8)
        
        # Min-Max normalized features
        for col in ['Temperature Data', 'ECG Data', 'Pressure Data']:
            min_val = df[col].min()
            max_val = df[col].max()
            df[f'{col}_Normalized'] = (df[col] - min_val) / (max_val - min_val + 1e-8)
        
        # Percentile ranks
        for col in ['Temperature Data', 'ECG Data', 'Pressure Data']:
            df[f'{col}_Percentile'] = df[col].rank(pct=True) * 100
        
        print(f"   ✅ Created 9 statistical features")
        return df
    
    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio features between vitals"""
        print(f"\n🔢 Creating Ratio Features...")
        
        # Temperature to ECG ratio
        df['Temp_ECG_Ratio'] = df['Temperature Data'] / (df['ECG Data'] + 1)  # +1 to avoid division by zero
        
        # ECG to Pressure ratio
        df['ECG_Pressure_Ratio'] = df['ECG Data'] / (df['Pressure Data'] + 1)
        
        # Temperature to Pressure ratio
        df['Temp_Pressure_Ratio'] = df['Temperature Data'] / df['Pressure Data']
        
        # Combined vital efficiency
        df['Vital_Efficiency'] = (df['Temperature Data'] * df['Pressure Data']) / (df['ECG Data'] + 1)
        
        print(f"   ✅ Created 4 ratio features")
        return df
    
    def _create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features (squared, cubed)"""
        print(f"\n📐 Creating Polynomial Features...")
        
        # Squared features
        df['Temp_Squared'] = df['Temperature Data'] ** 2
        df['ECG_Squared'] = df['ECG Data'] ** 2
        df['Pressure_Squared'] = df['Pressure Data'] ** 2
        
        # Cubed features
        df['Temp_Cubed'] = df['Temperature Data'] ** 3
        df['ECG_Cubed'] = df['ECG Data'] ** 3
        
        # Square root features
        df['Temp_Sqrt'] = np.sqrt(df['Temperature Data'])
        df['ECG_Sqrt'] = np.sqrt(np.abs(df['ECG Data']))  # abs for negative values
        df['Pressure_Sqrt'] = np.sqrt(df['Pressure Data'])
        
        print(f"   ✅ Created 8 polynomial features")
        return df
    
    def _create_binary_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary indicator features"""
        print(f"\n🎯 Creating Binary Indicators...")
        
        # Zero indicators
        df['ECG_Is_Zero'] = (df['ECG Data'] == 0).astype(int)
        df['ECG_Is_NonZero'] = (df['ECG Data'] > 0).astype(int)
        
        # Range indicators
        df['Temp_In_Normal_Range'] = ((df['Temperature Data'] >= 36.5) & 
                                       (df['Temperature Data'] <= 37.5)).astype(int)
        df['ECG_In_Low_Range'] = (df['ECG Data'] <= 10).astype(int)
        df['ECG_In_High_Range'] = (df['ECG Data'] > 50).astype(int)
        
        # Extreme value indicators
        df['Temp_Extreme'] = ((df['Temperature Data'] < 35) | 
                              (df['Temperature Data'] > 40)).astype(int)
        df['ECG_Extreme'] = (df['ECG Data'] > 100).astype(int)
        
        print(f"   ✅ Created 7 binary indicators")
        return df
    
    def _create_anomaly_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create anomaly detection scores"""
        print(f"\n🚨 Creating Anomaly Scores...")
        
        # Distance from median
        for col in ['Temperature Data', 'ECG Data', 'Pressure Data']:
            median = df[col].median()
            df[f'{col}_Distance_From_Median'] = np.abs(df[col] - median)
        
        # IQR-based anomaly score
        for col in ['Temperature Data', 'ECG Data', 'Pressure Data']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[f'{col}_Is_Outlier'] = ((df[col] < lower_bound) | 
                                       (df[col] > upper_bound)).astype(int)
        
        # Combined anomaly score
        df['Total_Anomaly_Score'] = (df['Temperature Data_Is_Outlier'] + 
                                     df['ECG Data_Is_Outlier'] + 
                                     df['Pressure Data_Is_Outlier'])
        
        print(f"   ✅ Created 7 anomaly features")
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between vitals"""
        print(f"\n🔗 Creating Interaction Features...")
        
        # Multiplicative interactions
        df['Temp_ECG_Interaction'] = df['Temperature Data'] * df['ECG Data']
        df['Temp_Pressure_Interaction'] = df['Temperature Data'] * df['Pressure Data']
        df['ECG_Pressure_Interaction'] = df['ECG Data'] * df['Pressure Data']
        df['Three_Way_Interaction'] = (df['Temperature Data'] * 
                                       df['ECG Data'] * 
                                       df['Pressure Data'])
        
        # Additive interactions
        df['Temp_Plus_ECG'] = df['Temperature Data'] + df['ECG Data']
        df['All_Vitals_Sum'] = (df['Temperature Data'] + 
                                df['ECG Data'] + 
                                df['Pressure Data'])
        
        print(f"   ✅ Created 6 interaction features")
        return df
    
    def _create_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregate features"""
        print(f"\n📈 Creating Aggregate Features...")
        
        vitals = ['Temperature Data', 'ECG Data', 'Pressure Data']
        
        # Mean of all vitals
        df['Vitals_Mean'] = df[vitals].mean(axis=1)
        
        # Std of all vitals
        df['Vitals_Std'] = df[vitals].std(axis=1)
        
        # Min and Max
        df['Vitals_Min'] = df[vitals].min(axis=1)
        df['Vitals_Max'] = df[vitals].max(axis=1)
        
        # Range
        df['Vitals_Range'] = df['Vitals_Max'] - df['Vitals_Min']
        
        # Coefficient of variation
        df['Vitals_CV'] = df['Vitals_Std'] / (df['Vitals_Mean'] + 1e-8)
        
        print(f"   ✅ Created 6 aggregate features")
        return df
    
    def save_feature_report(self, df: pd.DataFrame, output_path: Path = REPORTS_DIR / 'feature_engineering_report.json'):
        """Save feature engineering report"""
        print(f"\n💾 Saving Feature Report...")
        
        # Create report
        report = {
            'total_features': len(self.feature_names),
            'feature_categories': {
                'medical': 14,
                'statistical': 9,
                'ratio': 4,
                'polynomial': 8,
                'binary': 7,
                'anomaly': 7,
                'interaction': 6,
                'aggregate': 6
            },
            'feature_list': self.feature_names,
            'sample_statistics': {}
        }
        
        # Add statistics for each feature
        for feature in self.feature_names[:10]:  # First 10 features as sample
            if feature in df.columns:
                report['sample_statistics'][feature] = {
                    'mean': float(df[feature].mean()),
                    'std': float(df[feature].std()),
                    'min': float(df[feature].min()),
                    'max': float(df[feature].max())
                }
        
        # Save report
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"   ✅ Report saved: {output_path}")
        
        return report
    
    def visualize_feature_importance(self, df: pd.DataFrame):
        """Visualize feature statistics"""
        print(f"\n📊 Creating Feature Visualizations...")
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Select numerical features only
        feature_cols = [col for col in self.feature_names 
                       if df[col].dtype in ['int64', 'float64']][:20]  # Top 20 features
        
        # Create correlation with target
        if 'Target' in df.columns:
            correlations = df[feature_cols + ['Target']].corr()['Target'].drop('Target').sort_values(ascending=False)
            
            # Plot top correlations
            fig, ax = plt.subplots(figsize=(10, 8))
            correlations.head(15).plot(kind='barh', ax=ax, color='skyblue', edgecolor='black')
            ax.set_title('Top 15 Feature Correlations with Target', fontsize=14, fontweight='bold')
            ax.set_xlabel('Correlation Coefficient')
            ax.set_ylabel('Feature')
            ax.grid(alpha=0.3, axis='x')
            plt.tight_layout()
            
            # Save
            output_path = REPORTS_DIR / 'feature_correlations.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved: {output_path}")
            
            # Print top features
            print(f"\n   🎯 Top 10 Features by Correlation:")
            for idx, (feat, corr) in enumerate(correlations.head(10).items(), 1):
                print(f"      {idx}. {feat}: {corr:.3f}")
        
        plt.close('all')


def main():
    """Main execution"""
    # Load expanded dataset
    print("📂 Loading expanded dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    original_count = len(df)

    # Drop serial column if present
    if 'Sl.No' in df.columns:
        df = df.drop(columns=['Sl.No'])

    # Remove duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count:
        print(f"   ⚠️  Found {duplicate_count} duplicate rows. Removing...")
        df = df.drop_duplicates()

    print(f"   Loaded: {len(df)} samples (originally {original_count})")
    
    # Initialize feature engineer
    engineer = AdvancedFeatureEngineer()
    
    # Engineer features
    df_engineered = engineer.engineer_all_features(df)
    
    # Save engineered dataset
    output_file = OUTPUT_ENGINEERED
    df_engineered.to_csv(output_file, index=False)
    print(f"\n💾 Engineered dataset saved: {output_file}")
    
    # Save report
    report = engineer.save_feature_report(df_engineered)
    
    # Visualize
    engineer.visualize_feature_importance(df_engineered)
    
    print(f"\n" + "="*70)
    print(f"🎉 Feature Engineering Complete!")
    print(f"="*70)
    print(f"\n   📊 Summary:")
    print(f"      Original features: 3")
    print(f"      Engineered features: {report['total_features']}")
    print(f"      Total samples: {len(df_engineered)}")
    print(f"\n   📈 Feature Breakdown:")
    for category, count in report['feature_categories'].items():
        print(f"      {category.capitalize()}: {count}")
    
    print(f"\n   🚀 Next step: Retrain models with {report['total_features']} features")
    
    return df_engineered


if __name__ == "__main__":
    main()
