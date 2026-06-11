"""
PulseAI - Data Augmentation Module
Expands dataset using SMOTE, ADASYN, and Gaussian noise injection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from pathlib import Path
import json

class DataAugmenter:
    """Data augmentation using multiple techniques"""
    
    def __init__(self, data_path: str, random_state: int = 42):
        self.data_path = data_path
        self.random_state = random_state
        self.df = None
        self.augmented_df = None
        
    def load_data(self):
        """Load original dataset"""
        print("="*70)
        print("PulseAI Data Augmentation")
        print("="*70)
        
        self.df = pd.read_csv(self.data_path)
        print(f"\n✅ Loaded original dataset: {len(self.df)} records")
        
        # Show class distribution
        print(f"\n📊 Original Class Distribution:")
        for target in sorted(self.df['Target'].unique()):
            count = (self.df['Target'] == target).sum()
            print(f"   Class {target}: {count} ({count/len(self.df)*100:.1f}%)")
        
        return self
    
    def augment_with_smote(self, target_samples: int = 200):
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique)
        Creates synthetic samples by interpolating between existing samples
        """
        print(f"\n🔄 Applying SMOTE augmentation...")
        
        # Prepare data
        feature_cols = ['Temperature Data', 'ECG Data', 'Pressure Data']
        X = self.df[feature_cols].values
        y = self.df['Target'].values
        
        # Get current class counts
        unique, counts = np.unique(y, return_counts=True)
        current_max = counts.max()
        
        # Calculate target samples per class (must be >= current max)
        samples_per_class = max(target_samples // 3, current_max + 10)
        
        # Apply SMOTE
        smote = SMOTE(
            sampling_strategy={
                0: samples_per_class,  # Low risk
                1: samples_per_class,  # Medium risk
                2: samples_per_class   # High risk
            },
            random_state=self.random_state,
            k_neighbors=min(5, counts.min() - 1)  # Ensure k_neighbors < minority class size
        )
        
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Create augmented dataframe
        smote_df = pd.DataFrame(X_resampled, columns=feature_cols)
        smote_df['Target'] = y_resampled
        smote_df['Patient ID'] = range(1, len(smote_df) + 1)
        smote_df['Augmentation_Method'] = 'SMOTE'
        
        print(f"   ✅ SMOTE complete: {len(smote_df)} samples")
        print(f"   Class distribution:")
        for target in sorted(smote_df['Target'].unique()):
            count = (smote_df['Target'] == target).sum()
            print(f"      Class {target}: {count}")
        
        return smote_df
    
    def augment_with_adasyn(self, target_samples: int = 200):
        """
        Apply ADASYN (Adaptive Synthetic Sampling)
        Focuses more on harder-to-learn samples
        """
        print(f"\n🔄 Applying ADASYN augmentation...")
        
        # Prepare data
        feature_cols = ['Temperature Data', 'ECG Data', 'Pressure Data']
        X = self.df[feature_cols].values
        y = self.df['Target'].values
        
        # Get current class counts
        unique, counts = np.unique(y, return_counts=True)
        current_max = counts.max()
        
        # Calculate target samples per class (must be >= current max)
        samples_per_class = max(target_samples // 3, current_max + 10)
        
        try:
            # Apply ADASYN
            adasyn = ADASYN(
                sampling_strategy={
                    0: samples_per_class,
                    1: samples_per_class,
                    2: samples_per_class
                },
                random_state=self.random_state,
                n_neighbors=min(5, counts.min() - 1)
            )
            
            X_resampled, y_resampled = adasyn.fit_resample(X, y)
            
            # Create augmented dataframe
            adasyn_df = pd.DataFrame(X_resampled, columns=feature_cols)
            adasyn_df['Target'] = y_resampled
            adasyn_df['Patient ID'] = range(1, len(adasyn_df) + 1)
            adasyn_df['Augmentation_Method'] = 'ADASYN'
            
            print(f"   ✅ ADASYN complete: {len(adasyn_df)} samples")
            print(f"   Class distribution:")
            for target in sorted(adasyn_df['Target'].unique()):
                count = (adasyn_df['Target'] == target).sum()
                print(f"      Class {target}: {count}")
            
            return adasyn_df
            
        except ValueError as e:
            print(f"   ⚠️  ADASYN failed: {e}")
            print(f"   Using SMOTE as fallback...")
            return self.augment_with_smote(target_samples)
    
    def augment_with_gaussian_noise(self, num_samples: int = 100, noise_level: float = 0.05):
        """
        Add Gaussian noise to existing samples
        Creates realistic variations
        """
        print(f"\n🔄 Applying Gaussian noise augmentation...")
        
        feature_cols = ['Temperature Data', 'ECG Data', 'Pressure Data']
        augmented_samples = []
        
        # Generate samples per class
        samples_per_class = num_samples // 3
        
        for target_class in [0, 1, 2]:
            class_data = self.df[self.df['Target'] == target_class][feature_cols]
            
            for _ in range(samples_per_class):
                # Select random sample
                base_sample = class_data.sample(n=1, random_state=self.random_state + _).values[0]
                
                # Add Gaussian noise
                noise = np.random.normal(0, noise_level * np.abs(base_sample), size=base_sample.shape)
                noisy_sample = base_sample + noise
                
                # Ensure values stay within reasonable bounds
                noisy_sample = np.clip(noisy_sample, 
                                      [30, 0, 70],  # Min values
                                      [50, 150, 80])  # Max values
                
                augmented_samples.append({
                    'Temperature Data': noisy_sample[0],
                    'ECG Data': noisy_sample[1],
                    'Pressure Data': noisy_sample[2],
                    'Target': target_class,
                    'Augmentation_Method': 'Gaussian_Noise'
                })
        
        noise_df = pd.DataFrame(augmented_samples)
        noise_df['Patient ID'] = range(1, len(noise_df) + 1)
        
        print(f"   ✅ Gaussian noise complete: {len(noise_df)} samples")
        print(f"   Class distribution:")
        for target in sorted(noise_df['Target'].unique()):
            count = (noise_df['Target'] == target).sum()
            print(f"      Class {target}: {count}")
        
        return noise_df
    
    def create_combined_dataset(self, smote_samples: int = 200, 
                               adasyn_samples: int = 150, 
                               noise_samples: int = 100):
        """
        Create final augmented dataset combining all methods
        """
        print(f"\n🎯 Creating combined augmented dataset...")
        
        # Original data with augmentation tag
        original_df = self.df.copy()
        original_df['Augmentation_Method'] = 'Original'
        
        # Apply augmentation techniques
        smote_df = self.augment_with_smote(smote_samples)
        adasyn_df = self.augment_with_adasyn(adasyn_samples)
        noise_df = self.augment_with_gaussian_noise(noise_samples, noise_level=0.03)
        
        # Combine all
        self.augmented_df = pd.concat([original_df, smote_df, adasyn_df, noise_df], 
                                     ignore_index=True)
        
        # Reorder columns
        cols = ['Patient ID', 'Temperature Data', 'ECG Data', 'Pressure Data', 
                'Target', 'Augmentation_Method']
        self.augmented_df = self.augmented_df[cols]
        
        print(f"\n✅ Combined dataset created!")
        print(f"   Total samples: {len(self.augmented_df)}")
        print(f"\n   Breakdown by method:")
        for method in self.augmented_df['Augmentation_Method'].unique():
            count = (self.augmented_df['Augmentation_Method'] == method).sum()
            print(f"      {method}: {count}")
        
        print(f"\n   Final class distribution:")
        for target in sorted(self.augmented_df['Target'].unique()):
            count = (self.augmented_df['Target'] == target).sum()
            print(f"      Class {target}: {count} ({count/len(self.augmented_df)*100:.1f}%)")
        
        return self.augmented_df
    
    def save_augmented_data(self, output_path: str = 'dataset_augmented.csv'):
        """Save augmented dataset"""
        self.augmented_df.to_csv(output_path, index=False)
        print(f"\n💾 Augmented dataset saved: {output_path}")
        
        # Save statistics
        stats = {
            'original_samples': int(len(self.df)),
            'augmented_samples': int(len(self.augmented_df)),
            'increase_factor': float(len(self.augmented_df) / len(self.df)),
            'methods_used': list(self.augmented_df['Augmentation_Method'].unique()),
            'class_distribution': {
                f'class_{target}': int((self.augmented_df['Target'] == target).sum())
                for target in sorted(self.augmented_df['Target'].unique())
            }
        }
        
        stats_path = Path('reports/augmentation_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        
        print(f"💾 Statistics saved: {stats_path}")
        
        return output_path
    
    def validate_augmentation(self):
        """Validate augmented data quality"""
        print(f"\n🔍 Validating augmented data...")
        
        feature_cols = ['Temperature Data', 'ECG Data', 'Pressure Data']
        
        # Check for NaN/Inf values
        nan_count = self.augmented_df[feature_cols].isna().sum().sum()
        inf_count = np.isinf(self.augmented_df[feature_cols]).sum().sum()
        
        print(f"   - NaN values: {nan_count}")
        print(f"   - Inf values: {inf_count}")
        
        # Check value ranges
        print(f"\n   Value ranges:")
        for col in feature_cols:
            min_val = self.augmented_df[col].min()
            max_val = self.augmented_df[col].max()
            print(f"      {col}: [{min_val:.2f}, {max_val:.2f}]")
        
        # Check class balance
        print(f"\n   Class balance ratio:")
        class_counts = self.augmented_df['Target'].value_counts()
        max_count = class_counts.max()
        min_count = class_counts.min()
        balance_ratio = max_count / min_count
        print(f"      Ratio: {balance_ratio:.2f}:1")
        
        if balance_ratio <= 1.5:
            print(f"      ✅ Well balanced")
        else:
            print(f"      ⚠️  Slightly imbalanced")
        
        if nan_count == 0 and inf_count == 0:
            print(f"\n   ✅ Validation passed!")
        else:
            print(f"\n   ⚠️  Data quality issues detected!")
        
        return nan_count == 0 and inf_count == 0


def main():
    """Main execution"""
    # Initialize augmenter
    augmenter = DataAugmenter('iot_dataset_expanded.csv', random_state=42)
    
    # Load data
    augmenter.load_data()
    
    # Create augmented dataset (target: 500+ samples)
    augmented_df = augmenter.create_combined_dataset(
        smote_samples=200,   # SMOTE: 200 samples
        adasyn_samples=150,  # ADASYN: 150 samples
        noise_samples=100    # Gaussian: 100 samples
    )
    
    # Validate
    augmenter.validate_augmentation()
    
    # Save
    output_file = augmenter.save_augmented_data('dataset_augmented.csv')
    
    print(f"\n" + "="*70)
    print(f"🎉 Data Augmentation Complete!")
    print(f"="*70)
    print(f"\n   Original: 150 samples")
    print(f"   Augmented: {len(augmented_df)} samples")
    print(f"   Increase: {len(augmented_df)/150:.1f}x")
    print(f"\n   Next step: Retrain models on augmented data")
    
    return augmented_df


if __name__ == "__main__":
    main()
