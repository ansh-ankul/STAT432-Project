import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SHAPAnalyzer:
    def __init__(self, model, X_train, sample_size=1000):
        """
        Initialize SHAP analyzer
        
        Args:
            model: Trained model with predict_proba method
            X_train: Training features
            sample_size: Number of samples to use for SHAP analysis (for performance)
        """
        self.model = model
        self.X_train = X_train
        self.sample_size = sample_size
        
        # Sample data for faster SHAP computation
        if len(X_train) > sample_size:
            self.X_sample = X_train.sample(n=sample_size, random_state=42)
        else:
            self.X_sample = X_train
        
        # Initialize SHAP explainer
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        
        # Initialize explainer
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer based on model type"""
        try:
            if hasattr(self.model, 'predict_proba'):
                # For tree-based models, use TreeExplainer
                if hasattr(self.model, 'estimators_') or hasattr(self.model, 'estimator'):
                    self.explainer = shap.TreeExplainer(self.model)
                else:
                    # For other models, use KernelExplainer
                    self.explainer = shap.KernelExplainer(self.model.predict_proba, self.X_sample)
                
                # Calculate SHAP values
                self.shap_values = self.explainer.shap_values(self.X_sample)
                if isinstance(self.shap_values, list):
                    # For multi-class, take the mean across classes
                    self.shap_values = np.mean(self.shap_values, axis=0)
                
                # Get expected value
                if hasattr(self.explainer, 'expected_value'):
                    self.expected_value = self.explainer.expected_value
                else:
                    self.expected_value = np.mean(self.model.predict_proba(self.X_sample), axis=0)
                
                print("SHAP explainer initialized successfully!")
                
            else:
                print("Model doesn't support probability predictions. SHAP analysis may not work properly.")
                
        except Exception as e:
            print(f"Error initializing SHAP explainer: {str(e)}")
            print("Falling back to feature importance analysis...")
    
    def get_top_features(self, n=10):
        """Get top N most important features based on SHAP values"""
        if self.shap_values is None:
            print("SHAP values not available. Returning feature importance instead.")
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': self.X_train.columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                return feature_importance.head(n).to_dict('records')
            else:
                return []
        
        # Calculate mean absolute SHAP values for each feature
        mean_shap_values = np.mean(np.abs(self.shap_values), axis=0)
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'shap_importance': mean_shap_values
        }).sort_values('shap_importance', ascending=False)
        
        return feature_importance.head(n).to_dict('records')
    
    def get_feature_effects(self, feature_name, n_samples=100):
        """Get SHAP effects for a specific feature"""
        if self.shap_values is None:
            return None
        
        if feature_name not in self.X_train.columns:
            print(f"Feature '{feature_name}' not found in dataset.")
            return None
        
        # Get feature index
        feature_idx = self.X_train.columns.get_loc(feature_name)
        
        # Get feature values and SHAP values
        feature_values = self.X_sample.iloc[:n_samples, feature_idx]
        shap_values = self.shap_values[:n_samples, feature_idx]
        
        return {
            'feature_values': feature_values.tolist(),
            'shap_values': shap_values.tolist(),
            'feature_name': feature_name
        }
    
    def plot_summary_plot(self, save_path=None):
        """Generate SHAP summary plot"""
        if self.shap_values is None:
            print("SHAP values not available.")
            return None
        
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                self.shap_values, 
                self.X_sample,
                plot_type="bar",
                show=False
            )
            plt.title("SHAP Feature Importance Summary")
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            print(f"Error generating summary plot: {str(e)}")
    
    def plot_waterfall_plot(self, sample_idx=0, save_path=None):
        """Generate SHAP waterfall plot for a specific sample"""
        if self.shap_values is None:
            print("SHAP values not available.")
            return None
        
        try:
            plt.figure(figsize=(12, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=self.shap_values[sample_idx],
                    base_values=self.expected_value,
                    data=self.X_sample.iloc[sample_idx],
                    feature_names=self.X_sample.columns
                ),
                show=False
            )
            plt.title(f"SHAP Waterfall Plot - Sample {sample_idx}")
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            print(f"Error generating waterfall plot: {str(e)}")
    
    def plot_dependence_plot(self, feature_name, save_path=None):
        """Generate SHAP dependence plot for a specific feature"""
        if self.shap_values is None:
            print("SHAP values not available.")
            return None
        
        if feature_name not in self.X_train.columns:
            print(f"Feature '{feature_name}' not found in dataset.")
            return None
        
        try:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature_name,
                self.shap_values,
                self.X_sample,
                show=False
            )
            plt.title(f"SHAP Dependence Plot - {feature_name}")
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            print(f"Error generating dependence plot: {str(e)}")
    
    def plot_force_plot(self, sample_idx=0, save_path=None):
        """Generate SHAP force plot for a specific sample"""
        if self.shap_values is None:
            print("SHAP values not available.")
            return None
        
        try:
            plt.figure(figsize=(12, 6))
            shap.force_plot(
                self.expected_value,
                self.shap_values[sample_idx],
                self.X_sample.iloc[sample_idx],
                show=False
            )
            plt.title(f"SHAP Force Plot - Sample {sample_idx}")
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            print(f"Error generating force plot: {str(e)}")
    
    def get_comprehensive_analysis(self):
        """Get comprehensive SHAP analysis results"""
        analysis_results = {
            'top_features': self.get_top_features(10),
            'feature_effects': {},
            'summary_stats': {}
        }
        
        # Get effects for top 5 features
        top_features = self.get_top_features(5)
        for feature_info in top_features:
            feature_name = feature_info['feature']
            effects = self.get_feature_effects(feature_name, n_samples=50)
            if effects:
                analysis_results['feature_effects'][feature_name] = effects
        
        # Calculate summary statistics
        if self.shap_values is not None:
            analysis_results['summary_stats'] = {
                'total_features': len(self.X_train.columns),
                'samples_analyzed': len(self.X_sample),
                'mean_abs_shap': np.mean(np.abs(self.shap_values)),
                'std_abs_shap': np.std(np.abs(self.shap_values))
            }
        
        return analysis_results
    
    def explain_prediction(self, X_sample):
        """Explain prediction for a specific sample"""
        if self.explainer is None:
            print("SHAP explainer not available.")
            return None
        
        try:
            # Get SHAP values for the sample
            sample_shap_values = self.explainer.shap_values(X_sample)
            if isinstance(sample_shap_values, list):
                sample_shap_values = np.mean(sample_shap_values, axis=0)
            
            # Get prediction
            prediction = self.model.predict(X_sample)[0]
            probability = self.model.predict_proba(X_sample)[0]
            
            # Get feature contributions
            feature_contributions = []
            for i, feature in enumerate(X_sample.columns):
                contribution = sample_shap_values[0, i]
                feature_contributions.append({
                    'feature': feature,
                    'value': X_sample.iloc[0, i],
                    'contribution': contribution,
                    'abs_contribution': abs(contribution)
                })
            
            # Sort by absolute contribution
            feature_contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
            
            return {
                'prediction': int(prediction),
                'probability': probability.tolist(),
                'feature_contributions': feature_contributions,
                'expected_value': self.expected_value.tolist() if isinstance(self.expected_value, np.ndarray) else self.expected_value
            }
            
        except Exception as e:
            print(f"Error explaining prediction: {str(e)}")
            return None
    
    def get_interaction_effects(self, feature1, feature2):
        """Get SHAP interaction effects between two features"""
        if self.shap_values is None:
            print("SHAP values not available.")
            return None
        
        if feature1 not in self.X_train.columns or feature2 not in self.X_train.columns:
            print("One or both features not found in dataset.")
            return None
        
        try:
            # Calculate interaction effects
            interaction_values = shap.TreeExplainer(self.model).shap_interaction_values(self.X_sample)
            
            # Get feature indices
            idx1 = self.X_train.columns.get_loc(feature1)
            idx2 = self.X_train.columns.get_loc(feature2)
            
            # Extract interaction values
            interaction_effect = interaction_values[:, idx1, idx2]
            
            return {
                'feature1': feature1,
                'feature2': feature2,
                'interaction_values': interaction_effect.tolist(),
                'mean_interaction': np.mean(interaction_effect),
                'std_interaction': np.std(interaction_effect)
            }
            
        except Exception as e:
            print(f"Error calculating interaction effects: {str(e)}")
            return None 