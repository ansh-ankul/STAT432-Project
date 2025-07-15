import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        
    def train_model(self, X_train, y_train, X_val=None, y_val=None, model_type='random_forest'):
        """Train the specified model type"""
        print(f"Training {model_type} model...")
        
        if model_type == 'random_forest':
            model = self._train_random_forest(X_train, y_train, X_val, y_val)
        elif model_type == 'gradient_boosting':
            model = self._train_gradient_boosting(X_train, y_train, X_val, y_val)
        elif model_type == 'logistic_regression':
            model = self._train_logistic_regression(X_train, y_train, X_val, y_val)
        elif model_type == 'svm':
            model = self._train_svm(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.best_model = model
        self.models[model_type] = model
        
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        print(f"{model_type} training completed!")
        return model
    
    def _train_random_forest(self, X_train, y_train, X_val=None, y_val=None):
        """Train Random Forest model with hyperparameter tuning"""
        print("Training Random Forest...")
        
        # Define parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'random_state': [42]
        }
        
        # Initialize Random Forest
        rf = RandomForestClassifier()
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _train_gradient_boosting(self, X_train, y_train, X_val=None, y_val=None):
        """Train Gradient Boosting model"""
        print("Training Gradient Boosting...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 0.05],
            'max_depth': [3, 5],
            'random_state': [42]
        }
        
        gb = GradientBoostingClassifier()
        
        grid_search = GridSearchCV(
            estimator=gb,
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _train_logistic_regression(self, X_train, y_train, X_val=None, y_val=None):
        """Train Logistic Regression model"""
        print("Training Logistic Regression...")
        
        param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'random_state': [42]
        }
        
        lr = LogisticRegression(max_iter=1000)
        
        grid_search = GridSearchCV(
            estimator=lr,
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _train_svm(self, X_train, y_train, X_val=None, y_val=None):
        """Train SVM model"""
        print("Training SVM...")
        
        # For large datasets, use LinearSVC instead of SVC
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        
        param_grid = {
            'C': [0.1, 1, 10],
            'random_state': [42]
        }
        
        svc = LinearSVC()
        
        grid_search = GridSearchCV(
            estimator=svc,
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Wrap with CalibratedClassifierCV to get probability estimates
        calibrated_svc = CalibratedClassifierCV(grid_search.best_estimator_)
        calibrated_svc.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return calibrated_svc
    
    def evaluate_model(self, model, X, y):
        """Evaluate model performance"""
        print("Evaluating model...")
        
        # Make predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y, y_pred)
        
        # Calculate ROC AUC for each class
        roc_auc_scores = {}
        if y_prob is not None:
            y_bin = label_binarize(y, classes=sorted(y.unique()))
            n_classes = y_bin.shape[1]
            
            for i in range(n_classes):
                roc_auc_scores[f'class_{i+1}'] = roc_auc_score(y_bin[:, i], y_prob[:, i])
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'roc_auc_scores': roc_auc_scores
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y, y_pred))
        
        return metrics
    
    def get_feature_importance(self, top_n=20):
        """Get top N most important features"""
        if self.feature_importance is None:
            print("No feature importance available. Train a model first.")
            return None
        
        return self.feature_importance.head(top_n)
    
    def plot_feature_importance(self, top_n=20, save_path=None):
        """Plot feature importance"""
        if self.feature_importance is None:
            print("No feature importance available. Train a model first.")
            return None
        
        top_features = self.feature_importance.head(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return top_features
    
    def plot_confusion_matrix(self, X, y, save_path=None):
        """Plot confusion matrix"""
        if self.best_model is None:
            print("No trained model available.")
            return None
        
        y_pred = self.best_model.predict(X)
        conf_matrix = confusion_matrix(y, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return conf_matrix
    
    def plot_roc_curves(self, X, y, save_path=None):
        """Plot ROC curves for all classes"""
        if self.best_model is None:
            print("No trained model available.")
            return None
        
        if not hasattr(self.best_model, 'predict_proba'):
            print("Model doesn't support probability predictions.")
            return None
        
        y_prob = self.best_model.predict_proba(X)
        y_bin = label_binarize(y, classes=sorted(y.unique()))
        n_classes = y_bin.shape[1]
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            auc_score = roc_auc_score(y_bin[:, i], y_prob[:, i])
            plt.plot(fpr, tpr, label=f'Class {i+1} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Classes')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.best_model is None:
            print("No trained model to save.")
            return False
        
        import joblib
        joblib.dump(self.best_model, filepath)
        print(f"Model saved to {filepath}")
        return True
    
    def load_model(self, filepath):
        """Load a trained model"""
        import joblib
        self.best_model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return self.best_model 