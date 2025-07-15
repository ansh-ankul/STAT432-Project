#!/usr/bin/env python3
"""
Test script for the ML Pipeline
This script tests the basic functionality of the ML pipeline components.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_sample_data():
    """Create sample data for testing"""
    print("Creating sample data...")
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=2,
        n_classes=4,
        random_state=42
    )
    
    # Create feature names
    feature_names = [
        'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
        'Distance(mi)', 'Start_Lat', 'Start_Lng', 'Zipcode', 'Airport_Code',
        'Weather_Condition', 'City_Freq_Enc', 'State', 'Wind_Direction',
        'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['Severity'] = y + 1  # Make severity 1-4 instead of 0-3
    
    # Add some categorical data
    df['Weather_Condition'] = np.random.choice(['Clear', 'Rainy', 'Cloudy', 'Snowy'], size=len(df))
    df['State'] = np.random.choice(['CA', 'FL'], size=len(df))
    df['City'] = np.random.choice(['Los Angeles', 'San Francisco', 'Miami', 'Orlando'], size=len(df))
    
    # Add some realistic constraints
    df['Temperature(F)'] = np.clip(df['Temperature(F)'] * 10 + 70, -20, 120)
    df['Humidity(%)'] = np.clip(df['Humidity(%)'] * 10 + 50, 0, 100)
    df['Visibility(mi)'] = np.clip(df['Visibility(mi)'] * 2 + 10, 0, 20)
    df['Wind_Speed(mph)'] = np.clip(df['Wind_Speed(mph)'] * 5 + 10, 0, 50)
    
    return df

def test_data_processor():
    """Test the data processor"""
    print("\n=== Testing Data Processor ===")
    
    try:
        from ml_pipeline.data_processor import DataProcessor
        
        # Create sample data
        sample_data = create_sample_data()
        
        # Initialize processor
        processor = DataProcessor()
        
        # Test preprocessing
        processed_data = processor.preprocess_data(sample_data)
        print(f"✓ Data preprocessing completed. Shape: {processed_data.shape}")
        
        # Test data splitting
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(processed_data)
        print(f"✓ Data splitting completed:")
        print(f"  - Train: {X_train.shape[0]} samples")
        print(f"  - Validation: {X_val.shape[0]} samples")
        print(f"  - Test: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    except Exception as e:
        print(f"✗ Data processor test failed: {str(e)}")
        return None

def test_model_trainer(X_train, y_train, X_val, y_val):
    """Test the model trainer"""
    print("\n=== Testing Model Trainer ===")
    
    try:
        from ml_pipeline.model_trainer import ModelTrainer
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Test model training (with reduced parameters for speed)
        print("Training Random Forest model...")
        model = trainer.train_model(X_train, y_train, X_val, y_val, model_type='random_forest')
        print("✓ Model training completed")
        
        # Test model evaluation
        metrics = trainer.evaluate_model(model, X_val, y_val)
        print(f"✓ Model evaluation completed. Accuracy: {metrics['accuracy']:.4f}")
        
        # Test feature importance
        feature_importance = trainer.get_feature_importance(10)
        if feature_importance is not None:
            print(f"✓ Feature importance extracted. Top features: {len(feature_importance)}")
        
        return model, trainer
        
    except Exception as e:
        print(f"✗ Model trainer test failed: {str(e)}")
        return None, None

def test_shap_analyzer(model, X_train):
    """Test the SHAP analyzer"""
    print("\n=== Testing SHAP Analyzer ===")
    
    try:
        from ml_pipeline.shap_analyzer import SHAPAnalyzer
        
        # Initialize SHAP analyzer with smaller sample size
        shap_analyzer = SHAPAnalyzer(model, X_train, sample_size=100)
        
        # Test top features
        top_features = shap_analyzer.get_top_features(10)
        print(f"✓ SHAP analysis completed. Top features: {len(top_features)}")
        
        # Test comprehensive analysis
        analysis = shap_analyzer.get_comprehensive_analysis()
        print(f"✓ Comprehensive SHAP analysis completed")
        
        return shap_analyzer
        
    except Exception as e:
        print(f"✗ SHAP analyzer test failed: {str(e)}")
        return None

def test_visualizer():
    """Test the visualizer"""
    print("\n=== Testing Visualizer ===")
    
    try:
        from ml_pipeline.visualizer import Visualizer
        
        # Create sample data
        california_data = create_sample_data()
        florida_data = create_sample_data()
        
        # Initialize visualizer
        visualizer = Visualizer()
        
        # Test visualization generation
        viz_data = visualizer.generate_all_visualizations(california_data, florida_data)
        print(f"✓ Visualization generation completed")
        print(f"  - Data overview: {'data_overview' in viz_data}")
        print(f"  - Severity distribution: {'severity_distribution' in viz_data}")
        print(f"  - Weather analysis: {'weather_analysis' in viz_data}")
        
        return visualizer
        
    except Exception as e:
        print(f"✗ Visualizer test failed: {str(e)}")
        return None

def test_flask_app():
    """Test Flask app imports"""
    print("\n=== Testing Flask App ===")
    
    try:
        # Test if Flask app can be imported
        import app
        print("✓ Flask app imports successful")
        
        # Test if routes are defined
        routes = [rule.rule for rule in app.app.url_map.iter_rules()]
        expected_routes = ['/', '/train_model', '/test_florida', '/get_visualizations', '/get_shap_analysis']
        
        for route in expected_routes:
            if route in routes:
                print(f"✓ Route {route} found")
            else:
                print(f"✗ Route {route} missing")
        
        return True
        
    except Exception as e:
        print(f"✗ Flask app test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting ML Pipeline Tests")
    print("=" * 50)
    
    # Test data processor
    result = test_data_processor()
    if result is None:
        print("❌ Data processor test failed. Stopping tests.")
        return
    
    X_train, X_val, X_test, y_train, y_val, y_test = result
    
    # Test model trainer
    model, trainer = test_model_trainer(X_train, y_train, X_val, y_val)
    if model is None:
        print("❌ Model trainer test failed. Stopping tests.")
        return
    
    # Test SHAP analyzer
    shap_analyzer = test_shap_analyzer(model, X_train)
    if shap_analyzer is None:
        print("⚠️  SHAP analyzer test failed. Continuing with other tests.")
    
    # Test visualizer
    visualizer = test_visualizer()
    if visualizer is None:
        print("⚠️  Visualizer test failed. Continuing with other tests.")
    
    # Test Flask app
    flask_ok = test_flask_app()
    if not flask_ok:
        print("⚠️  Flask app test failed. Continuing with other tests.")
    
    print("\n" + "=" * 50)
    print("🎉 Test Summary:")
    print("✓ Data Processor: Working")
    print("✓ Model Trainer: Working")
    if shap_analyzer:
        print("✓ SHAP Analyzer: Working")
    else:
        print("⚠️  SHAP Analyzer: Issues detected")
    if visualizer:
        print("✓ Visualizer: Working")
    else:
        print("⚠️  Visualizer: Issues detected")
    if flask_ok:
        print("✓ Flask App: Working")
    else:
        print("⚠️  Flask App: Issues detected")
    
    print("\n🚀 Pipeline is ready to use!")
    print("Run 'python app.py' to start the Flask application.")

if __name__ == "__main__":
    main() 