
import os
import json
import joblib
import pandas as pd
import numpy as np
from ml_pipeline.data_processor import DataProcessor
from ml_pipeline.model_trainer import ModelTrainer
from ml_pipeline.shap_analyzer import SHAPAnalyzer
from ml_pipeline.visualizer import Visualizer

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# Paths
CALIFORNIA_CSV = "data/california_data.csv"
MODEL_PATH = "models/accident_model.pkl"
METRICS_PATH = "models/metrics.json"
SHAP_PATH = "models/shap_summary.json"
VIS_PATH = "models/visualizations.json"

# 1. Load and preprocess data
processor = DataProcessor()
df = processor.load_data(CALIFORNIA_CSV)
df = processor.preprocess_data(df)
X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(df)

# 2. Train or load model
if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    model = joblib.load(MODEL_PATH)
else:
    print("Training new model...")
    trainer = ModelTrainer()
    model = trainer.train_model(X_train, y_train, X_val, y_val)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# 3. Compute evaluation metrics
trainer = ModelTrainer()
trainer.best_model = model
metrics = {
    "train": trainer.evaluate_model(model, X_train, y_train),
    "validation": trainer.evaluate_model(model, X_val, y_val),
    "test": trainer.evaluate_model(model, X_test, y_test)
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to {METRICS_PATH}")

# 4. SHAP analysis
shap_analyzer = SHAPAnalyzer(model, X_train, sample_size=500)
shap_summary = shap_analyzer.get_comprehensive_analysis()

# Convert numpy types for JSON serialization
shap_summary = convert_numpy_types(shap_summary)

with open(SHAP_PATH, "w") as f:
    json.dump(shap_summary, f, indent=2)
print(f"SHAP summary saved to {SHAP_PATH}")

# 5. Visualizations
visualizer = Visualizer()
visualizations = visualizer.generate_all_visualizations(
    pd.read_csv(CALIFORNIA_CSV),  # raw for overview
    pd.read_csv("data/florida_data.csv") if os.path.exists("data/florida_data.csv") else pd.DataFrame(),
    shap_analyzer
)

# Convert numpy types to native Python types for JSON serialization
visualizations = convert_numpy_types(visualizations)

with open(VIS_PATH, "w") as f:
    json.dump(visualizations, f, indent=2)
print(f"Visualizations saved to {VIS_PATH}")