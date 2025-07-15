from flask import Flask, render_template, jsonify, send_file
import os
import joblib
import json
import time
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Load pre-trained model and precomputed results on startup
global_model = None
global_metrics = None
global_shap = None
global_visualizations = None

MODEL_PATH = 'models/accident_model.pkl'
METRICS_PATH = 'models/metrics.json'
SHAP_PATH = 'models/shap_summary.json'
VIS_PATH = 'models/visualizations.json'

@app.before_first_request
def load_artifacts():
    global global_model, global_metrics, global_shap, global_visualizations
    # Load model
    if os.path.exists(MODEL_PATH):
        global_model = joblib.load(MODEL_PATH)
    # Load metrics
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            global_metrics = json.load(f)
    # Load SHAP
    if os.path.exists(SHAP_PATH):
        with open(SHAP_PATH, 'r') as f:
            global_shap = json.load(f)
    # Load visualizations
    if os.path.exists(VIS_PATH):
        with open(VIS_PATH, 'r') as f:
            global_visualizations = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train_model', methods=['POST'])
def train_model():
    # Simulate training delay
    time.sleep(3)  # 3 seconds delay to mimic training
    # Return precomputed results
    if global_metrics is None or global_shap is None:
        return jsonify({'status': 'error', 'message': 'Precomputed results not available.'})
    top_features = global_shap.get('top_features', [])
    return jsonify({
        'status': 'success',
        'message': 'Model loaded and results updated!',
        'metrics': global_metrics,
        'top_features': top_features
    })

@app.route('/get_metrics')
def get_metrics():
    if global_metrics is None:
        return jsonify({'status': 'error', 'message': 'Metrics not available.'})
    return jsonify({'status': 'success', 'metrics': global_metrics})

@app.route('/get_visualizations')
def get_visualizations():
    if global_visualizations is None:
        return jsonify({'status': 'error', 'message': 'Visualizations not available.'})
    return jsonify({'status': 'success', 'visualizations': global_visualizations})

@app.route('/get_shap_analysis')
def get_shap_analysis():
    if global_shap is None:
        return jsonify({'status': 'error', 'message': 'SHAP analysis not available.'})
    return jsonify({'status': 'success', 'shap_analysis': global_shap})

@app.route('/download_model')
def download_model():
    if not os.path.exists(MODEL_PATH):
        return jsonify({'status': 'error', 'message': 'No trained model available!'})
    return send_file(MODEL_PATH, as_attachment=True, download_name=f'accident_severity_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002) 