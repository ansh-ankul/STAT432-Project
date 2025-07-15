// Dashboard JavaScript for Accident Severity Prediction ML Pipeline

class Dashboard {
    constructor() {
        this.initializeEventListeners();
        this.charts = {};
        this.currentData = null;
        this.modelLoaded = false;
    }

    initializeEventListeners() {
        document.getElementById('trainBtn').addEventListener('click', () => this.fakeTrainModel());
        document.getElementById('evalBtn').addEventListener('click', () => this.displayEvaluationResults());
        document.getElementById('visualizeBtn').addEventListener('click', () => this.generateVisualizations());
        document.getElementById('shapBtn').addEventListener('click', () => this.generateSHAPAnalysis());
    }

    showLoading(message = 'Processing...') {
        document.getElementById('loading').style.display = 'block';
        document.getElementById('loadingText').textContent = message;
    }

    hideLoading() {
        document.getElementById('loading').style.display = 'none';
    }

    showAlert(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        const container = document.querySelector('.container-fluid');
        container.insertBefore(alertDiv, container.firstChild);
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }

    async fakeTrainModel() {
        this.showLoading('Training model (loading weights)...');
        try {
            const response = await fetch('/train_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const result = await response.json();
            if (result.status === 'success') {
                this.showAlert('Model loaded and results updated!', 'success');
                this.updateMetrics(result);
                this.enableButtons();
                this.displayModelPerformance(result.metrics);
                this.modelLoaded = true;
            } else {
                this.showAlert(`Error: ${result.message}`, 'danger');
            }
        } catch (error) {
            this.showAlert(`Network error: ${error.message}`, 'danger');
        } finally {
            this.hideLoading();
        }
    }

    async displayEvaluationResults() {
        if (!this.modelLoaded) {
            this.showAlert('Please load the model first by clicking Train Model.', 'warning');
            return;
        }
        this.showLoading('Loading evaluation results...');
        try {
            const response = await fetch('/get_metrics');
            const result = await response.json();
            if (result.status === 'success') {
                this.displayModelPerformance(result.metrics);
            } else {
                this.showAlert(`Error: ${result.message}`, 'danger');
            }
        } catch (error) {
            this.showAlert(`Network error: ${error.message}`, 'danger');
        } finally {
            this.hideLoading();
        }
    }

    async generateVisualizations() {
        if (!this.modelLoaded) {
            this.showAlert('Please load the model first by clicking Train Model.', 'warning');
            return;
        }
        this.showLoading('Generating visualizations...');
        try {
            const response = await fetch('/get_visualizations');
            const result = await response.json();
            if (result.status === 'success') {
                this.showAlert('Visualizations generated successfully!', 'success');
                this.displayVisualizations(result.visualizations);
            } else {
                this.showAlert(`Error: ${result.message}`, 'danger');
            }
        } catch (error) {
            this.showAlert(`Network error: ${error.message}`, 'danger');
        } finally {
            this.hideLoading();
        }
    }

    async generateSHAPAnalysis() {
        if (!this.modelLoaded) {
            this.showAlert('Please load the model first by clicking Train Model.', 'warning');
            return;
        }
        this.showLoading('Generating SHAP analysis...');
        try {
            const response = await fetch('/get_shap_analysis');
            const result = await response.json();
            if (result.status === 'success') {
                this.showAlert('SHAP analysis generated successfully!', 'success');
                this.displaySHAPAnalysis(result.shap_analysis);
            } else {
                this.showAlert(`Error: ${result.message}`, 'danger');
            }
        } catch (error) {
            this.showAlert(`Network error: ${error.message}`, 'danger');
        } finally {
            this.hideLoading();
        }
    }

    updateMetrics(result) {
        document.getElementById('modelStatus').textContent = 'Loaded';
        document.getElementById('modelStatus').style.color = '#27ae60';
        if (result.metrics && result.metrics.test) {
            const accuracy = (result.metrics.test.accuracy * 100).toFixed(2);
            document.getElementById('accuracy').textContent = `${accuracy}%`;
        }
        if (result.top_features) {
            document.getElementById('topFeatures').textContent = result.top_features.length;
        }
    }

    enableButtons() {
        document.getElementById('evalBtn').disabled = false;
        document.getElementById('visualizeBtn').disabled = false;
        document.getElementById('shapBtn').disabled = false;
    }

    displayModelPerformance(metrics) {
        const container = document.getElementById('modelPerformance');
        
        let html = '<div class="row">';
        
        // Training metrics
        if (metrics.train) {
            html += `
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">
                            <i class="fas fa-graduation-cap me-2"></i>Training Metrics
                        </div>
                        <div class="card-body">
                            <div class="metric-card">
                                <div class="metric-value">${(metrics.train.accuracy * 100).toFixed(2)}%</div>
                                <div class="metric-label">Accuracy</div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Validation metrics
        if (metrics.validation) {
            html += `
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">
                            <i class="fas fa-check-circle me-2"></i>Validation Metrics
                        </div>
                        <div class="card-body">
                            <div class="metric-card">
                                <div class="metric-value">${(metrics.validation.accuracy * 100).toFixed(2)}%</div>
                                <div class="metric-label">Accuracy</div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Test metrics
        if (metrics.test) {
            html += `
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">
                            <i class="fas fa-vial me-2"></i>Test Metrics
                        </div>
                        <div class="card-body">
                            <div class="metric-card">
                                <div class="metric-value">${(metrics.test.accuracy * 100).toFixed(2)}%</div>
                                <div class="metric-label">Accuracy</div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        html += '</div>';
        
        // Add confusion matrix if available
        if (metrics.test && metrics.test.confusion_matrix) {
            html += `
                <div class="row mt-4">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">
                                <i class="fas fa-table me-2"></i>Confusion Matrix (Test Set)
                            </div>
                            <div class="card-body">
                                <div class="table-responsive">
                                    <table class="table table-bordered">
                                        <thead>
                                            <tr>
                                                <th>Actual/Predicted</th>
                                                <th>Severity 1</th>
                                                <th>Severity 2</th>
                                                <th>Severity 3</th>
                                                <th>Severity 4</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            ${this.generateConfusionMatrixRows(metrics.test.confusion_matrix)}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        container.innerHTML = html;
    }

    generateConfusionMatrixRows(matrix) {
        let rows = '';
        for (let i = 0; i < matrix.length; i++) {
            rows += '<tr>';
            rows += `<td><strong>Severity ${i + 1}</strong></td>`;
            for (let j = 0; j < matrix[i].length; j++) {
                const cellClass = i === j ? 'table-success' : 'table-danger';
                rows += `<td class="${cellClass}">${matrix[i][j]}</td>`;
            }
            rows += '</tr>';
        }
        return rows;
    }

    displayFloridaResults(result) {
        const container = document.getElementById('floridaContent');
        
        let html = '<div class="row">';
        
        // Prediction distribution
        if (result.prediction_distribution) {
            html += `
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <i class="fas fa-chart-pie me-2"></i>Prediction Distribution
                        </div>
                        <div class="card-body">
                            <div class="chart-container">
                                <canvas id="floridaPredictionChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Metrics if available
        if (result.metrics && result.metrics.accuracy) {
            html += `
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <i class="fas fa-chart-line me-2"></i>Florida Performance
                        </div>
                        <div class="card-body">
                            <div class="metric-card">
                                <div class="metric-value">${(result.metrics.accuracy * 100).toFixed(2)}%</div>
                                <div class="metric-label">Accuracy on Florida Data</div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        html += '</div>';
        
        container.innerHTML = html;
        
        // Create prediction distribution chart
        if (result.prediction_distribution) {
            this.createPredictionChart(result.prediction_distribution);
        }
    }

    displayVisualizations(visualizations) {
        const container = document.getElementById('visualizationsContent');
        let html = '<div class="row">';
        // Severity distribution
        if (visualizations.severity_distribution) {
            html += `
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <i class="fas fa-chart-bar me-2"></i>Severity Distribution
                        </div>
                        <div class="card-body">
                            <div class="chart-container">
                                <canvas id="severityChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        // Weather analysis
        if (visualizations.weather_analysis) {
            html += `
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <i class="fas fa-cloud me-2"></i>Weather Analysis
                        </div>
                        <div class="card-body">
                            <div class="chart-container">
                                <canvas id="weatherChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        html += '</div>';
        // EDA Visualizations Section
        if (visualizations.eda_analysis) {
            html += '<div class="row mt-4">';
            html += '<div class="col-12"><h4><i class="fas fa-search me-2"></i>Exploratory Data Analysis (EDA)</h4></div>';
            // Elbow plot
            if (visualizations.eda_analysis.kmeans_elbow_plot) {
                html += `<div class="col-md-6"><div class="card mb-3"><div class="card-header">K-Means Elbow Plot</div><div class="card-body"><img src="${visualizations.eda_analysis.kmeans_elbow_plot}" class="img-fluid"/></div></div></div>`;
            }
            // Cluster map
            if (visualizations.eda_analysis.kmeans_cluster_map) {
                html += `<div class="col-md-6"><div class="card mb-3"><div class="card-header">K-Means Cluster Map</div><div class="card-body"><img src="${visualizations.eda_analysis.kmeans_cluster_map}" class="img-fluid"/></div></div></div>`;
            }
            // Skewness bar
            if (visualizations.eda_analysis.skewness_bar) {
                html += `<div class="col-md-6"><div class="card mb-3"><div class="card-header">Feature Skewness</div><div class="card-body"><img src="${visualizations.eda_analysis.skewness_bar}" class="img-fluid"/></div></div></div>`;
            }
            // Show a few histograms
            if (visualizations.eda_analysis.histograms) {
                let count = 0;
                for (const [col, img] of Object.entries(visualizations.eda_analysis.histograms)) {
                    if (count >= 2) break;
                    html += `<div class="col-md-6"><div class="card mb-3"><div class="card-header">Histogram: ${col}</div><div class="card-body"><img src="${img}" class="img-fluid"/></div></div></div>`;
                    count++;
                }
            }
            // Show a few boxplots
            if (visualizations.eda_analysis.boxplots) {
                let count = 0;
                for (const [col, img] of Object.entries(visualizations.eda_analysis.boxplots)) {
                    if (count >= 2) break;
                    html += `<div class="col-md-6"><div class="card mb-3"><div class="card-header">Boxplot: ${col}</div><div class="card-body"><img src="${img}" class="img-fluid"/></div></div></div>`;
                    count++;
                }
            }
            html += '</div>';
        }
        container.innerHTML = html;
        // Create charts
        if (visualizations.severity_distribution) {
            this.createSeverityChart(visualizations.severity_distribution);
        }
        if (visualizations.weather_analysis) {
            this.createWeatherChart(visualizations.weather_analysis);
        }
    }

    displaySHAPAnalysis(shapAnalysis) {
        const container = document.getElementById('shapContent');
        
        let html = '<div class="row">';
        
        // Top features
        if (shapAnalysis.top_features) {
            html += `
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <i class="fas fa-list-ol me-2"></i>Top 10 Feature Importance
                        </div>
                        <div class="card-body">
                            ${this.generateFeatureImportanceList(shapAnalysis.top_features)}
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Feature effects
        if (shapAnalysis.feature_effects) {
            html += `
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <i class="fas fa-chart-line me-2"></i>Feature Effects
                        </div>
                        <div class="card-body">
                            <div class="chart-container">
                                <canvas id="featureEffectsChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        html += '</div>';
        
        container.innerHTML = html;
        
        // Create feature effects chart
        if (shapAnalysis.feature_effects) {
            this.createFeatureEffectsChart(shapAnalysis.feature_effects);
        }
    }

    generateFeatureImportanceList(features) {
        let html = '';
        features.forEach((feature, index) => {
            html += `
                <div class="feature-importance-item">
                    <div class="d-flex align-items-center">
                        <strong>${index + 1}. ${feature.feature}</strong>
                    </div>
                </div>
            `;
        });
        return html;
    }

    createPredictionChart(predictionDistribution) {
        const ctx = document.getElementById('floridaPredictionChart').getContext('2d');
        
        const labels = Object.keys(predictionDistribution).map(key => `Severity ${key}`);
        const data = Object.values(predictionDistribution);
        
        this.charts.floridaPrediction = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: [
                        '#27ae60',
                        '#f39c12',
                        '#e67e22',
                        '#e74c3c'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    createSeverityChart(severityData) {
        const ctx = document.getElementById('severityChart').getContext('2d');
        
        let labels = [];
        let data = [];
        
        if (severityData.california_severity) {
            labels = severityData.california_severity.labels.map(l => `Severity ${l}`);
            data = severityData.california_severity.values;
        }
        
        this.charts.severity = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'California Accidents',
                    data: data,
                    backgroundColor: '#3498db'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    createWeatherChart(weatherData) {
        const ctx = document.getElementById('weatherChart').getContext('2d');
        
        let labels = [];
        let data = [];
        
        if (weatherData.weather_distribution) {
            labels = weatherData.weather_distribution.labels;
            data = weatherData.weather_distribution.values;
        }
        
        this.charts.weather = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: [
                        '#3498db',
                        '#e74c3c',
                        '#f39c12',
                        '#27ae60',
                        '#9b59b6'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    createFeatureEffectsChart(featureEffects) {
        const ctx = document.getElementById('featureEffectsChart').getContext('2d');
        
        // Get the first feature's effects for demonstration
        const firstFeature = Object.keys(featureEffects)[0];
        const effects = featureEffects[firstFeature];
        
        if (effects && effects.feature_values && effects.shap_values) {
            this.charts.featureEffects = new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: firstFeature,
                        data: effects.feature_values.map((x, i) => ({
                            x: x,
                            y: effects.shap_values[i]
                        })),
                        backgroundColor: '#3498db'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: `SHAP Values vs ${firstFeature}`
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: firstFeature
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'SHAP Value'
                            }
                        }
                    }
                }
            });
        }
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new Dashboard();
}); 