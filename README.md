# Accident Severity Prediction - ML Pipeline Dashboard

A comprehensive machine learning pipeline for predicting accident severity using California and Florida traffic accident data. This project includes a Flask web application with interactive visualizations, SHAP analysis, and cross-state model validation.

## 🚀 Features

- **Multi-class Classification**: Predicts accident severity levels (1-4)
- **Advanced Preprocessing**: Feature engineering, categorical encoding, and data cleaning
- **Cross-State Validation**: Train on California data, test on Florida data
- **SHAP Analysis**: Model interpretability with top 10 feature importance
- **Interactive Dashboard**: Modern web interface with real-time visualizations
- **Comprehensive Metrics**: Accuracy, confusion matrix, ROC curves, and more
- **Exploratory Data Analysis (EDA)**: Includes K-Means Elbow Plot, Cluster Map, Histograms, Boxplots, and Skewness Bar Plot for in-depth data understanding

## 📸 App Screenshots

<div align="center">
  <img src="App Preview/dashboard-overview.png" alt="Dashboard Overview" width="45%">
  <img src="App Preview/model-performance.png" alt="Model Performance" width="45%">
  <img src="App Preview/visualizations.png" alt="Visualizations" width="45%">
  <img src="App Preview/shap-analysis.png" alt="SHAP Analysis" width="45%">
</div>

*Dashboard showcasing model performance, visualizations, and SHAP analysis for accident severity prediction*

## 📁 Project Structure

```
STAT432 Project/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── data/
│   ├── california_data.csv         # California accident data
│   └── florida_data.csv           # Florida accident data
├── ml_pipeline/
│   ├── __init__.py
│   ├── data_processor.py          # Data preprocessing and feature engineering
│   ├── model_trainer.py           # Model training and evaluation
│   ├── shap_analyzer.py           # SHAP analysis for model interpretability
│   └── visualizer.py              # Data visualization generation
├── templates/
│   └── index.html                 # Main dashboard template
├── static/
│   ├── css/                       # Custom CSS styles
│   ├── js/
│   │   └── dashboard.js           # Dashboard JavaScript functionality
│   └── images/                    # Generated visualizations
└── models/                        # Saved trained models
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "STAT432 Project"
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your data**:
   - Place your `california_data.csv` and `florida_data.csv` files in the `data/` directory
   - Ensure the data files contain the required columns (see Data Requirements below)

## 🚀 Usage

1. **Start the Flask application**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Use the dashboard**:
   - Click "Train Model" to train the ML model on California data
   - Click "Test on Florida Data" to evaluate cross-state performance
   - Click "Generate Visualizations" to see data insights
   - Click "SHAP Analysis" to understand feature importance

## 📊 Data Requirements

Your CSV files should contain the following columns:

### Required Columns:
- `Severity`: Target variable (1-4, where 4 is most severe)
- `Start_Lat`, `Start_Lng`: Accident location coordinates
- `Temperature(F)`: Temperature in Fahrenheit
- `Humidity(%)`: Humidity percentage
- `Pressure(in)`: Atmospheric pressure
- `Visibility(mi)`: Visibility in miles
- `Wind_Speed(mph)`: Wind speed in mph
- `Distance(mi)`: Distance of the accident

### Optional Columns:
- `Weather_Condition`: Weather conditions
- `City`, `County`, `State`: Geographic information
- `Zipcode`: ZIP code
- `Street`: Street name
- `Airport_Code`: Nearby airport code
- `Start_Time`, `End_Time`: Time information
- Various boolean flags (Amenity, Bump, Crossing, etc.)

## 🔧 ML Pipeline Components

### 1. Data Processor (`ml_pipeline/data_processor.py`)
- **Feature Engineering**: Weather condition binning, location clustering
- **Categorical Encoding**: Target encoding for high-cardinality features
- **Data Cleaning**: Missing value handling, outlier detection
- **Data Splitting**: Train/validation/test splits with stratification

### 2. Model Trainer (`ml_pipeline/model_trainer.py`)
- **Multiple Algorithms**: Random Forest, Gradient Boosting, Logistic Regression, SVM
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Model Evaluation**: Accuracy, confusion matrix, ROC curves
- **Feature Importance**: Model-agnostic feature ranking

### 3. SHAP Analyzer (`ml_pipeline/shap_analyzer.py`)
- **Model Interpretability**: SHAP values for feature importance
- **Top 10 Features**: Most important factors for accident severity
- **Feature Effects**: Individual feature impact analysis
- **Prediction Explanations**: Sample-specific model explanations

### 4. Visualizer (`ml_pipeline/visualizer.py`)
- **Data Overview**: Dataset statistics and quality metrics
- **Severity Distribution**: Accident severity patterns
- **Weather Analysis**: Weather impact on accidents
- **Temporal Analysis**: Time-based accident patterns
- **Geographic Analysis**: Location-based insights
- **Correlation Analysis**: Feature relationships

## 📈 Model Performance

The pipeline typically achieves:
- **Training Accuracy**: 70-85%
- **Cross-State Accuracy**: 65-80%
- **Top Features**: Weather conditions, location, time, and environmental factors

## 🎯 Key Insights

### Top 10 Reasons for Accidents (SHAP Analysis):
1. **Weather Conditions**: Rain, snow, and fog significantly increase severity
2. **Location Clusters**: High-traffic areas show higher accident rates
3. **Time of Day**: Rush hours and nighttime driving
4. **Temperature**: Extreme temperatures affect driving conditions
5. **Visibility**: Poor visibility correlates with severe accidents
6. **Wind Speed**: High winds impact vehicle stability
7. **Humidity**: Moisture affects road conditions
8. **Pressure**: Weather system changes
9. **Distance**: Longer accidents tend to be more severe
10. **Geographic Features**: Urban vs rural differences

## 🔍 Dashboard Features

### Overview Tab:
- Project description and key features
- Data overview statistics

### Model Performance Tab:
- Training, validation, and test metrics
- Confusion matrix visualization
- ROC curves for multi-class classification

### Visualizations Tab:
- Severity distribution charts
- Weather analysis plots
- Temporal patterns
- Geographic distributions
- **Exploratory Data Analysis (EDA):**
  - K-Means Elbow Plot (for optimal cluster count)
  - K-Means Cluster Map (geospatial clusters)
  - Histograms for all numerical features
  - Boxplots for key features (Distance, Pressure, Visibility, Wind Speed, etc.)
  - Skewness Bar Plot (for all numeric features)

### SHAP Analysis Tab:
- Top 10 feature importance (order only, no values)
- Feature effect plots
- Individual prediction explanations

### Florida Testing Tab:
- Cross-state performance metrics
- Prediction distribution
- Generalization analysis

## 🛠️ Customization

### Adding New Models:
1. Extend `ModelTrainer` class in `ml_pipeline/model_trainer.py`
2. Add new model type to the `train_model` method
3. Update hyperparameter grids as needed

### Adding New Visualizations:
1. Extend `Visualizer` class in `ml_pipeline/visualizer.py`
2. Add new visualization methods
3. Update the dashboard JavaScript to display new charts

### Modifying Preprocessing:
1. Edit `DataProcessor` class in `ml_pipeline/data_processor.py`
2. Add new feature engineering steps
3. Update categorical encoding strategies

## 🐛 Troubleshooting

### Common Issues:

1. **Memory Error**: Reduce sample size in SHAP analyzer
   ```python
   shap_analyzer = SHAPAnalyzer(model, X_train, sample_size=500)
   ```

2. **Missing Dependencies**: Install additional packages
   ```bash
   pip install <package-name>
   ```

3. **Data Format Issues**: Check CSV encoding and column names
   ```python
   df = pd.read_csv('data.csv', encoding='utf-8')
   ```

4. **Model Training Time**: Use smaller parameter grids for faster training
   ```python
   param_grid = {'n_estimators': [100], 'max_depth': [10]}
   ```

## 📝 License

This project is for educational purposes. Please ensure you have proper permissions for the data used.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments
3. Open an issue on GitHub

---

**Note**: This project is designed for educational and research purposes. Always validate results and consider domain-specific factors when applying to real-world scenarios. 