import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import io
import warnings
warnings.filterwarnings('ignore')

class Visualizer:
    def __init__(self):
        self.setup_plotting_style()
    
    def setup_plotting_style(self):
        """Setup consistent plotting style"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def generate_eda_analysis(self, data):
        """Generate EDA visualizations: elbow plot, cluster map, histograms, boxplots, skewness bar plot"""
        eda_viz = {}
        # 1. K-Means Elbow Plot (WCSS vs K)
        if 'Start_Lat' in data.columns and 'Start_Lng' in data.columns:
            X = data[['Start_Lat', 'Start_Lng']].dropna()
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            wcss = []
            max_clusters = 10
            for i in range(1, max_clusters + 1):
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=i, random_state=0)
                kmeans.fit(X_scaled)
                wcss.append(kmeans.inertia_)
            elbow_fig = self.create_plotly_figure(
                'line',
                pd.DataFrame({'K': list(range(1, max_clusters + 1)), 'WCSS': wcss}),
                x='K', y='WCSS', markers=True, title='K-Means Elbow Plot'
            )
            eda_viz['kmeans_elbow_plot'] = self.save_plot_as_base64(elbow_fig)
            # 2. K-Means Cluster Map
            k = 6
            kmeans = KMeans(n_clusters=k, random_state=0)
            clusters = kmeans.fit_predict(X_scaled)
            sample_idx = np.random.choice(len(X), min(10000, len(X)), replace=False)
            sample = X.iloc[sample_idx]
            sample_clusters = clusters[sample_idx]
            cluster_fig = px.scatter(
                sample, x='Start_Lng', y='Start_Lat', color=sample_clusters.astype(str),
                title='K-Means Cluster Map (Lat/Lng)', labels={'color': 'Cluster'}
            )
            eda_viz['kmeans_cluster_map'] = self.save_plot_as_base64(cluster_fig)
        # 3. Histograms for all numerical features
        num_cols = data.select_dtypes(include=[np.number]).columns
        histograms = {}
        for col in num_cols:
            fig = px.histogram(data, x=col, nbins=30, title=f'Histogram: {col}')
            histograms[col] = self.save_plot_as_base64(fig)
        eda_viz['histograms'] = histograms
        # 4. Boxplots for selected features
        box_cols = [c for c in ['Distance(mi)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)'] if c in data.columns]
        boxplots = {}
        for col in box_cols:
            fig = px.box(data, y=col, title=f'Boxplot: {col}')
            boxplots[col] = self.save_plot_as_base64(fig)
        eda_viz['boxplots'] = boxplots
        # 5. Skewness Bar Plot
        skew_vals = data[num_cols].skew()
        skew_fig = px.bar(x=skew_vals.index, y=skew_vals.values, title='Feature Skewness', labels={'x': 'Feature', 'y': 'Skewness'})
        eda_viz['skewness_bar'] = self.save_plot_as_base64(skew_fig)
        return eda_viz

    def generate_all_visualizations(self, california_data, florida_data, shap_analyzer=None):
        """Generate all visualizations for the dashboard"""
        viz_data = {}
        
        # Data overview visualizations
        viz_data['data_overview'] = self.generate_data_overview(california_data, florida_data)
        
        # Severity distribution
        viz_data['severity_distribution'] = self.generate_severity_distribution(california_data, florida_data)
        
        # Weather analysis
        viz_data['weather_analysis'] = self.generate_weather_analysis(california_data)
        
        # Temporal analysis
        viz_data['temporal_analysis'] = self.generate_temporal_analysis(california_data)
        
        # Geographic analysis
        viz_data['geographic_analysis'] = self.generate_geographic_analysis(california_data)
        
        # Feature correlation
        viz_data['correlation_analysis'] = self.generate_correlation_analysis(california_data)
        
        # SHAP analysis (if available)
        if shap_analyzer:
            viz_data['shap_analysis'] = self.generate_shap_visualizations(shap_analyzer)
        
        viz_data['eda_analysis'] = self.generate_eda_analysis(california_data)
        
        return viz_data
    
    def generate_data_overview(self, california_data, florida_data):
        """Generate data overview visualizations"""
        overview = {}
        
        # Dataset sizes
        overview['dataset_sizes'] = {
            'california': len(california_data),
            'florida': len(florida_data),
            'total': len(california_data) + len(florida_data)
        }
        
        # Feature comparison
        ca_features = set(california_data.columns)
        fl_features = set(florida_data.columns)
        common_features = ca_features.intersection(fl_features)
        
        overview['feature_comparison'] = {
            'california_only': list(ca_features - fl_features),
            'florida_only': list(fl_features - ca_features),
            'common': list(common_features),
            'total_california': len(ca_features),
            'total_florida': len(fl_features)
        }
        
        # Data quality metrics
        overview['data_quality'] = {
            'california_missing': california_data.isnull().sum().sum(),
            'florida_missing': florida_data.isnull().sum().sum(),
            'california_duplicates': california_data.duplicated().sum(),
            'florida_duplicates': florida_data.duplicated().sum()
        }
        
        return overview
    
    def generate_severity_distribution(self, california_data, florida_data):
        """Generate severity distribution visualizations"""
        severity_viz = {}
        
        # California severity distribution
        if 'Severity' in california_data.columns:
            ca_severity_counts = california_data['Severity'].value_counts().sort_index()
            severity_viz['california_severity'] = {
                'labels': ca_severity_counts.index.tolist(),
                'values': ca_severity_counts.values.tolist(),
                'percentages': (ca_severity_counts / len(california_data) * 100).tolist()
            }
        
        # Florida severity distribution
        if 'Severity' in florida_data.columns:
            fl_severity_counts = florida_data['Severity'].value_counts().sort_index()
            severity_viz['florida_severity'] = {
                'labels': fl_severity_counts.index.tolist(),
                'values': fl_severity_counts.values.tolist(),
                'percentages': (fl_severity_counts / len(florida_data) * 100).tolist()
            }
        
        # Comparison plot data
        if 'Severity' in california_data.columns and 'Severity' in florida_data.columns:
            comparison_data = []
            for severity in sorted(set(california_data['Severity'].unique()) | set(florida_data['Severity'].unique())):
                ca_count = len(california_data[california_data['Severity'] == severity])
                fl_count = len(florida_data[florida_data['Severity'] == severity])
                comparison_data.append({
                    'severity': severity,
                    'california': ca_count,
                    'florida': fl_count
                })
            severity_viz['comparison'] = comparison_data
        
        return severity_viz
    
    def generate_weather_analysis(self, data):
        """Generate weather-related visualizations"""
        weather_viz = {}
        
        if 'Weather_Condition' in data.columns:
            # Weather condition distribution
            weather_counts = data['Weather_Condition'].value_counts().head(10)
            weather_viz['weather_distribution'] = {
                'labels': weather_counts.index.tolist(),
                'values': weather_counts.values.tolist()
            }
            
            # Weather vs Severity
            if 'Severity' in data.columns:
                weather_severity = data.groupby(['Weather_Condition', 'Severity']).size().unstack(fill_value=0)
                weather_viz['weather_severity'] = {
                    'weather_conditions': weather_severity.index.tolist(),
                    'severity_data': weather_severity.to_dict('records')
                }
        
        # Temperature analysis
        if 'Temperature(F)' in data.columns:
            temp_stats = data['Temperature(F)'].describe()
            weather_viz['temperature_stats'] = {
                'mean': temp_stats['mean'],
                'std': temp_stats['std'],
                'min': temp_stats['min'],
                'max': temp_stats['max'],
                'q25': temp_stats['25%'],
                'q75': temp_stats['75%']
            }
            
            # Temperature vs Severity
            if 'Severity' in data.columns:
                temp_severity = data.groupby('Severity')['Temperature(F)'].agg(['mean', 'std']).reset_index()
                weather_viz['temperature_severity'] = temp_severity.to_dict('records')
        
        return weather_viz
    
    def generate_temporal_analysis(self, data):
        """Generate temporal analysis visualizations"""
        temporal_viz = {}
        
        # Convert time columns if they exist
        time_columns = ['Start_Time', 'End_Time', 'Weather_Timestamp']
        for col in time_columns:
            if col in data.columns:
                try:
                    data[col] = pd.to_datetime(data[col])
                    data[f'{col}_hour'] = data[col].dt.hour
                    data[f'{col}_day'] = data[col].dt.day_name()
                    data[f'{col}_month'] = data[col].dt.month
                    data[f'{col}_year'] = data[col].dt.year
                except:
                    continue
        
        # Hourly distribution
        if 'Start_Time_hour' in data.columns:
            hourly_counts = data['Start_Time_hour'].value_counts().sort_index()
            temporal_viz['hourly_distribution'] = {
                'hours': hourly_counts.index.tolist(),
                'counts': hourly_counts.values.tolist()
            }
            
            # Hour vs Severity
            if 'Severity' in data.columns:
                hour_severity = data.groupby(['Start_Time_hour', 'Severity']).size().unstack(fill_value=0)
                temporal_viz['hour_severity'] = {
                    'hours': hour_severity.index.tolist(),
                    'severity_data': hour_severity.to_dict('records')
                }
        
        # Daily distribution
        if 'Start_Time_day' in data.columns:
            daily_counts = data['Start_Time_day'].value_counts()
            temporal_viz['daily_distribution'] = {
                'days': daily_counts.index.tolist(),
                'counts': daily_counts.values.tolist()
            }
        
        # Monthly distribution
        if 'Start_Time_month' in data.columns:
            monthly_counts = data['Start_Time_month'].value_counts().sort_index()
            temporal_viz['monthly_distribution'] = {
                'months': monthly_counts.index.tolist(),
                'counts': monthly_counts.values.tolist()
            }
        
        return temporal_viz
    
    def generate_geographic_analysis(self, data):
        """Generate geographic analysis visualizations"""
        geo_viz = {}
        
        # County analysis
        if 'County' in data.columns:
            county_counts = data['County'].value_counts().head(20)
            geo_viz['top_counties'] = {
                'counties': county_counts.index.tolist(),
                'counts': county_counts.values.tolist()
            }
            
            # County vs Severity
            if 'Severity' in data.columns:
                county_severity = data.groupby(['County', 'Severity']).size().unstack(fill_value=0)
                top_counties = county_counts.head(10).index
                county_severity_top = county_severity.loc[top_counties]
                geo_viz['county_severity'] = {
                    'counties': county_severity_top.index.tolist(),
                    'severity_data': county_severity_top.to_dict('records')
                }
        
        # City analysis
        if 'City' in data.columns:
            city_counts = data['City'].value_counts().head(20)
            geo_viz['top_cities'] = {
                'cities': city_counts.index.tolist(),
                'counts': city_counts.values.tolist()
            }
        
        # Location clustering (if coordinates exist)
        if 'Start_Lat' in data.columns and 'Start_Lng' in data.columns:
            # Sample for visualization
            sample_data = data.sample(min(10000, len(data)), random_state=42)
            geo_viz['location_sample'] = {
                'latitudes': sample_data['Start_Lat'].tolist(),
                'longitudes': sample_data['Start_Lng'].tolist(),
                'severities': sample_data['Severity'].tolist() if 'Severity' in sample_data.columns else None
            }
        
        return geo_viz
    
    def generate_correlation_analysis(self, data):
        """Generate correlation analysis visualizations"""
        correlation_viz = {}
        
        # Select numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) > 1:
            # Calculate correlation matrix
            corr_matrix = data[numerical_cols].corr()
            
            # Get top correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.1:  # Only significant correlations
                        corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            # Sort by absolute correlation
            corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
            correlation_viz['top_correlations'] = corr_pairs[:20]
            
            # Correlation with target
            if 'Severity' in numerical_cols:
                severity_corr = corr_matrix['Severity'].drop('Severity').sort_values(key=abs, ascending=False)
                correlation_viz['severity_correlations'] = {
                    'features': severity_corr.index.tolist(),
                    'correlations': severity_corr.values.tolist()
                }
        
        return correlation_viz
    
    def generate_shap_visualizations(self, shap_analyzer):
        """Generate SHAP-related visualizations"""
        shap_viz = {}
        
        # Get top features
        top_features = shap_analyzer.get_top_features(10)
        shap_viz['top_features'] = top_features
        
        # Get feature effects for top features
        feature_effects = {}
        for feature_info in top_features[:5]:
            feature_name = feature_info['feature']
            effects = shap_analyzer.get_feature_effects(feature_name, n_samples=100)
            if effects:
                feature_effects[feature_name] = effects
        
        shap_viz['feature_effects'] = feature_effects
        
        return shap_viz
    
    def create_plotly_figure(self, fig_type, data, **kwargs):
        """Create a Plotly figure based on type and data"""
        if fig_type == 'bar':
            fig = px.bar(data, **kwargs)
        elif fig_type == 'line':
            fig = px.line(data, **kwargs)
        elif fig_type == 'scatter':
            fig = px.scatter(data, **kwargs)
        elif fig_type == 'histogram':
            fig = px.histogram(data, **kwargs)
        elif fig_type == 'box':
            fig = px.box(data, **kwargs)
        elif fig_type == 'heatmap':
            fig = px.imshow(data, **kwargs)
        else:
            raise ValueError(f"Unknown figure type: {fig_type}")
        
        fig.update_layout(
            template='plotly_white',
            font=dict(size=12),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    
    def save_plot_as_base64(self, fig, format='png'):
        """Save a Plotly figure as base64 string"""
        img_bytes = fig.to_image(format=format)
        img_base64 = base64.b64encode(img_bytes).decode()
        return f"data:image/{format};base64,{img_base64}"
    
    def create_matplotlib_plot(self, plot_func, save_path=None, **kwargs):
        """Create a matplotlib plot and optionally save it"""
        plt.figure(figsize=(10, 6))
        plot_func(**kwargs)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_base64}" 