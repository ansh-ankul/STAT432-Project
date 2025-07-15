import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from category_encoders import TargetEncoder
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.target_encoder = TargetEncoder(smoothing=0.8)
        self.kmeans = KMeans(n_clusters=10, random_state=42)
        self.label_encoders = {}
        
    def load_data(self, file_path):
        """Load data from CSV file"""
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def preprocess_data(self, df):
        """Main preprocessing pipeline"""
        print("Starting data preprocessing...")
        
        # Step 1: Drop non-useful columns
        df = self._drop_unnecessary_columns(df)
        
        # Step 2: Handle categorical variables
        df = self._handle_categorical_variables(df)
        
        # Step 3: Handle numerical variables
        df = self._handle_numerical_variables(df)
        
        # Step 4: Handle location data
        df = self._handle_location_data(df)
        
        # Step 5: Handle time-based features
        df = self._handle_time_features(df)
        
        # Step 6: Final cleaning
        df = self._final_cleaning(df)
        
        print(f"Preprocessing complete. Final shape: {df.shape}")
        return df
    
    def _drop_unnecessary_columns(self, df):
        """Drop columns that are not useful for modeling"""
        columns_to_drop = [
            'Start_Time', 'End_Time', 'ID', 'Weather_Timestamp', 
            'Description', 'Country', 'Timezone', 'Source', 'End_Lat', 'End_Lng',
            'Wind_Chill(F)', 'Precipitation(in)', 'Sunrise_Sunset', 
            'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight'
        ]
        
        # Only drop columns that exist
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(columns=existing_columns)
        
        return df
    
    def _handle_categorical_variables(self, df):
        """Handle categorical variables with various encoding strategies"""
        
        # Weather condition binning
        if 'Weather_Condition' in df.columns:
            df = self._bin_weather_conditions(df)
        
        # City frequency encoding
        if 'City' in df.columns:
            # Handle missing cities
            df['City'] = df['City'].fillna('Unknown')
            city_freq = df['City'].value_counts() / len(df)
            df['City_Freq_Enc'] = df['City'].map(city_freq)
            df = df.drop(columns=['City'])
        
        # Target encoding for high-cardinality categorical variables
        high_cardinality_cols = ['Zipcode', 'Airport_Code', 'Street', 'County']
        for col in high_cardinality_cols:
            if col in df.columns:
                # Handle missing values before target encoding
                df[col] = df[col].fillna('Unknown')
                # Only apply target encoding if we have enough data
                if len(df[col].unique()) > 1:
                    try:
                        df[col] = self.target_encoder.fit_transform(df[[col]], df['Severity'])
                    except Exception as e:
                        print(f"Warning: Target encoding failed for {col}: {e}")
                        # Fallback to label encoding
                        df[col] = self._label_encode_column(df[col])
                else:
                    # If only one unique value, just encode it
                    df[col] = 0
        
        # Label encoding for low-cardinality categorical variables
        low_cardinality_cols = ['State', 'Wind_Direction']
        for col in low_cardinality_cols:
            if col in df.columns:
                # Handle missing values
                df[col] = df[col].fillna('Unknown')
                df[col] = self._label_encode_column(df[col])
        
        return df
    
    def _bin_weather_conditions(self, df):
        """Bin weather conditions into major categories"""
        weather_bins = {
            'Clear': ['Clear', 'Fair'],
            'Cloudy': ['Cloudy', 'Mostly Cloudy', 'Partly Cloudy', 'Scattered Clouds'],
            'Rainy': ['Light Rain', 'Rain', 'Light Freezing Drizzle', 'Light Drizzle', 
                     'Heavy Rain', 'Light Freezing Rain', 'Drizzle', 'Light Freezing Fog', 
                     'Light Rain Showers', 'Showers in the Vicinity', 'T-Storm', 'Thunder', 
                     'Patches of Fog', 'Heavy T-Storm', 'Heavy Thunderstorms and Rain', 
                     'Funnel Cloud', 'Heavy T-Storm / Windy', 'Heavy Thunderstorms and Snow', 
                     'Rain / Windy', 'Heavy Rain / Windy', 'Squalls', 'Heavy Ice Pellets', 
                     'Thunder / Windy', 'Drizzle and Fog', 'T-Storm / Windy', 'Smoke / Windy', 
                     'Haze / Windy', 'Light Drizzle / Windy', 'Widespread Dust / Windy', 
                     'Wintry Mix', 'Wintry Mix / Windy', 'Light Snow with Thunder', 
                     'Fog / Windy', 'Snow and Thunder', 'Sleet / Windy', 
                     'Heavy Freezing Rain / Windy', 'Squalls / Windy', 
                     'Light Rain Shower / Windy', 'Snow and Thunder / Windy', 
                     'Light Sleet / Windy', 'Sand / Dust Whirlwinds', 'Mist / Windy', 
                     'Drizzle / Windy', 'Duststorm', 'Sand / Dust Whirls Nearby', 
                     'Thunder and Hail', 'Freezing Rain / Windy', 
                     'Light Snow Shower / Windy', 'Partial Fog', 
                     'Thunder / Wintry Mix / Windy', 'Patches of Fog / Windy', 
                     'Rain and Sleet', 'Light Snow Grains', 'Partial Fog / Windy', 
                     'Sand / Dust Whirlwinds / Windy', 'Heavy Snow with Thunder', 
                     'Heavy Blowing Snow', 'Low Drifting Snow', 'Light Hail', 
                     'Light Thunderstorm', 'Heavy Freezing Drizzle', 'Light Blowing Snow', 
                     'Thunderstorms and Snow', 'Heavy Rain Showers', 'Rain Shower / Windy', 
                     'Sleet and Thunder', 'Heavy Sleet and Thunder', 'Drifting Snow / Windy', 
                     'Shallow Fog / Windy', 'Thunder and Hail / Windy', 'Heavy Sleet / Windy', 
                     'Sand / Windy', 'Heavy Rain Shower / Windy', 'Blowing Snow Nearby', 
                     'Blowing Sand', 'Heavy Rain Shower', 'Drifting Snow', 
                     'Heavy Thunderstorms with Small Hail'],
            'Snowy': ['Light Snow', 'Snow', 'Light Snow / Windy', 'Snow Grains', 
                     'Snow Showers', 'Snow / Windy', 'Light Snow and Sleet', 
                     'Snow and Sleet', 'Light Snow and Sleet / Windy', 'Snow and Sleet / Windy'],
            'Windy': ['Blowing Dust / Windy', 'Fair / Windy', 'Mostly Cloudy / Windy', 
                     'Light Rain / Windy', 'T-Storm / Windy', 'Blowing Snow / Windy', 
                     'Freezing Rain / Windy', 'Light Snow and Sleet / Windy', 
                     'Sleet and Thunder / Windy', 'Blowing Snow Nearby', 
                     'Heavy Rain Shower / Windy'],
            'Hail': ['Hail'],
            'Volcanic Ash': ['Volcanic Ash'],
            'Tornado': ['Tornado']
        }
        
        def map_weather_to_bins(weather):
            for bin_name, bin_values in weather_bins.items():
                if weather in bin_values:
                    return bin_name
            return 'Other'
        
        df['Weather_Condition'] = df['Weather_Condition'].apply(map_weather_to_bins)
        df['Weather_Condition'] = self._label_encode_column(df['Weather_Condition'])
        
        return df
    
    def _label_encode_column(self, series):
        """Label encode a categorical column"""
        le = LabelEncoder()
        return le.fit_transform(series.astype(str))
    
    def _handle_numerical_variables(self, df):
        """Handle numerical variables with scaling and transformations"""
        numerical_cols = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 
                         'Visibility(mi)', 'Wind_Speed(mph)', 'Distance(mi)']
        
        # Handle missing values in numerical columns before scaling
        for col in numerical_cols:
            if col in df.columns:
                # Fill missing values with median for numerical columns
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                # Scale numerical variables
                df[col] = self.scaler.fit_transform(df[[col]])
        
        return df
    
    def _handle_location_data(self, df):
        """Handle location data with clustering"""
        if 'Start_Lat' in df.columns and 'Start_Lng' in df.columns:
            # Handle missing coordinates by dropping those rows
            location_mask = df['Start_Lat'].notna() & df['Start_Lng'].notna()
            df = df[location_mask].copy()
            
            if len(df) == 0:
                print("Warning: No valid location data found after cleaning")
                return df
            
            # Create location clusters
            location_data = df[['Start_Lat', 'Start_Lng']].copy()
            location_data_scaled = self.scaler.fit_transform(location_data)
            
            # Apply K-means clustering
            df['Location_Cluster'] = self.kmeans.fit_predict(location_data_scaled)
            
            # Drop original coordinates
            df = df.drop(columns=['Start_Lat', 'Start_Lng'])
        
        return df
    
    def _handle_time_features(self, df):
        """Extract time-based features if available"""
        # This would be implemented if we have time data
        # For now, we'll skip this step
        return df
    
    def _final_cleaning(self, df):
        """Final data cleaning steps"""
        print(f"Before final cleaning: {df.shape}")
        
        # Handle missing values more intelligently
        # For categorical columns, fill with mode or 'Unknown'
        categorical_cols = ['Weather_Condition', 'Wind_Direction', 'State']
        for col in categorical_cols:
            if col in df.columns:
                if df[col].isnull().sum() > 0:
                    mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                    df[col] = df[col].fillna(mode_val)
        
        # For high-cardinality columns, fill with a default value
        high_cardinality_cols = ['Zipcode', 'Airport_Code', 'Street', 'County']
        for col in high_cardinality_cols:
            if col in df.columns:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna('Unknown')
        
        # For boolean columns, fill with False (assuming missing means no)
        boolean_cols = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 
                       'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 
                       'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
        for col in boolean_cols:
            if col in df.columns:
                df[col] = df[col].fillna(False)
        
        # Ensure all columns are numeric (except Severity)
        for col in df.columns:
            if col != 'Severity':  # Keep target as is
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    print(f"Warning: Could not convert column {col} to numeric")
        
        # Remove any remaining rows with NaN values in critical columns
        critical_cols = ['Severity']  # Only drop if target is missing
        df = df.dropna(subset=critical_cols)
        
        # Final cleanup: remove any remaining NaN values
        # Fill any remaining NaN values with 0 for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'Severity':  # Don't modify the target
                df[col] = df[col].fillna(0)
        
        # Double-check: if any NaN values remain, drop those rows
        if df.isnull().any().any():
            print("Warning: Some NaN values remain, dropping those rows")
            df = df.dropna()
        
        print(f"After final cleaning: {df.shape}")
        
        if len(df) == 0:
            raise ValueError("No data remaining after preprocessing. Check your data quality.")
        
        return df
    
    def split_data(self, df, test_size=0.2, val_size=0.2, random_state=42):
        """Split data into train, validation, and test sets"""
        print("Splitting data into train, validation, and test sets...")
        
        # Separate features and target
        X = df.drop('Severity', axis=1)
        y = df['Severity']
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def balance_data(self, X, y, method='undersample'):
        """Balance the dataset to handle class imbalance"""
        print("Balancing dataset...")
        
        if method == 'undersample':
            from sklearn.utils import resample
            
            # Get class counts
            class_counts = y.value_counts()
            min_class_count = class_counts.min()
            
            # Resample each class to have the same number of samples
            balanced_data = []
            for class_label in y.unique():
                class_data = X[y == class_label]
                class_labels = y[y == class_label]
                
                if len(class_data) > min_class_count:
                    # Undersample majority classes
                    class_data_resampled, class_labels_resampled = resample(
                        class_data, class_labels,
                        n_samples=min_class_count,
                        random_state=42
                    )
                else:
                    # Oversample minority classes
                    class_data_resampled, class_labels_resampled = resample(
                        class_data, class_labels,
                        n_samples=min_class_count,
                        random_state=42
                    )
                
                balanced_data.append(pd.concat([class_data_resampled, class_labels_resampled], axis=1))
            
            balanced_df = pd.concat(balanced_data, ignore_index=True)
            X_balanced = balanced_df.drop('Severity', axis=1)
            y_balanced = balanced_df['Severity']
            
            print(f"Balanced dataset shape: {X_balanced.shape}")
            return X_balanced, y_balanced
        
        return X, y 