"""
Data preprocessing module for exoplanet analysis.

This module handles loading, cleaning, and preparing the Kepler dataset
for machine learning analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_tabular_data():
    """
    Main preprocessing function that merges tabular datasets, cleans data,
    selects numeric features, maps labels, and saves merged_tabular.csv.
    """
    print("Starting tabular data preprocessing...")
    
    # 1. Load CSVs
    print("Loading CSV files...")
    kepler_tab1 = pd.read_csv('data/cumulative_2025.10.04_07.34.55.csv', comment='#')
    kepler_tab2 = pd.read_csv('data/k2pandc_2025.10.04_07.36.56.csv', comment='#')
    
    print(f"Kepler cumulative dataset: {kepler_tab1.shape}")
    print(f"K2 dataset: {kepler_tab2.shape}")
    
    # 2. Normalize column names
    kepler_tab1.columns = kepler_tab1.columns.str.lower().str.strip()
    kepler_tab2.columns = kepler_tab2.columns.str.lower().str.strip()
    
    # 3. Merge datasets
    print("Merging datasets...")
    # Check available merge keys
    if 'kepid' in kepler_tab1.columns and 'kepid' in kepler_tab2.columns:
        print("Merging on 'kepid'")
        merged = pd.merge(kepler_tab1, kepler_tab2, on='kepid', how='outer')
    elif 'kepoi_name' in kepler_tab1.columns and 'pl_name' in kepler_tab2.columns:
        print("Merging on name columns")
        # Try to match by name (this might be tricky, so we'll use outer join)
        merged = pd.concat([kepler_tab1, kepler_tab2], ignore_index=True, sort=False)
    else:
        print("No common keys found, concatenating datasets")
        merged = pd.concat([kepler_tab1, kepler_tab2], ignore_index=True, sort=False)
    
    print(f"Merged dataset shape: {merged.shape}")
    
    # 4. Select numeric features and disposition column
    # Look for relevant numeric columns based on what's available
    potential_numeric_cols = [
        'koi_period', 'pl_orbper',  # orbital period
        'koi_prad', 'pl_rade',     # planet radius  
        'koi_ecc', 'pl_orbeccen',  # eccentricity
        'koi_smass', 'st_mass',    # stellar mass
        'koi_steff', 'st_teff',    # stellar temperature
        'koi_srad', 'st_rad',      # stellar radius
        'koi_kepmag', 'sy_kepmag', # kepler magnitude
        'koi_dor', 'pl_ratdor'     # distance over radius ratio
    ]
    
    # Find which columns actually exist
    available_numeric_cols = [col for col in potential_numeric_cols if col in merged.columns]
    print(f"Available numeric columns: {available_numeric_cols}")
    
    # Look for disposition columns
    disposition_cols = [col for col in merged.columns if 'disposition' in col or 'disp' in col]
    print(f"Available disposition columns: {disposition_cols}")
    
    # Select the best available columns
    if not available_numeric_cols:
        # Fallback to any numeric columns
        numeric_dtypes = merged.select_dtypes(include=[np.number]).columns.tolist()
        available_numeric_cols = numeric_dtypes[:8]  # Take first 8 numeric columns
        print(f"Using fallback numeric columns: {available_numeric_cols}")
    
    # Select best disposition column
    target_col = None
    if 'koi_disposition' in merged.columns:
        target_col = 'koi_disposition'
    elif 'disposition' in merged.columns:
        target_col = 'disposition'
    elif 'tfopwg_disp' in merged.columns:
        target_col = 'tfopwg_disp'
    
    if target_col:
        selected_cols = available_numeric_cols + [target_col]
        merged = merged[selected_cols]
        print(f"Selected columns: {selected_cols}")
    else:
        print("Warning: No disposition column found!")
        merged = merged[available_numeric_cols]
    
    # 5. Handle missing values
    print("Handling missing values...")
    initial_rows = len(merged)
    
    # Drop rows missing more than 50% of numeric features
    if available_numeric_cols:
        merged = merged.dropna(thresh=len(available_numeric_cols)//2)
        print(f"Rows after dropping sparse data: {len(merged)} (removed {initial_rows - len(merged)})")
        
        # Fill remaining missing values with median
        for col in available_numeric_cols:
            if merged[col].isnull().any():
                median_val = merged[col].median()
                merged[col].fillna(median_val, inplace=True)
                print(f"Filled {col} missing values with median: {median_val}")
    
    # 6. Map labels
    if target_col and target_col in merged.columns:
        print("Mapping disposition labels...")
        print(f"Original disposition values: {merged[target_col].value_counts()}")
        
        # Create a more comprehensive label mapping
        label_map = {
            'CONFIRMED': 1, 'confirmed': 1,
            'CANDIDATE': 0, 'candidate': 0, 
            'FALSE POSITIVE': 2, 'false positive': 2,
            'PC': 0,  # Planet Candidate
            'CP': 1,  # Confirmed Planet  
            'FP': 2,  # False Positive
            'KP': 1,  # Known Planet
        }
        
        # Apply mapping
        merged['label'] = merged[target_col].astype(str).str.upper().map(label_map)
        
        # Handle unmapped values
        unmapped_mask = merged['label'].isnull()
        if unmapped_mask.any():
            print(f"Warning: {unmapped_mask.sum()} rows with unmapped dispositions")
            print(f"Unmapped values: {merged.loc[unmapped_mask, target_col].unique()}")
            # Drop unmapped rows or assign default label
            merged = merged.dropna(subset=['label'])
        
        merged.drop(columns=[target_col], inplace=True)
        print(f"Final label distribution: {merged['label'].value_counts()}")
    
    # 7. Save processed data
    output_path = 'data/merged_tabular.csv'
    merged.to_csv(output_path, index=False)
    print(f'Saved merged_tabular.csv with shape {merged.shape}')
    
    return merged


class ExoplanetPreprocessor:
    """
    Preprocessor for Kepler exoplanet dataset.
    
    Handles data loading, cleaning, feature engineering, and train/test splits.
    """
    
    def __init__(self, data_path="../data/"):
        """
        Initialize preprocessor with data path.
        
        Args:
            data_path (str): Path to directory containing CSV files
        """
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self, filename):
        """
        Load CSV data from the data directory.
        
        Args:
            filename (str): Name of CSV file to load
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            filepath = f"{self.data_path}{filename}"
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            logger.error(f"File {filename} not found in {self.data_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            return None
    
    def clean_data(self, df):
        """
        Clean the dataset by handling missing values and outliers.
        
        Args:
            df (pd.DataFrame): Raw dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        # For numerical columns, fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        logger.info(f"Data cleaned: {df.shape[0]} rows remaining")
        return df
    
    def engineer_features(self, df):
        """
        Create new features from existing data.
        
        Args:
            df (pd.DataFrame): Cleaned dataframe
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        # This method should be customized based on the specific columns
        # in your Kepler dataset
        
        # Example feature engineering (adjust based on actual columns):
        # if 'koi_period' in df.columns and 'koi_ror' in df.columns:
        #     df['period_ror_ratio'] = df['koi_period'] / df['koi_ror']
        
        logger.info("Feature engineering completed")
        return df
    
    def encode_target(self, df, target_column='koi_disposition'):
        """
        Encode categorical target variable.
        
        Args:
            df (pd.DataFrame): Dataframe with target column
            target_column (str): Name of target column
            
        Returns:
            tuple: (dataframe, encoded_target)
        """
        if target_column in df.columns:
            y = self.label_encoder.fit_transform(df[target_column])
            logger.info(f"Target classes: {self.label_encoder.classes_}")
            return df.drop(columns=[target_column]), y
        else:
            logger.warning(f"Target column '{target_column}' not found")
            return df, None
    
    def scale_features(self, X_train, X_test=None):
        """
        Scale numerical features using StandardScaler.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame, optional): Test features
            
        Returns:
            tuple: (scaled_train, scaled_test) or just scaled_train
        """
        # Select only numerical columns
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        # Fit scaler on training data
        X_train_scaled = X_train.copy()
        X_train_scaled[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def prepare_data(self, filename, target_column='koi_disposition', test_size=0.2, random_state=42):
        """
        Complete preprocessing pipeline.
        
        Args:
            filename (str): CSV filename to process
            target_column (str): Target variable column name
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Load and preprocess data
        df = self.load_data(filename)
        if df is None:
            return None, None, None, None
            
        df = self.clean_data(df)
        df = self.engineer_features(df)
        
        # Encode target and split features
        X, y = self.encode_target(df, target_column)
        
        if y is None:
            logger.error("Could not encode target variable")
            return None, None, None, None
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        logger.info(f"Data preparation complete:")
        logger.info(f"Training set: {X_train_scaled.shape}")
        logger.info(f"Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    # Run the main preprocessing function
    print("Running tabular data preprocessing...")
    merged_data = preprocess_tabular_data()
    
    # Also demonstrate the ExoplanetPreprocessor class usage
    print("\nTesting ExoplanetPreprocessor class...")
    preprocessor = ExoplanetPreprocessor()
    
    # Try to load and preprocess the merged dataset
    try:
        X_train, X_test, y_train, y_test = preprocessor.prepare_data('merged_tabular.csv')
        
        if X_train is not None:
            print("ExoplanetPreprocessor successful!")
            print(f"Training features shape: {X_train.shape}")
            print(f"Training labels shape: {y_train.shape}")
        else:
            print("ExoplanetPreprocessor failed. Please check your data files.")
    except Exception as e:
        print(f"Error with ExoplanetPreprocessor: {str(e)}")
        print("The main preprocess_tabular_data() function completed successfully.")