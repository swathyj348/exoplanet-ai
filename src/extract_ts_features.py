"""
Time series feature extraction module for exoplanet analysis.

This module extracts meaningful features from Kepler time series data
that can be used for machine learning classification.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks, periodogram
from scipy.fft import fft, fftfreq
import logging
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_ts_features_from_available_data():
    """
    Extract time series-like features from available datasets.
    Since we don't have traditional lightcurve data, we'll extract
    features from time-related and magnitude columns.
    """
    print("Starting time series feature extraction...")
    
    # Load the datasets
    try:
        toi_data = pd.read_csv('data/TOI_2025.10.04_07.29.48.csv', comment='#')
        cumulative_data = pd.read_csv('data/cumulative_2025.10.04_07.34.55.csv', comment='#')
        print(f"TOI dataset shape: {toi_data.shape}")
        print(f"Cumulative dataset shape: {cumulative_data.shape}")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None
    
    # Normalize column names
    toi_data.columns = toi_data.columns.str.lower().str.strip()
    cumulative_data.columns = cumulative_data.columns.str.lower().str.strip()
    
    # Extract features from TOI data (which has more time-series related columns)
    features_list = []
    
    # Define columns that could represent time-series characteristics
    time_related_cols = [
        'pl_orbper', 'pl_orbpererr1', 'pl_orbpererr2',  # orbital period and errors
        'pl_trandur', 'pl_trandurh',  # transit duration
        'pl_tranmid', 'pl_tranmiderr1', 'pl_tranmiderr2',  # transit midpoint
    ]
    
    magnitude_cols = [
        'st_tmag', 'st_tmagerr1', 'st_tmagerr2',  # TESS magnitude
        'st_jmag', 'st_hmag', 'st_kmag',  # J, H, K magnitudes
    ]
    
    depth_cols = [
        'pl_trandep', 'pl_trandeperr1', 'pl_trandeperr2',  # transit depth
        'pl_ratror', 'pl_ratrorerr1', 'pl_ratrorerr2',  # radius ratio
    ]
    
    # Find available columns
    available_time_cols = [col for col in time_related_cols if col in toi_data.columns]
    available_mag_cols = [col for col in magnitude_cols if col in toi_data.columns]
    available_depth_cols = [col for col in depth_cols if col in toi_data.columns]
    
    print(f"Available time-related columns: {available_time_cols}")
    print(f"Available magnitude columns: {available_mag_cols}")
    print(f"Available depth columns: {available_depth_cols}")
    
    # Extract features for each object
    for idx, row in toi_data.iterrows():
        features = {'id': row.get('toi', idx)}
        
        # Time-series statistical features
        if available_time_cols:
            time_vals = row[available_time_cols].dropna().values
            # Convert to numeric, handling any string values
            time_vals = pd.to_numeric(time_vals, errors='coerce')
            time_vals = time_vals[~np.isnan(time_vals)]  # Remove NaN values
            
            if len(time_vals) > 0:
                features.update({
                    'time_mean': float(np.mean(time_vals)),
                    'time_std': float(np.std(time_vals)),
                    'time_median': float(np.median(time_vals)),
                    'time_min': float(np.min(time_vals)),
                    'time_max': float(np.max(time_vals)),
                    'time_range': float(np.max(time_vals) - np.min(time_vals)),
                    'time_skew': float(stats.skew(time_vals)) if len(time_vals) > 2 else 0.0,
                    'time_kurtosis': float(stats.kurtosis(time_vals)) if len(time_vals) > 2 else 0.0,
                })
            else:
                features.update({f'time_{stat}': 0.0 for stat in ['mean', 'std', 'median', 'min', 'max', 'range', 'skew', 'kurtosis']})
        
        # Magnitude-based features
        if available_mag_cols:
            mag_vals = row[available_mag_cols].dropna().values
            # Convert to numeric, handling any string values
            mag_vals = pd.to_numeric(mag_vals, errors='coerce')
            mag_vals = mag_vals[~np.isnan(mag_vals)]
            
            if len(mag_vals) > 0:
                features.update({
                    'mag_mean': float(np.mean(mag_vals)),
                    'mag_std': float(np.std(mag_vals)),
                    'mag_median': float(np.median(mag_vals)),
                    'mag_range': float(np.max(mag_vals) - np.min(mag_vals)),
                    'mag_brightness_var': float(np.std(mag_vals) / np.mean(mag_vals)) if np.mean(mag_vals) != 0 else 0.0,
                })
            else:
                features.update({f'mag_{stat}': 0.0 for stat in ['mean', 'std', 'median', 'range', 'brightness_var']})
        
        # Transit depth features
        if available_depth_cols:
            depth_vals = row[available_depth_cols].dropna().values
            # Convert to numeric, handling any string values
            depth_vals = pd.to_numeric(depth_vals, errors='coerce')
            depth_vals = depth_vals[~np.isnan(depth_vals)]
            
            if len(depth_vals) > 0:
                features.update({
                    'depth_mean': float(np.mean(depth_vals)),
                    'depth_std': float(np.std(depth_vals)),
                    'depth_max': float(np.max(depth_vals)),
                    'depth_variability': float(np.std(depth_vals) / np.mean(depth_vals)) if np.mean(depth_vals) != 0 else 0.0,
                })
            else:
                features.update({f'depth_{stat}': 0.0 for stat in ['mean', 'std', 'max', 'variability']})
        
        # Synthetic frequency domain features based on available data
        # Use period and duration to create frequency-like features
        if 'pl_orbper' in row and not pd.isna(row['pl_orbper']):
            period = float(row['pl_orbper'])
            features['frequency'] = 1.0 / period if period > 0 else 0.0
            features['log_period'] = np.log10(period) if period > 0 else 0.0
        else:
            features['frequency'] = 0.0
            features['log_period'] = 0.0
        
        if 'pl_trandurh' in row and not pd.isna(row['pl_trandurh']):
            duration = float(row['pl_trandurh'])
            period = features.get('frequency', 0)
            if period > 0:
                features['duration_ratio'] = duration / (1.0/period)
            else:
                features['duration_ratio'] = 0.0
        else:
            features['duration_ratio'] = 0.0
        
        features_list.append(features)
    
    # Convert to DataFrame
    ts_features = pd.DataFrame(features_list)
    
    # Handle any remaining NaN values
    ts_features = ts_features.fillna(0)
    
    # Save the features
    output_path = 'data/ts_features.csv'
    ts_features.to_csv(output_path, index=False)
    print(f'Saved ts_features.csv with shape {ts_features.shape}')
    print(f'Features extracted: {list(ts_features.columns)}')
    
    return ts_features


def create_synthetic_lightcurve_features(n_objects=100, n_timepoints=1000):
    """
    Create synthetic lightcurve data for demonstration purposes.
    This generates realistic-looking time series data that could represent
    stellar brightness variations with potential transit signals.
    """
    print(f"Creating synthetic lightcurve features for {n_objects} objects...")
    
    features_list = []
    
    for obj_id in range(n_objects):
        # Generate synthetic time series
        time = np.linspace(0, 100, n_timepoints)  # 100 day observation
        
        # Base stellar brightness with noise
        base_flux = 1.0 + 0.01 * np.random.randn(n_timepoints)
        
        # Add potential transit signals for some objects
        if np.random.random() < 0.3:  # 30% chance of transit
            period = np.random.uniform(1, 50)  # days
            depth = np.random.uniform(0.001, 0.1)  # transit depth
            duration = np.random.uniform(0.1, 0.5)  # fraction of period
            
            # Add periodic transits
            for i in range(int(100 / period)):
                transit_time = i * period
                transit_mask = np.abs(time - transit_time) < (period * duration / 2)
                base_flux[transit_mask] -= depth
        
        # Extract features from synthetic lightcurve
        features = extract_lightcurve_features(time, base_flux, obj_id)
        features_list.append(features)
    
    # Convert to DataFrame and save
    synthetic_features = pd.DataFrame(features_list)
    output_path = 'data/synthetic_ts_features.csv'
    synthetic_features.to_csv(output_path, index=False)
    print(f'Saved synthetic_ts_features.csv with shape {synthetic_features.shape}')
    
    return synthetic_features


def extract_lightcurve_features(time, flux, obj_id):
    """
    Extract features from a single lightcurve.
    This follows the pattern from your original specification.
    """
    features = {'id': obj_id}
    
    # Remove NaN values
    mask = ~(np.isnan(time) | np.isnan(flux))
    time_clean = time[mask]
    flux_clean = flux[mask]
    
    if len(flux_clean) == 0:
        return {f: 0 for f in ['id', 'mean', 'std', 'median', 'min', 'max', 'p5', 'p25', 'p75', 'p95', 'skew', 'kurt', 'fft1', 'fft2', 'fft3']}
    
    # Basic statistical features
    features.update({
        'mean': np.mean(flux_clean),
        'std': np.std(flux_clean),
        'median': np.median(flux_clean),
        'min': np.min(flux_clean),
        'max': np.max(flux_clean),
        'p5': np.percentile(flux_clean, 5),
        'p25': np.percentile(flux_clean, 25),
        'p75': np.percentile(flux_clean, 75),
        'p95': np.percentile(flux_clean, 95),
    })
    
    # Higher order statistics
    if len(flux_clean) > 2:
        features['skew'] = stats.skew(flux_clean)
        features['kurt'] = stats.kurtosis(flux_clean)
    else:
        features['skew'] = 0
        features['kurt'] = 0
    
    # FFT features - top 3 magnitudes
    try:
        fft_vals = np.fft.fft(flux_clean - np.mean(flux_clean))
        fft_mag = np.abs(fft_vals)
        # Get top 3 frequencies (excluding DC component)
        top3_indices = np.argsort(fft_mag[1:])[-3:] + 1
        top3_mags = fft_mag[top3_indices]
        
        features['fft1'] = top3_mags[0] if len(top3_mags) > 0 else 0
        features['fft2'] = top3_mags[1] if len(top3_mags) > 1 else 0
        features['fft3'] = top3_mags[2] if len(top3_mags) > 2 else 0
    except:
        features['fft1'] = features['fft2'] = features['fft3'] = 0
    
    return features


class TimeSeriesFeatureExtractor:
    """
    Extract features from Kepler time series light curves.
    
    Features include statistical, frequency domain, and transit-specific metrics.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.feature_names = []
    
    def load_time_series(self, filepath):
        """
        Load time series data from CSV.
        
        Args:
            filepath (str): Path to time series CSV file
            
        Returns:
            pd.DataFrame: Time series dataframe
        """
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded time series data: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading time series: {str(e)}")
            return None
    
    def extract_statistical_features(self, flux_values):
        """
        Extract basic statistical features from flux time series.
        
        Args:
            flux_values (array-like): Flux measurements
            
        Returns:
            dict: Statistical features
        """
        flux = np.array(flux_values)
        flux = flux[~np.isnan(flux)]  # Remove NaN values
        
        if len(flux) == 0:
            return {}
        
        features = {
            'mean_flux': np.mean(flux),
            'std_flux': np.std(flux),
            'var_flux': np.var(flux),
            'median_flux': np.median(flux),
            'min_flux': np.min(flux),
            'max_flux': np.max(flux),
            'range_flux': np.max(flux) - np.min(flux),
            'q25_flux': np.percentile(flux, 25),
            'q75_flux': np.percentile(flux, 75),
            'iqr_flux': np.percentile(flux, 75) - np.percentile(flux, 25),
            'skewness': stats.skew(flux),
            'kurtosis': stats.kurtosis(flux),
            'rms': np.sqrt(np.mean(flux**2))
        }
        
        return features
    
    def extract_variability_features(self, flux_values):
        """
        Extract variability and noise features.
        
        Args:
            flux_values (array-like): Flux measurements
            
        Returns:
            dict: Variability features
        """
        flux = np.array(flux_values)
        flux = flux[~np.isnan(flux)]
        
        if len(flux) < 3:
            return {}
        
        # Calculate differences
        diff1 = np.diff(flux)
        diff2 = np.diff(flux, n=2)
        
        features = {
            'mean_abs_deviation': np.mean(np.abs(flux - np.mean(flux))),
            'median_abs_deviation': np.median(np.abs(flux - np.median(flux))),
            'coefficient_variation': np.std(flux) / np.mean(flux) if np.mean(flux) != 0 else 0,
            'first_diff_mean': np.mean(diff1),
            'first_diff_std': np.std(diff1),
            'second_diff_mean': np.mean(diff2),
            'second_diff_std': np.std(diff2),
            'autocorr_lag1': np.corrcoef(flux[:-1], flux[1:])[0, 1] if len(flux) > 1 else 0
        }
        
        return features
    
    def extract_frequency_features(self, time_values, flux_values, max_freq=None):
        """
        Extract frequency domain features using FFT and periodogram.
        
        Args:
            time_values (array-like): Time measurements
            flux_values (array-like): Flux measurements
            max_freq (float): Maximum frequency to consider
            
        Returns:
            dict: Frequency domain features
        """
        time = np.array(time_values)
        flux = np.array(flux_values)
        
        # Remove NaN values
        mask = ~(np.isnan(time) | np.isnan(flux))
        time = time[mask]
        flux = flux[mask]
        
        if len(flux) < 10:
            return {}
        
        features = {}
        
        try:
            # FFT analysis
            fft_vals = fft(flux - np.mean(flux))
            fft_freq = fftfreq(len(flux), d=np.median(np.diff(time)))
            fft_power = np.abs(fft_vals)**2
            
            # Keep only positive frequencies
            pos_mask = fft_freq > 0
            if max_freq:
                pos_mask &= (fft_freq <= max_freq)
            
            if np.sum(pos_mask) > 0:
                pos_freq = fft_freq[pos_mask]
                pos_power = fft_power[pos_mask]
                
                features['dominant_freq'] = pos_freq[np.argmax(pos_power)]
                features['max_power'] = np.max(pos_power)
                features['total_power'] = np.sum(pos_power)
                features['power_ratio'] = np.max(pos_power) / np.sum(pos_power) if np.sum(pos_power) > 0 else 0
                
                # Spectral centroid
                features['spectral_centroid'] = np.sum(pos_freq * pos_power) / np.sum(pos_power) if np.sum(pos_power) > 0 else 0
        
        except Exception as e:
            logger.warning(f"Error in frequency analysis: {str(e)}")
        
        try:
            # Periodogram for more robust period detection
            freqs, powers = periodogram(flux, fs=1/np.median(np.diff(time)))
            
            if len(powers) > 0:
                features['periodogram_peak_freq'] = freqs[np.argmax(powers)]
                features['periodogram_peak_power'] = np.max(powers)
        
        except Exception as e:
            logger.warning(f"Error in periodogram analysis: {str(e)}")
        
        return features
    
    def extract_transit_features(self, flux_values, threshold_sigma=3):
        """
        Extract transit-like features from light curve.
        
        Args:
            flux_values (array-like): Flux measurements
            threshold_sigma (float): Sigma threshold for anomaly detection
            
        Returns:
            dict: Transit-related features
        """
        flux = np.array(flux_values)
        flux = flux[~np.isnan(flux)]
        
        if len(flux) < 10:
            return {}
        
        features = {}
        
        # Detect dips (potential transits)
        mean_flux = np.mean(flux)
        std_flux = np.std(flux)
        threshold = mean_flux - threshold_sigma * std_flux
        
        dips = flux < threshold
        features['num_dips'] = np.sum(dips)
        features['dip_fraction'] = np.sum(dips) / len(flux)
        
        if np.sum(dips) > 0:
            features['mean_dip_depth'] = np.mean(flux[dips] - mean_flux)
            features['max_dip_depth'] = np.min(flux[dips]) - mean_flux
        else:
            features['mean_dip_depth'] = 0
            features['max_dip_depth'] = 0
        
        # Detect peaks (potential flares)
        peak_threshold = mean_flux + threshold_sigma * std_flux
        peaks = flux > peak_threshold
        features['num_peaks'] = np.sum(peaks)
        features['peak_fraction'] = np.sum(peaks) / len(flux)
        
        # Use scipy to find peaks with more sophisticated detection
        try:
            peak_indices, _ = find_peaks(-flux, height=-peak_threshold)  # Inverted for dips
            features['num_detected_transits'] = len(peak_indices)
        except:
            features['num_detected_transits'] = 0
        
        return features
    
    def extract_all_features(self, time_values, flux_values, object_id=None):
        """
        Extract all features from a single light curve.
        
        Args:
            time_values (array-like): Time measurements
            flux_values (array-like): Flux measurements
            object_id (str): Identifier for the object
            
        Returns:
            dict: All extracted features
        """
        all_features = {}
        
        # Add object ID if provided
        if object_id:
            all_features['object_id'] = object_id
        
        # Extract different types of features
        stat_features = self.extract_statistical_features(flux_values)
        var_features = self.extract_variability_features(flux_values)
        freq_features = self.extract_frequency_features(time_values, flux_values)
        transit_features = self.extract_transit_features(flux_values)
        
        # Combine all features
        all_features.update(stat_features)
        all_features.update(var_features)
        all_features.update(freq_features)
        all_features.update(transit_features)
        
        return all_features
    
    def process_multiple_lightcurves(self, time_series_df, time_col='time', flux_col='flux', id_col='kepid'):
        """
        Process multiple light curves from a dataframe.
        
        Args:
            time_series_df (pd.DataFrame): DataFrame with time series data
            time_col (str): Name of time column
            flux_col (str): Name of flux column
            id_col (str): Name of object ID column
            
        Returns:
            pd.DataFrame: Features for all objects
        """
        features_list = []
        
        # Group by object ID and process each light curve
        for obj_id, group in time_series_df.groupby(id_col):
            features = self.extract_all_features(
                group[time_col].values,
                group[flux_col].values,
                obj_id
            )
            features_list.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Store feature names
        self.feature_names = [col for col in features_df.columns if col != 'object_id']
        
        logger.info(f"Extracted {len(self.feature_names)} features from {len(features_df)} objects")
        
        return features_df
    
    def save_features(self, features_df, output_path):
        """
        Save extracted features to CSV.
        
        Args:
            features_df (pd.DataFrame): Features dataframe
            output_path (str): Output file path
        """
        try:
            features_df.to_csv(output_path, index=False)
            logger.info(f"Features saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")


if __name__ == "__main__":
    print("Starting time series feature extraction...")
    
    # Method 1: Extract features from available datasets
    print("\n=== Extracting features from available data ===")
    ts_features = extract_ts_features_from_available_data()
    
    # Method 2: Create synthetic lightcurve data for demonstration
    print("\n=== Creating synthetic lightcurve features ===")
    synthetic_features = create_synthetic_lightcurve_features(n_objects=500, n_timepoints=1000)
    
    # Method 3: Demonstrate the TimeSeriesFeatureExtractor class
    print("\n=== Testing TimeSeriesFeatureExtractor class ===")
    extractor = TimeSeriesFeatureExtractor()
    
    # Try to load traditional time series data if available
    try:
        ts_data = extractor.load_time_series('data/kepler_time_series.csv')
        
        if ts_data is not None:
            print("Time series data loaded successfully!")
            print(f"Shape: {ts_data.shape}")
            print(f"Columns: {ts_data.columns.tolist()}")
            
            # Process the data (adjust column names as needed)
            features_df = extractor.process_multiple_lightcurves(ts_data)
            extractor.save_features(features_df, 'data/extracted_features.csv')
        else:
            print("No traditional time series data found.")
    except Exception as e:
        print(f"Traditional time series processing failed: {e}")
    
    print("\nTime series feature extraction completed!")
    print("Generated files:")
    print("- data/ts_features.csv (from available data)")
    print("- data/synthetic_ts_features.csv (synthetic data)")
    if 'features_df' in locals():
        print("- data/extracted_features.csv (traditional time series)")