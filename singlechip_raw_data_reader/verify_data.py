"""
verify_data.py

Data verification utilities for mmWave captures.

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/

Converted from MATLAB: verify_data.m
"""

import os
import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy.io import loadmat
import matplotlib
# Use TkAgg backend for interactive plotting (fallback to Agg if not available)
try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt


def verify_data(
    data_file: str,
    expected_params: Optional[Dict[str, Any]] = None,
    plot_on: bool = True,
) -> Dict[str, Any]:
    """
    Verify captured radar data for validity and consistency.
    
    Args:
        data_file: Path to data file (MAT or binary)
        expected_params: Expected parameters for validation
        plot_on: Enable visualization
        
    Returns:
        Dictionary with verification results
    """
    results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'stats': {},
    }
    
    # Determine file type
    _, ext = os.path.splitext(data_file)
    
    if ext.lower() == '.mat':
        results = _verify_mat_file(data_file, expected_params, results)
    elif ext.lower() == '.bin':
        results = _verify_bin_file(data_file, expected_params, results)
    else:
        results['errors'].append(f"Unknown file type: {ext}")
        results['valid'] = False
        return results
    
    if plot_on and results['valid']:
        _plot_verification(data_file, results)
    
    return results


def _verify_mat_file(file_path: str, expected: Optional[Dict],
                     results: Dict) -> Dict:
    """Verify MAT file contents."""
    try:
        data = loadmat(file_path)
    except Exception as e:
        results['errors'].append(f"Failed to load MAT file: {e}")
        results['valid'] = False
        return results
    
    # Check for expected data structures
    if 'adcRawData' in data:
        raw_data = data['adcRawData']
        results['stats']['type'] = 'raw_adc'
        # Additional validation...
    elif 'radarCube' in data:
        cube_data = data['radarCube']
        results['stats']['type'] = 'radar_cube'
    else:
        results['warnings'].append("Unknown data structure in MAT file")
    
    # Check for NaN/Inf values
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            if np.any(np.isnan(value)):
                results['warnings'].append(f"NaN values found in {key}")
            if np.any(np.isinf(value)):
                results['warnings'].append(f"Inf values found in {key}")
    
    return results


def _verify_bin_file(file_path: str, expected: Optional[Dict],
                     results: Dict) -> Dict:
    """Verify binary file contents."""
    if not os.path.exists(file_path):
        results['errors'].append(f"File not found: {file_path}")
        results['valid'] = False
        return results
    
    file_size = os.path.getsize(file_path)
    results['stats']['file_size_bytes'] = file_size
    
    if file_size == 0:
        results['errors'].append("File is empty")
        results['valid'] = False
        return results
    
    # Check for expected frame size
    if expected:
        expected_frame_size = _calculate_frame_size(expected)
        if expected_frame_size > 0:
            num_frames = file_size // expected_frame_size
            remainder = file_size % expected_frame_size
            
            results['stats']['expected_frame_size'] = expected_frame_size
            results['stats']['num_frames'] = num_frames
            
            if remainder != 0:
                results['warnings'].append(
                    f"File size not multiple of frame size. "
                    f"Remainder: {remainder} bytes"
                )
    
    # Sample the binary data
    with open(file_path, 'rb') as f:
        sample_data = np.fromfile(f, dtype=np.uint16, count=min(1000, file_size // 2))
    
    # Check for constant values (potential issue)
    unique_vals = len(np.unique(sample_data))
    if unique_vals < 10:
        results['warnings'].append(
            f"Very few unique values in data ({unique_vals}). "
            "Possible DC offset or capture issue."
        )
    
    # Check value range
    results['stats']['min_value'] = int(np.min(sample_data))
    results['stats']['max_value'] = int(np.max(sample_data))
    results['stats']['mean_value'] = float(np.mean(sample_data))
    
    return results


def _calculate_frame_size(params: Dict) -> int:
    """Calculate expected frame size from parameters."""
    num_samples = params.get('numAdcSamples', 256)
    num_chirps = params.get('numChirpsPerFrame', 64)
    num_rx = params.get('numRxChan', 4)
    
    # Complex data: 2 samples per IQ, 2 bytes per sample
    return num_samples * num_chirps * num_rx * 4


def _plot_verification(file_path: str, results: Dict) -> None:
    """Create verification plots."""
    _, ext = os.path.splitext(file_path)
    
    if ext.lower() == '.mat':
        data = loadmat(file_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Data Verification: {os.path.basename(file_path)}")
        
        # Find array data to plot
        arrays = {k: v for k, v in data.items() 
                  if isinstance(v, np.ndarray) and not k.startswith('_')}
        
        if arrays:
            first_array = list(arrays.values())[0]
            if first_array.ndim >= 1:
                ax = axes[0, 0]
                flat = first_array.flatten()[:1000]
                ax.plot(np.real(flat) if np.iscomplexobj(flat) else flat)
                ax.set_title('First 1000 Samples')
                ax.set_xlabel('Sample Index')
                ax.grid(True)
                
                ax = axes[0, 1]
                ax.hist(np.real(flat) if np.iscomplexobj(flat) else flat, bins=50)
                ax.set_title('Value Distribution')
                ax.set_xlabel('Value')
                ax.grid(True)
        
        # Stats text
        ax = axes[1, 0]
        ax.axis('off')
        stats_text = "Verification Results:\n"
        stats_text += f"  Valid: {results['valid']}\n"
        for key, value in results['stats'].items():
            stats_text += f"  {key}: {value}\n"
        if results['warnings']:
            stats_text += "\nWarnings:\n"
            for w in results['warnings'][:5]:
                stats_text += f"  - {w}\n"
        ax.text(0.1, 0.9, stats_text, va='top', fontfamily='monospace',
                transform=ax.transAxes)
        
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()


def compare_data(file1: str, file2: str, tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Compare two data files for equivalence.
    
    Useful for validating MATLAB to Python conversion.
    
    Args:
        file1: First data file
        file2: Second data file
        tolerance: Numerical tolerance for comparison
        
    Returns:
        Comparison results dictionary
    """
    results = {
        'equivalent': True,
        'differences': [],
    }
    
    try:
        data1 = loadmat(file1)
        data2 = loadmat(file2)
    except Exception as e:
        results['equivalent'] = False
        results['differences'].append(f"Load error: {e}")
        return results
    
    # Compare keys
    keys1 = set(k for k in data1.keys() if not k.startswith('_'))
    keys2 = set(k for k in data2.keys() if not k.startswith('_'))
    
    if keys1 != keys2:
        results['differences'].append(f"Different keys: {keys1 ^ keys2}")
        results['equivalent'] = False
    
    # Compare arrays
    for key in keys1 & keys2:
        arr1 = data1[key]
        arr2 = data2[key]
        
        if isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
            if arr1.shape != arr2.shape:
                results['differences'].append(f"{key}: Shape mismatch {arr1.shape} vs {arr2.shape}")
                results['equivalent'] = False
            else:
                max_diff = np.max(np.abs(arr1 - arr2))
                if max_diff > tolerance:
                    results['differences'].append(f"{key}: Max difference = {max_diff}")
                    results['equivalent'] = False
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Verification')
    parser.add_argument('--file', required=True, help='Data file to verify')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    
    args = parser.parse_args()
    
    results = verify_data(args.file, plot_on=not args.no_plot)
    
    print("\nVerification Results:")
    print(f"  Valid: {results['valid']}")
    print(f"  Stats: {results['stats']}")
    if results['warnings']:
        print(f"  Warnings: {results['warnings']}")
    if results['errors']:
        print(f"  Errors: {results['errors']}")
