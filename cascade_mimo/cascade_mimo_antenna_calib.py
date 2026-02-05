"""
cascade_mimo_antenna_calib.py

Antenna calibration for TI 4-chip cascade MIMO radar.

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/

Converted from MATLAB: cascade_MIMO_antennaCalib.m

This script generates calibration matrices from corner reflector data.
"""

import os
import numpy as np
from scipy.io import savemat, loadmat
from typing import Optional, Dict, Any

from .utils.data_parse import (
    get_unique_file_idx, get_bin_file_names_with_idx,
    read_adc_bin_tda2_separate_files
)
from .utils.json_parser import parse_mmwave_config, get_tx_enable_table, find_mmwave_json

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.constants import TI_CASCADE_RX_ID, DEFAULT_CALIBRATION_INTERP
from common.config import FileNameStruct


def run_antenna_calibration(
    data_folder_calib: str,
    target_range_m: float = 1.5,
    output_file: Optional[str] = None,
    calibration_interp: int = DEFAULT_CALIBRATION_INTERP,
    phase_only: bool = False,
    plot_on: bool = True,
) -> Dict[str, Any]:
    """
    Generate antenna calibration matrices from corner reflector data.
    
    Args:
        data_folder_calib: Path to calibration data folder
        target_range_m: Expected range to corner reflector (meters)
        output_file: Output .mat file path (default: calibrationData.mat in data folder)
        calibration_interp: Interpolation factor for calibration
        phase_only: Only generate phase calibration (skip frequency)
        plot_on: Enable visualization
        
    Returns:
        Dictionary containing calibration results
    """
    print(f"Starting antenna calibration...")
    print(f"  Data folder: {data_folder_calib}")
    print(f"  Target range: {target_range_m} m")
    
    # Find JSON configuration
    json_file = find_mmwave_json(data_folder_calib)
    if json_file is None:
        raise FileNotFoundError(f"No .mmwave.json file found in {data_folder_calib}")

    # Parse configuration
    params = parse_mmwave_config(json_file)
    tx_enable_table, tx_channel_enabled = get_tx_enable_table(params)
    
    dev_config = params['DevConfig'][1]
    profile = dev_config['Profile'][1]
    frame_config = dev_config['FrameConfig']
    
    # Calculate parameters
    num_samples = profile['NumSamples']
    sample_rate = profile['SamplingRate']  # Hz
    freq_slope = profile['FreqSlope'] * 1e6  # Hz/s
    start_freq = profile['StartFreq'] * 1e9  # Hz
    
    # Range resolution
    chirp_time = num_samples / sample_rate
    bandwidth = freq_slope * chirp_time
    c = 3e8
    range_resolution = c / (2 * bandwidth)
    
    # Target range bin
    range_fft_size = 2 ** int(np.ceil(np.log2(num_samples)))
    range_bin_size = range_resolution * num_samples / range_fft_size
    target_range_bin = int(round(target_range_m / range_bin_size))
    
    print(f"  Range resolution: {range_resolution:.4f} m")
    print(f"  Target range bin: {target_range_bin}")
    
    # Get unique file indices
    file_idx_unique = get_unique_file_idx(data_folder_calib)
    if len(file_idx_unique) == 0:
        raise FileNotFoundError("No data files found")
    
    # Use first file for calibration
    file_struct = get_bin_file_names_with_idx(data_folder_calib, file_idx_unique[0])
    
    # Read calibration frame (frame 2, skip first)
    num_chirps_per_loop = len(params['TxToEnable'])
    num_loops = frame_config['NumChirpLoops']
    
    print(f"  Reading frame 2...")
    adc_data = read_adc_bin_tda2_separate_files(
        file_struct, frame_idx=2,
        num_sample_per_chirp=num_samples,
        num_chirp_per_loop=num_chirps_per_loop,
        num_loops=num_loops,
        num_rx_per_device=4,
        num_devices=1
    )
    
    # adc_data shape: (samples, loops, rx, chirps_per_loop)
    print(f"  ADC data shape: {adc_data.shape}")
    
    # Average across loops for better SNR
    adc_avg = np.mean(adc_data, axis=1)  # (samples, rx, tx)
    
    # Range FFT
    range_fft = np.fft.fft(adc_avg, n=range_fft_size, axis=0)
    
    # Extract target responses
    num_rx = adc_avg.shape[1]
    num_tx = adc_avg.shape[2]
    
    # Search for peak around expected range bin
    search_range = 5
    search_start = max(0, target_range_bin - search_range)
    search_end = min(range_fft_size, target_range_bin + search_range)
    
    # Find actual peak
    power_profile = np.sum(np.abs(range_fft[search_start:search_end]) ** 2, axis=(1, 2))
    peak_offset = np.argmax(power_profile)
    actual_peak_bin = search_start + peak_offset
    
    print(f"  Actual peak bin: {actual_peak_bin}")
    
    # Extract target response at peak
    target_response = range_fft[actual_peak_bin]  # (rx, tx)
    
    # Generate phase calibration matrix
    # Reference: RX0, TX0
    ref_phase = np.angle(target_response[0, 0])
    
    phase_calib_matrix = np.zeros((num_tx, num_rx), dtype=complex)
    for tx in range(num_tx):
        for rx in range(num_rx):
            phase = np.angle(target_response[rx, tx])
            phase_calib_matrix[tx, rx] = np.exp(1j * (ref_phase - phase))
    
    # Generate frequency calibration (per-sample phase correction)
    freq_calib_matrix = None
    if not phase_only:
        # Frequency calibration from slope across range bins
        freq_calib_matrix = np.ones(range_fft_size, dtype=complex)
        
        for sample in range(range_fft_size):
            if sample != actual_peak_bin:
                # Calculate expected phase based on range
                expected_range = sample * range_bin_size
                expected_delay = 2 * expected_range / c
                expected_phase = 2 * np.pi * start_freq * expected_delay
                
                # Simplified frequency calibration
                freq_calib_matrix[sample] = np.exp(-1j * expected_phase)
    
    # Calculate RX mismatch (for compatibility with existing calibration format)
    rx_mismatch = np.zeros((num_tx, num_rx), dtype=complex)
    for tx in range(num_tx):
        ref_mag = np.abs(target_response[0, tx])
        ref_phase_tx = np.angle(target_response[0, tx])
        for rx in range(num_rx):
            mag_ratio = ref_mag / (np.abs(target_response[rx, tx]) + 1e-10)
            phase_diff = ref_phase_tx - np.angle(target_response[rx, tx])
            rx_mismatch[tx, rx] = mag_ratio * np.exp(1j * phase_diff)
    
    # Compile results
    calib_result = {
        'phaseCalibrationMatrix': phase_calib_matrix,
        'freqCalibrationMatrix': freq_calib_matrix,
        'RxMismatch': rx_mismatch,
        'targetRangeBin': actual_peak_bin,
        'targetRange': actual_peak_bin * range_bin_size,
        'rangeResolution': range_resolution,
        'numTx': num_tx,
        'numRx': num_rx,
        'TxToEnable': params['TxToEnable'],
        'RxToEnable': params['RxToEnable'],
    }
    
    # Visualization
    if plot_on:
        import matplotlib
        # Use TkAgg backend for interactive plotting
        try:
            matplotlib.use('TkAgg')
        except:
            pass
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Range profile
        ax = axes[0, 0]
        range_axis = np.arange(range_fft_size) * range_bin_size
        power_db = 10 * np.log10(np.sum(np.abs(range_fft) ** 2, axis=(1, 2)) + 1)
        ax.plot(range_axis, power_db)
        ax.axvline(actual_peak_bin * range_bin_size, color='r', linestyle='--', label='Peak')
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Power (dB)')
        ax.set_title('Range Profile (All RX/TX)')
        ax.legend()
        ax.grid(True)
        
        # Phase calibration matrix
        ax = axes[0, 1]
        phase_matrix = np.angle(phase_calib_matrix) * 180 / np.pi
        im = ax.imshow(phase_matrix, cmap='RdBu', vmin=-180, vmax=180)
        ax.set_xlabel('RX Channel')
        ax.set_ylabel('TX Channel')
        ax.set_title('Phase Calibration (degrees)')
        plt.colorbar(im, ax=ax)
        
        # RX mismatch magnitude
        ax = axes[1, 0]
        mag_matrix = 20 * np.log10(np.abs(rx_mismatch) + 1e-10)
        im = ax.imshow(mag_matrix, cmap='viridis')
        ax.set_xlabel('RX Channel')
        ax.set_ylabel('TX Channel')
        ax.set_title('RX Mismatch Magnitude (dB)')
        plt.colorbar(im, ax=ax)
        
        # RX mismatch phase
        ax = axes[1, 1]
        phase_mismatch = np.angle(rx_mismatch) * 180 / np.pi
        im = ax.imshow(phase_mismatch, cmap='RdBu', vmin=-180, vmax=180)
        ax.set_xlabel('RX Channel')
        ax.set_ylabel('TX Channel')
        ax.set_title('RX Mismatch Phase (degrees)')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.show()
    
    # Save calibration
    if output_file is None:
        output_file = os.path.join(data_folder_calib, 'calibrationData.mat')
    
    print(f"Saving calibration to: {output_file}")
    savemat(output_file, {
        'calibResult': calib_result,
        'phaseCalibrationMatrix': phase_calib_matrix,
        'freqCalibrationMatrix': freq_calib_matrix if freq_calib_matrix is not None else np.array([]),
    })
    
    # Also save as npz
    npz_file = output_file.replace('.mat', '.npz')
    np.savez_compressed(npz_file,
        calibResult=calib_result,
        phaseCalibrationMatrix=phase_calib_matrix,
        freqCalibrationMatrix=freq_calib_matrix if freq_calib_matrix is not None else np.array([]),
    )
    print(f"Also saved to: {npz_file}")
    
    print("Calibration complete!")
    return calib_result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Antenna Calibration')
    parser.add_argument('--data_folder', required=True, help='Calibration data folder')
    parser.add_argument('--target_range', type=float, default=1.5, help='Target range (m)')
    parser.add_argument('--output', default=None, help='Output file path')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--phase-only', action='store_true', help='Phase calibration only')
    
    args = parser.parse_args()
    
    run_antenna_calibration(
        data_folder_calib=args.data_folder,
        target_range_m=args.target_range,
        output_file=args.output,
        plot_on=not args.no_plot,
        phase_only=args.phase_only,
    )
