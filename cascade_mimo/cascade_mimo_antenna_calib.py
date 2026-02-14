"""
cascade_mimo_antenna_calib.py

Antenna calibration for TI 4-chip cascade MIMO radar.

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/

Converted from MATLAB: cascade_MIMO_antennaCalib.m and dataPath.m

This script generates calibration matrices from corner reflector data.
Rewritten to match MATLAB implementation exactly.
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
from common.constants import TI_CASCADE_RX_ID, DEFAULT_CALIBRATION_INTERP, SPEED_OF_LIGHT
from common.config import FileNameStruct


def hann_local(length: int) -> np.ndarray:
    """
    Local implementation of Hanning window to match MATLAB behavior.
    
    Args:
        length: Window length
        
    Returns:
        win: Generated windowing coefficients
    """
    win = np.arange(1, length + 1) / (length + 1)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * win)
    return win


def average_ph(ph_arr_rad: np.ndarray) -> float:
    """
    Average phase values in radians, handling phase wrapping.
    
    Takes input an array of angles in RADIAN units.
    Output is average of the angles in RADIAN units.
    
    Args:
        ph_arr_rad: Array of angles in RADIAN units
        
    Returns:
        avg_ph: Average of the angles in RADIAN units
    """
    diff_ph = np.angle(np.exp(1j * (ph_arr_rad - ph_arr_rad[0])))
    ph_arr_rad = ph_arr_rad[0] + diff_ph
    avg_ph = np.mean(ph_arr_rad)
    return avg_ph


def radar_fft_find_peak(
    rx_data: np.ndarray,
    range_bin_search_min: int,
    range_bin_search_max: int,
    interp_fact: int
) -> tuple:
    """
    Find the peak location and complex value that corresponds to the calibration target.
    
    Args:
        rx_data: Input ADC data
        range_bin_search_min: Start of the range bin to search for peak
        range_bin_search_max: End of the range bin to search for peak
        interp_fact: Interpolation factor for FFT
        
    Returns:
        rx_fft: Complex FFT values
        angle_fft_peak: Phase at highest peak (in degrees)
        val_fft_peak: Complex value at highest peak
        fund_range_index: Bin number for highest peak
    """
    effective_num_samples = len(rx_data)
    wind = hann_local(effective_num_samples)
    wind = wind / np.sqrt(np.mean(wind**2))  # RMS normalization
    
    rx_data_prefft = rx_data * wind
    rx_fft = np.fft.fft(rx_data_prefft, interp_fact * effective_num_samples)
    
    rx_fft_searchwindow = np.abs(rx_fft[range_bin_search_min:range_bin_search_max])
    fund_range_index = np.argmax(rx_fft_searchwindow)
    fund_range_index = fund_range_index + range_bin_search_min
    
    angle_fft_peak = np.angle(rx_fft[fund_range_index]) * 180 / np.pi
    val_fft_peak = rx_fft[fund_range_index]
    
    return rx_fft, angle_fft_peak, val_fft_peak, fund_range_index


def run_antenna_calibration(
    data_folder_calib: str,
    target_range_m: float = 1.5,
    output_file: Optional[str] = None,
    calibration_interp: int = 5,
    frame_idx: int = 2,
    plot_on: bool = True,
) -> Dict[str, Any]:
    """
    Generate antenna calibration results from corner reflector data.
    MATLAB-compatible implementation of dataPath.m
    
    Args:
        data_folder_calib: Path to calibration data folder
        target_range_m: Expected range to corner reflector (meters)
        output_file: Output .mat file path (default: calibrateResults_high.mat)
        calibration_interp: Interpolation factor for calibration
        frame_idx: Frame index to use (1-indexed, default=2)
        plot_on: Enable visualization
        
    Returns:
        Dictionary containing calibration results matching MATLAB output
    """
    print(f"Starting antenna calibration (MATLAB-compatible)...")
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
    
    # Extract parameters (match MATLAB variable names)
    samples_per_chirp = profile['NumSamples']
    # SamplingRate from JSON is in ksps, convert to sps (Hz)
    sampling_rate_sps = profile['SamplingRate'] * 1000  # ksps -> Hz
    # FreqSlope is in MHz/us, convert to Hz/s
    slope_calib = profile['FreqSlope'] * 1e12  # MHz/us -> Hz/s
    start_freq = profile['StartFreq'] * 1e9  # Hz
    
    tx_to_enable = params['TxToEnable']
    num_tx = len(tx_to_enable)
    num_rx = len(TI_CASCADE_RX_ID)
    nchirp_loops = frame_config['NumChirpLoops']
    
    # Check that all 12 TX are enabled
    if num_tx < 12:
        raise ValueError(
            'This data set cannot be used for calibration, '
            'all 12 channels should be enabled'
        )
    
    # Range resolution
    chirp_time = samples_per_chirp / sampling_rate_sps
    bandwidth = slope_calib * chirp_time
    range_resolution = SPEED_OF_LIGHT / (2 * bandwidth)
    
    print(f"  Range resolution: {range_resolution:.4f} m")
    print(f"  Number of TX: {num_tx}, Number of RX: {num_rx}")
    
    # Calculate range bin search window (MATLAB formula)
    range_bin_search_min = int(round(
        samples_per_chirp * calibration_interp * 
        ((target_range_m - 2) * 2 * slope_calib / (SPEED_OF_LIGHT * sampling_rate_sps))
    )) + 1
    
    range_bin_search_max = int(round(
        samples_per_chirp * calibration_interp * 
        ((target_range_m + 2) * 2 * slope_calib / (SPEED_OF_LIGHT * sampling_rate_sps))
    )) + 1
    
    print(f"  Search range: bins {range_bin_search_min} to {range_bin_search_max}")
    
    # Get unique file indices
    file_idx_unique = get_unique_file_idx(data_folder_calib)
    if len(file_idx_unique) == 0:
        raise FileNotFoundError("No data files found")
    
    # Get binary file names
    file_struct = get_bin_file_names_with_idx(data_folder_calib, file_idx_unique[0])
    
    # Read ADC data
    num_chirp_per_loop = len(tx_to_enable)
    num_loops = nchirp_loops
    num_rx_per_device = 4
    
    print(f"  Reading frame {frame_idx}...")
    radar_data_rxchain = read_adc_bin_tda2_separate_files(
        file_struct, frame_idx,
        samples_per_chirp, num_chirp_per_loop, num_loops,
        num_rx_per_device, 1
    )
    
    # radar_data_rxchain shape: (samples, loops, rx, tx)
    print(f"  ADC data shape: {radar_data_rxchain.shape}")
    
    # Re-ordering of the TX data (convert to 0-indexed)
    tx_indices = [tx - 1 for tx in tx_to_enable]
    radar_data_rxchain = radar_data_rxchain[:, :, :, tx_indices]
    
    # Find the peak location and complex value for each TX-RX pair
    # This is the core MATLAB algorithm
    angle_mat = np.zeros((num_tx, num_rx))
    range_mat = np.zeros((num_tx, num_rx))
    peak_val_mat = np.zeros((num_tx, num_rx), dtype=complex)
    
    # Initialize rx_fft
    fft_size = samples_per_chirp * calibration_interp
    rx_fft = np.zeros((fft_size, num_rx, num_tx), dtype=complex)
    
    print(f"  Processing {num_tx} TX x {num_rx} RX channels...")
    for i_tx in range(num_tx):
        for i_rx in range(num_rx):
            # Average chirps within a frame (average across loops dimension)
            rx_data = np.mean(radar_data_rxchain[:, :, i_rx, i_tx], axis=1)
            
            # Find peak for this channel
            rx_fft_curr, angle, val_peak, rangebin = radar_fft_find_peak(
                rx_data, range_bin_search_min, range_bin_search_max, calibration_interp
            )
            
            rx_fft[:, i_rx, i_tx] = rx_fft_curr
            angle_mat[i_tx, i_rx] = angle
            range_mat[i_tx, i_rx] = rangebin
            peak_val_mat[i_tx, i_rx] = val_peak
    
    # Calculate RX and TX mismatch (MATLAB algorithm)
    rx_mismatch = []
    tx_mismatch = []
    
    # TX index used for TX calibration (first enabled TX, 0-indexed)
    tx_ind_calib = 0
    
    # Calculate RX mismatch
    in_mat = angle_mat
    num_rxs = in_mat.shape[1]
    temp = in_mat - in_mat[:, [0]]  # Subtract first RX column
    
    for i in range(num_rxs):
        rx_mismatch.append(average_ph(temp[:, i] * np.pi / 180) * 180 / np.pi)
    
    rx_mismatch = np.array(rx_mismatch)
    
    # Calculate TX mismatch
    num_txs = in_mat.shape[0]
    temp = in_mat - in_mat[[tx_ind_calib], :]  # Subtract reference TX row
    
    for i in range(num_txs):
        tx_mismatch.append(average_ph(temp[i, :] * np.pi / 180) * 180 / np.pi)
    
    # MATLAB iterates in reverse order compared to Python
    # Reverse to match MATLAB's iteration order
    tx_mismatch = np.array(tx_mismatch)[::-1]
    
    # MATLAB reshapes column-wise (Fortran order) and the result is (3, 4)
    tx_mismatch = tx_mismatch.reshape(3, 4, order='F')
    
    # Estimate target range from results
    target_range_est = (
        np.floor(np.mean(range_mat) / calibration_interp) * range_resolution
    )
    print(f"  Target is estimated at range {target_range_est:.3f} m")
    print(f"  Mean range bin: {np.mean(range_mat):.2f}")
    
    # Compile results (MATLAB-compatible structure)
    calib_result = {
        'AngleMat': angle_mat,
        'RangeMat': range_mat,
        'PeakValMat': peak_val_mat,
        'RxMismatch': rx_mismatch,
        'TxMismatch': tx_mismatch,
        'Rx_fft': rx_fft
    }
    # Compile results (MATLAB-compatible structure)
    calib_result = {
        'AngleMat': angle_mat,
        'RangeMat': range_mat,
        'PeakValMat': peak_val_mat,
        'RxMismatch': rx_mismatch,
        'TxMismatch': tx_mismatch,
        'Rx_fft': rx_fft
    }
    
    # Visualization
    if plot_on:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot 1: Peak index across all channels
            ax = axes[0]
            ax.plot(range_mat.flatten(), 'o-')
            ax.grid(True)
            ax.set_title('Peak index across all channels')
            ax.set_xlabel('Channel Index (TX*16 + RX)')
            ax.set_ylabel('Range Bin')
            
            # Plot 2: FFT magnitude for first RX across all TX
            ax = axes[1]
            ax.plot(np.abs(rx_fft[:, 0, :]))
            ax.set_title('Range FFT (RX 0, All TX)')
            ax.set_xlabel('Range Bin')
            ax.set_ylabel('Magnitude')
            ax.grid(True)
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"  Warning: Could not create plots: {e}")
    
    # Prepare params for saving (compatibility with MATLAB)
    params_dict = {
        'Slope_MHzperus': slope_calib / 1e12,
        'Sampling_Rate_sps': sampling_rate_sps
    }
    
    # Save calibration data
    if output_file is None:
        output_file = os.path.join(data_folder_calib, 'calibrateResults_high.mat')
    
    print(f"  Saving calibration to: {output_file}")
    savemat(output_file, {
        'calibResult': calib_result,
        'params': params_dict
    })
    
    # Also save as npz for Python
    npz_file = output_file.replace('.mat', '.npz')
    np.savez_compressed(
        npz_file,
        calibResult=calib_result,
        params=params_dict,
        AngleMat=angle_mat,
        RangeMat=range_mat,
        PeakValMat=peak_val_mat,
        RxMismatch=rx_mismatch,
        TxMismatch=tx_mismatch,
        Rx_fft=rx_fft
    )
    print(f"  Also saved to: {npz_file}")
    
    print("Calibration complete!")
    
    return calib_result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Antenna Calibration (MATLAB-compatible)')
    parser.add_argument('--data_folder', required=True, help='Calibration data folder')
    parser.add_argument('--target_range', type=float, default=5.0, help='Target range (m)')
    parser.add_argument('--output', default=None, help='Output file path')
    parser.add_argument('--frame_idx', type=int, default=2, help='Frame index (default: 2)')
    parser.add_argument('--calibration_interp', type=int, default=5, 
                        help='Calibration interpolation factor (default: 5)')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    
    args = parser.parse_args()
    
    run_antenna_calibration(
        data_folder_calib=args.data_folder,
        target_range_m=args.target_range,
        output_file=args.output,
        calibration_interp=args.calibration_interp,
        frame_idx=args.frame_idx,
        plot_on=not args.no_plot,
    )
