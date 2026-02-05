"""
cascade_txbf_signal_processing.py

TX beamforming signal processing for TI 4-chip cascade radar.

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/

Converted from MATLAB: cascade_TxBF_signalProcessing.m
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
from scipy.io import savemat

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cascade_mimo.modules import RangeProcessor, DopplerProcessor, CFARDetector
from cascade_mimo.modules.range_proc import RangeProcConfig
from cascade_mimo.modules.doppler_proc import DopplerProcConfig
from cascade_mimo.modules.cfar_caso import CFARConfig
from cascade_mimo.utils.data_parse import (
    get_unique_file_idx, get_bin_file_names_with_idx,
    read_adc_bin_tda2_separate_files, get_valid_num_frames
)
from cascade_mimo.utils.json_parser import parse_mmwave_config, find_mmwave_json
from cascade_mimo.utils.plotting import plot_range_profile, plot_range_doppler


def run_txbf_signal_processing(
    data_folder_test: str,
    steer_angle_deg: float = 0.0,
    output_folder: Optional[str] = None,
    plot_on: bool = True,
    save_output: bool = False,
    num_frames_to_run: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run TX beamforming signal processing.
    
    TX beamforming uses all TX antennas simultaneously with phase shifts
    to steer the transmit beam. This script processes the resulting data.
    
    Args:
        data_folder_test: Path to test data folder
        steer_angle_deg: Expected steering angle (for reference)
        output_folder: Output folder for results
        plot_on: Enable plotting
        save_output: Save results to file
        num_frames_to_run: Limit number of frames
        
    Returns:
        Dictionary with processing results
    """
    print(f"Starting TX Beamforming signal processing...")
    print(f"  Data folder: {data_folder_test}")
    print(f"  Expected steer angle: {steer_angle_deg}°")
    
    # Find and parse configuration
    json_file = find_mmwave_json(data_folder_test)
    if json_file is None:
        raise FileNotFoundError(f"No .mmwave.json file found in {data_folder_test}")
    
    params = parse_mmwave_config(json_file)
    dev_config = params['DevConfig'][1]
    profile = dev_config['Profile'][1]
    frame_config = dev_config['FrameConfig']
    
    # Calculate RF parameters
    num_samples = profile['NumSamples']
    sample_rate = profile['SamplingRate']
    freq_slope = profile['FreqSlope'] * 1e6  # Hz/s
    
    chirp_time = num_samples / sample_rate
    bandwidth = freq_slope * chirp_time
    c = 3e8
    range_resolution = c / (2 * bandwidth)
    
    range_fft_size = 2 ** int(np.ceil(np.log2(num_samples)))
    doppler_fft_size = 2 ** int(np.ceil(np.log2(frame_config['NumChirpLoops'])))
    
    range_bin_size = range_resolution * num_samples / range_fft_size
    
    # Initialize processors
    range_proc = RangeProcessor(RangeProcConfig(
        rangeFFTSize=range_fft_size,
        numSamplePerChirp=num_samples,
    ))
    
    doppler_proc = DopplerProcessor(DopplerProcConfig(
        dopplerFFTSize=doppler_fft_size,
        numChirpsPerFrame=frame_config['NumChirpLoops'],
    ))
    
    cfar = CFARDetector(CFARConfig(
        rangeBinSize=range_bin_size,
    ))
    
    # Get data files
    file_idx_unique = get_unique_file_idx(data_folder_test)
    print(f"  Found {len(file_idx_unique)} data file(s)")
    
    # Results storage
    all_results = {
        'range_profiles': [],
        'doppler_maps': [],
        'detections': [],
    }
    
    frame_count = 0
    
    if plot_on:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for i_file, file_idx in enumerate(file_idx_unique):
        file_struct = get_bin_file_names_with_idx(data_folder_test, file_idx)
        
        idx_file_path = os.path.join(data_folder_test, file_struct.masterIdxFile)
        num_valid_frames, _ = get_valid_num_frames(idx_file_path)
        
        if num_frames_to_run:
            num_valid_frames = min(num_valid_frames, num_frames_to_run)
        
        print(f"\nProcessing file {i_file + 1}: {num_valid_frames} frames")
        
        # For TX beamforming, all TXs fire simultaneously (1 chirp type)
        num_chirps_per_loop = 1  # TX BF uses single chirp with all TX
        num_loops = frame_config['NumChirpLoops']
        
        for frame_idx in range(2, num_valid_frames + 1):
            try:
                # Read ADC data
                adc_data = read_adc_bin_tda2_separate_files(
                    file_struct, frame_idx,
                    num_samples, num_chirps_per_loop, num_loops,
                    num_rx_per_device=4, num_devices=1
                )
                
                # Sum across RX (combine coherently)
                # For TX BF, antenna pattern is in TX side, RX just captures
                adc_sum = adc_data[:, :, :, 0] if adc_data.ndim > 3 else adc_data
                
                # Range FFT
                range_fft = range_proc.datapath(adc_sum)
                
                # Doppler FFT
                doppler_fft = doppler_proc.datapath(range_fft)
                
                # Sum power across RX for detection
                power_map = np.sum(np.abs(doppler_fft) ** 2, axis=2)
                
                all_results['range_profiles'].append(np.sum(np.abs(range_fft) ** 2, axis=(1, 2)))
                all_results['doppler_maps'].append(power_map)
                
                # Detection (using power map only)
                dets = cfar.detect(doppler_fft)
                all_results['detections'].append(dets)
                
                frame_count += 1
                
                if plot_on and frame_idx % 5 == 0:
                    _update_txbf_plots(fig, axes, range_fft, doppler_fft, dets,
                                       range_bin_size, frame_idx, steer_angle_deg)
                    plt.pause(0.01)
                    
            except Exception as e:
                print(f"    Error frame {frame_idx}: {e}")
    
    print(f"\nProcessed {frame_count} frames")
    
    if save_output and output_folder:
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, 'txbf_results.mat')
        savemat(output_file, {
            'range_profiles': np.array([r for r in all_results['range_profiles']]),
            'steer_angle': steer_angle_deg,
        })
        print(f"Results saved to: {output_file}")
        
        # Also save as npz
        npz_file = os.path.join(output_folder, 'txbf_results.npz')
        np.savez_compressed(npz_file,
            range_profiles=np.array([r for r in all_results['range_profiles']]),
            steer_angle=steer_angle_deg,
        )
        print(f"Also saved to: {npz_file}")
    
    if plot_on:
        plt.show()
    
    return all_results


def _update_txbf_plots(fig, axes, range_fft, doppler_fft, detections,
                       range_bin_size, frame_idx, steer_angle):
    """Update visualization for TX beamforming processing."""
    for ax in axes.flat:
        ax.clear()
    
    # Range profile
    ax = axes[0, 0]
    range_power = np.sum(np.abs(range_fft) ** 2, axis=(1, 2))
    power_db = 10 * np.log10(range_power + 1e-10)
    range_axis = np.arange(len(power_db)) * range_bin_size
    ax.plot(range_axis, power_db)
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Power (dB)')
    ax.set_title(f'Range Profile - Frame {frame_idx}')
    ax.grid(True)
    
    # Range-Doppler map
    ax = axes[0, 1]
    power_map = np.sum(np.abs(doppler_fft) ** 2, axis=2)
    power_map_db = 10 * np.log10(power_map + 1e-10)
    ax.imshow(power_map_db, aspect='auto', origin='lower', cmap='jet')
    ax.set_xlabel('Doppler Bin')
    ax.set_ylabel('Range Bin')
    ax.set_title('Range-Doppler Map')
    
    # Detection overlay
    ax = axes[1, 0]
    ax.imshow(power_map_db, aspect='auto', origin='lower', cmap='jet')
    for det in detections:
        ax.plot(det.dopplerInd, det.rangeInd, 'wo', markersize=10, markeredgecolor='k')
    ax.set_xlabel('Doppler Bin')
    ax.set_ylabel('Range Bin')
    ax.set_title(f'Detections (n={len(detections)})')
    
    # Info text
    ax = axes[1, 1]
    ax.axis('off')
    info_text = f"TX Beamforming Processing\n" \
                f"Frame: {frame_idx}\n" \
                f"Expected Steer Angle: {steer_angle}°\n" \
                f"Detections: {len(detections)}"
    ax.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=14,
            transform=ax.transAxes)
    
    fig.tight_layout()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='TX Beamforming Signal Processing')
    parser.add_argument('--data_folder', required=True, help='Test data folder')
    parser.add_argument('--steer_angle', type=float, default=0.0, help='Expected steer angle')
    parser.add_argument('--output', default=None, help='Output folder')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--frames', type=int, default=None, help='Number of frames')
    
    args = parser.parse_args()
    
    run_txbf_signal_processing(
        data_folder_test=args.data_folder,
        steer_angle_deg=args.steer_angle,
        output_folder=args.output,
        plot_on=not args.no_plot,
        num_frames_to_run=args.frames,
    )
