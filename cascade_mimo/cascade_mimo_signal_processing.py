"""
cascade_mimo_signal_processing.py

Main MIMO signal processing chain for TI 4-chip cascade radar.

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/

Converted from MATLAB: cascade_MIMO_signalProcessing.m

This script processes raw ADC data through the following chain:
1. ADC data calibration
2. Range FFT
3. Doppler FFT  
4. CFAR detection
5. DOA estimation
6. 3D point cloud generation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from scipy.io import savemat

# Import modules
from .modules import (
    RangeProcessor, DopplerProcessor, CFARDetector,
    CalibrationCascade, DOACascade, SimTopCascade
)
from .modules.range_proc import RangeProcConfig
from .modules.doppler_proc import DopplerProcConfig
from .modules.cfar_caso import CFARConfig, Detection
from .modules.calibration_cascade import CalibrationConfig
from .modules.doa_cascade import DOAConfig, AngleEstimate, angles_to_xyz
from .modules.sim_top import SimTopConfig

from .utils.data_parse import get_unique_file_idx, get_bin_file_names_with_idx, get_valid_num_frames
from .utils.json_parser import parse_mmwave_config, get_tx_enable_table, find_mmwave_json
from .utils.plotting import plot_range_azimuth_2d, plot_3d_point_cloud, plot_range_profile


def run_mimo_signal_processing(
    data_folder_test: str,
    data_folder_calib: str,
    output_folder: Optional[str] = None,
    plot_on: bool = True,
    save_output: bool = False,
    log_scale: bool = True,
    num_frames_to_run: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run the complete MIMO signal processing chain.
    
    Args:
        data_folder_test: Path to test data folder
        data_folder_calib: Path to calibration file (.mat)
        output_folder: Output folder for results (optional)
        plot_on: Enable plotting
        save_output: Save results to file
        log_scale: Use log scale for plots
        num_frames_to_run: Limit number of frames (None = all)
        
    Returns:
        Dictionary containing processing results
    """
    print(f"Starting MIMO signal processing...")
    print(f"  Test data: {data_folder_test}")
    print(f"  Calibration: {data_folder_calib}")
    
    # Find JSON configuration file
    json_file = find_mmwave_json(data_folder_test)
    if json_file is None:
        raise FileNotFoundError(f"No .mmwave.json file found in {data_folder_test}")
    
    # Parse configuration
    print(f"  Parsing config: {json_file}")
    params = parse_mmwave_config(json_file)
    tx_enable_table, tx_channel_enabled = get_tx_enable_table(params)
    
    # Get device configuration
    dev_config = params['DevConfig'][1]
    profile = dev_config['Profile'][1]
    frame_config = dev_config['FrameConfig']
    
    # Initialize processing modules
    sim_top = SimTopCascade(SimTopConfig(
        numSamplePerChirp=profile['NumSamples'],
        adcSampleRate=profile['SamplingRate'],
        startFreqConst=profile['StartFreq'] * 1e9,
        chirpSlope=profile['FreqSlope'] * 1e12,
        chirpIdleTime=profile['IdleTime'] * 1e-6,
        chirpRampEndTime=profile['RampEndTime'] * 1e-6,
        nchirp_loops=frame_config['NumChirpLoops'],
        numTxAnt=len(params['TxToEnable']),
    ))
    
    range_proc = RangeProcessor(RangeProcConfig(
        rangeFFTSize=sim_top.rangeFFTSize,
        numSamplePerChirp=profile['NumSamples'],
    ))
    
    doppler_proc = DopplerProcessor(DopplerProcConfig(
        dopplerFFTSize=sim_top.dopplerFFTSize,
        numChirpsPerFrame=frame_config['NumChirpLoops'],
    ))
    
    cfar = CFARDetector(CFARConfig(
        rangeBinSize=sim_top.rangeBinSize,
        dopplerBinSize=sim_top.dopplerBinSize,
        numVirtualAntennas=sim_top.numVirtualAntennas,
    ))
    
    calibration = CalibrationCascade(CalibrationConfig(
        calibrationFilePath=data_folder_calib,
        numSamplePerChirp=profile['NumSamples'],
        nchirp_loops=frame_config['NumChirpLoops'],
        TxToEnable=params['TxToEnable'],
    ))
    
    doa = DOACascade(DOAConfig(
        numTxAntennas=len(params['TxToEnable']),
        numRxAntennas=len(params['RxToEnable']),
    ))
    
    # Get unique file indices
    file_idx_unique = get_unique_file_idx(data_folder_test)
    print(f"  Found {len(file_idx_unique)} data file(s)")
    
    # Storage for results
    all_results = {
        'detection_results': [],
        'angle_estimates': [],
        'xyz_points': [],
    }
    
    frame_count_global = 0
    
    if plot_on:
        fig = plt.figure(figsize=(16, 10))
    
    # Process each file
    for i_file, file_idx in enumerate(file_idx_unique):
        print(f"\nProcessing file {i_file + 1}/{len(file_idx_unique)}: index {file_idx}")
        
        # Get file names
        file_struct = get_bin_file_names_with_idx(data_folder_test, file_idx)
        calibration.set_bin_file_path(file_struct)
        
        # Get valid frames
        idx_file_path = os.path.join(data_folder_test, file_struct.masterIdxFile)
        num_valid_frames, _ = get_valid_num_frames(idx_file_path)
        
        if num_frames_to_run is not None:
            num_valid_frames = min(num_valid_frames, num_frames_to_run)
        
        print(f"  Processing {num_valid_frames} frames...")
        
        # Skip first frame (TDA2 artifact)
        for frame_idx in range(2, num_valid_frames + 1):
            calibration.config.frameIdx = frame_idx
            frame_count_global += 1
            
            if frame_idx % 10 == 1:
                print(f"    Frame {frame_idx}...")
            
            try:
                # Read and calibrate ADC data
                adc_data = calibration.datapath()
                
                # Process each TX
                range_fft_out = []
                doppler_fft_out = []
                
                num_tx = adc_data.shape[3] if adc_data.ndim > 3 else 1
                
                for i_tx in range(num_tx):
                    if num_tx > 1:
                        tx_data = adc_data[:, :, :, i_tx]
                    else:
                        tx_data = adc_data
                    
                    # Range FFT
                    r_fft = range_proc.datapath(tx_data)
                    range_fft_out.append(r_fft)
                    
                    # Doppler FFT
                    d_fft = doppler_proc.datapath(r_fft)
                    doppler_fft_out.append(d_fft)
                
                # Combine TX data for MIMO processing
                doppler_combined = np.stack(doppler_fft_out, axis=-1)
                doppler_combined = doppler_combined.reshape(
                    doppler_combined.shape[0], doppler_combined.shape[1], -1
                )
                
                # CFAR detection
                detections = cfar.detect(doppler_combined)
                all_results['detection_results'].append(detections)
                
                # DOA estimation
                if len(detections) > 0:
                    angle_ests = doa.process_detections(detections, doppler_combined)
                    all_results['angle_estimates'].append(angle_ests)
                    
                    # Convert to XYZ
                    xyz_points = []
                    for est in angle_ests:
                        x, y, z = angles_to_xyz(est.range, est.angles[0], est.angles[1])
                        xyz_points.append({
                            'x': x, 'y': y, 'z': z,
                            'velocity': est.doppler_corr,
                            'snr': est.estSNR,
                            'range': est.range,
                        })
                    all_results['xyz_points'].append(xyz_points)
                else:
                    all_results['angle_estimates'].append([])
                    all_results['xyz_points'].append([])
                
                # Plotting
                if plot_on and len(detections) > 0:
                    _update_plots(fig, doppler_combined, detections, xyz_points,
                                 sim_top, cfar.config, log_scale, frame_idx)
                    plt.pause(0.01)
                    
            except Exception as e:
                print(f"    Error processing frame {frame_idx}: {e}")
                continue
    
    print(f"\nProcessing complete. Processed {frame_count_global} frames.")
    
    # Save results
    if save_output and output_folder:
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, 'processing_results.mat')
        _save_results(output_file, all_results)
        print(f"Results saved to: {output_file}")
    
    if plot_on:
        plt.show()
    
    return all_results


def _update_plots(fig, doppler_fft: np.ndarray, detections: List[Detection],
                  xyz_points: List[Dict], sim_top: SimTopCascade,
                  cfar_config: CFARConfig, log_scale: bool, frame_idx: int):
    """Update visualization plots."""
    fig.clf()
    
    # Range profile
    ax1 = fig.add_subplot(2, 2, 1)
    sig_integrate = 10 * np.log10(np.sum(np.abs(doppler_fft) ** 2, axis=2) + 1)
    range_axis = np.arange(sig_integrate.shape[0]) * cfar_config.rangeBinSize
    
    # Zero-Doppler line
    zero_doppler = sig_integrate.shape[1] // 2
    ax1.plot(range_axis, sig_integrate[:, zero_doppler], 'g-', linewidth=2, label='Zero Doppler')
    
    # Mark detections
    for det in detections:
        ax1.plot(det.range, sig_integrate[det.rangeInd, zero_doppler], 
                'ko', markersize=8, markerfacecolor='lime')
    
    ax1.set_xlabel('Range (m)')
    ax1.set_ylabel('Power (dB)')
    ax1.set_title(f'Range Profile: Frame {frame_idx}')
    ax1.grid(True)
    ax1.legend()
    
    # Range-Doppler map
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(sig_integrate, aspect='auto', origin='lower', cmap='jet')
    ax2.set_xlabel('Doppler Bin')
    ax2.set_ylabel('Range Bin')
    ax2.set_title('Range-Velocity Map')
    
    # 3D Point cloud
    if len(xyz_points) > 0:
        ax3 = fig.add_subplot(2, 2, 4, projection='3d')
        x = [p['x'] for p in xyz_points]
        y = [p['y'] for p in xyz_points]
        z = [p['z'] for p in xyz_points]
        v = [p['velocity'] for p in xyz_points]
        
        scatter = ax3.scatter(x, y, z, c=v, cmap='RdBu', s=45)
        ax3.set_xlim(-20, 20)
        ax3.set_ylim(1, 50)
        ax3.set_zlim(-5, 5)
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_zlabel('Z (m)')
        ax3.set_title('3D Point Cloud')
        plt.colorbar(scatter, ax=ax3, label='Velocity (m/s)')


def _save_results(output_file: str, results: Dict[str, Any]):
    """Save results to MAT file."""
    # Convert to MATLAB-compatible format
    save_data = {
        'num_frames': len(results['detection_results']),
    }
    
    # Convert detections
    all_detections = []
    for frame_dets in results['detection_results']:
        frame_data = []
        for det in frame_dets:
            frame_data.append([det.rangeInd, det.dopplerInd, det.range, det.doppler, det.estSNR])
        all_detections.append(np.array(frame_data) if frame_data else np.array([]))
    save_data['detections'] = all_detections
    
    # Convert XYZ points
    all_xyz = []
    for frame_xyz in results['xyz_points']:
        if frame_xyz:
            xyz_array = np.array([[p['x'], p['y'], p['z'], p['velocity'], p['snr']] for p in frame_xyz])
        else:
            xyz_array = np.array([])
        all_xyz.append(xyz_array)
    save_data['xyz_points'] = all_xyz
    
    savemat(output_file, save_data, do_compression=True)
    
    # Also save as npz
    npz_file = output_file.replace('.mat', '.npz')
    np.savez_compressed(npz_file, **save_data)
    print(f"Also saved to: {npz_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='MIMO Signal Processing')
    parser.add_argument('--data_folder', required=True, help='Test data folder')
    parser.add_argument('--calib_file', required=True, help='Calibration .mat file')
    parser.add_argument('--output', default=None, help='Output folder')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--save', action='store_true', help='Save results')
    parser.add_argument('--frames', type=int, default=None, help='Number of frames to process')
    
    args = parser.parse_args()
    
    run_mimo_signal_processing(
        data_folder_test=args.data_folder,
        data_folder_calib=args.calib_file,
        output_folder=args.output,
        plot_on=not args.no_plot,
        save_output=args.save,
        num_frames_to_run=args.frames,
    )
