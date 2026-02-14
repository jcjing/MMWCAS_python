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
import matplotlib
# Use TkAgg backend for interactive plotting (fallback to Agg if not available)
try:
    matplotlib.use('TkAgg')
except:
    pass
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

# Import antenna position constants for range-azimuth plotting
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.constants import (
    TI_CASCADE_TX_POSITION_AZI, TI_CASCADE_TX_POSITION_ELE,
    TI_CASCADE_RX_POSITION_AZI, TI_CASCADE_RX_POSITION_ELE,
    TI_CASCADE_RX_ID
)


def compute_azimuth_only_antennas(num_tx: int, num_rx: int) -> np.ndarray:
    """
    Compute azimuth-only antenna indices from virtual MIMO array.
    Matches MATLAB implementation where antenna_azimuthonly is computed from
    antenna positions with elevation = 0.
    
    Args:
        num_tx: Number of TX antennas
        num_rx: Number of RX antennas
        
    Returns:
        Array of antenna indices for azimuth-only processing
    """
    # Build virtual array positions using the RX channel order from TI cascade EVM
    # MATLAB: D_RX = TI_Cascade_RX_position_azi(RxForMIMOProcess)
    # where RxForMIMOProcess = [13 14 15 16 1 2 3 4 9 10 11 12 5 6 7 8]
    virtual_ant_azi = []
    virtual_ant_ele = []
    
    for tx in range(num_tx):
        for rx_idx in range(num_rx):
            # Use TI_CASCADE_RX_ID to get the actual RX channel order (convert 1-based to 0-based)
            rx = TI_CASCADE_RX_ID[rx_idx] - 1
            virtual_ant_azi.append(TI_CASCADE_TX_POSITION_AZI[tx] + TI_CASCADE_RX_POSITION_AZI[rx])
            virtual_ant_ele.append(TI_CASCADE_TX_POSITION_ELE[tx] + TI_CASCADE_RX_POSITION_ELE[rx])
    
    # Find antennas with elevation = 0 (azimuth-only)
    # MATLAB: ind = find(D(:,2)==0); [val ID_unique] = unique(D(ind,1)); antenna_azimuthonly = ind(ID_unique);
    ind = [i for i, ele in enumerate(virtual_ant_ele) if ele == 0]
    ind_azi = [virtual_ant_azi[i] for i in ind]
    
    # MATLAB unique returns sorted unique values and indices of first occurrence
    unique_azi, unique_indices = np.unique(ind_azi, return_index=True)
    
    # Map back to global antenna indices: antenna_azimuthonly = ind(ID_unique)
    antenna_azimuthonly = [ind[idx] for idx in unique_indices]
    
    return np.array(antenna_azimuthonly)


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
        adcSampleRate=profile['SamplingRate'] * 1000,  # Convert ksps to sps
        startFreqConst=profile['StartFreq'] * 1e9,  # Convert GHz to Hz
        chirpSlope=profile['FreqSlope'] * 1e12,  # Convert MHz/us to Hz/s
        chirpIdleTime=profile['IdleTime'] * 1e-6,  # Convert us to s
        chirpRampEndTime=profile['RampEndTime'] * 1e-6,  # Convert us to s
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
        detectMethod=1,  # Dual-pass CASO-CFAR
        numAntenna=sim_top.numVirtualAntennas,
        refWinSize=[8, 4],  # [range, doppler] training cells
        guardWinSize=[8, 0],  # [range, doppler] guard cells
        K0=[5.0, 3.0],  # [range, doppler] threshold scale in LINEAR (not dB!)
        rangeBinSize=sim_top.rangeBinSize,  # Calculated from radar parameters
        velocityBinSize=sim_top.dopplerBinSize,  # Calculated from radar parameters
        dopplerFFTSize=sim_top.dopplerFFTSize,
        discardCellLeft=10,
        discardCellRight=20,
        numRxAnt=len(params['RxToEnable']),
        TDM_MIMO_numTX=len(params['TxToEnable']),
    ))
    
    calibration = CalibrationCascade(CalibrationConfig(
        calibrationFilePath=data_folder_calib,
        numSamplePerChirp=profile['NumSamples'],
        nchirp_loops=frame_config['NumChirpLoops'],
        TxToEnable=params['TxToEnable'],
        chirpSlope=profile['FreqSlope'] * 1e12,  # Convert MHz/us to Hz/s
        Slope_calib=profile['FreqSlope'] * 1e12,
        Sampling_Rate_sps=profile['SamplingRate'] * 1000,  # Convert ksps to sps
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
        'magdatastatic': [],
        'magdatadynamic': [],
        'xa': [],
        'ya': [],
        'Dopplerout': [],  # For comparison with MATLAB
    }
    
    # Compute azimuth-only antenna indices (used for range-azimuth heatmap)
    num_tx_antennas = len(params['TxToEnable'])
    num_rx_antennas = len(params['RxToEnable'])
    antenna_azimuthonly = compute_azimuth_only_antennas(num_tx_antennas, num_rx_antennas)
    
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
        num_valid_frames = get_valid_num_frames(idx_file_path)
        
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
                # Match MATLAB's RX-major ordering: reshape (512, 64, 16, 12) -> (512, 64, 192)
                # MATLAB's column-major reshape: v = k + 16*l where k=RX (dim 3), l=TX (dim 4)
                # This means RX varies fastest: RX0-TX0, RX1-TX0, RX2-TX0, ..., RX15-TX0, RX0-TX1, ...
                # Python's C-order reshape has rightmost dimension varying fastest
                # So we need to transpose to (512, 64, 12, 16) then reshape to get TX varying fastest
                # which gives the same as MATLAB's RX varying fastest
                
                doppler_stacked = np.stack(doppler_fft_out, axis=-1)  # Shape: (512, 64, 16, 12)
                doppler_transposed = np.transpose(doppler_stacked, (0, 1, 3, 2))  # (512, 64, 12, 16)
                doppler_combined = doppler_transposed.reshape(
                    doppler_transposed.shape[0], 
                    doppler_transposed.shape[1], 
                    -1
                )  # (512, 64, 192) with RX-major: index v = RX + 16*TX
                
                # CFAR detection
                detections = cfar.datapath(doppler_combined)
                all_results['detection_results'].append(detections)
                
                # Compute range-azimuth heatmap (MATLAB: plot_range_azimuth_2D)
                STATIC_ONLY = 1  # MATLAB default
                minRangeBinKeep = 5
                rightRangeBinDiscard = 20
                mag_data_static, mag_data_dynamic, y_axis, x_axis = plot_range_azimuth_2d(
                    range_resolution=cfar.config.rangeBinSize,
                    radar_data_pre_3dfft=doppler_combined,
                    TDM_MIMO_numTX=num_tx_antennas,
                    numRxAnt=num_rx_antennas,
                    antenna_azimuthonly=antenna_azimuthonly,
                    LOG=log_scale,
                    STATIC_ONLY=STATIC_ONLY,
                    PLOT_ON=False,  # We'll plot separately in _update_plots
                    minRangeBinKeep=minRangeBinKeep,
                    rightRangeBinDiscard=rightRangeBinDiscard,
                )
                
                # Store range-azimuth results (like MATLAB)
                all_results['magdatastatic'].append(mag_data_static)
                all_results['magdatadynamic'].append(mag_data_dynamic)
                all_results['xa'].append(x_axis)
                all_results['ya'].append(y_axis)
                all_results['Dopplerout'].append(doppler_combined)
                all_results['Dopplerout'].append(doppler_combined)
                
                # DOA estimation
                if len(detections) > 0:
                    angle_ests = doa.datapath(detections)
                    all_results['angle_estimates'].append(angle_ests)
                    
                    # Convert to XYZ (MATLAB format: flip azimuth and elevation signs)
                    xyz_points = []
                    for est in angle_ests:
                        # MATLAB flips azimuth: x = range * sin(-azimuth) * cos(elevation)
                        # MATLAB flips elevation: z = range * sin(-elevation)
                        azim_rad = np.radians(-est.angles[0])
                        elev_rad = np.radians(-est.angles[1])
                        x = est.range * np.sin(azim_rad) * np.cos(elev_rad)
                        y = est.range * np.cos(azim_rad) * np.cos(elev_rad)
                        z = est.range * np.sin(elev_rad)
                        
                        xyz_points.append({
                            'x': x, 'y': y, 'z': z,
                            'velocity': est.doppler_corr,
                            'snr': est.estSNR,
                            'range': est.range,
                            'doppler_corr_overlap': est.doppler_corr_overlap,
                            'doppler_corr_FFT': est.doppler_corr_FFT,
                            'dopplerInd_org': est.dopplerInd_org,
                        })
                    all_results['xyz_points'].append(xyz_points)
                else:
                    all_results['angle_estimates'].append([])
                    all_results['xyz_points'].append([])
                
                # Plotting
                if plot_on:
                    _update_plots(fig, doppler_combined, detections, xyz_points,
                                 sim_top, cfar.config, log_scale, frame_idx,
                                 mag_data_static, y_axis, x_axis)
                    plt.pause(0.01)
                    
            except Exception as e:
                print(f"    Error processing frame {frame_idx}: {e}")
                continue
    
    print(f"\nProcessing complete. Processed {frame_count_global} frames.")
    
    # Save results
    if save_output:
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            # Extract test name from folder path (MATLAB-compatible naming)
            folder_name = os.path.basename(data_folder_test.rstrip('/\\'))
            output_file = os.path.join(output_folder, f'newOutput_{folder_name}.mat')
        else:
            # Save to parent directory with MATLAB-compatible naming (like MATLAB does)
            folder_name = os.path.basename(data_folder_test.rstrip('/\\'))
            parent_dir = os.path.dirname(os.path.abspath(data_folder_test))
            output_file = os.path.join(parent_dir, f'newOutput_{folder_name}.mat')
        _save_results(output_file, all_results)
    
    if plot_on:
        plt.show()
    
    return all_results


def _update_plots(fig, doppler_fft: np.ndarray, detections: List[Detection],
                  xyz_points: List[Dict], sim_top: SimTopCascade,
                  cfar_config: CFARConfig, log_scale: bool, frame_idx: int,
                  mag_data_static: np.ndarray, y_axis: np.ndarray, x_axis: np.ndarray):
    """Update visualization plots."""
    fig.clf()
    
    # Range profile
    ax1 = fig.add_subplot(2, 2, 1)
    sig_integrate = 10 * np.log10(np.sum(np.abs(doppler_fft) ** 2, axis=2) + 1)
    range_axis = np.arange(sig_integrate.shape[0]) * cfar_config.rangeBinSize
    
    # Plot all doppler bins (like MATLAB)
    for ii in range(sig_integrate.shape[1]):
        ax1.plot(range_axis, sig_integrate[:, ii], linewidth=0.5)
        # Mark detections for this doppler bin
        for det in detections:
            if det.dopplerInd == ii:
                ax1.plot(det.range, sig_integrate[det.rangeInd, ii], 
                        'o', linewidth=0.5, markeredgecolor='k', 
                        markerfacecolor=[0.49, 1, 0.63], markersize=6)
    
    # Zero-Doppler line (thick green line on top)
    zero_doppler = sig_integrate.shape[1] // 2
    ax1.plot(range_axis, sig_integrate[:, zero_doppler], '-', c='limegreen', linewidth=1, label='Zero Doppler')
    
    ax1.set_xlabel('Range (m)')
    ax1.set_ylabel('Receive Power (dB)')
    ax1.set_title(f'Range Profile (zero Doppler - thick green line): frameID {frame_idx}')
    ax1.grid(True)
    
    # Range-Doppler map (matches MATLAB's imagesc)
    ax2 = fig.add_subplot(2, 2, 2)
    # MATLAB's imagesc displays matrix as-is: rows=Y-axis (range), columns=X-axis (doppler)
    # Python's imshow does the same, but we need origin='upper' to match MATLAB
    im = ax2.imshow(sig_integrate, aspect='auto', origin='upper', cmap='jet')
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Relative Power (dB)')
    ax2.set_xlabel('Doppler Bin')
    ax2.set_ylabel('Range Bin')
    ax2.set_title('Range/Velocity Plot')
    
    # Range-Azimuth plot using pre-computed data
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Apply non-linear scaling for visualization (MATLAB uses ^0.4)
    mag_data_display = mag_data_static ** 0.4
    
    # MATLAB uses surf(y_axis, x_axis, mag_data) with view(2)
    # This makes y_axis horizontal and x_axis vertical
    surf = ax3.pcolormesh(y_axis, x_axis, mag_data_display, cmap='jet', shading='auto')
    ax3.set_xlabel('meters')
    ax3.set_ylabel('meters')
    ax3.set_title('range/azimuth heat map static objects')
    ax3.set_ylim(bottom=0)
    plt.colorbar(surf, ax=ax3)
    
    # 3D Point cloud
    if len(xyz_points) > 0:
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        x = [p['x'] for p in xyz_points]
        y = [p['y'] for p in xyz_points]
        z = [p['z'] for p in xyz_points]
        v = [p['velocity'] for p in xyz_points]
        
        scatter = ax4.scatter(x, y, z, c=v, cmap='RdBu', s=45)
        ax4.set_xlim(-20, 20)
        ax4.set_ylim(1, 50)
        ax4.set_zlim(-5, 5)
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.set_zlabel('Z (m)')
        ax4.set_title('3D Point Cloud')
        plt.colorbar(scatter, ax=ax4, label='Velocity (m/s)')


def _save_results(output_file: str, results: Dict[str, Any]):
    """Save results to MAT file in MATLAB-compatible format."""
    num_frames = len(results['detection_results'])
    
    # Create cell arrays (object arrays in Python/scipy)
    detection_results_all = np.empty((1, num_frames), dtype=object)
    angles_all_all = np.empty((1, num_frames), dtype=object)
    xyz_all = np.empty((1, num_frames), dtype=object)
    magdatastatic = np.empty((1, num_frames), dtype=object)
    magdatadynamic = np.empty((1, num_frames), dtype=object)
    xa = np.empty((1, num_frames), dtype=object)
    ya = np.empty((1, num_frames), dtype=object)
    Dopplerout = np.empty((1, num_frames), dtype=object)
    
    for i in range(num_frames):
        # Convert detection results to structured array
        detections = results['detection_results'][i]
        if len(detections) > 0:
            # Create structured array with proper dtype
            # Each field must be a 2D array to match MATLAB format
            det_dtype = np.dtype([
                ('rangeInd', 'O'),
                ('range', 'O'),
                ('dopplerInd_org', 'O'),
                ('dopplerInd', 'O'),
                ('doppler', 'O'),
                ('doppler_corr', 'O'),
                ('noise_var', 'O'),
                ('bin_val', 'O'),
                ('estSNR', 'O'),
                ('doppler_corr_overlap', 'O'),
                ('doppler_corr_FFT', 'O'),
            ])
            det_array = np.zeros((len(detections),), dtype=det_dtype)
            for j, det in enumerate(detections):
                # Wrap each scalar in a 2D array to match MATLAB format
                det_array[j]['rangeInd'] = np.array([[det.rangeInd]])
                det_array[j]['range'] = np.array([[det.range]])
                det_array[j]['dopplerInd_org'] = np.array([[det.dopplerInd_org]])
                det_array[j]['dopplerInd'] = np.array([[det.dopplerInd]])
                det_array[j]['doppler'] = np.array([[det.doppler]])
                det_array[j]['doppler_corr'] = np.array([[det.doppler_corr]])
                det_array[j]['noise_var'] = np.array([[det.noise_var]])
                det_array[j]['bin_val'] = det.bin_val.reshape(-1, 1)
                det_array[j]['estSNR'] = np.array([[det.estSNR]])
                det_array[j]['doppler_corr_overlap'] = np.array([[det.doppler_corr_overlap]])
                det_array[j]['doppler_corr_FFT'] = np.array([[det.doppler_corr_FFT]])
            detection_results_all[0, i] = det_array
        else:
            detection_results_all[0, i] = np.array([])
        
        # Store angle estimates
        angles_est = results['angle_estimates'][i]
        if len(angles_est) > 0:
            # Build angles matrix: [azimuth, elevation, SNR, rangeInd, doppler_corr, range]
            angles_matrix = np.zeros((len(angles_est), 6))
            for j, est in enumerate(angles_est):
                angles_matrix[j, 0] = est.angles[0]  # azimuth
                angles_matrix[j, 1] = est.angles[1]  # elevation  
                angles_matrix[j, 2] = est.estSNR
                angles_matrix[j, 3] = est.rangeInd
                angles_matrix[j, 4] = est.doppler_corr
                angles_matrix[j, 5] = est.range
            angles_all_all[0, i] = angles_matrix
        else:
            angles_all_all[0, i] = np.array([])
        
        # Store XYZ points
        if len(results['xyz_points'][i]) > 0:
            # Build XYZ matrix: [x, y, z, velocity, range, SNR, doppler_corr_overlap, doppler_corr_FFT, dopplerInd_org]
            xyz_matrix = np.zeros((len(results['xyz_points'][i]), 9))
            for j, xyz_data in enumerate(results['xyz_points'][i]):
                xyz_matrix[j, 0] = xyz_data['x']
                xyz_matrix[j, 1] = xyz_data['y']
                xyz_matrix[j, 2] = xyz_data['z']
                xyz_matrix[j, 3] = xyz_data['velocity']
                xyz_matrix[j, 4] = xyz_data['range']
                xyz_matrix[j, 5] = xyz_data['snr']
                xyz_matrix[j, 6] = xyz_data.get('doppler_corr_overlap', 0)
                xyz_matrix[j, 7] = xyz_data.get('doppler_corr_FFT', 0)
                xyz_matrix[j, 8] = xyz_data.get('dopplerInd_org', 0)
            xyz_all[0, i] = xyz_matrix
        else:
            xyz_all[0, i] = np.array([])
        
        # Store range-azimuth heatmap data
        magdatastatic[0, i] = results['magdatastatic'][i]
        magdatadynamic[0, i] = results['magdatadynamic'][i]
        xa[0, i] = results['xa'][i]
        ya[0, i] = results['ya'][i]
        Dopplerout[0, i] = results['Dopplerout'][i]
    
    # Save in MATLAB format
    save_data = {
        'detection_results_all': detection_results_all,
        'angles_all_all': angles_all_all,
        'xyz_all': xyz_all,
        'magdatastatic': magdatastatic,
        'magdatadynamic': magdatadynamic,
        'xa': xa,
        'ya': ya,
        'Dopplerout': Dopplerout,
    }
    
    savemat(output_file, save_data, do_compression=True)
    print(f"Saved results to: {output_file}")


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
