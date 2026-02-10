#!/usr/bin/env python3
"""
cascade_tx_phase_calibration.py

Copyright (C) 2020 Texas Instruments Incorporated - http://www.ti.com/

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

  Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the
  distribution.

  Neither the name of Texas Instruments Incorporated nor the names of
  its contributors may be used to endorse or promote products derived
  from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Top level main test chain to perform TX phase shifter calibration.
Output is saved to calibrateTXPhaseResults.npz

Usage: modify the dataFolder_calib_data_path
"""

import os
import re
import json
import time
import glob
from scipy.io import savemat
import numpy as np
import matplotlib
# Use TkAgg backend for interactive plotting (fallback to Agg if not available)
try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


# ==============================================================================
# Configuration Constants
# ==============================================================================

DEBUG_PLOTS = False  # optionally display debug plots while calibration data is being processed

NUM_DEVICES = 4                  # number of AWRx devices being processed
NUM_TX = 3                       # number of AWRx TX channels being processed
NUM_RX = 16                      # number of MMWCAS-RF-EVM RX channels being processed
NUM_PHASE_SHIFTER_OFFSETS = 64   # number of phase-shifter offset increments being processed
NUM_CHIRPS_LOOPS_PER_FRAME = 12  # number of chirp-loops per frame
NUM_CHIRPS_PER_LOOP = 64         # number of chirps per TX active phase
NUM_SAMPLES_PER_CHIRP = 256      # number of samples per chirp
SEARCH_BINS_SKIP = 50            # number of bins to skip when looking for peak

# Reference TX/Device channel for computing offsets
REF_TX = 0          # 0-indexed (was 1 in MATLAB)
REF_RX = 0          # 0-indexed (was 1 in MATLAB)
REF_PHASE_OFFSET = 0  # 0-indexed (was 1 in MATLAB)

# Estimated corner-reflector target range (in meters)
TARGET_RANGE = 5.0

# TI Cascade board constants
TI_CASCADE_TX_POSITION_AZI = [11, 10, 9, 32, 28, 24, 20, 16, 12, 8, 4, 0]
TI_CASCADE_TX_POSITION_ELE = [6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
TI_CASCADE_RX_POSITION_ELE = [0] * 16
TI_CASCADE_RX_POSITION_AZI = list(range(11, 15)) + list(range(50, 54)) + list(range(46, 50)) + list(range(0, 4))
TI_CASCADE_RX_ID = [13, 14, 15, 16, 1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8]
TI_CASCADE_ANTENNA_DESIGN_FREQ = 76.8  # GHz

SPEED_OF_LIGHT = 3e8


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class FileNameStruct:
    """Structure to hold binary file names for each device."""
    master: str = ""
    slave1: str = ""
    slave2: str = ""
    slave3: str = ""
    masterIdxFile: str = ""
    slave1IdxFile: str = ""
    slave2IdxFile: str = ""
    slave3IdxFile: str = ""
    dataFolderName: str = ""


@dataclass
class GenCalibrationMatrixParams:
    """Parameters for calibration matrix generation."""
    calibrateFileName: str = ""
    targetRange: float = 0.0
    frameIdx: int = 1
    numSamplePerChirp: int = 0
    nchirp_loops: int = 0
    numChirpsPerFrame: int = 0
    TxToEnable: List[int] = field(default_factory=list)
    Slope_calib: float = 0.0
    Sampling_Rate_sps: float = 0.0
    calibrationInterp: int = 0
    TI_Cascade_RX_ID: List[int] = field(default_factory=list)
    RxForMIMOProcess: List[int] = field(default_factory=list)
    TxForMIMOProcess: List[int] = field(default_factory=list)
    numRxToEnable: int = 0
    rangeResolution: float = 0.0
    dataPlatform: str = "TDA2"
    NumDevices: int = 4
    binDataFile: Optional[FileNameStruct] = None
    RxOrder: List[int] = field(default_factory=list)


# ==============================================================================
# Utility Functions
# ==============================================================================

def get_unique_file_idx(data_folder: str) -> List[str]:
    """
    Find all unique binary file indices in the data folder.
    
    Args:
        data_folder: Path to the data folder
        
    Returns:
        List of unique file indices as zero-padded strings (e.g., ['0000', '0001'])
    """
    pattern = os.path.join(data_folder, "*_data.bin")
    files = glob.glob(pattern)
    
    bin_file_ids = set()
    for file_path in files:
        file_name = os.path.basename(file_path)
        # Find underscores and extract the numerical value between first two underscores
        underscores = [i for i, c in enumerate(file_name) if c == '_']
        if len(underscores) >= 2:
            try:
                bin_id = int(file_name[underscores[0] + 1:underscores[1]])
                bin_file_ids.add(bin_id)
            except ValueError:
                continue
    
    # Sort and format as 4-digit strings
    return [f"{idx:04d}" for idx in sorted(bin_file_ids)]


def get_bin_file_names_with_idx(folder_name: str, file_idx: str) -> FileNameStruct:
    """
    Get binary file names for master and slave devices.
    
    Args:
        folder_name: Path to the data folder
        file_idx: File index string (e.g., '0000')
        
    Returns:
        FileNameStruct containing the file names
    """
    data_pattern = os.path.join(folder_name, f"*{file_idx}_data.bin")
    idx_pattern = os.path.join(folder_name, f"*{file_idx}_idx.bin")
    
    data_files = glob.glob(data_pattern)
    idx_files = glob.glob(idx_pattern)
    
    if len(data_files) > 4:
        raise ValueError("Too many binary files corresponding to the same fileIdx")
    
    file_struct = FileNameStruct(dataFolderName=folder_name)
    
    for data_file in data_files:
        name = os.path.basename(data_file)
        if 'master' in name:
            file_struct.master = name
        elif 'slave1' in name:
            file_struct.slave1 = name
        elif 'slave2' in name:
            file_struct.slave2 = name
        elif 'slave3' in name:
            file_struct.slave3 = name
    
    for idx_file in idx_files:
        name = os.path.basename(idx_file)
        if 'master' in name:
            file_struct.masterIdxFile = name
        elif 'slave1' in name:
            file_struct.slave1IdxFile = name
        elif 'slave2' in name:
            file_struct.slave2IdxFile = name
        elif 'slave3' in name:
            file_struct.slave3IdxFile = name
    
    return file_struct


def read_bin_file(file_path: str, frame_idx: int, num_sample_per_chirp: int,
                  num_chirp_per_loop: int, num_loops: int, num_rx_per_device: int,
                  num_devices: int) -> np.ndarray:
    """
    Read binary ADC data file for a single device.
    
    Args:
        file_path: Full path to the binary file
        frame_idx: Frame index (1-indexed as in MATLAB)
        num_sample_per_chirp: Number of samples per chirp
        num_chirp_per_loop: Number of chirps per loop
        num_loops: Number of loops
        num_rx_per_device: Number of RX channels per device
        num_devices: Number of devices
        
    Returns:
        Complex ADC data array with shape (num_sample_per_chirp, num_loops, num_rx_per_device, num_chirp_per_loop)
    """
    expected_samples_per_frame = num_sample_per_chirp * num_chirp_per_loop * num_loops * num_rx_per_device * 2
    
    with open(file_path, 'rb') as fp:
        # Seek to the frame position (frame_idx is 1-indexed)
        fp.seek((frame_idx - 1) * expected_samples_per_frame * 2)
        
        # Read data as uint16
        adc_data = np.fromfile(fp, dtype=np.uint16, count=expected_samples_per_frame)
    
    # Convert to signed (handle negative values stored as 2's complement)
    neg_mask = (adc_data & 0x8000) != 0
    adc_data = adc_data.astype(np.int32)
    adc_data[neg_mask] = adc_data[neg_mask] - 65536
    
    # Convert to complex (I + jQ)
    adc_complex = adc_data[0::2] + 1j * adc_data[1::2]
    
    # Reshape: (numLoops, numChirpPerLoop, numSamplePerChirp, numRXPerDevice)
    adc_complex = adc_complex.reshape(num_loops, num_chirp_per_loop, 
                                   num_sample_per_chirp, num_rx_per_device)

    # Permute to: (numSamplePerChirp, numLoops, numRXPerDevice, numChirpPerLoop)
    adc_complex = np.transpose(adc_complex, (2, 0, 3, 1))
    
    return adc_complex


def read_adc_bin_tda2_separate_files(file_name_cascade: FileNameStruct, frame_idx: int,
                                      num_sample_per_chirp: int, num_chirp_per_loop: int,
                                      num_loops: int, num_rx_per_device: int,
                                      num_devices: int) -> np.ndarray:
    """
    Read raw ADC data from separate files for each device.
    
    Args:
        file_name_cascade: FileNameStruct with file paths
        frame_idx: Frame index (1-indexed)
        num_sample_per_chirp: Number of samples per chirp
        num_chirp_per_loop: Number of chirps per loop  
        num_loops: Number of loops
        num_rx_per_device: Number of RX per device (typically 4)
        num_devices: Number of devices
        
    Returns:
        Combined radar data with shape (num_sample_per_chirp, num_loops, 16, num_chirp_per_loop)
    """
    data_folder = file_name_cascade.dataFolderName
    
    # Read data from each device
    master_data = read_bin_file(os.path.join(data_folder, file_name_cascade.master),
                                frame_idx, num_sample_per_chirp, num_chirp_per_loop,
                                num_loops, num_rx_per_device, num_devices)
    slave1_data = read_bin_file(os.path.join(data_folder, file_name_cascade.slave1),
                                frame_idx, num_sample_per_chirp, num_chirp_per_loop,
                                num_loops, num_rx_per_device, num_devices)
    slave2_data = read_bin_file(os.path.join(data_folder, file_name_cascade.slave2),
                                frame_idx, num_sample_per_chirp, num_chirp_per_loop,
                                num_loops, num_rx_per_device, num_devices)
    slave3_data = read_bin_file(os.path.join(data_folder, file_name_cascade.slave3),
                                frame_idx, num_sample_per_chirp, num_chirp_per_loop,
                                num_loops, num_rx_per_device, num_devices)
    
    # Combine all channels: Master (1-4), Slave1 (5-8), Slave2 (9-12), Slave3 (13-16)
    radar_data = np.concatenate([master_data, slave1_data, slave2_data, slave3_data], axis=2)
    
    return radar_data


def json_parser(json_file_path: str) -> Dict[str, Any]:
    """
    Parse mmWave JSON configuration file.
    
    Args:
        json_file_path: Path to the JSON file
        
    Returns:
        Dictionary with parsed parameters
    """
    with open(json_file_path, 'r') as f:
        mmwave_json = json.load(f)
    
    mmwave_devices_config = mmwave_json['mmWaveDevices']
    
    params = {
        'NumDevices': len(mmwave_devices_config),
        'TxToEnable': [],
        'RxToEnable': [],
        'DevConfig': {}
    }
    
    for count, device_config in enumerate(mmwave_devices_config):
        dev_id = device_config['mmWaveDeviceId'] + 1  # 1-indexed
        
        # Frame type
        waveform_type = device_config['rfConfig']['waveformType']
        if waveform_type == 'singleFrameChirp':
            params['FrameType'] = 0
        elif waveform_type == 'advancedFrameChirp':
            params['FrameType'] = 1
        elif waveform_type == 'continuousWave':
            params['FrameType'] = 2
        
        # TX channel enable
        tx_channel_en = int(device_config['rfConfig']['rlChanCfg_t']['txChannelEn'], 16)
        for tx_channel in range(3):
            if tx_channel_en & (1 << tx_channel):
                params['TxToEnable'].append(3 * (dev_id - 1) + tx_channel + 1)
        
        # RX channel enable
        rx_channel_en = int(device_config['rfConfig']['rlChanCfg_t']['rxChannelEn'], 16)
        for rx_channel in range(4):
            if rx_channel_en & (1 << rx_channel):
                params['RxToEnable'].append(4 * (dev_id - 1) + rx_channel + 1)
        
        # Initialize device config
        if dev_id not in params['DevConfig']:
            params['DevConfig'][dev_id] = {'Profile': {}, 'Chirp': {}, 'FrameConfig': {}}
        
        # Profile configuration
        profiles = device_config['rfConfig']['rlProfiles']
        params['DevConfig'][dev_id]['NumProfiles'] = len(profiles)
        
        for profile_idx, profile in enumerate(profiles):
            profile_cfg = profile['rlProfileCfg_t']
            params['DevConfig'][dev_id]['Profile'][profile_idx + 1] = {
                'ProfileId': profile_cfg['profileId'],
                'StartFreq': profile_cfg['startFreqConst_GHz'],
                'FreqSlope': profile_cfg['freqSlopeConst_MHz_usec'],
                'IdleTime': profile_cfg['idleTimeConst_usec'],
                'AdcStartTime': profile_cfg['adcStartTimeConst_usec'],
                'RampEndTime': profile_cfg['rampEndTime_usec'],
                'NumSamples': profile_cfg['numAdcSamples'],
                'SamplingRate': profile_cfg['digOutSampleRate'],
            }
        
        # Chirp configuration
        chirps = device_config['rfConfig']['rlChirps']
        params['DevConfig'][dev_id]['NumChirps'] = 0
        
        for chirp_block in chirps:
            chirp_cfg = chirp_block['rlChirpCfg_t']
            start_idx = chirp_cfg['chirpStartIdx']
            end_idx = chirp_cfg['chirpEndIdx']
            params['DevConfig'][dev_id]['NumChirps'] += end_idx - start_idx + 1
            
            tx_enable = int(chirp_cfg['txEnable'], 16)
            for chirp_id in range(start_idx + 1, end_idx + 2):
                params['DevConfig'][dev_id]['Chirp'][chirp_id] = {
                    'ChirpIdx': chirp_id,
                    'ProfileId': chirp_cfg['profileId'],
                    'Tx0Enable': (tx_enable >> 0) & 1,
                    'Tx1Enable': (tx_enable >> 1) & 1,
                    'Tx2Enable': (tx_enable >> 2) & 1,
                }
        
        # Frame configuration
        if params.get('FrameType', 0) == 0:
            frame_cfg = device_config['rfConfig']['rlFrameCfg_t']
            params['DevConfig'][dev_id]['FrameConfig'] = {
                'ChirpIdx': frame_cfg['chirpStartIdx'],
                'ChirpEndIdx': frame_cfg['chirpEndIdx'],
                'NumChirpLoops': frame_cfg['numLoops'],
                'NumFrames': frame_cfg['numFrames'],
                'Periodicity': frame_cfg['framePeriodicity_msec'],
            }
    
    return params


def parse_calibration_params_from_json(data_folder: str, data_platform: str) -> GenCalibrationMatrixParams:
    """
    Parse calibration parameters from JSON file in the data folder.
    
    Args:
        data_folder: Path to the calibration data folder
        data_platform: Data platform (e.g., 'TDA2')
        
    Returns:
        GenCalibrationMatrixParams object with parsed parameters
    """
    # Find JSON file
    json_files = glob.glob(os.path.join(data_folder, "*.mmwave.json"))
    if len(json_files) != 1:
        raise ValueError(f"Expected exactly one .mmwave.json file in {data_folder}, found {len(json_files)}")
    
    json_file = json_files[0]
    print(f"paramFile= {json_file}")
    
    params_chirp = json_parser(json_file)
    
    # Build TX enable table
    num_chirp_config = params_chirp['DevConfig'][1]['NumChirps']
    num_tx_per_dev = 3
    tot_tx = num_tx_per_dev * params_chirp['NumDevices']
    
    tx_enable_table = np.zeros((num_chirp_config, tot_tx), dtype=int)
    
    for i_dev in range(1, params_chirp['NumDevices'] + 1):
        for i_config in range(1, num_chirp_config + 1):
            chirp = params_chirp['DevConfig'][i_dev]['Chirp'].get(i_config, {})
            tx_enable_table[i_config - 1, 0 + (i_dev - 1) * num_tx_per_dev] = chirp.get('Tx0Enable', 0)
            tx_enable_table[i_config - 1, 1 + (i_dev - 1) * num_tx_per_dev] = chirp.get('Tx1Enable', 0)
            tx_enable_table[i_config - 1, 2 + (i_dev - 1) * num_tx_per_dev] = chirp.get('Tx2Enable', 0)
    
    # Find enabled TX channel for each chirp config
    tx_channel_enabled = []
    for i_config in range(num_chirp_config):
        channel_ids = np.where(tx_enable_table[i_config, :] != 0)[0]
        if len(channel_ids) > 0:
            tx_channel_enabled.append(channel_ids[0] + 1)  # 1-indexed
        else:
            tx_channel_enabled.append(0)
    
    # Extract profile parameters
    profile = params_chirp['DevConfig'][1]['Profile'][1]
    frame_config = params_chirp['DevConfig'][1]['FrameConfig']
    
    num_adc_sample = profile['NumSamples']
    adc_sample_rate = profile['SamplingRate'] * 1e3  # Hz
    chirp_slope = profile['FreqSlope'] * 1e12  # Hz/s
    idle_time = profile['IdleTime'] * 1e-6  # s
    ramp_end_time = profile['RampEndTime'] * 1e-6  # s
    nchirp_loops = frame_config['NumChirpLoops']
    
    # Derived parameters
    chirp_ramp_time = num_adc_sample / adc_sample_rate
    chirp_bandwidth = chirp_slope * chirp_ramp_time
    num_sample_per_chirp = round(chirp_ramp_time * adc_sample_rate)
    range_resolution = SPEED_OF_LIGHT / 2 / chirp_bandwidth
    
    # Create calibration params object
    cal_params = GenCalibrationMatrixParams(
        dataPlatform=data_platform,
        NumDevices=params_chirp['NumDevices'],
        numSamplePerChirp=num_sample_per_chirp,
        Sampling_Rate_sps=adc_sample_rate,
        nchirp_loops=nchirp_loops,
        numChirpsPerFrame=nchirp_loops * len(tx_channel_enabled),
        TxToEnable=tx_channel_enabled,
        Slope_calib=chirp_slope,
        calibrationInterp=5,
        TI_Cascade_RX_ID=TI_CASCADE_RX_ID,
        RxForMIMOProcess=TI_CASCADE_RX_ID,
        TxForMIMOProcess=tx_channel_enabled,
        numRxToEnable=len(params_chirp['RxToEnable']),
        rangeResolution=range_resolution,
        RxOrder=TI_CASCADE_RX_ID,
    )
    
    return cal_params


def cascade_read_tx_cal_data(obj: GenCalibrationMatrixParams) -> np.ndarray:
    """
    Read TX calibration data from binary files.
    
    Args:
        obj: GenCalibrationMatrixParams object with configuration
        
    Returns:
        Radar data array
    """
    if obj.dataPlatform == 'TDA2':
        num_chirp_per_loop = len(obj.TxToEnable)
        num_loops = obj.nchirp_loops
        num_rx_per_device = 4
        
        radar_data = read_adc_bin_tda2_separate_files(
            obj.binDataFile,
            obj.frameIdx,
            obj.numSamplePerChirp,
            num_chirp_per_loop,
            num_loops,
            num_rx_per_device,
            1
        )
    else:
        raise ValueError(f"Not supported data capture platform: {obj.dataPlatform}")
    
    return radar_data


# ==============================================================================
# Main Processing Function
# ==============================================================================

def run_tx_phase_calibration(
    data_folder_calib_data_path: str,
    data_platform: str = 'TDA2',
    target_range: float = TARGET_RANGE,
    debug_plots: bool = DEBUG_PLOTS,
    num_phase_shifter_offsets: int = NUM_PHASE_SHIFTER_OFFSETS,
    num_chirps_loops_per_frame: int = NUM_CHIRPS_LOOPS_PER_FRAME,
    num_chirps_per_loop: int = NUM_CHIRPS_PER_LOOP,
    num_samples_per_chirp: int = NUM_SAMPLES_PER_CHIRP,
    num_rx: int = NUM_RX,
    search_bins_skip: int = SEARCH_BINS_SKIP,
    ref_tx: int = REF_TX,
    ref_rx: int = REF_RX,
    ref_phase_offset: int = REF_PHASE_OFFSET,
) -> Dict[str, np.ndarray]:
    """
    Run TX phase shifter calibration.
    
    Args:
        data_folder_calib_data_path: Path to folder containing calibration datasets
        data_platform: Data capture platform (default: 'TDA2')
        target_range: Estimated corner-reflector target range in meters
        debug_plots: Whether to display debug plots
        num_phase_shifter_offsets: Number of phase-shifter offset increments
        num_chirps_loops_per_frame: Number of chirp-loops per frame
        num_chirps_per_loop: Number of chirps per TX active phase
        num_samples_per_chirp: Number of samples per chirp
        num_rx: Number of RX channels
        search_bins_skip: Number of bins to skip when looking for peak
        ref_tx: Reference TX channel (0-indexed)
        ref_rx: Reference RX channel (0-indexed)
        ref_phase_offset: Reference phase offset index (0-indexed)
        
    Returns:
        Dictionary containing phase calibration results
    """
    start_time = time.time()
    
    print(f"dataPlatform = {data_platform}")
    
    # Initialize result arrays
    phase_values = np.zeros((num_chirps_loops_per_frame, num_rx, num_phase_shifter_offsets))
    phase_values_bin = np.zeros((num_chirps_loops_per_frame, num_rx, num_phase_shifter_offsets))
    phase_values_target_distance = np.zeros((num_chirps_loops_per_frame, num_rx, num_phase_shifter_offsets))
    
    # Setup debug plots if enabled
    if debug_plots:
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
        fig4, ax4 = plt.subplots(1, 1, figsize=(10, 6))
        fig5, ax5 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Get list of subdirectories in calibration folder
    calib_data_info = []
    for entry in os.scandir(data_folder_calib_data_path):
        if entry.is_dir() and entry.name not in ['.', '..']:
            calib_data_info.append(entry)
    
    calib_data_info.sort(key=lambda x: x.name)
    
    if len(calib_data_info) < num_phase_shifter_offsets:
        print(f"Warning: Found only {len(calib_data_info)} folders, expected {num_phase_shifter_offsets}")
        num_phase_shifter_offsets = len(calib_data_info)
    
    # Process each phase shifter offset dataset
    for idx_ps, folder_entry in enumerate(calib_data_info[:num_phase_shifter_offsets]):
        print(f"dataFolder_calib_data_info(folderIdx).name = {folder_entry.name}")
        
        # Create folder structure for this iteration
        data_folder_name = os.path.join(folder_entry.path, '')
        
        # Get unique file indices
        file_idx_unique = get_unique_file_idx(data_folder_name)
        if not file_idx_unique:
            print(f"Warning: No data files found in {data_folder_name}")
            continue
        
        # Get file names
        file_name_struct = get_bin_file_names_with_idx(data_folder_name, file_idx_unique[0])
        
        # Parse parameters from JSON
        cal_params = parse_calibration_params_from_json(data_folder_name, data_platform)
        cal_params.calibrateFileName = data_folder_name
        cal_params.targetRange = target_range
        cal_params.binDataFile = file_name_struct
        
        # Use second frame for calibration
        cal_params.frameIdx = 2
        
        # Read calibration data
        cal_data = cascade_read_tx_cal_data(cal_params)
        
        # Process each TX and RX channel
        for idx_tx in range(num_chirps_loops_per_frame):
            for idx_rx in range(num_rx):
                # 1D FFT
                cal_data_1dfft = np.fft.fftshift(
                    np.fft.fft(cal_data[:, :, idx_rx, idx_tx], n=cal_params.numSamplePerChirp, axis=0),
                    axes=0
                )
                
                # 2D FFT
                cal_data_2dfft = (1.0 / num_chirps_per_loop) * np.fft.fftshift(
                    np.fft.fft(cal_data_1dfft, n=num_chirps_per_loop, axis=1),
                    axes=1
                )
                
                # Find target peak bin in 2D-FFT 0-velocity bin
                zero_velocity_bin_idx = num_chirps_per_loop // 2
                search_range = np.abs(cal_data_2dfft[search_bins_skip:num_samples_per_chirp, zero_velocity_bin_idx])
                target_bin_value = np.max(search_range)
                target_bin_idx = np.argmax(search_range) + search_bins_skip
                
                phase_values_bin[idx_tx, idx_rx, idx_ps] = target_bin_idx
                phase_values_target_distance[idx_tx, idx_rx, idx_ps] = target_bin_idx * cal_params.rangeResolution
                
                # Record phase at target peak bin
                phase_values[idx_tx, idx_rx, idx_ps] = np.angle(
                    cal_data_2dfft[target_bin_idx, zero_velocity_bin_idx]
                ) * 180 / np.pi
                
                # Debug plots
                if debug_plots:
                    ax1.clear()
                    ax1.plot(10 * np.log10(np.abs(cal_data_2dfft[:, zero_velocity_bin_idx]) + 1e-10))
                    ax1.plot(target_bin_idx, 10 * np.log10(target_bin_value + 1e-10), 'ro')
                    ax1.set_title('Calibration Target Power vs. IF bins')
                    ax1.set_xlabel('1D-FFT Spectrum (bins)')
                    ax1.set_ylabel('1D-FFT Magnitude (dB)')
                    
                    ax2.clear()
                    ax2.plot(np.angle(cal_data_2dfft[:, zero_velocity_bin_idx]) * 180 / np.pi)
                    ax2.plot(target_bin_idx, phase_values[idx_tx, idx_rx, idx_ps], 'ro')
                    ax2.set_title('Calibration Target Phase vs. IF bins')
                    ax2.set_xlabel('1D-FFT Spectrum (bins)')
                    ax2.set_ylabel('1D-FFT Phase (degrees)')
                    
                    ax3.clear()
                    ax3.plot(phase_values_bin[idx_tx, idx_rx, :idx_ps + 1])
                    ax3.set_title('Calibration Target Detected Index')
                    ax3.set_xlabel('Phase Shifter Offset (5.625 degrees/LSB)')
                    ax3.set_ylabel('Calibration Target Sampled IF Index')
                    
                    ax4.clear()
                    ax4.plot(phase_values_target_distance[idx_tx, idx_rx, :idx_ps + 1])
                    ax4.set_title('Calibration Target Distance')
                    ax4.set_xlabel('Phase Shifter Offset (5.625 degrees/LSB)')
                    ax4.set_ylabel('Target Distance (meters)')
                    
                    ax5.clear()
                    ax5.plot(phase_values[idx_tx, idx_rx, :idx_ps + 1])
                    ax5.set_title('Calibration Target Phase vs. IF bins')
                    ax5.set_xlabel('Phase Shifter Offset (5.625 degrees/LSB)')
                    ax5.set_ylabel('Target 1D-FFT Phase (degrees)')
                    
                    plt.pause(0.01)
    
    # Compute phase offsets and phase errors
    phase_offset_values = np.zeros_like(phase_values)
    phase_offset_error = np.zeros_like(phase_values)
    
    for idx_tx in range(num_chirps_loops_per_frame):
        for idx_rx in range(num_rx):
            for idx_ps in range(num_phase_shifter_offsets):
                # Reference the phase-shifter 0 setting from channel ref_tx, ref_rx
                phase_offset_values[idx_tx, idx_rx, idx_ps] = (
                    phase_values[idx_tx, idx_rx, idx_ps] - 
                    phase_values[ref_tx, idx_rx, ref_phase_offset]
                )
                
                # Find error between actual phase offset and expected (ideal) phase offset
                # idx_ps is 0-indexed, so we add 1 to match MATLAB's 1-indexed behavior
                phase_offset_error[idx_tx, idx_rx, idx_ps] = (
                    phase_values[idx_tx, idx_rx, idx_ps] - (idx_ps + 1) * 5.625
                )
    
    # Create result plots
    fig6, ax6 = plt.subplots(1, 1, figsize=(12, 8))
    fig7, ax7 = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot phase values for RX channel 1 (index 1, 0-indexed)
    rx_idx_for_plot = 1
    for tx_idx in range(num_chirps_loops_per_frame):
        color = (tx_idx / num_chirps_loops_per_frame, tx_idx / num_chirps_loops_per_frame, rx_idx_for_plot * 0.0625)
        ax6.plot(phase_values[tx_idx, rx_idx_for_plot, :], 
                 color=color, label=f"TX{tx_idx + 1}")
    
    ax6.set_title('TX Channel Target Phase Value (deg) vs. TX Phase-Shifter Offset Value (5.625 deg increment)')
    ax6.set_xlabel('TX Phase-Shifter Offset Value (5.625 deg increment)')
    ax6.set_ylabel('TX Channel Absolute Phase (deg)')
    ax6.legend()
    ax6.grid(True)
    
    # Plot phase offset values for RX channel 0 (index 0, 0-indexed)
    rx_idx_for_offset_plot = 0
    for tx_idx in range(num_chirps_loops_per_frame):
        color = (tx_idx / num_chirps_loops_per_frame, tx_idx / num_chirps_loops_per_frame, rx_idx_for_offset_plot * 0.0625)
        ax7.plot(phase_offset_values[tx_idx, rx_idx_for_offset_plot, :], 
                 color=color, label=f"TX{tx_idx + 1}")
    
    ax7.set_title('TX Channel Phase Offset (deg) vs. TX Phase-Shifter Offset Value (5.625 deg increment)')
    ax7.set_xlabel('TX Phase-Shifter Offset Value (5.625 deg increment)')
    ax7.set_ylabel('TX Channel Phase Offset (deg)')
    ax7.legend()
    ax7.grid(True)
    
    # Save results
    results_file = os.path.join(data_folder_calib_data_path, 'calibrateTXPhaseResults.npz')
    np.savez(results_file,
             phaseOffsetValues=phase_offset_values,
             phaseValues=phase_values,
             phaseOffsetError=phase_offset_error)
    print(f"Saved calibration results to: {results_file}")
    
    # Save full debug data
    full_results_file = os.path.join(data_folder_calib_data_path, 'calibrateTXPhaseResultsFileFull.npz')
    np.savez(full_results_file,
             phaseOffsetValues=phase_offset_values,
             phaseValues=phase_values,
             phaseOffsetError=phase_offset_error,
             phaseValuesBin=phase_values_bin,
             phaseValuesTargetDistance=phase_values_target_distance)
    print(f"Saved full debug (numpy) data to: {full_results_file}")
    
    # Save full debug data as .mat file (MATLAB compatible)
    full_results_file = os.path.join(data_folder_calib_data_path, 'calibrateTXPhaseResultsFileFull.mat')
    savemat(full_results_file, {
        'phaseOffsetValues': phase_offset_values,
        'phaseValues': phase_values,
        'phaseOffsetError': phase_offset_error,
        'phaseValuesBin': phase_values_bin,
        'phaseValuesTargetDistance': phase_values_target_distance,
    })
    print(f"Saved full debug (matlab) data to: {full_results_file}")
    
    elapsed_time = time.time() - start_time
    print(f"Processing time: {elapsed_time:.2f} seconds")
    
    plt.show()
    
    return {
        'phaseOffsetValues': phase_offset_values,
        'phaseValues': phase_values,
        'phaseOffsetError': phase_offset_error,
        'phaseValuesBin': phase_values_bin,
        'phaseValuesTargetDistance': phase_values_target_distance,
    }


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == '__main__':
    # Example configuration - modify these paths for your setup
    PRO_PATH = r'C:\ti\mmwave_studio_03_00_00_14\mmWaveStudio\MatlabExamples\4chip_cascade_MIMO_example'
    DATA_FOLDER_CALIB_DATA_PATH = r'C:\ti\mmwave_studio_03_00_00_14\mmWaveStudio\PostProc\cal1'
    DATA_PLATFORM = 'TDA2'
    
    # Run calibration
    results = run_tx_phase_calibration(
        data_folder_calib_data_path=DATA_FOLDER_CALIB_DATA_PATH,
        data_platform=DATA_PLATFORM,
        target_range=5.0,
        debug_plots=False,
    )
