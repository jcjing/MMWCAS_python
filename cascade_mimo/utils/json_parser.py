"""
json_parser.py

mmWave JSON configuration file parser.

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/

Converted from MATLAB: JsonParser.m
"""

import json
import os
from typing import Dict, Any, List, Optional


def load_mmwave_json(json_file_path: str) -> Dict[str, Any]:
    """
    Load and parse an mmWave JSON configuration file.
    
    Args:
        json_file_path: Path to the JSON file
        
    Returns:
        Parsed JSON as dictionary
    """
    with open(json_file_path, 'r') as f:
        return json.load(f)


def parse_mmwave_config(json_file_path: str) -> Dict[str, Any]:
    """
    Parse mmWave JSON configuration file and extract radar parameters.
    
    Converts from MATLAB JsonParser.m
    
    Args:
        json_file_path: Path to the .mmwave.json file
        
    Returns:
        Dictionary containing parsed parameters:
        - NumDevices: Number of mmWave devices
        - TxToEnable: List of enabled TX channels
        - RxToEnable: List of enabled RX channels
        - FrameType: 0=single, 1=advanced, 2=continuous
        - DevConfig: Per-device configuration
    """
    mmwave_json = load_mmwave_json(json_file_path)
    mmwave_devices_config = mmwave_json['mmWaveDevices']
    
    params = {
        'NumDevices': len(mmwave_devices_config),
        'TxToEnable': [],
        'RxToEnable': [],
        'DevConfig': {}
    }
    
    dev_id_map = []
    
    for count, device_config in enumerate(mmwave_devices_config):
        dev_id = device_config['mmWaveDeviceId'] + 1  # 1-indexed
        dev_id_map.append(dev_id)
        
        # Frame type
        waveform_type = device_config['rfConfig']['waveformType']
        if waveform_type == 'singleFrameChirp':
            params['FrameType'] = 0
        elif waveform_type == 'advancedFrameChirp':
            params['FrameType'] = 1
        elif waveform_type == 'continuousWave':
            params['FrameType'] = 2
        
        # TX channel enable
        tx_channel_en_str = device_config['rfConfig']['rlChanCfg_t']['txChannelEn']
        tx_channel_en = int(tx_channel_en_str, 16) if tx_channel_en_str.startswith('0x') else int(tx_channel_en_str)
        for tx_channel in range(3):
            if tx_channel_en & (1 << tx_channel):
                params['TxToEnable'].append(3 * (dev_id - 1) + tx_channel + 1)
        
        # RX channel enable
        rx_channel_en_str = device_config['rfConfig']['rlChanCfg_t']['rxChannelEn']
        rx_channel_en = int(rx_channel_en_str, 16) if rx_channel_en_str.startswith('0x') else int(rx_channel_en_str)
        for rx_channel in range(4):
            if rx_channel_en & (1 << rx_channel):
                params['RxToEnable'].append(4 * (dev_id - 1) + rx_channel + 1)
        
        # Initialize device config
        if dev_id not in params['DevConfig']:
            params['DevConfig'][dev_id] = {
                'Profile': {},
                'Chirp': {},
                'FrameConfig': {},
                'NumProfiles': 0,
                'NumChirps': 0
            }
        
        # Profile configuration
        profiles = device_config['rfConfig']['rlProfiles']
        if isinstance(profiles, list):
            params['DevConfig'][dev_id]['NumProfiles'] = len(profiles)
            for profile_idx, profile in enumerate(profiles):
                profile_cfg = profile['rlProfileCfg_t']
                params['DevConfig'][dev_id]['Profile'][profile_idx + 1] = _parse_profile(profile_cfg)
        else:
            # Single profile
            params['DevConfig'][dev_id]['NumProfiles'] = 1
            params['DevConfig'][dev_id]['Profile'][1] = _parse_profile(profiles['rlProfileCfg_t'])
        
        # Chirp configuration
        chirps = device_config['rfConfig']['rlChirps']
        if isinstance(chirps, list):
            for chirp_block in chirps:
                _parse_chirp_block(chirp_block, params['DevConfig'][dev_id])
        else:
            _parse_chirp_block(chirps, params['DevConfig'][dev_id])
        
        # Frame configuration
        if params.get('FrameType', 0) == 0:
            frame_cfg = device_config['rfConfig']['rlFrameCfg_t']
            params['DevConfig'][dev_id]['FrameConfig'] = {
                'ChirpIdx': frame_cfg['chirpStartIdx'],
                'ChirpEndIdx': frame_cfg['chirpEndIdx'],
                'NumChirpLoops': frame_cfg['numLoops'],
                'NumFrames': frame_cfg['numFrames'],
                'Periodicity': frame_cfg.get('framePeriodicity_msec', 
                              frame_cfg.get('framePeriodicity', 0)),
            }
    
    return params


def _parse_profile(profile_cfg: Dict) -> Dict[str, Any]:
    """Parse a single profile configuration."""
    return {
        'ProfileId': profile_cfg.get('profileId', 0),
        'StartFreq': profile_cfg.get('startFreqConst_GHz', profile_cfg.get('startFreq', 77.0)),
        'FreqSlope': profile_cfg.get('freqSlopeConst_MHz_usec', profile_cfg.get('freqSlope', 70.0)),
        'IdleTime': profile_cfg.get('idleTimeConst_usec', profile_cfg.get('idleTime', 7.0)),
        'AdcStartTime': profile_cfg.get('adcStartTimeConst_usec', profile_cfg.get('adcStartTime', 6.0)),
        'RampEndTime': profile_cfg.get('rampEndTime_usec', profile_cfg.get('rampEndTime', 60.0)),
        'NumSamples': profile_cfg.get('numAdcSamples', 256),
        'SamplingRate': profile_cfg.get('digOutSampleRate', 10000),
        'TxStartTime': profile_cfg.get('txStartTime_usec', 0),
    }


def _parse_chirp_block(chirp_block: Dict, dev_config: Dict) -> None:
    """Parse a chirp configuration block and add to device config."""
    chirp_cfg = chirp_block['rlChirpCfg_t']
    start_idx = chirp_cfg['chirpStartIdx']
    end_idx = chirp_cfg['chirpEndIdx']
    dev_config['NumChirps'] += end_idx - start_idx + 1
    
    tx_enable_str = chirp_cfg['txEnable']
    tx_enable = int(tx_enable_str, 16) if isinstance(tx_enable_str, str) and tx_enable_str.startswith('0x') else int(tx_enable_str)
    
    for chirp_id in range(start_idx + 1, end_idx + 2):
        dev_config['Chirp'][chirp_id] = {
            'ChirpIdx': chirp_id,
            'ProfileId': chirp_cfg['profileId'],
            'Tx0Enable': (tx_enable >> 0) & 1,
            'Tx1Enable': (tx_enable >> 1) & 1,
            'Tx2Enable': (tx_enable >> 2) & 1,
            'StartFreqVar': chirp_cfg.get('startFreqVar_MHz', 0),
            'FreqSlopeVar': chirp_cfg.get('freqSlopeVar_KHz_usec', 0),
            'IdleTimeVar': chirp_cfg.get('idleTimeVar_usec', 0),
            'AdcStartTimeVar': chirp_cfg.get('adcStartTimeVar_usec', 0),
        }


def get_tx_enable_table(params: Dict[str, Any]) -> tuple:
    """
    Build TX enable table from parsed parameters.
    
    Args:
        params: Parsed mmWave parameters
        
    Returns:
        Tuple of (tx_enable_table, tx_channel_enabled)
    """
    import numpy as np
    
    num_devices = params['NumDevices']
    num_tx_per_dev = 3
    tot_tx = num_tx_per_dev * num_devices
    
    # Get number of chirp configs from first device
    num_chirp_config = params['DevConfig'][1]['NumChirps']
    
    tx_enable_table = np.zeros((num_chirp_config, tot_tx), dtype=int)
    
    for i_dev in range(1, num_devices + 1):
        for i_config in range(1, num_chirp_config + 1):
            chirp = params['DevConfig'][i_dev]['Chirp'].get(i_config, {})
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
    
    return tx_enable_table, tx_channel_enabled


def find_mmwave_json(data_folder: str) -> Optional[str]:
    """
    Find the .mmwave.json file in a data folder.
    
    Args:
        data_folder: Path to search
        
    Returns:
        Full path to JSON file, or None if not found
    """
    import glob
    json_files = glob.glob(os.path.join(data_folder, "*.mmwave.json"))
    if len(json_files) == 1:
        return json_files[0]
    elif len(json_files) > 1:
        raise ValueError(f"Multiple .mmwave.json files found in {data_folder}")
    return None
