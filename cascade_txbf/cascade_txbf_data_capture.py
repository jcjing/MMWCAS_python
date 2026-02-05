"""
cascade_txbf_data_capture.py

TX beamforming data capture configuration.

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/

Converted from MATLAB: cascade_TxBF_dataCapture.m
"""

import json
import os
from typing import Dict, Any, Optional, List
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cascade_mimo.txbf.txbf_calc_phase_settings import calculate_phase_settings
from common.constants import NUM_DEVICES_CASCADE, NUM_TX_PER_DEVICE


def configure_txbf_capture(
    base_config_file: str,
    steer_angles: List[float],
    center_freq_ghz: float = 77.0,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Configure TX beamforming data capture.
    
    Modifies a base mmWave JSON configuration to set up TX beamforming
    with configurable steering angles.
    
    Args:
        base_config_file: Path to base JSON configuration
        steer_angles: List of steering angles to configure
        center_freq_ghz: Center frequency in GHz
        output_file: Output file path for modified configuration
        
    Returns:
        Modified configuration dictionary
    """
    # Load base configuration
    with open(base_config_file, 'r') as f:
        config = json.load(f)
    
    # Calculate phase settings for each angle
    angle_configs = []
    for angle in steer_angles:
        ps_result = calculate_phase_settings(angle, center_freq_ghz)
        angle_configs.append({
            'angle': angle,
            'phase_codes': ps_result['phase_shifter_codes'].tolist(),
        })
    
    # Modify configuration for TX beamforming
    for device in config.get('mmWaveDevices', []):
        rf_config = device.get('rfConfig', {})
        
        # Enable all TX simultaneously
        rf_config['rlChanCfg_t']['txChannelEn'] = '0x7'  # All 3 TX enabled
        
        # Configure chirps with phase settings
        # For beam steering, we modify the phase shifter settings
        device_id = device.get('mmWaveDeviceId', 0)
        
        if len(angle_configs) > 0:
            # Use first angle configuration (for single-angle capture)
            first_config = angle_configs[0]
            phase_codes = first_config['phase_codes']
            
            # Get codes for this device
            start_idx = device_id * NUM_TX_PER_DEVICE
            end_idx = start_idx + NUM_TX_PER_DEVICE
            device_codes = phase_codes[start_idx:end_idx] if end_idx <= len(phase_codes) else [0, 0, 0]
            
            # Add phase shifter configuration
            if 'rlRfPhaseShiftCfg_t' not in device:
                device['rlRfPhaseShiftCfg_t'] = {}
            
            ps_config = device['rlRfPhaseShiftCfg_t']
            ps_config['tx0PhShiftDeg'] = device_codes[0] * 5.625 if len(device_codes) > 0 else 0
            ps_config['tx1PhShiftDeg'] = device_codes[1] * 5.625 if len(device_codes) > 1 else 0
            ps_config['tx2PhShiftDeg'] = device_codes[2] * 5.625 if len(device_codes) > 2 else 0
    
    # Add beamforming metadata
    config['txBeamformingConfig'] = {
        'enabled': True,
        'steerAngles': steer_angles,
        'angleConfigs': angle_configs,
        'centerFreqGHz': center_freq_ghz,
    }
    
    # Save if output path provided
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to: {output_file}")
    
    return config


def create_angle_sweep_config(
    base_config_file: str,
    angle_range: tuple = (-45, 45),
    angle_step: float = 5.0,
    center_freq_ghz: float = 77.0,
    output_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Create multiple configurations for angle sweep.
    
    Args:
        base_config_file: Path to base configuration
        angle_range: (min, max) angle range in degrees
        angle_step: Step between angles
        center_freq_ghz: Center frequency
        output_dir: Directory to save configurations
        
    Returns:
        List of configuration dictionaries
    """
    min_angle, max_angle = angle_range
    angles = np.arange(min_angle, max_angle + angle_step, angle_step)
    
    configs = []
    
    for angle in angles:
        output_file = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f'txbf_angle_{angle:+06.1f}.json')
        
        config = configure_txbf_capture(
            base_config_file,
            steer_angles=[angle],
            center_freq_ghz=center_freq_ghz,
            output_file=output_file,
        )
        configs.append(config)
    
    return configs


# Chirp profile configurations from MATLAB examples
CHIRP_PROFILES = {
    'LRR': {  # Long Range Radar
        'name': 'Long Range Radar',
        'startFreqGHz': 77.0,
        'freqSlopeMHzPerUs': 70.0,
        'idleTimeUs': 7.0,
        'adcStartTimeUs': 6.0,
        'rampEndTimeUs': 60.0,
        'numAdcSamples': 256,
        'sampleRateKsps': 10000,
    },
    'SMRR': {  # Short-Medium Range Radar
        'name': 'Short-Medium Range Radar',
        'startFreqGHz': 77.0,
        'freqSlopeMHzPerUs': 100.0,
        'idleTimeUs': 5.0,
        'adcStartTimeUs': 4.0,
        'rampEndTimeUs': 40.0,
        'numAdcSamples': 256,
        'sampleRateKsps': 12500,
    },
    'USRR': {  # Ultra-Short Range Radar
        'name': 'Ultra-Short Range Radar',
        'startFreqGHz': 77.0,
        'freqSlopeMHzPerUs': 200.0,
        'idleTimeUs': 3.0,
        'adcStartTimeUs': 2.0,
        'rampEndTimeUs': 20.0,
        'numAdcSamples': 128,
        'sampleRateKsps': 15000,
    },
}


def get_chirp_profile(profile_name: str) -> Dict[str, Any]:
    """
    Get predefined chirp profile configuration.
    
    Args:
        profile_name: 'LRR', 'SMRR', or 'USRR'
        
    Returns:
        Chirp profile dictionary
    """
    if profile_name not in CHIRP_PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}. Options: {list(CHIRP_PROFILES.keys())}")
    
    profile = CHIRP_PROFILES[profile_name].copy()
    
    # Calculate derived parameters
    c = 3e8
    chirp_time_us = profile['numAdcSamples'] / (profile['sampleRateKsps'] / 1000)
    bandwidth_mhz = profile['freqSlopeMHzPerUs'] * chirp_time_us
    
    profile['chirpTimeUs'] = chirp_time_us
    profile['bandwidthMHz'] = bandwidth_mhz
    profile['rangeResolutionM'] = c / (2 * bandwidth_mhz * 1e6)
    profile['maxRangeM'] = (c * profile['sampleRateKsps'] * 1e3 * chirp_time_us * 1e-6) / (2 * bandwidth_mhz * 1e6)
    
    return profile


if __name__ == '__main__':
    # Example: Print chirp profiles
    print("TX Beamforming Chirp Profiles:")
    print("=" * 50)
    
    for name in ['LRR', 'SMRR', 'USRR']:
        profile = get_chirp_profile(name)
        print(f"\n{profile['name']}:")
        print(f"  Range resolution: {profile['rangeResolutionM']:.3f} m")
        print(f"  Max range: {profile['maxRangeM']:.1f} m")
        print(f"  Bandwidth: {profile['bandwidthMHz']:.1f} MHz")
