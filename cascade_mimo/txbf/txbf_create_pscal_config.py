"""
txbf_create_pscal_config.py

Create phase shifter calibration advanced frame configuration.

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/

Converted from MATLAB: TXBF_Create_PSCal_Advanced_Frame_Config.m
"""

import json
import os
from typing import Dict, Any, List, Optional
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common.constants import PHASE_SHIFTER_RESOLUTION, NUM_DEVICES_CASCADE, NUM_TX_PER_DEVICE


def create_pscal_advanced_frame_config(
    base_config_file: str,
    phase_settings: List[Dict[str, Any]],
    output_file: Optional[str] = None,
    num_frames: int = 1,
    frame_periodicity_ms: float = 100.0,
) -> Dict[str, Any]:
    """
    Create advanced frame configuration for phase shifter calibration.
    
    This generates a JSON configuration that sweeps through different
    phase shifter settings for calibration purposes.
    
    Args:
        base_config_file: Path to base mmWave JSON configuration
        phase_settings: List of phase shifter settings per frame
        output_file: Output JSON file path
        num_frames: Number of frames to configure
        frame_periodicity_ms: Frame periodicity in milliseconds
        
    Returns:
        Modified configuration dictionary
    """
    # Load base configuration
    with open(base_config_file, 'r') as f:
        config = json.load(f)
    
    # Modify for advanced frame mode
    for device in config.get('mmWaveDevices', []):
        rf_config = device.get('rfConfig', {})
        
        # Set to advanced frame mode
        rf_config['waveformType'] = 'advancedFrameChirp'
        
        # Configure advanced frame
        if 'rlAdvFrameCfg_t' not in rf_config:
            rf_config['rlAdvFrameCfg_t'] = {}
        
        adv_frame = rf_config['rlAdvFrameCfg_t']
        adv_frame['numOfSubFrames'] = len(phase_settings)
        adv_frame['forceProfile'] = 0
        adv_frame['numFrames'] = num_frames
        adv_frame['triggerSelect'] = 1  # Software trigger
        adv_frame['frameTrigDelay'] = 0
        
        # Configure sub-frames with different phase settings
        sub_frames = []
        for i, ps_setting in enumerate(phase_settings):
            sub_frame = {
                'subFrameIdx': i,
                'numLoops': 64,
                'numOfBurst': 1,
                'numOfBurstLoops': 1,
                'chirpStartIdx': 0,
                'chirpEndIdx': 11,  # Assuming 12 chirps per sub-frame
                'burstPeriodicity': frame_periodicity_ms * 1000,  # microseconds
                'chirpStartIdxOffset': 0,
                'numOfBursts': 1,
            }
            
            # Add phase shifter settings
            sub_frame['phaseShifterConfig'] = ps_setting
            sub_frames.append(sub_frame)
        
        rf_config['rlAdvFrameSubCfg_t'] = sub_frames
    
    # Save if output path provided
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to: {output_file}")
    
    return config


def generate_phase_sweep_settings(
    num_settings: int = 64,
    ps_resolution_deg: float = PHASE_SHIFTER_RESOLUTION,
    tx_indices: List[int] = None,
) -> List[Dict[str, Any]]:
    """
    Generate phase shifter settings for a calibration sweep.
    
    Sweeps one TX through all phase settings while keeping others at 0.
    
    Args:
        num_settings: Number of phase settings (typically 64 for 5.625° resolution)
        ps_resolution_deg: Phase shifter resolution
        tx_indices: TX indices to calibrate (default: all 12)
        
    Returns:
        List of phase shifter configurations
    """
    if tx_indices is None:
        tx_indices = list(range(12))
    
    settings = []
    
    for tx_idx in tx_indices:
        for phase_code in range(num_settings):
            setting = {
                'txIndex': tx_idx,
                'phaseCode': phase_code,
                'phaseDeg': phase_code * ps_resolution_deg,
                'deviceSettings': []
            }
            
            # Generate per-device settings
            for dev_id in range(NUM_DEVICES_CASCADE):
                dev_setting = {
                    'deviceId': dev_id,
                    'tx0PhaseShifter': 0,
                    'tx1PhaseShifter': 0,
                    'tx2PhaseShifter': 0,
                }
                
                # Set the phase for the target TX
                dev_tx_start = dev_id * NUM_TX_PER_DEVICE
                dev_tx_end = dev_tx_start + NUM_TX_PER_DEVICE
                
                if dev_tx_start <= tx_idx < dev_tx_end:
                    local_tx = tx_idx - dev_tx_start
                    if local_tx == 0:
                        dev_setting['tx0PhaseShifter'] = phase_code
                    elif local_tx == 1:
                        dev_setting['tx1PhaseShifter'] = phase_code
                    elif local_tx == 2:
                        dev_setting['tx2PhaseShifter'] = phase_code
                
                setting['deviceSettings'].append(dev_setting)
            
            settings.append(setting)
    
    return settings


def generate_beamsteering_settings(
    angles_deg: List[float],
    center_freq_ghz: float = 77.0,
) -> List[Dict[str, Any]]:
    """
    Generate phase shifter settings for beam steering calibration.
    
    Args:
        angles_deg: List of steering angles
        center_freq_ghz: Center frequency in GHz
        
    Returns:
        List of phase shifter configurations
    """
    from .txbf_calc_phase_settings import calculate_phase_settings
    
    settings = []
    
    for angle in angles_deg:
        result = calculate_phase_settings(angle, center_freq_ghz)
        
        setting = {
            'steerAngleDeg': angle,
            'phaseCodes': result['phase_shifter_codes'].tolist(),
            'deviceSettings': []
        }
        
        # Generate per-device settings
        for dev_id in range(NUM_DEVICES_CASCADE):
            dev_tx_start = dev_id * NUM_TX_PER_DEVICE
            
            dev_setting = {
                'deviceId': dev_id,
                'tx0PhaseShifter': int(result['phase_shifter_codes'][dev_tx_start]) if dev_tx_start < len(result['phase_shifter_codes']) else 0,
                'tx1PhaseShifter': int(result['phase_shifter_codes'][dev_tx_start + 1]) if dev_tx_start + 1 < len(result['phase_shifter_codes']) else 0,
                'tx2PhaseShifter': int(result['phase_shifter_codes'][dev_tx_start + 2]) if dev_tx_start + 2 < len(result['phase_shifter_codes']) else 0,
            }
            setting['deviceSettings'].append(dev_setting)
        
        settings.append(setting)
    
    return settings


if __name__ == '__main__':
    # Example: Generate phase sweep settings
    print("Generating phase sweep settings for TX0...")
    sweep_settings = generate_phase_sweep_settings(
        num_settings=8,  # Simplified for example
        tx_indices=[0],
    )
    
    for s in sweep_settings[:3]:
        print(f"  TX{s['txIndex']}, Phase code {s['phaseCode']}: {s['phaseDeg']:.1f}°")
    print(f"  ... ({len(sweep_settings)} total)")
    
    # Example: Generate beam steering settings
    print("\nGenerating beam steering settings...")
    angles = [-30, -15, 0, 15, 30]
    steer_settings = generate_beamsteering_settings(angles)
    
    for s in steer_settings:
        print(f"  Angle {s['steerAngleDeg']:+6.1f}°: codes = {s['phaseCodes'][:4]}...")
