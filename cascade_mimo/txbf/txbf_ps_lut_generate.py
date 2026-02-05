"""
txbf_ps_lut_generate.py

Generate TX beamforming phase shifter lookup table.

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/

Converted from MATLAB: TXBF_PS_LUT_Generate.m
"""

import numpy as np
from typing import List, Dict, Any, Optional
import json
import os

from .txbf_calc_phase_settings import calculate_phase_settings

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common.constants import PHASE_SHIFTER_RESOLUTION, NUM_TX_PER_DEVICE, NUM_DEVICES_CASCADE


def generate_phase_shifter_lut(
    angle_range: tuple = (-60, 60),
    angle_step: float = 1.0,
    center_freq_ghz: float = 77.0,
    num_tx: int = 12,
    ps_resolution_deg: float = PHASE_SHIFTER_RESOLUTION,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a lookup table for TX beamforming phase shifter settings.
    
    Args:
        angle_range: (min_angle, max_angle) steering range in degrees
        angle_step: Step size between angles
        center_freq_ghz: Center frequency in GHz
        num_tx: Number of TX antennas
        ps_resolution_deg: Phase shifter resolution in degrees
        output_file: Optional path to save LUT as JSON
        
    Returns:
        Dictionary containing:
        - angles: Array of steering angles
        - phase_codes: 2D array of phase shifter codes (angles x TX)
        - quantization_errors: 2D array of errors
    """
    min_angle, max_angle = angle_range
    angles = np.arange(min_angle, max_angle + angle_step, angle_step)
    
    num_angles = len(angles)
    phase_codes = np.zeros((num_angles, num_tx), dtype=int)
    quantization_errors = np.zeros((num_angles, num_tx))
    phase_values = np.zeros((num_angles, num_tx))
    
    for i, angle in enumerate(angles):
        result = calculate_phase_settings(
            steer_angle_deg=angle,
            center_freq_ghz=center_freq_ghz,
            num_tx=num_tx,
            ps_resolution_deg=ps_resolution_deg,
        )
        phase_codes[i] = result['phase_shifter_codes']
        quantization_errors[i] = result['quantization_error_deg']
        phase_values[i] = result['phase_settings_deg']
    
    lut = {
        'angles': angles.tolist(),
        'phase_codes': phase_codes.tolist(),
        'phase_values': phase_values.tolist(),
        'quantization_errors': quantization_errors.tolist(),
        'config': {
            'angle_range': angle_range,
            'angle_step': angle_step,
            'center_freq_ghz': center_freq_ghz,
            'num_tx': num_tx,
            'ps_resolution_deg': ps_resolution_deg,
        }
    }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(lut, f, indent=2)
        print(f"LUT saved to: {output_file}")
    
    return lut


def load_phase_shifter_lut(file_path: str) -> Dict[str, Any]:
    """Load a phase shifter LUT from JSON file."""
    with open(file_path, 'r') as f:
        lut = json.load(f)
    
    # Convert lists back to numpy arrays
    lut['angles'] = np.array(lut['angles'])
    lut['phase_codes'] = np.array(lut['phase_codes'])
    lut['phase_values'] = np.array(lut['phase_values'])
    lut['quantization_errors'] = np.array(lut['quantization_errors'])
    
    return lut


def lookup_phase_codes(lut: Dict[str, Any], steer_angle_deg: float) -> np.ndarray:
    """
    Look up phase shifter codes for a given steering angle.
    
    Args:
        lut: Phase shifter LUT dictionary
        steer_angle_deg: Desired steering angle
        
    Returns:
        Array of phase shifter codes for each TX
    """
    angles = np.array(lut['angles'])
    phase_codes = np.array(lut['phase_codes'])
    
    # Find closest angle
    idx = np.argmin(np.abs(angles - steer_angle_deg))
    
    return phase_codes[idx]


def format_phase_codes_for_mmwave_studio(
    phase_codes: np.ndarray,
    num_devices: int = NUM_DEVICES_CASCADE,
    num_tx_per_device: int = NUM_TX_PER_DEVICE,
) -> List[Dict[str, Any]]:
    """
    Format phase codes for mmWave Studio JSON configuration.
    
    Args:
        phase_codes: Phase shifter codes for each TX
        num_devices: Number of cascade devices
        num_tx_per_device: Number of TX per device
        
    Returns:
        List of per-device configurations
    """
    device_configs = []
    
    for dev_id in range(num_devices):
        tx_start = dev_id * num_tx_per_device
        tx_end = tx_start + num_tx_per_device
        
        dev_codes = phase_codes[tx_start:tx_end] if tx_end <= len(phase_codes) else np.zeros(num_tx_per_device, dtype=int)
        
        device_configs.append({
            'deviceId': dev_id,
            'tx0PhaseShifter': int(dev_codes[0]) if len(dev_codes) > 0 else 0,
            'tx1PhaseShifter': int(dev_codes[1]) if len(dev_codes) > 1 else 0,
            'tx2PhaseShifter': int(dev_codes[2]) if len(dev_codes) > 2 else 0,
        })
    
    return device_configs


if __name__ == '__main__':
    # Generate example LUT
    print("Generating TX Beamforming Phase Shifter LUT...")
    lut = generate_phase_shifter_lut(
        angle_range=(-45, 45),
        angle_step=5.0,
        center_freq_ghz=77.0,
        num_tx=12,
    )
    
    print(f"\nGenerated LUT with {len(lut['angles'])} angles")
    print(f"Angle range: {lut['angles'][0]}° to {lut['angles'][-1]}°")
    
    # Example lookup
    test_angle = 15.0
    codes = lookup_phase_codes(lut, test_angle)
    print(f"\nPhase codes for {test_angle}°: {codes}")
