"""
txbf_calc_phase_settings.py

Calculate TX beamforming phase shifter settings.

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/

Converted from MATLAB: TXBF_Calc_Phase_Settings.m
"""

import numpy as np
from typing import List, Tuple, Dict, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common.constants import (
    PHASE_SHIFTER_RESOLUTION, SPEED_OF_LIGHT,
    TI_CASCADE_TX_POSITION_AZI, TI_CASCADE_TX_POSITION_ELE
)


def calculate_phase_settings(
    steer_angle_deg: float,
    center_freq_ghz: float = 77.0,
    num_tx: int = 12,
    tx_positions: np.ndarray = None,
    ps_resolution_deg: float = PHASE_SHIFTER_RESOLUTION,
) -> Dict[str, Any]:
    """
    Calculate phase shifter settings for TX beamforming to a given angle.
    
    Args:
        steer_angle_deg: Desired steering angle in degrees (positive = right)
        center_freq_ghz: Center frequency in GHz
        num_tx: Number of TX antennas
        tx_positions: TX antenna positions (if None, uses TI cascade defaults)
        ps_resolution_deg: Phase shifter resolution in degrees
        
    Returns:
        Dictionary containing:
        - phase_settings_deg: Continuous phase values (degrees)
        - phase_settings_quantized: Quantized phase values (degrees)
        - phase_shifter_codes: Integer codes for phase shifters
        - quantization_error_deg: Quantization error per TX
    """
    if tx_positions is None:
        tx_positions = TI_CASCADE_TX_POSITION_AZI[:num_tx].astype(float)
    
    # Wavelength
    wavelength = SPEED_OF_LIGHT / (center_freq_ghz * 1e9)
    
    # Antenna spacing (normalized by wavelength/2)
    d = wavelength / 2
    
    # Calculate required phase shift for each TX
    # Phase = 2*pi * d * sin(theta) / wavelength
    steer_angle_rad = np.radians(steer_angle_deg)
    
    # Relative positions from first TX
    relative_positions = tx_positions - tx_positions[0]
    
    # Phase shift in radians
    phase_shift_rad = 2 * np.pi * relative_positions * (d / wavelength) * np.sin(steer_angle_rad)
    
    # Convert to degrees
    phase_shift_deg = np.degrees(phase_shift_rad)
    
    # Wrap to [0, 360)
    phase_shift_deg = phase_shift_deg % 360
    
    # Quantize to phase shifter resolution
    phase_shifter_codes = np.round(phase_shift_deg / ps_resolution_deg).astype(int)
    phase_shifter_codes = phase_shifter_codes % int(360 / ps_resolution_deg)
    
    # Quantized phase values
    phase_quantized_deg = phase_shifter_codes * ps_resolution_deg
    
    # Quantization error
    quant_error_deg = phase_shift_deg - phase_quantized_deg
    
    return {
        'phase_settings_deg': phase_shift_deg,
        'phase_settings_quantized': phase_quantized_deg,
        'phase_shifter_codes': phase_shifter_codes,
        'quantization_error_deg': quant_error_deg,
        'steer_angle_deg': steer_angle_deg,
        'center_freq_ghz': center_freq_ghz,
    }


def calculate_phase_settings_2d(
    azimuth_deg: float,
    elevation_deg: float = 0.0,
    center_freq_ghz: float = 77.0,
    num_tx: int = 12,
    tx_positions_azi: np.ndarray = None,
    tx_positions_ele: np.ndarray = None,
    ps_resolution_deg: float = PHASE_SHIFTER_RESOLUTION,
) -> Dict[str, Any]:
    """
    Calculate phase shifter settings for 2D TX beamforming.
    
    Args:
        azimuth_deg: Azimuth steering angle in degrees
        elevation_deg: Elevation steering angle in degrees
        center_freq_ghz: Center frequency in GHz
        num_tx: Number of TX antennas
        tx_positions_azi: TX azimuth positions (normalized)
        tx_positions_ele: TX elevation positions (normalized)
        ps_resolution_deg: Phase shifter resolution
        
    Returns:
        Dictionary with phase settings
    """
    if tx_positions_azi is None:
        tx_positions_azi = TI_CASCADE_TX_POSITION_AZI[:num_tx].astype(float)
    if tx_positions_ele is None:
        tx_positions_ele = TI_CASCADE_TX_POSITION_ELE[:num_tx].astype(float)
    
    wavelength = SPEED_OF_LIGHT / (center_freq_ghz * 1e9)
    d = wavelength / 2
    
    az_rad = np.radians(azimuth_deg)
    el_rad = np.radians(elevation_deg)
    
    # Direction cosines
    u = np.sin(az_rad) * np.cos(el_rad)
    v = np.sin(el_rad)
    
    # Relative positions
    rel_azi = tx_positions_azi - tx_positions_azi[0]
    rel_ele = tx_positions_ele - tx_positions_ele[0]
    
    # Combined phase shift
    phase_rad = 2 * np.pi * (d / wavelength) * (rel_azi * u + rel_ele * v)
    phase_deg = np.degrees(phase_rad) % 360
    
    # Quantize
    ps_codes = np.round(phase_deg / ps_resolution_deg).astype(int)
    ps_codes = ps_codes % int(360 / ps_resolution_deg)
    phase_quantized = ps_codes * ps_resolution_deg
    
    return {
        'phase_settings_deg': phase_deg,
        'phase_settings_quantized': phase_quantized,
        'phase_shifter_codes': ps_codes,
        'azimuth_deg': azimuth_deg,
        'elevation_deg': elevation_deg,
        'center_freq_ghz': center_freq_ghz,
    }


def wrap_to_360(angle_deg: float) -> float:
    """Wrap angle to [0, 360) range."""
    return angle_deg % 360


def wrap_to_180(angle_deg: float) -> float:
    """Wrap angle to [-180, 180) range."""
    angle = angle_deg % 360
    if angle >= 180:
        angle -= 360
    return angle


if __name__ == '__main__':
    # Example usage
    angles = [-30, -15, 0, 15, 30]
    
    print("TX Beamforming Phase Settings Calculator")
    print("=" * 50)
    
    for angle in angles:
        result = calculate_phase_settings(angle)
        print(f"\nSteering angle: {angle}°")
        print(f"  Phase shifter codes: {result['phase_shifter_codes']}")
        print(f"  Quantized phases: {result['phase_settings_quantized']}")
