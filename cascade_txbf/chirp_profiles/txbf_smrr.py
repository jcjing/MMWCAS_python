"""
txbf_smrr.py

Short-Medium Range Radar chirp profile for TX beamforming.

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/

Converted from MATLAB: chirpProfile_TxBF_SMRR.m
"""

from typing import Dict, Any

# Short-Medium Range Radar profile configuration
CHIRP_PROFILE_SMRR = {
    'name': 'Short-Medium Range Radar (SMRR)',
    'description': 'Balanced profile for medium-range applications',
    
    # RF parameters
    'startFreqGHz': 77.0,
    'freqSlopeMHzPerUs': 100.0,  # 100 MHz/us slope
    
    # Timing parameters
    'idleTimeUs': 5.0,
    'adcStartTimeUs': 4.0,
    'rampEndTimeUs': 40.0,
    
    # ADC parameters
    'numAdcSamples': 256,
    'sampleRateKsps': 12500,  # 12.5 Msps
    
    # Frame parameters
    'numLoops': 128,
    'framePeriodMs': 40.0,
    
    # TX configuration
    'txEnable': 0x7,
}


def get_smrr_config(
    num_loops: int = 128,
    frame_period_ms: float = 40.0,
    num_frames: int = 0,
) -> Dict[str, Any]:
    """
    Get Short-Medium Range Radar configuration.
    
    Args:
        num_loops: Number of chirp loops per frame
        frame_period_ms: Frame periodicity
        num_frames: Number of frames (0 = infinite)
        
    Returns:
        Configuration dictionary
    """
    config = CHIRP_PROFILE_SMRR.copy()
    config['numLoops'] = num_loops
    config['framePeriodMs'] = frame_period_ms
    config['numFrames'] = num_frames
    
    # Calculate performance
    c = 3e8
    chirp_time_us = config['numAdcSamples'] / (config['sampleRateKsps'] / 1000)
    bandwidth_mhz = config['freqSlopeMHzPerUs'] * chirp_time_us
    
    config['chirpTimeUs'] = chirp_time_us
    config['bandwidthMHz'] = bandwidth_mhz
    config['rangeResolutionM'] = c / (2 * bandwidth_mhz * 1e6)
    
    wavelength = c / (config['startFreqGHz'] * 1e9)
    chirp_interval_us = config['idleTimeUs'] + config['rampEndTimeUs']
    
    config['wavelengthM'] = wavelength
    config['chirpIntervalUs'] = chirp_interval_us
    config['maxVelocityMps'] = wavelength / (4 * chirp_interval_us * 1e-6)
    config['velocityResolutionMps'] = wavelength / (2 * num_loops * chirp_interval_us * 1e-6)
    
    return config


if __name__ == '__main__':
    config = get_smrr_config()
    print("Short-Medium Range Radar Configuration:")
    print(f"  Range resolution: {config['rangeResolutionM']:.4f} m")
    print(f"  Bandwidth: {config['bandwidthMHz']:.1f} MHz")
    print(f"  Max velocity: {config['maxVelocityMps']:.2f} m/s")
