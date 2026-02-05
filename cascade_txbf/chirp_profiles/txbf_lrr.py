"""
txbf_lrr.py

Long Range Radar chirp profile for TX beamforming.

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/

Converted from MATLAB: chirpProfile_TxBF_LRR.m
"""

from typing import Dict, Any

# Long Range Radar profile configuration
CHIRP_PROFILE_LRR = {
    'name': 'Long Range Radar (LRR)',
    'description': 'Optimized for detection at longer ranges with good range resolution',
    
    # RF parameters
    'startFreqGHz': 77.0,
    'freqSlopeMHzPerUs': 70.0,  # 70 MHz/us slope
    
    # Timing parameters
    'idleTimeUs': 7.0,
    'adcStartTimeUs': 6.0,
    'rampEndTimeUs': 60.0,
    
    # ADC parameters
    'numAdcSamples': 256,
    'sampleRateKsps': 10000,  # 10 Msps
    
    # Frame parameters
    'numLoops': 128,
    'framePeriodMs': 50.0,
    
    # TX configuration for beamforming
    'txEnable': 0x7,  # All 3 TX enabled
    
    # Expected performance (calculated)
    'rangeResolutionM': 0.0375,  # Approximate
    'maxUnambiguousRangeM': 80.0,  # Approximate
    'velocityResolutionMps': 0.07,  # Approximate
    'maxVelocityMps': 4.4,  # Approximate
}


def get_lrr_config(
    num_loops: int = 128,
    frame_period_ms: float = 50.0,
    num_frames: int = 0,  # 0 = infinite
) -> Dict[str, Any]:
    """
    Get Long Range Radar configuration with customizations.
    
    Args:
        num_loops: Number of chirp loops per frame
        frame_period_ms: Frame periodicity in milliseconds
        num_frames: Number of frames (0 = infinite)
        
    Returns:
        Complete configuration dictionary
    """
    config = CHIRP_PROFILE_LRR.copy()
    
    # Apply customizations
    config['numLoops'] = num_loops
    config['framePeriodMs'] = frame_period_ms
    config['numFrames'] = num_frames
    
    # Recalculate performance metrics
    c = 3e8
    chirp_time_us = config['numAdcSamples'] / (config['sampleRateKsps'] / 1000)
    bandwidth_mhz = config['freqSlopeMHzPerUs'] * chirp_time_us
    
    config['chirpTimeUs'] = chirp_time_us
    config['bandwidthMHz'] = bandwidth_mhz
    config['rangeResolutionM'] = c / (2 * bandwidth_mhz * 1e6)
    
    # Velocity parameters
    wavelength = c / (config['startFreqGHz'] * 1e9)
    chirp_interval_us = config['idleTimeUs'] + config['rampEndTimeUs']
    
    config['wavelengthM'] = wavelength
    config['chirpIntervalUs'] = chirp_interval_us
    config['maxVelocityMps'] = wavelength / (4 * chirp_interval_us * 1e-6)
    config['velocityResolutionMps'] = wavelength / (2 * num_loops * chirp_interval_us * 1e-6)
    
    return config


def generate_lrr_json_params() -> Dict[str, Any]:
    """
    Generate JSON parameters for mmWave Studio configuration.
    
    Returns:
        Dictionary formatted for mmWave Studio JSON
    """
    config = get_lrr_config()
    
    return {
        'rlProfileCfg_t': {
            'profileId': 0,
            'startFreqConst_GHz': config['startFreqGHz'],
            'freqSlopeConst_MHz_usec': config['freqSlopeMHzPerUs'],
            'idleTimeConst_usec': config['idleTimeUs'],
            'adcStartTimeConst_usec': config['adcStartTimeUs'],
            'rampEndTime_usec': config['rampEndTimeUs'],
            'numAdcSamples': config['numAdcSamples'],
            'digOutSampleRate': config['sampleRateKsps'] * 1000,
        },
        'rlFrameCfg_t': {
            'chirpStartIdx': 0,
            'chirpEndIdx': 11,  # 12 chirps for MIMO
            'numLoops': config['numLoops'],
            'numFrames': config['numFrames'],
            'framePeriodicity_msec': config['framePeriodMs'],
        },
    }


if __name__ == '__main__':
    config = get_lrr_config()
    print("Long Range Radar Configuration:")
    print(f"  Range resolution: {config['rangeResolutionM']:.4f} m")
    print(f"  Bandwidth: {config['bandwidthMHz']:.1f} MHz")
    print(f"  Max velocity: {config['maxVelocityMps']:.2f} m/s")
    print(f"  Velocity resolution: {config['velocityResolutionMps']:.4f} m/s")
