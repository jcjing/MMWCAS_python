"""
sim_top.py

Top-level simulation configuration for cascade MIMO radar.

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/

Converted from MATLAB: simTopCascade.m
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common.constants import SPEED_OF_LIGHT


@dataclass
class SimTopConfig:
    """Top-level simulation configuration."""
    platform: str = "TI_4Chip_CASCADE"
    dataPlatform: str = "TDA2"
    
    # Frame configuration
    totNumFrames: int = 100
    startFrame: int = 1
    
    # ADC configuration
    numSamplePerChirp: int = 256
    adcSampleRate: float = 10e6  # Hz
    
    # Chirp configuration
    startFreqConst: float = 77e9  # Hz
    chirpSlope: float = 70e12  # Hz/s
    chirpIdleTime: float = 7e-6  # s
    chirpRampEndTime: float = 60e-6  # s
    adcStartTimeConst: float = 6e-6  # s
    
    # Frame timing
    nchirp_loops: int = 64
    numTxAnt: int = 12
    framePeriodicity: float = 50e-3  # s
    
    # Processing flags
    plotOn: bool = True
    saveOutput: bool = False
    logScale: bool = True


class SimTopCascade:
    """
    Top-level simulation object for cascade MIMO radar.
    
    Aggregates parameters and provides derived values.
    """
    
    def __init__(self, config: Optional[SimTopConfig] = None,
                 pfile: Optional[str] = None, **kwargs):
        """
        Initialize simulation top object.
        
        Args:
            config: SimTopConfig object
            pfile: Path to parameter file (for MATLAB compatibility)
            **kwargs: Config parameters
        """
        if config is not None:
            self.config = config
        else:
            self.config = SimTopConfig(**kwargs)
        
        # Calculate derived parameters
        self._calculate_derived_params()
    
    def _calculate_derived_params(self) -> None:
        """Calculate derived RF parameters."""
        cfg = self.config
        
        # Chirp timing
        self.chirpRampTime = cfg.numSamplePerChirp / cfg.adcSampleRate
        self.chirpBandwidth = cfg.chirpSlope * self.chirpRampTime
        self.chirpInterval = cfg.chirpRampEndTime + cfg.chirpIdleTime
        
        # Center frequency
        self.carrierFrequency = (cfg.startFreqConst + 
                                 (cfg.adcStartTimeConst + self.chirpRampTime / 2) * cfg.chirpSlope)
        self.wavelength = SPEED_OF_LIGHT / self.carrierFrequency
        
        # Range parameters
        self.maxRange = (SPEED_OF_LIGHT * cfg.adcSampleRate * self.chirpRampTime / 
                        (2 * self.chirpBandwidth))
        self.rangeResolution = SPEED_OF_LIGHT / (2 * self.chirpBandwidth)
        
        # FFT sizes
        self.rangeFFTSize = 2 ** int(np.ceil(np.log2(cfg.numSamplePerChirp)))
        self.dopplerFFTSize = 2 ** int(np.ceil(np.log2(cfg.nchirp_loops)))
        
        # Velocity parameters  
        self.maxVelocity = self.wavelength / (self.chirpInterval * 4)
        self.velocityResolution = (self.wavelength / 
                                   (2 * cfg.nchirp_loops * self.chirpInterval * cfg.numTxAnt))
        
        # Frame parameters
        self.numChirpsPerFrame = cfg.nchirp_loops * cfg.numTxAnt
        self.numVirtualAntennas = cfg.numTxAnt * 16  # Assuming 16 RX
    
    @property
    def platform(self) -> str:
        return self.config.platform
    
    @property
    def totNumFrames(self) -> int:
        return self.config.totNumFrames
    
    @property
    def rangeBinSize(self) -> float:
        """Range bin size in meters, accounting for FFT size."""
        return self.rangeResolution * self.config.numSamplePerChirp / self.rangeFFTSize
    
    @property
    def dopplerBinSize(self) -> float:
        """Doppler bin size in m/s, accounting for FFT size."""
        return self.velocityResolution * self.config.nchirp_loops / self.dopplerFFTSize
    
    def get_rf_params(self) -> Dict[str, Any]:
        """Get RF parameters as dictionary."""
        return {
            'startFreq': self.config.startFreqConst,
            'chirpSlope': self.config.chirpSlope,
            'adcSampleRate': self.config.adcSampleRate,
            'numSamples': self.config.numSamplePerChirp,
            'chirpRampTime': self.chirpRampTime,
            'chirpBandwidth': self.chirpBandwidth,
            'carrierFrequency': self.carrierFrequency,
            'wavelength': self.wavelength,
            'rangeResolution': self.rangeResolution,
            'maxRange': self.maxRange,
            'velocityResolution': self.velocityResolution,
            'maxVelocity': self.maxVelocity,
            'rangeFFTSize': self.rangeFFTSize,
            'dopplerFFTSize': self.dopplerFFTSize,
        }


# Need numpy for calculations
import numpy as np
