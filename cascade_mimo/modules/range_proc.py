"""
range_proc.py

Range FFT processing module.

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/

Converted from MATLAB: rangeProcCascade.m
"""

import numpy as np
from typing import Optional, Literal
from dataclasses import dataclass


@dataclass
class RangeProcConfig:
    """Range processing configuration."""
    rangeFFTSize: int = 256
    numSamplePerChirp: int = 256
    windowType: Literal['hann', 'blackman', 'hamming', 'rect'] = 'hann'
    scaleFFT: bool = True
    rangeResolution: float = 0.0  # meters, calculated from RF params


class RangeProcessor:
    """
    Range FFT processing module.
    
    Performs windowed FFT on ADC samples to generate range profiles.
    """
    
    def __init__(self, config: Optional[RangeProcConfig] = None, **kwargs):
        """
        Initialize range processor.
        
        Args:
            config: RangeProcConfig object
            **kwargs: Alternative way to pass config parameters
        """
        if config is not None:
            self.config = config
        else:
            self.config = RangeProcConfig(**kwargs)
        
        # Pre-compute window
        self._window = self._generate_window()
    
    def _generate_window(self) -> np.ndarray:
        """Generate the windowing function."""
        n = self.config.numSamplePerChirp
        
        if self.config.windowType == 'hann':
            return np.hanning(n)
        elif self.config.windowType == 'blackman':
            return np.blackman(n)
        elif self.config.windowType == 'hamming':
            return np.hamming(n)
        else:  # rect
            return np.ones(n)
    
    def process(self, adc_data: np.ndarray) -> np.ndarray:
        """
        Perform range FFT on ADC data.
        
        Args:
            adc_data: Input ADC data with shape (..., num_samples)
                      Last axis should be the sample axis
                      
        Returns:
            Range FFT output with shape (..., rangeFFTSize)
        """
        # Apply window along last axis
        windowed = adc_data * self._window
        
        # Perform FFT
        range_fft = np.fft.fft(windowed, n=self.config.rangeFFTSize, axis=-1)
        
        if self.config.scaleFFT:
            range_fft = range_fft / self.config.numSamplePerChirp
        
        return range_fft
    
    def datapath(self, adc_data: np.ndarray) -> np.ndarray:
        """
        MATLAB-compatible datapath interface.
        
        Args:
            adc_data: ADC data with shape (num_samples, num_chirps, num_rx)
            
        Returns:
            Range FFT with shape (rangeFFTSize, num_chirps, num_rx)
        """
        # Transpose to have samples last
        # Input: (samples, chirps, rx) -> (chirps, rx, samples)
        data_transposed = np.transpose(adc_data, (1, 2, 0))
        
        # Process
        range_fft = self.process(data_transposed)
        
        # Transpose back: (chirps, rx, range) -> (range, chirps, rx)
        return np.transpose(range_fft, (2, 0, 1))


def range_fft(adc_data: np.ndarray, fft_size: int = 256,
              window_type: str = 'hann') -> np.ndarray:
    """
    Simple function interface for range FFT.
    
    Args:
        adc_data: ADC samples, last axis is sample axis
        fft_size: FFT size
        window_type: Window type ('hann', 'blackman', 'hamming', 'rect')
        
    Returns:
        Range FFT output
    """
    n_samples = adc_data.shape[-1]
    
    # Generate window
    if window_type == 'hann':
        window = np.hanning(n_samples)
    elif window_type == 'blackman':
        window = np.blackman(n_samples)
    elif window_type == 'hamming':
        window = np.hamming(n_samples)
    else:
        window = np.ones(n_samples)
    
    # Apply window and FFT
    windowed = adc_data * window
    return np.fft.fft(windowed, n=fft_size, axis=-1)
