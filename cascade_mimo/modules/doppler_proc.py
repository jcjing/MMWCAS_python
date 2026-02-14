"""
doppler_proc.py

Doppler FFT processing with clutter removal.

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/

Converted from MATLAB: DopplerProcClutterRemove.m
"""

import numpy as np
from typing import Optional, Literal
from dataclasses import dataclass


@dataclass
class DopplerProcConfig:
    """Doppler processing configuration."""
    dopplerFFTSize: int = 64
    numChirpsPerFrame: int = 64
    windowType: Literal['hann', 'blackman', 'hamming', 'rect'] = 'rect'  # Match MATLAB dopplerWindowEnable=0
    clutterRemoval: bool = False  # Match MATLAB clutterRemove=0
    scaleFFT: bool = False  # Match MATLAB FFTOutScaleOn=0
    dopplerResolution: float = 0.0  # m/s, calculated from RF params


class DopplerProcessor:
    """
    Doppler FFT processing module with optional clutter removal.
    
    Performs windowed FFT across chirps to estimate velocity.
    """
    
    def __init__(self, config: Optional[DopplerProcConfig] = None, **kwargs):
        """
        Initialize Doppler processor.
        
        Args:
            config: DopplerProcConfig object
            **kwargs: Alternative way to pass config parameters
        """
        if config is not None:
            self.config = config
        else:
            self.config = DopplerProcConfig(**kwargs)
        
        # Pre-compute window
        self._window = self._generate_window()
    
    def _generate_window(self) -> np.ndarray:
        """Generate the windowing function."""
        n = self.config.numChirpsPerFrame
        
        if self.config.windowType == 'hann':
            return np.hanning(n)
        elif self.config.windowType == 'blackman':
            return np.blackman(n)
        elif self.config.windowType == 'hamming':
            return np.hamming(n)
        else:  # rect
            return np.ones(n)
    
    def remove_clutter(self, range_data: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Remove static clutter by subtracting mean across chirps.
        
        This is a simple MTI (Moving Target Indication) filter.
        
        Args:
            range_data: Range FFT data with chirp axis
            axis: Axis along which to remove clutter (chirp axis)
            
        Returns:
            Clutter-removed data
        """
        mean_val = np.mean(range_data, axis=axis, keepdims=True)
        return range_data - mean_val
    
    def process(self, range_data: np.ndarray, 
                chirp_axis: int = -1) -> np.ndarray:
        """
        Perform Doppler FFT on range-processed data.
        
        Args:
            range_data: Range FFT output with chirp axis
            chirp_axis: Axis corresponding to chirps
                      
        Returns:
            Doppler FFT output (range, Doppler, ...)
        """
        # Move chirp axis to last position for processing
        data = np.moveaxis(range_data, chirp_axis, -1)
        
        # Remove clutter if enabled
        if self.config.clutterRemoval:
            data = self.remove_clutter(data, axis=-1)
        
        # Apply window along chirp axis
        windowed = data * self._window
        
        # Perform FFT and shift
        doppler_fft = np.fft.fftshift(
            np.fft.fft(windowed, n=self.config.dopplerFFTSize, axis=-1),
            axes=-1
        )
        
        if self.config.scaleFFT:
            doppler_fft = doppler_fft / self.config.numChirpsPerFrame
        
        return doppler_fft
    
    def datapath(self, range_data: np.ndarray) -> np.ndarray:
        """
        MATLAB-compatible datapath interface.
        
        Args:
            range_data: Range FFT with shape (rangeFFTSize, num_chirps, num_rx)
            
        Returns:
            Doppler FFT with shape (rangeFFTSize, dopplerFFTSize, num_rx)
        """
        # Input shape: (range, chirps, rx)
        # Process with chirps as axis 1
        
        # Transpose to (chirps, range, rx) then back
        data = np.transpose(range_data, (1, 0, 2))  # (chirps, range, rx)
        
        # Clutter removal along chirps (axis 0)
        if self.config.clutterRemoval:
            mean_val = np.mean(data, axis=0, keepdims=True)
            data = data - mean_val
        
        # Window
        window = self._window[:, np.newaxis, np.newaxis]
        windowed = data * window
        
        # FFT along chirps
        doppler_fft = np.fft.fftshift(
            np.fft.fft(windowed, n=self.config.dopplerFFTSize, axis=0),
            axes=0
        )
        
        if self.config.scaleFFT:
            doppler_fft = doppler_fft / self.config.numChirpsPerFrame
        
        # Output shape: (doppler, range, rx) -> (range, doppler, rx)
        return np.transpose(doppler_fft, (1, 0, 2))


def doppler_fft(range_data: np.ndarray, fft_size: int = 64,
                window_type: str = 'hann', 
                remove_clutter: bool = True,
                chirp_axis: int = 1) -> np.ndarray:
    """
    Simple function interface for Doppler FFT.
    
    Args:
        range_data: Range FFT data
        fft_size: Doppler FFT size
        window_type: Window type
        remove_clutter: Whether to remove static clutter
        chirp_axis: Axis corresponding to chirps
        
    Returns:
        Doppler FFT output
    """
    # Move chirp axis to last
    data = np.moveaxis(range_data, chirp_axis, -1)
    n_chirps = data.shape[-1]
    
    # Clutter removal
    if remove_clutter:
        data = data - np.mean(data, axis=-1, keepdims=True)
    
    # Generate window
    if window_type == 'hann':
        window = np.hanning(n_chirps)
    elif window_type == 'blackman':
        window = np.blackman(n_chirps)
    elif window_type == 'hamming':
        window = np.hamming(n_chirps)
    else:
        window = np.ones(n_chirps)
    
    # Apply window and FFT
    windowed = data * window
    doppler = np.fft.fftshift(np.fft.fft(windowed, n=fft_size, axis=-1), axes=-1)
    
    return doppler
