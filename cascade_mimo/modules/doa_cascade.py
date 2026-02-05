"""
doa_cascade.py

Direction of Arrival estimation for cascade MIMO radar.

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/

Converted from MATLAB: DOACascade.m, DOA_beamformingFFT_2D.m
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common.constants import (
    TI_CASCADE_TX_POSITION_AZI, TI_CASCADE_TX_POSITION_ELE,
    TI_CASCADE_RX_POSITION_AZI, TI_CASCADE_RX_POSITION_ELE,
    TI_CASCADE_ANTENNA_DESIGN_FREQ, SPEED_OF_LIGHT
)
from .cfar_caso import Detection


@dataclass
class DOAConfig:
    """DOA estimation configuration."""
    azimuthFFTSize: int = 64
    elevationFFTSize: int = 16
    numTxAntennas: int = 12
    numRxAntennas: int = 16
    TxAntPos_azi: np.ndarray = field(default_factory=lambda: TI_CASCADE_TX_POSITION_AZI)
    TxAntPos_ele: np.ndarray = field(default_factory=lambda: TI_CASCADE_TX_POSITION_ELE)
    RxAntPos_azi: np.ndarray = field(default_factory=lambda: TI_CASCADE_RX_POSITION_AZI.astype(float))
    RxAntPos_ele: np.ndarray = field(default_factory=lambda: TI_CASCADE_RX_POSITION_ELE.astype(float))
    designFreq: float = TI_CASCADE_ANTENNA_DESIGN_FREQ  # GHz
    centerFreq: float = 77.0  # GHz
    peakThreshold: float = 6.0  # dB above noise


@dataclass
class AngleEstimate:
    """Angle estimation result."""
    rangeInd: int = 0
    dopplerInd: int = 0
    dopplerInd_org: int = 0
    range: float = 0.0
    doppler: float = 0.0
    doppler_corr: float = 0.0
    doppler_corr_overlap: float = 0.0
    doppler_corr_FFT: float = 0.0
    estSNR: float = 0.0
    angles: Tuple[float, float] = (0.0, 0.0)  # (azimuth, elevation) in degrees


class DOACascade:
    """
    Direction of Arrival estimation using 2D FFT beamforming.
    
    Supports both azimuth-only and azimuth-elevation estimation.
    """
    
    def __init__(self, config: Optional[DOAConfig] = None, **kwargs):
        """
        Initialize DOA estimator.
        
        Args:
            config: DOAConfig object
            **kwargs: Config parameters
        """
        if config is not None:
            self.config = config
        else:
            self.config = DOAConfig(**kwargs)
        
        # Build virtual antenna array
        self._build_virtual_array()
    
    def _build_virtual_array(self) -> None:
        """Build virtual antenna array from TX and RX positions."""
        num_tx = self.config.numTxAntennas
        num_rx = self.config.numRxAntennas
        
        # Virtual array positions
        self.virtual_ant_azi = np.zeros(num_tx * num_rx)
        self.virtual_ant_ele = np.zeros(num_tx * num_rx)
        
        for tx in range(num_tx):
            for rx in range(num_rx):
                idx = tx * num_rx + rx
                self.virtual_ant_azi[idx] = (self.config.TxAntPos_azi[tx] + 
                                              self.config.RxAntPos_azi[rx])
                self.virtual_ant_ele[idx] = (self.config.TxAntPos_ele[tx] + 
                                              self.config.RxAntPos_ele[rx])
        
        # Scale by wavelength ratio
        wavelength_ratio = self.config.designFreq / self.config.centerFreq
        self.virtual_ant_azi *= wavelength_ratio
        self.virtual_ant_ele *= wavelength_ratio
        
        # Find azimuth-only virtual antennas (elevation = 0)
        self.azimuth_only_mask = self.virtual_ant_ele == 0
        self.azimuth_only_indices = np.where(self.azimuth_only_mask)[0]
    
    def beamforming_fft_1d(self, signal: np.ndarray, 
                           ant_positions: np.ndarray,
                           fft_size: int) -> np.ndarray:
        """
        1D beamforming using FFT.
        
        Args:
            signal: Complex signal from virtual antennas
            ant_positions: Antenna positions (normalized to wavelength/2)
            fft_size: FFT size for angle estimation
            
        Returns:
            Beamforming output
        """
        # Create steering matrix
        angles = np.arcsin(np.linspace(-1, 1, fft_size)) * 180 / np.pi
        
        # Zero-pad signal to FFT size based on antenna positions
        # Map antenna positions to FFT bins
        max_pos = np.max(np.abs(ant_positions))
        if max_pos > 0:
            bin_positions = np.round(ant_positions * (fft_size - 1) / (2 * max_pos) + fft_size // 2).astype(int)
        else:
            bin_positions = np.ones(len(ant_positions), dtype=int) * (fft_size // 2)
        
        # Create padded signal
        padded = np.zeros(fft_size, dtype=complex)
        for i, pos in enumerate(bin_positions):
            if 0 <= pos < fft_size:
                padded[pos] += signal[i]
        
        # FFT
        bf_out = np.fft.fftshift(np.fft.fft(padded))
        
        return bf_out
    
    def beamforming_fft_2d(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        2D beamforming for azimuth and elevation.
        
        Args:
            signal: Complex signal from virtual antennas, shape (num_virtual_ant,)
            
        Returns:
            Tuple of (azimuth_spectrum, elevation_spectrum)
        """
        # Azimuth beamforming
        az_spectrum = self.beamforming_fft_1d(
            signal, self.virtual_ant_azi, self.config.azimuthFFTSize
        )
        
        # Elevation beamforming
        el_spectrum = self.beamforming_fft_1d(
            signal, self.virtual_ant_ele, self.config.elevationFFTSize
        )
        
        return az_spectrum, el_spectrum
    
    def estimate_angles(self, signal: np.ndarray) -> Tuple[float, float, float]:
        """
        Estimate azimuth and elevation angles from virtual antenna signal.
        
        Args:
            signal: Complex signal from virtual antennas
            
        Returns:
            Tuple of (azimuth_deg, elevation_deg, peak_power)
        """
        az_spectrum, el_spectrum = self.beamforming_fft_2d(signal)
        
        # Find azimuth peak
        az_power = np.abs(az_spectrum) ** 2
        az_peak_idx = np.argmax(az_power)
        az_angle = np.arcsin(2 * (az_peak_idx - self.config.azimuthFFTSize // 2) / 
                            self.config.azimuthFFTSize) * 180 / np.pi
        
        # Find elevation peak
        el_power = np.abs(el_spectrum) ** 2
        el_peak_idx = np.argmax(el_power)
        el_angle = np.arcsin(2 * (el_peak_idx - self.config.elevationFFTSize // 2) / 
                            self.config.elevationFFTSize) * 180 / np.pi
        
        peak_power = az_power[az_peak_idx]
        
        return az_angle, el_angle, peak_power
    
    def process_detections(self, detections: List[Detection],
                          doppler_fft_out: np.ndarray) -> List[AngleEstimate]:
        """
        Estimate angles for each CFAR detection.
        
        Args:
            detections: List of CFAR detections
            doppler_fft_out: Doppler FFT data, shape (range, doppler, virtual_ant)
            
        Returns:
            List of AngleEstimate objects
        """
        angle_estimates = []
        
        for det in detections:
            # Extract signal from all virtual antennas at detection location
            if det.rangeInd < doppler_fft_out.shape[0] and det.dopplerInd < doppler_fft_out.shape[1]:
                signal = doppler_fft_out[det.rangeInd, det.dopplerInd, :]
            else:
                continue
            
            # Estimate angles
            az_angle, el_angle, peak_power = self.estimate_angles(signal)
            
            angle_est = AngleEstimate(
                rangeInd=det.rangeInd,
                dopplerInd=det.dopplerInd,
                dopplerInd_org=det.dopplerInd_org,
                range=det.range,
                doppler=det.doppler,
                doppler_corr=det.doppler,
                doppler_corr_overlap=det.doppler,
                doppler_corr_FFT=det.doppler,
                estSNR=det.estSNR,
                angles=(az_angle, el_angle)
            )
            angle_estimates.append(angle_est)
        
        return angle_estimates
    
    def datapath(self, detections: List[Detection],
                 doppler_fft_out: Optional[np.ndarray] = None) -> List[AngleEstimate]:
        """
        MATLAB-compatible datapath interface.
        
        Args:
            detections: List of CFAR detections
            doppler_fft_out: Optional Doppler FFT data
            
        Returns:
            List of angle estimates
        """
        if doppler_fft_out is None:
            # Just copy detection info without angle estimation
            return [AngleEstimate(
                rangeInd=d.rangeInd,
                dopplerInd=d.dopplerInd,
                dopplerInd_org=d.dopplerInd_org,
                range=d.range,
                doppler=d.doppler,
                estSNR=d.estSNR,
                angles=(0.0, 0.0)
            ) for d in detections]
        
        return self.process_detections(detections, doppler_fft_out)


def angles_to_xyz(range_m: float, azimuth_deg: float, elevation_deg: float) -> Tuple[float, float, float]:
    """
    Convert spherical coordinates to Cartesian.
    
    Args:
        range_m: Range in meters
        azimuth_deg: Azimuth angle in degrees
        elevation_deg: Elevation angle in degrees
        
    Returns:
        Tuple of (x, y, z) in meters
    """
    az_rad = np.radians(azimuth_deg)
    el_rad = np.radians(elevation_deg)
    
    x = range_m * np.sin(az_rad) * np.cos(el_rad)
    y = range_m * np.cos(az_rad) * np.cos(el_rad)
    z = range_m * np.sin(el_rad)
    
    return x, y, z
