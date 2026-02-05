"""
cfar_caso.py

CFAR Cell-Averaging Smallest Of (CASO) detection.

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/

Converted from MATLAB: CFAR_CASO.m, CFAR_CASO_Range.m, CFAR_CASO_Doppler_overlap.m
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class CFARConfig:
    """CFAR detection configuration."""
    # Range CFAR parameters
    guardCellsRange: int = 4
    trainCellsRange: int = 8
    thresholdScaleRange: float = 10.0  # dB
    
    # Doppler CFAR parameters
    guardCellsDoppler: int = 2
    trainCellsDoppler: int = 4
    thresholdScaleDoppler: float = 10.0  # dB
    
    # General parameters
    peakGrouping: bool = True
    maxNumDetections: int = 200
    
    # Range/velocity resolution for output
    rangeBinSize: float = 0.0375  # meters
    dopplerBinSize: float = 0.1  # m/s
    
    # Antenna configuration for SNR estimation
    numVirtualAntennas: int = 192


@dataclass
class Detection:
    """Single CFAR detection."""
    rangeInd: int = 0
    dopplerInd: int = 0
    dopplerInd_org: int = 0
    range: float = 0.0
    doppler: float = 0.0
    estSNR: float = 0.0
    noise: float = 0.0
    peakVal: float = 0.0


class CFARDetector:
    """
    CFAR-CASO (Cell Averaging Smallest Of) detector.
    
    Performs 2D CFAR detection on range-Doppler maps.
    """
    
    def __init__(self, config: Optional[CFARConfig] = None, **kwargs):
        """
        Initialize CFAR detector.
        
        Args:
            config: CFARConfig object
            **kwargs: Alternative way to pass config parameters
        """
        if config is not None:
            self.config = config
        else:
            self.config = CFARConfig(**kwargs)
    
    def cfar_1d(self, signal: np.ndarray, 
                guard_cells: int, train_cells: int,
                threshold_scale: float = 10.0,
                wrap_around: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        1D CFAR detection using CA-CFAR.
        
        Args:
            signal: 1D signal magnitude (linear scale)
            guard_cells: Number of guard cells on each side
            train_cells: Number of training cells on each side
            threshold_scale: Threshold scale in dB
            wrap_around: Whether to wrap around at edges
            
        Returns:
            Tuple of (detection_mask, noise_estimate)
        """
        n = len(signal)
        threshold_linear = 10 ** (threshold_scale / 10)
        
        noise_estimate = np.zeros(n)
        
        for i in range(n):
            # Left training cells
            left_start = i - guard_cells - train_cells
            left_end = i - guard_cells
            
            # Right training cells  
            right_start = i + guard_cells + 1
            right_end = i + guard_cells + train_cells + 1
            
            if wrap_around:
                left_indices = np.arange(left_start, left_end) % n
                right_indices = np.arange(right_start, right_end) % n
                left_sum = np.sum(signal[left_indices])
                right_sum = np.sum(signal[right_indices])
            else:
                # Handle edges by using available cells
                left_indices = np.arange(max(0, left_start), max(0, left_end))
                right_indices = np.arange(min(n, right_start), min(n, right_end))
                left_sum = np.sum(signal[left_indices]) if len(left_indices) > 0 else 0
                right_sum = np.sum(signal[right_indices]) if len(right_indices) > 0 else 0
            
            # CASO: use smaller of the two sides
            left_avg = left_sum / max(len(left_indices), 1)
            right_avg = right_sum / max(len(right_indices), 1)
            noise_estimate[i] = min(left_avg, right_avg)
        
        # Threshold
        threshold = noise_estimate * threshold_linear
        detection_mask = signal > threshold
        
        return detection_mask, noise_estimate
    
    def cfar_range(self, range_doppler: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        CFAR detection along range dimension.
        
        Args:
            range_doppler: Range-Doppler map, shape (num_range, num_doppler)
            
        Returns:
            Tuple of (detection_mask, noise_estimate)
        """
        num_range, num_doppler = range_doppler.shape
        detection_mask = np.zeros_like(range_doppler, dtype=bool)
        noise_estimate = np.zeros_like(range_doppler)
        
        for d in range(num_doppler):
            mask, noise = self.cfar_1d(
                range_doppler[:, d],
                self.config.guardCellsRange,
                self.config.trainCellsRange,
                self.config.thresholdScaleRange
            )
            detection_mask[:, d] = mask
            noise_estimate[:, d] = noise
        
        return detection_mask, noise_estimate
    
    def cfar_doppler(self, range_doppler: np.ndarray, 
                     range_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        CFAR detection along Doppler dimension for range cells that passed.
        
        Args:
            range_doppler: Range-Doppler map, shape (num_range, num_doppler)
            range_mask: Mask of detections from range CFAR
            
        Returns:
            Tuple of (detection_mask, noise_estimate)
        """
        num_range, num_doppler = range_doppler.shape
        detection_mask = np.zeros_like(range_doppler, dtype=bool)
        noise_estimate = np.zeros_like(range_doppler)
        
        # Only process range bins with detections
        range_indices = np.any(range_mask, axis=1)
        
        for r in np.where(range_indices)[0]:
            mask, noise = self.cfar_1d(
                range_doppler[r, :],
                self.config.guardCellsDoppler,
                self.config.trainCellsDoppler,
                self.config.thresholdScaleDoppler,
                wrap_around=True  # Doppler wraps around
            )
            detection_mask[r, :] = mask & range_mask[r, :]
            noise_estimate[r, :] = noise
        
        return detection_mask, noise_estimate
    
    def peak_grouping(self, range_doppler: np.ndarray,
                      detection_mask: np.ndarray) -> np.ndarray:
        """
        Group nearby detections and keep only peaks.
        
        Args:
            range_doppler: Range-Doppler map
            detection_mask: Boolean detection mask
            
        Returns:
            Grouped detection mask
        """
        from scipy.ndimage import label, maximum_filter
        
        # Label connected regions
        labeled, num_features = label(detection_mask)
        
        # Find local maxima
        local_max = maximum_filter(range_doppler, size=3) == range_doppler
        
        # Keep only local maxima within detection regions
        grouped_mask = detection_mask & local_max
        
        return grouped_mask
    
    def detect(self, doppler_fft_out: np.ndarray) -> List[Detection]:
        """
        Perform 2D CFAR detection.
        
        Args:
            doppler_fft_out: Doppler FFT output, shape (num_range, num_doppler, num_antennas)
            
        Returns:
            List of Detection objects
        """
        # Sum power across antennas
        power = np.sum(np.abs(doppler_fft_out) ** 2, axis=2)
        
        num_range, num_doppler = power.shape
        
        # Range CFAR
        range_mask, range_noise = self.cfar_range(power)
        
        # Doppler CFAR
        detection_mask, doppler_noise = self.cfar_doppler(power, range_mask)
        
        # Peak grouping
        if self.config.peakGrouping:
            detection_mask = self.peak_grouping(power, detection_mask)
        
        # Extract detections
        detections = []
        range_inds, doppler_inds = np.where(detection_mask)
        
        for i, (r, d) in enumerate(zip(range_inds, doppler_inds)):
            if len(detections) >= self.config.maxNumDetections:
                break
            
            # Calculate SNR
            signal_power = power[r, d]
            noise_power = doppler_noise[r, d]
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            # Convert indices to physical values
            range_val = r * self.config.rangeBinSize
            doppler_org = d - num_doppler // 2
            doppler_val = doppler_org * self.config.dopplerBinSize
            
            det = Detection(
                rangeInd=r,
                dopplerInd=d,
                dopplerInd_org=doppler_org,
                range=range_val,
                doppler=doppler_val,
                estSNR=snr,
                noise=noise_power,
                peakVal=signal_power
            )
            detections.append(det)
        
        return detections
    
    def datapath(self, doppler_fft_out: np.ndarray) -> List[Detection]:
        """MATLAB-compatible datapath interface."""
        return self.detect(doppler_fft_out)


def cfar_2d(range_doppler: np.ndarray,
            guard_range: int = 4, train_range: int = 8,
            guard_doppler: int = 2, train_doppler: int = 4,
            threshold_db: float = 10.0) -> np.ndarray:
    """
    Simple 2D CFAR function.
    
    Args:
        range_doppler: 2D range-Doppler map
        guard_range: Guard cells in range
        train_range: Training cells in range
        guard_doppler: Guard cells in Doppler
        train_doppler: Training cells in Doppler
        threshold_db: Detection threshold in dB
        
    Returns:
        Boolean detection mask
    """
    config = CFARConfig(
        guardCellsRange=guard_range,
        trainCellsRange=train_range,
        guardCellsDoppler=guard_doppler,
        trainCellsDoppler=train_doppler,
        thresholdScaleRange=threshold_db,
        thresholdScaleDoppler=threshold_db,
        peakGrouping=True
    )
    detector = CFARDetector(config)
    
    # Add dummy antenna dimension
    if range_doppler.ndim == 2:
        range_doppler = range_doppler[:, :, np.newaxis]
    
    detections = detector.detect(range_doppler)
    
    # Convert to mask
    mask = np.zeros(range_doppler.shape[:2], dtype=bool)
    for det in detections:
        mask[det.rangeInd, det.dopplerInd] = True
    
    return mask
