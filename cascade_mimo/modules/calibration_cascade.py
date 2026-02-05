"""
calibration_cascade.py

ADC data calibration module for cascade MIMO radar.

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/

Converted from MATLAB: calibrationCascade.m, genCalibrationMatrixCascade.m
"""

import os
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from scipy.io import loadmat

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common.constants import TI_CASCADE_RX_ID, SPEED_OF_LIGHT
from common.config import FileNameStruct


@dataclass
class CalibrationConfig:
    """Calibration configuration."""
    binfilePath: Optional[FileNameStruct] = None
    calibrationFilePath: str = ""
    frameIdx: int = 1
    numSamplePerChirp: int = 256
    nchirp_loops: int = 64
    numChirpsPerFrame: int = 0
    TxToEnable: List[int] = field(default_factory=list)
    RxForMIMOProcess: List[int] = field(default_factory=lambda: TI_CASCADE_RX_ID.tolist())
    IdTxForMIMOProcess: List[int] = field(default_factory=list)
    Slope_calib: float = 0.0
    Sampling_Rate_sps: float = 0.0
    calibrationInterp: int = 5
    dataPlatform: str = 'TDA2'
    NumDevices: int = 4
    adcCalibrationOn: bool = True
    phaseCalibOnly: bool = False


class CalibrationCascade:
    """
    ADC data calibration for 4-chip cascade MIMO radar.
    
    Applies frequency and phase calibration matrices to raw ADC data.
    """
    
    def __init__(self, config: Optional[CalibrationConfig] = None, 
                 calibration_file_path: str = "",
                 **kwargs):
        """
        Initialize calibration module.
        
        Args:
            config: CalibrationConfig object
            calibration_file_path: Path to calibration .mat file
            **kwargs: Config parameters
        """
        if config is not None:
            self.config = config
        else:
            self.config = CalibrationConfig(**kwargs)
        
        if calibration_file_path:
            self.config.calibrationFilePath = calibration_file_path
        
        # Load calibration data
        self.calib_data = None
        self.freq_calib_matrix = None
        self.phase_calib_matrix = None
        
        if self.config.calibrationFilePath and os.path.exists(self.config.calibrationFilePath):
            self._load_calibration()
    
    def _load_calibration(self) -> None:
        """Load calibration matrices from file."""
        try:
            self.calib_data = loadmat(self.config.calibrationFilePath)
            
            # Extract calibration matrices
            if 'calibResult' in self.calib_data:
                calib_result = self.calib_data['calibResult']
                if isinstance(calib_result, np.ndarray) and calib_result.dtype.names:
                    # Structured array
                    if 'RxMismatch' in calib_result.dtype.names:
                        self.phase_calib_matrix = calib_result['RxMismatch'][0, 0]
                else:
                    self.phase_calib_matrix = calib_result
            
            if 'phaseCalibrationMatrix' in self.calib_data:
                self.phase_calib_matrix = self.calib_data['phaseCalibrationMatrix']
            
            if 'freqCalibrationMatrix' in self.calib_data:
                self.freq_calib_matrix = self.calib_data['freqCalibrationMatrix']
                
        except Exception as e:
            print(f"Warning: Could not load calibration file: {e}")
    
    def set_bin_file_path(self, file_struct: FileNameStruct) -> None:
        """Set the binary file path structure."""
        self.config.binfilePath = file_struct
    
    def read_adc_data(self) -> np.ndarray:
        """
        Read raw ADC data from binary files.
        
        Returns:
            ADC data array with shape (samples, chirps, rx, tx)
        """
        from ..utils.data_parse import read_adc_bin_tda2_separate_files
        
        if self.config.binfilePath is None:
            raise ValueError("Binary file path not set")
        
        num_chirp_per_loop = len(self.config.TxToEnable)
        num_loops = self.config.nchirp_loops
        
        radar_data = read_adc_bin_tda2_separate_files(
            self.config.binfilePath,
            self.config.frameIdx,
            self.config.numSamplePerChirp,
            num_chirp_per_loop,
            num_loops,
            num_rx_per_device=4,
            num_devices=1
        )
        
        return radar_data
    
    def apply_calibration(self, adc_data: np.ndarray) -> np.ndarray:
        """
        Apply calibration to ADC data.
        
        Args:
            adc_data: Raw ADC data, shape (samples, chirps, rx, tx)
            
        Returns:
            Calibrated ADC data
        """
        if not self.config.adcCalibrationOn:
            return adc_data
        
        calibrated = adc_data.copy()
        
        # Apply phase calibration
        if self.phase_calib_matrix is not None:
            # Phase calibration is applied per RX-TX virtual antenna
            num_rx = adc_data.shape[2]
            num_tx = adc_data.shape[3] if adc_data.ndim > 3 else 1
            
            for rx in range(num_rx):
                for tx in range(num_tx):
                    if tx < self.phase_calib_matrix.shape[0] and rx < self.phase_calib_matrix.shape[1]:
                        phase_correction = np.exp(-1j * np.angle(self.phase_calib_matrix[tx, rx]))
                        if adc_data.ndim > 3:
                            calibrated[:, :, rx, tx] *= phase_correction
                        else:
                            calibrated[:, :, rx] *= phase_correction
        
        # Apply frequency calibration (range-dependent phase correction)
        if self.freq_calib_matrix is not None and not self.config.phaseCalibOnly:
            # Frequency calibration corrects for path length differences
            num_samples = adc_data.shape[0]
            for sample in range(num_samples):
                if sample < self.freq_calib_matrix.shape[0]:
                    freq_correction = self.freq_calib_matrix[sample]
                    calibrated[sample] *= freq_correction
        
        return calibrated
    
    def datapath(self) -> np.ndarray:
        """
        MATLAB-compatible datapath interface.
        
        Reads ADC data and applies calibration.
        
        Returns:
            Calibrated ADC data
        """
        adc_data = self.read_adc_data()
        
        # Reorder RX channels
        if self.config.RxForMIMOProcess:
            rx_order = [rx - 1 for rx in self.config.RxForMIMOProcess if rx - 1 < adc_data.shape[2]]
            adc_data = adc_data[:, :, rx_order, :]
        
        calibrated_data = self.apply_calibration(adc_data)
        
        return calibrated_data


def generate_calibration_matrix(
    adc_data: np.ndarray,
    target_range_bin: int,
    num_tx: int,
    num_rx: int,
    ref_tx: int = 0,
    ref_rx: int = 0
) -> np.ndarray:
    """
    Generate phase calibration matrix from calibration data.
    
    Assumes data was captured with a corner reflector at known range.
    
    Args:
        adc_data: ADC data from calibration capture
        target_range_bin: Range bin of calibration target
        num_tx: Number of TX antennas
        num_rx: Number of RX antennas
        ref_tx: Reference TX antenna index
        ref_rx: Reference RX antenna index
        
    Returns:
        Phase calibration matrix, shape (num_tx, num_rx)
    """
    # Perform range FFT
    range_fft = np.fft.fft(adc_data, axis=0)
    
    # Extract target bin
    target_response = range_fft[target_range_bin]
    
    # Initialize calibration matrix
    calib_matrix = np.zeros((num_tx, num_rx), dtype=complex)
    
    # Calculate relative phase with respect to reference
    ref_phase = np.angle(target_response[ref_rx, ref_tx] if target_response.ndim > 1 
                         else target_response[ref_rx])
    
    for tx in range(num_tx):
        for rx in range(num_rx):
            if target_response.ndim > 1:
                phase = np.angle(target_response[rx, tx])
            else:
                phase = np.angle(target_response[rx])
            
            # Calibration factor to align with reference
            calib_matrix[tx, rx] = np.exp(1j * (ref_phase - phase))
    
    return calib_matrix
