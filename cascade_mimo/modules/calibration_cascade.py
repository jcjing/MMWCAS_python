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
    chirpSlope: float = 0.0
    calibrationInterp: int = 5
    dataPlatform: str = 'TDA2'
    NumDevices: int = 4
    adcCalibrationOn: bool = True
    phaseCalibOnly: bool = True  # 1: only phase calibration; 0: phase and amplitude calibration


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
    
    def datapath(self) -> np.ndarray:
        """
        MATLAB-compatible datapath interface.
        
        Reads ADC data and applies calibration exactly as MATLAB does.
        
        Returns:
            Calibrated ADC data with shape (samples, loops, rx, tx)
        """
        # Read raw ADC data
        radar_data_rxchain = self.read_adc_data()
        
        # If calibration is off, return raw data
        if not self.config.adcCalibrationOn or self.calib_data is None:
            # Reorder RX channels
            if self.config.RxForMIMOProcess:
                rx_order = [rx - 1 for rx in self.config.RxForMIMOProcess if rx - 1 < radar_data_rxchain.shape[2]]
                radar_data_rxchain = radar_data_rxchain[:, :, rx_order, :]
            return radar_data_rxchain
        
        # Extract calibration matrices
        calib_result = self.calib_data['calibResult'][0, 0] if isinstance(self.calib_data['calibResult'], np.ndarray) else self.calib_data['calibResult']
        RangeMat = calib_result['RangeMat']
        PeakValMat = calib_result['PeakValMat']
        
        # Get parameters (need these from config)
        numSamplePerChirp = self.config.numSamplePerChirp
        nchirp_loops = self.config.nchirp_loops
        TxToEnable = self.config.TxToEnable
        calibrationInterp = self.config.calibrationInterp
        phaseCalibOnly = self.config.phaseCalibOnly
        
        # Need slope and sampling rate from params
        if 'params' in self.calib_data:
            params = self.calib_data['params'][0, 0] if isinstance(self.calib_data['params'], np.ndarray) else self.calib_data['params']
            Slope_calib = params['Slope_MHzperus'] * 1e12  # Convert to Hz/s
            fs_calib = Sampling_Rate_sps = params['Sampling_Rate_sps']
        else:
            # Fallback - these should be set in config
            Slope_calib = getattr(self.config, 'Slope_calib', 12.022e12)
            Sampling_Rate_sps = fs_calib = getattr(self.config, 'Sampling_Rate_sps', 11.11e6)
        
        # Get chirp slope from config (current system's slope)
        chirpSlope = getattr(self.config, 'chirpSlope', Slope_calib)
        
        numTX = len(TxToEnable)
        outData = np.zeros_like(radar_data_rxchain)
        
        # Use first TX as reference (MATLAB uses TxToEnable(1))
        TX_ref = TxToEnable[0]
        
        # Apply calibration per TX
        for iTX in range(numTX):
            # MATLAB uses 1-indexed TxToEnable
            TXind = TxToEnable[iTX]
            
            # Construct frequency compensation matrix
            # freq_calib = (RangeMat(TXind,:)-RangeMat(TX_ref,1))*fs_calib/Sampling_Rate_sps *chirpSlope/Slope_calib
            freq_calib = (RangeMat[TXind-1, :] - RangeMat[TX_ref-1, 0]) * fs_calib / Sampling_Rate_sps * chirpSlope / Slope_calib
            freq_calib = 2 * np.pi * freq_calib / (numSamplePerChirp * calibrationInterp)
            
            # correction_vec = exp(1i*((0:numSamplePerChirp-1)'*freq_calib))'
            sample_indices = np.arange(numSamplePerChirp).reshape(-1, 1)
            correction_vec = np.exp(1j * (sample_indices @ freq_calib.reshape(1, -1)))
            
            # freq_correction_mat = repmat(correction_vec, 1, 1, nchirp_loops)
            # freq_correction_mat = permute(freq_correction_mat, [2 3 1])
            # Shape: (samples, rx) -> (samples, loops, rx)
            freq_correction_mat = np.tile(correction_vec[:, np.newaxis, :], (1, nchirp_loops, 1))
            
            # Apply frequency correction
            outData1TX = radar_data_rxchain[:, :, :, iTX] * freq_correction_mat
            
            # Construct phase compensation matrix
            # phase_calib = PeakValMat(TX_ref,1)./PeakValMat(TXind,:)
            phase_calib = PeakValMat[TX_ref-1, 0] / PeakValMat[TXind-1, :]
            
            # Remove amplitude calibration if phase only
            if phaseCalibOnly:
                phase_calib = phase_calib / np.abs(phase_calib)
            
            # phase_correction_mat = repmat(phase_calib.', 1,numSamplePerChirp, nchirp_loops)
            # phase_correction_mat = permute(phase_correction_mat, [2 3 1])
            # Shape: (rx,) -> (samples, loops, rx)
            phase_correction_mat = np.tile(phase_calib.reshape(1, 1, -1), (numSamplePerChirp, nchirp_loops, 1))
            
            # Apply phase correction
            outData[:, :, :, iTX] = outData1TX * phase_correction_mat
        
        # Reorder RX channels
        if self.config.RxForMIMOProcess:
            rx_order = [rx - 1 for rx in self.config.RxForMIMOProcess if rx - 1 < outData.shape[2]]
            outData = outData[:, :, rx_order, :]
        
        return outData


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
