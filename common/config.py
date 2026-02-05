"""
config.py

Configuration dataclasses for TI mmWave radar signal processing.

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np

from .constants import (
    TI_CASCADE_RX_ID, DEFAULT_CALIBRATION_INTERP,
    DATA_PLATFORM_TDA2, NUM_DEVICES_CASCADE
)


@dataclass
class FileNameStruct:
    """Structure to hold binary file names for each device."""
    master: str = ""
    slave1: str = ""
    slave2: str = ""
    slave3: str = ""
    masterIdxFile: str = ""
    slave1IdxFile: str = ""
    slave2IdxFile: str = ""
    slave3IdxFile: str = ""
    dataFolderName: str = ""


@dataclass
class ADCDataParams:
    """ADC data configuration parameters."""
    dataFmt: int = 0
    iqSwap: int = 0
    chanInterleave: int = 0
    numChirpsPerFrame: int = 0
    adcBits: int = 2  # 16-bit ADC
    numRxChan: int = 4
    numAdcSamples: int = 256


@dataclass
class RadarCubeParams:
    """Radar cube configuration parameters."""
    iqSwap: int = 0
    numRxChan: int = 4
    numTxChan: int = 3
    numRangeBins: int = 256
    numDopplerChirps: int = 64
    radarCubeFmt: int = 1


@dataclass
class RFParams:
    """RF configuration parameters."""
    startFreq: float = 77.0  # GHz
    freqSlope: float = 70.0  # MHz/us
    sampleRate: float = 10.0  # Msps
    numRangeBins: int = 256
    numDopplerBins: int = 64
    bandwidth: float = 4000.0  # MHz
    rangeResolutionMeters: float = 0.0375
    dopplerResolutionMps: float = 0.1
    framePeriodicity: float = 50.0  # ms


@dataclass
class GenCalibrationMatrixParams:
    """Parameters for calibration matrix generation."""
    calibrateFileName: str = ""
    targetRange: float = 0.0
    frameIdx: int = 1
    numSamplePerChirp: int = 256
    nchirp_loops: int = 64
    numChirpsPerFrame: int = 0
    TxToEnable: List[int] = field(default_factory=list)
    Slope_calib: float = 0.0
    Sampling_Rate_sps: float = 0.0
    calibrationInterp: int = DEFAULT_CALIBRATION_INTERP
    TI_Cascade_RX_ID: List[int] = field(default_factory=lambda: TI_CASCADE_RX_ID.tolist())
    RxForMIMOProcess: List[int] = field(default_factory=lambda: TI_CASCADE_RX_ID.tolist())
    TxForMIMOProcess: List[int] = field(default_factory=list)
    numRxToEnable: int = 16
    rangeResolution: float = 0.0
    dataPlatform: str = DATA_PLATFORM_TDA2
    NumDevices: int = NUM_DEVICES_CASCADE
    binDataFile: Optional[FileNameStruct] = None
    RxOrder: List[int] = field(default_factory=lambda: TI_CASCADE_RX_ID.tolist())


@dataclass  
class RangeProcParams:
    """Range processing parameters."""
    rangeFFTSize: int = 256
    numSamplePerChirp: int = 256
    windowType: str = 'hann'  # 'hann', 'blackman', 'rect'
    scaleFFT: bool = True


@dataclass
class DopplerProcParams:
    """Doppler processing parameters."""
    dopplerFFTSize: int = 64
    numChirpsPerFrame: int = 64
    windowType: str = 'hann'
    clutterRemoval: bool = True


@dataclass
class CFARParams:
    """CFAR detection parameters."""
    guardCellsRange: int = 4
    guardCellsDoppler: int = 2
    trainCellsRange: int = 8
    trainCellsDoppler: int = 4
    thresholdScale: float = 10.0  # dB
    peakGrouping: bool = True


@dataclass
class DOAParams:
    """Direction of Arrival estimation parameters."""
    azimuthFFTSize: int = 64
    elevationFFTSize: int = 16
    numTxAntennas: int = 12
    numRxAntennas: int = 16
    peakThreshold: float = 6.0  # dB above noise floor


@dataclass
class SimTopParams:
    """Top-level simulation parameters."""
    platform: str = "TI_4Chip_CASCADE"
    numFrames: int = 100
    startFrame: int = 1
    plotOn: bool = True
    saveOutput: bool = False
    dataPlatform: str = DATA_PLATFORM_TDA2


@dataclass
class DetectionResult:
    """Single detection result."""
    rangeInd: int = 0
    dopplerInd: int = 0
    dopplerInd_org: int = 0
    range: float = 0.0
    doppler: float = 0.0
    doppler_corr: float = 0.0
    doppler_corr_overlap: float = 0.0
    doppler_corr_FFT: float = 0.0
    estSNR: float = 0.0
    noise: float = 0.0
    angles: List[float] = field(default_factory=lambda: [0.0, 0.0])  # [azimuth, elevation]


def parse_module_params(param_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """
    Extract parameters with a given prefix from a parameter dictionary.
    
    Args:
        param_dict: Dictionary containing all parameters
        prefix: Module prefix to filter by (e.g., 'genCalibrationMatrixCascade_')
        
    Returns:
        Dictionary with prefix stripped from keys
    """
    result = {}
    for key, value in param_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            result[new_key] = value
    return result
