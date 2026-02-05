"""
raw_data_reader.py

Raw data reader for single-chip TI mmWave devices with DCA1000 capture.

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/

Converted from MATLAB: rawDataReader.m

This script reads JSON configuration files and binary data captured from 
single-chip mmWave devices (xWR12xx/14xx/16xx/18xx/22xx/68xx) via DCA1000.
"""

import os
import json
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from scipy.io import savemat
import matplotlib.pyplot as plt


@dataclass
class ADCDataParams:
    """ADC data parameters."""
    dataFmt: int = 0
    iqSwap: int = 0
    chanInterleave: int = 0
    numChirpsPerFrame: int = 0
    adcBits: int = 2  # 16-bit
    numRxChan: int = 4
    numAdcSamples: int = 256


@dataclass
class RadarCubeParams:
    """Radar cube parameters."""
    iqSwap: int = 0
    numRxChan: int = 4
    numTxChan: int = 3
    numRangeBins: int = 256
    numDopplerChirps: int = 64
    radarCubeFmt: int = 1


@dataclass
class RFParams:
    """RF parameters."""
    startFreq: float = 77.0
    freqSlope: float = 70.0
    sampleRate: float = 10.0
    numRangeBins: int = 256
    numDopplerBins: int = 64
    bandwidth: float = 4000.0
    rangeResolutionMeters: float = 0.0375
    dopplerResolutionMps: float = 0.1
    framePeriodicity: float = 50.0


class RawDataReader:
    """
    Raw data reader for single-chip mmWave devices.
    
    Reads binary data captured via DCA1000 and processes it to radar cubes.
    """
    
    SUPPORTED_DEVICES = [
        'awr1642', 'iwr1642', 'awr1243', 'awr1443', 'iwr1443',
        'awr1843', 'iwr1843', 'iwr6843', 'awr2243'
    ]
    
    def __init__(self, setup_json_path: str):
        """
        Initialize raw data reader.
        
        Args:
            setup_json_path: Path to setup JSON file from Radar Studio
        """
        self.setup_json_path = setup_json_path
        self.setup_json = None
        self.mmwave_json = None
        self.params = {}
        
        self._load_config()
    
    def _load_config(self) -> None:
        """Load and parse configuration files."""
        # Load setup JSON
        with open(self.setup_json_path, 'r') as f:
            self.setup_json = json.load(f)
        
        # Load mmwave JSON
        mmwave_json_path = self.setup_json.get('configUsed', '')
        if mmwave_json_path and os.path.exists(mmwave_json_path):
            with open(mmwave_json_path, 'r') as f:
                self.mmwave_json = json.load(f)
        else:
            # Try relative path
            base_dir = os.path.dirname(self.setup_json_path)
            mmwave_json_path = os.path.join(base_dir, mmwave_json_path)
            if os.path.exists(mmwave_json_path):
                with open(mmwave_json_path, 'r') as f:
                    self.mmwave_json = json.load(f)
        
        self.device = self.setup_json.get('mmWaveDevice', 'unknown')
        print(f"mmWave Device: {self.device}")
        
        self._validate_config()
        self._generate_params()
    
    def _validate_config(self) -> bool:
        """Validate configuration."""
        if self.device.lower() not in [d.lower() for d in self.SUPPORTED_DEVICES]:
            print(f"Warning: Device {self.device} may not be fully supported")
        
        if self.setup_json.get('captureHardware') != 'DCA1000':
            print("Warning: Only DCA1000 capture is supported")
        
        return True
    
    def _generate_params(self) -> None:
        """Generate processing parameters from configuration."""
        if self.mmwave_json is None:
            return
        
        try:
            devices = self.mmwave_json.get('mmWaveDevices', [])
            if isinstance(devices, list) and len(devices) > 0:
                device_cfg = devices[0]
            else:
                device_cfg = devices
            
            rf_config = device_cfg.get('rfConfig', {})
            frame_cfg = rf_config.get('rlFrameCfg_t', {})
            profile_cfg = rf_config.get('rlProfiles', {})
            
            if isinstance(profile_cfg, list):
                profile_cfg = profile_cfg[0].get('rlProfileCfg_t', {})
            else:
                profile_cfg = profile_cfg.get('rlProfileCfg_t', {})
            
            # ADC parameters
            self.params['numAdcSamples'] = profile_cfg.get('numAdcSamples', 256)
            self.params['numChirpsPerFrame'] = (
                frame_cfg.get('numLoops', 64) * 
                (frame_cfg.get('chirpEndIdx', 0) - frame_cfg.get('chirpStartIdx', 0) + 1)
            )
            
            # RX channels
            rx_mask_str = rf_config.get('rlChanCfg_t', {}).get('rxChannelEn', '0xF')
            rx_mask = int(rx_mask_str, 16) if isinstance(rx_mask_str, str) else rx_mask_str
            self.params['numRxChan'] = bin(rx_mask).count('1')
            
            # TX channels
            self.params['numTxChan'] = frame_cfg.get('chirpEndIdx', 0) - frame_cfg.get('chirpStartIdx', 0) + 1
            
            # RF parameters
            self.params['startFreq'] = profile_cfg.get('startFreqConst_GHz', 77.0)
            self.params['freqSlope'] = profile_cfg.get('freqSlopeConst_MHz_usec', 70.0)
            self.params['sampleRate'] = profile_cfg.get('digOutSampleRate', 10000) / 1000
            self.params['idleTime'] = profile_cfg.get('idleTimeConst_usec', 7.0)
            self.params['rampEndTime'] = profile_cfg.get('rampEndTime_usec', 60.0)
            self.params['framePeriod'] = frame_cfg.get('framePeriodicity_msec', 50.0)
            self.params['numLoops'] = frame_cfg.get('numLoops', 64)
            
            # Derived parameters
            self.params['numRangeBins'] = 2 ** int(np.ceil(np.log2(self.params['numAdcSamples'])))
            self.params['numDopplerBins'] = frame_cfg.get('numLoops', 64)
            
            c = 3e8
            chirp_time = self.params['numAdcSamples'] / (self.params['sampleRate'] * 1e3)
            bandwidth = self.params['freqSlope'] * 1e6 * chirp_time
            
            self.params['rangeResolution'] = c / (2 * bandwidth)
            self.params['bandwidth'] = bandwidth / 1e6  # MHz
            
            # Check lane configuration
            lane_cfg = device_cfg.get('rawDataCaptureConfig', {}).get('rlDevLaneEnable_t', {})
            lane_en_str = lane_cfg.get('laneEn', '0xF')
            lane_en = int(lane_en_str, 16) if isinstance(lane_en_str, str) else lane_en_str
            self.params['numLanes'] = bin(lane_en).count('1')
            
            interleave_cfg = device_cfg.get('rawDataCaptureConfig', {}).get('rlDevDataFmtCfg_t', {})
            self.params['chInterleave'] = interleave_cfg.get('chInterleave', 0)
            self.params['iqSwap'] = interleave_cfg.get('iqSwapSel', 0)
            
            self._print_params()
            
        except Exception as e:
            print(f"Error parsing config: {e}")
    
    def _print_params(self) -> None:
        """Print parsed parameters."""
        print("\nParsed Parameters:")
        print(f"  ADC Samples: {self.params.get('numAdcSamples')}")
        print(f"  Chirps/Frame: {self.params.get('numChirpsPerFrame')}")
        print(f"  RX Channels: {self.params.get('numRxChan')}")
        print(f"  TX Channels: {self.params.get('numTxChan')}")
        print(f"  Range Bins: {self.params.get('numRangeBins')}")
        print(f"  Doppler Bins: {self.params.get('numDopplerBins')}")
        print(f"  Range Resolution: {self.params.get('rangeResolution', 0):.4f} m")
        print(f"  LVDS Lanes: {self.params.get('numLanes')}")
    
    def get_bin_file_paths(self) -> List[str]:
        """Get binary file paths from configuration."""
        base_path = self.setup_json.get('capturedFiles', {}).get('fileBasePath', '')
        files = self.setup_json.get('capturedFiles', {}).get('files', [])
        
        paths = []
        for f in files:
            file_name = f.get('processedFileName', f.get('rawFileName', ''))
            if file_name:
                paths.append(os.path.join(base_path, file_name))
        
        return paths
    
    def get_num_frames(self, bin_file_path: str) -> int :
        """Calculate number of frames in binary file."""
        if not os.path.exists(bin_file_path):
            return 0
        
        file_size = os.path.getsize(bin_file_path)
        
        # Calculate frame size
        num_samples = self.params.get('numAdcSamples', 256)
        num_chirps = self.params.get('numChirpsPerFrame', 64)
        num_rx = self.params.get('numRxChan', 4)
        
        # Complex data: 2 samples per IQ pair, 2 bytes per sample
        frame_size = num_samples * num_chirps * num_rx * 4  # bytes
        
        return file_size // frame_size
    
    def read_frame(self, bin_file_path: str, frame_idx: int) -> np.ndarray:
        """
        Read a single frame from binary file.
        
        Args:
            bin_file_path: Path to binary file
            frame_idx: Frame index (1-indexed)
            
        Returns:
            Complex ADC data with shape (num_chirps, num_rx, num_samples)
        """
        num_samples = self.params.get('numAdcSamples', 256)
        num_chirps = self.params.get('numChirpsPerFrame', 64)
        num_rx = self.params.get('numRxChan', 4)
        num_lanes = self.params.get('numLanes', 4)
        ch_interleave = self.params.get('chInterleave', 0)
        iq_swap = self.params.get('iqSwap', 0)
        
        frame_size = num_samples * num_chirps * num_rx * 4  # bytes
        
        with open(bin_file_path, 'rb') as f:
            f.seek((frame_idx - 1) * frame_size)
            raw_data = np.fromfile(f, dtype=np.uint16, count=frame_size // 2)
        
        # Convert to float and handle signed values
        raw_data = raw_data.astype(np.float32)
        raw_data[raw_data >= 2**15] -= 2**16
        
        # Reshape based on LVDS lanes
        if num_lanes == 4:
            raw_data_8 = raw_data.reshape(-1, 8)
            raw_i = raw_data_8[:, :4].flatten()
            raw_q = raw_data_8[:, 4:].flatten()
        else:  # 2 lanes
            raw_data_4 = raw_data.reshape(-1, 4)
            raw_i = raw_data_4[:, :2].flatten()
            raw_q = raw_data_4[:, 2:].flatten()
        
        # Handle IQ swap
        if iq_swap == 1:
            raw_i, raw_q = raw_q, raw_i
        
        # Combine to complex
        frame_complex = raw_i + 1j * raw_q
        
        # Reshape based on interleave mode
        if ch_interleave == 1:  # Non-interleaved
            temp = frame_complex.reshape(num_chirps, num_samples * num_rx)
            frame_data = np.zeros((num_chirps, num_rx, num_samples), dtype=complex)
            for chirp in range(num_chirps):
                frame_data[chirp] = temp[chirp].reshape(num_samples, num_rx).T
        else:  # Interleaved
            temp = frame_complex.reshape(num_chirps, num_samples * num_rx)
            frame_data = np.zeros((num_chirps, num_rx, num_samples), dtype=complex)
            for chirp in range(num_chirps):
                frame_data[chirp] = temp[chirp].reshape(num_rx, num_samples)
        
        return frame_data
    
    def compute_range_fft(self, frame_data: np.ndarray, 
                          window_type: int = 0) -> np.ndarray:
        """
        Compute range FFT on frame data.
        
        Args:
            frame_data: ADC data (num_chirps, num_rx, num_samples)
            window_type: 0=rect, 1=hann, 2=blackman
            
        Returns:
            Radar cube (num_chirps, num_rx, numRangeBins)
        """
        num_chirps, num_rx, num_samples = frame_data.shape
        num_range_bins = self.params.get('numRangeBins', 256)
        
        # Generate window
        if window_type == 1:
            window = np.hanning(num_samples)
        elif window_type == 2:
            window = np.blackman(num_samples)
        else:
            window = np.ones(num_samples)
        
        # Apply window and FFT
        radar_cube = np.zeros((num_chirps, num_rx, num_range_bins), dtype=complex)
        for chirp in range(num_chirps):
            for rx in range(num_rx):
                windowed = frame_data[chirp, rx, :] * window
                radar_cube[chirp, rx, :] = np.fft.fft(windowed, n=num_range_bins)
        
        return radar_cube
    
    def export_data(self, raw_data_file: Optional[str] = None,
                    radar_cube_file: Optional[str] = None) -> None:
        """
        Export raw ADC data and/or radar cube to MAT files.
        
        Args:
            raw_data_file: Output path for raw ADC data
            radar_cube_file: Output path for radar cube data
        """
        bin_files = self.get_bin_file_paths()
        if not bin_files:
            print("No binary files found")
            return
        
        all_raw = []
        all_cube = []
        
        for bin_file in bin_files:
            num_frames = self.get_num_frames(bin_file)
            print(f"Processing {bin_file}: {num_frames} frames")
            
            for frame_idx in range(1, num_frames + 1):
                frame_data = self.read_frame(bin_file, frame_idx)
                all_raw.append(frame_data)
                
                if radar_cube_file:
                    cube = self.compute_range_fft(frame_data)
                    all_cube.append(cube)
        
        if raw_data_file and all_raw:
            print(f"Saving raw data to: {raw_data_file}")
            savemat(raw_data_file, {
                'adcRawData': {
                    'rfParams': self.params,
                    'data': all_raw,
                    'dim': {
                        'numFrames': len(all_raw),
                        'numChirps': self.params.get('numChirpsPerFrame'),
                        'numRxChan': self.params.get('numRxChan'),
                        'numSamples': self.params.get('numAdcSamples'),
                    }
                }
            }, do_compression=True)
            
            # Also save as npz
            npz_file = raw_data_file.replace('.mat', '.npz')
            np.savez_compressed(npz_file,
                rfParams=self.params,
                data=all_raw,
                numFrames=len(all_raw),
                numChirps=self.params.get('numChirpsPerFrame'),
                numRxChan=self.params.get('numRxChan'),
                numSamples=self.params.get('numAdcSamples'),
            )
            print(f"Also saved to: {npz_file}")
        
        if radar_cube_file and all_cube:
            print(f"Saving radar cube to: {radar_cube_file}")
            savemat(radar_cube_file, {
                'radarCube': {
                    'rfParams': self.params,
                    'data': all_cube,
                    'dim': {
                        'numFrames': len(all_cube),
                        'numChirps': self.params.get('numChirpsPerFrame'),
                        'numRxChan': self.params.get('numRxChan'),
                        'numRangeBins': self.params.get('numRangeBins'),
                    }
                }
            }, do_compression=True)
            
            # Also save as npz
            npz_file = radar_cube_file.replace('.mat', '.npz')
            np.savez_compressed(npz_file,
                rfParams=self.params,
                data=all_cube,
                numFrames=len(all_cube),
                numChirps=self.params.get('numChirpsPerFrame'),
                numRxChan=self.params.get('numRxChan'),
                numRangeBins=self.params.get('numRangeBins'),
            )
            print(f"Also saved to: {npz_file}")


def read_raw_data(setup_json_path: str,
                  raw_data_file: Optional[str] = None,
                  radar_cube_file: Optional[str] = None,
                  debug_plot: bool = False) -> RawDataReader:
    """
    Main function to read raw mmWave data.
    
    Args:
        setup_json_path: Path to setup JSON file
        raw_data_file: Output path for raw data MAT file
        radar_cube_file: Output path for radar cube MAT file
        debug_plot: Enable debug plotting
        
    Returns:
        RawDataReader instance
    """
    reader = RawDataReader(setup_json_path)
    
    if raw_data_file or radar_cube_file:
        reader.export_data(raw_data_file, radar_cube_file)
    
    if debug_plot:
        _debug_plot(reader)
    
    return reader


def _debug_plot(reader: RawDataReader) -> None:
    """Create debug plots for the first frame."""
    bin_files = reader.get_bin_file_paths()
    if not bin_files:
        return
    
    frame_data = reader.read_frame(bin_files[0], 1)
    radar_cube = reader.compute_range_fft(frame_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Time domain - first chirp, first RX
    ax = axes[0, 0]
    ax.plot(np.real(frame_data[0, 0, :]), label='I')
    ax.plot(np.imag(frame_data[0, 0, :]), label='Q')
    ax.set_xlabel('Sample')
    ax.set_ylabel('ADC Value')
    ax.set_title('Time Domain (Chirp 0, RX 0)')
    ax.legend()
    ax.grid(True)
    
    # Range profile - first chirp
    ax = axes[0, 1]
    range_bins = reader.params.get('numRangeBins', 256)
    range_res = reader.params.get('rangeResolution', 1)
    range_axis = np.arange(range_bins) * range_res
    power = 10 * np.log10(np.sum(np.abs(radar_cube[0]) ** 2, axis=0) + 1e-10)
    ax.plot(range_axis, power)
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Power (dB)')
    ax.set_title('Range Profile (Chirp 0, All RX)')
    ax.grid(True)
    
    # Range-Doppler map
    ax = axes[1, 0]
    doppler_fft = np.fft.fftshift(np.fft.fft(radar_cube, axis=0), axes=0)
    rd_map = 10 * np.log10(np.sum(np.abs(doppler_fft) ** 2, axis=1) + 1e-10)
    ax.imshow(rd_map, aspect='auto', origin='lower', cmap='jet')
    ax.set_xlabel('Range Bin')
    ax.set_ylabel('Doppler Bin')
    ax.set_title('Range-Doppler Map')
    
    # RX comparison
    ax = axes[1, 1]
    for rx in range(min(4, reader.params.get('numRxChan', 4))):
        power = 10 * np.log10(np.abs(radar_cube[0, rx, :]) ** 2 + 1e-10)
        ax.plot(power, label=f'RX{rx}')
    ax.set_xlabel('Range Bin')
    ax.set_ylabel('Power (dB)')
    ax.set_title('Range Profile by RX Channel')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Raw Data Reader')
    parser.add_argument('--setup_json', required=True, help='Setup JSON file')
    parser.add_argument('--raw_output', default=None, help='Raw data output file')
    parser.add_argument('--cube_output', default=None, help='Radar cube output file')
    parser.add_argument('--debug', action='store_true', help='Enable debug plots')
    
    args = parser.parse_args()
    
    read_raw_data(
        args.setup_json,
        args.raw_output,
        args.cube_output,
        args.debug
    )
