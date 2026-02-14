"""
data_parse.py

Data parsing utilities for TI mmWave radar binary files.

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/

Converted from MATLAB:
- getUniqueFileIdx.m
- getBinFileNames_withIdx.m
- read_ADC_bin_TDA2_separateFiles.m
- getValidNumFrames.m
"""

import os
import glob
import struct
import numpy as np
from typing import List, Tuple, Optional
import ctypes

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common.config import FileNameStruct


def get_unique_file_idx(data_folder: str) -> List[str]:
    """
    Find all unique binary file indices in the data folder.
    
    Args:
        data_folder: Path to the data folder
        
    Returns:
        List of unique file indices as zero-padded strings (e.g., ['0000', '0001'])
    """
    pattern = os.path.join(data_folder, "*_data.bin")
    files = glob.glob(pattern)
    
    bin_file_ids = set()
    for file_path in files:
        file_name = os.path.basename(file_path)
        # Find underscores and extract the numerical value between first two underscores
        underscores = [i for i, c in enumerate(file_name) if c == '_']
        if len(underscores) >= 2:
            try:
                bin_id = int(file_name[underscores[0] + 1:underscores[1]])
                bin_file_ids.add(bin_id)
            except ValueError:
                continue
    
    # Sort and format as 4-digit strings
    return [f"{idx:04d}" for idx in sorted(bin_file_ids)]


def get_bin_file_names_with_idx(folder_name: str, file_idx: str) -> FileNameStruct:
    """
    Get binary file names for master and slave devices.
    
    Args:
        folder_name: Path to the data folder
        file_idx: File index string (e.g., '0000')
        
    Returns:
        FileNameStruct containing the file names
    """
    data_pattern = os.path.join(folder_name, f"*{file_idx}_data.bin")
    idx_pattern = os.path.join(folder_name, f"*{file_idx}_idx.bin")
    
    data_files = glob.glob(data_pattern)
    idx_files = glob.glob(idx_pattern)
    
    if len(data_files) > 4:
        raise ValueError("Too many binary files corresponding to the same fileIdx")
    
    file_struct = FileNameStruct(dataFolderName=folder_name)
    
    for data_file in data_files:
        name = os.path.basename(data_file)
        if 'master' in name.lower():
            file_struct.master = name
        elif 'slave1' in name.lower():
            file_struct.slave1 = name
        elif 'slave2' in name.lower():
            file_struct.slave2 = name
        elif 'slave3' in name.lower():
            file_struct.slave3 = name
    
    for idx_file in idx_files:
        name = os.path.basename(idx_file)
        if 'master' in name.lower():
            file_struct.masterIdxFile = name
        elif 'slave1' in name.lower():
            file_struct.slave1IdxFile = name
        elif 'slave2' in name.lower():
            file_struct.slave2IdxFile = name
        elif 'slave3' in name.lower():
            file_struct.slave3IdxFile = name
    
    return file_struct

class idx_header(ctypes.Structure):
    _fields_ = [
        ("tag", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("numIdx", ctypes.c_uint32),
        ("dataFileSize", ctypes.c_uint64)
    ]

    def __repr__(self) -> str:
        """
        Returns a detailed, developer-friendly string representation of the structure.
        """
        # Collect field names and their current values
        field_values = []
        for field in self._fields_:
            field_name = field[0]
            value = getattr(self, field_name)
            # Special handling for ctypes arrays to make them readable
            if isinstance(value, ctypes.Array):
                value_repr = list(value)
            else:
                value_repr = repr(value)
            field_values.append(f"{field_name}={value_repr}")
            
        # Format the output to look like a valid Python expression
        return f"{self.__class__.__name__}({', '.join(field_values)})"

class idx_buff(ctypes.Structure):
    _fields_ = [
        ("tag", ctypes.c_uint16),
        ("version", ctypes.c_uint16),
        ("flags", ctypes.c_uint32),
        ("width", ctypes.c_uint16),
        ("height", ctypes.c_uint16),
        ("pitch_or_meta_size", ctypes.c_uint32 * 4),
        ("size", ctypes.c_uint32),
        ("timestamp", ctypes.c_uint32),
        ("offset", ctypes.c_uint64),
    ]
    
    def __repr__(self) -> str:
        """
        Returns a detailed, developer-friendly string representation of the structure.
        """
        # Collect field names and their current values
        field_values = []
        for field in self._fields_:
            field_name = field[0]
            value = getattr(self, field_name)
            # Special handling for ctypes arrays to make them readable
            if isinstance(value, ctypes.Array):
                value_repr = list(value)
            else:
                value_repr = repr(value)
            field_values.append(f"{field_name}={value_repr}")
            
        # Format the output to look like a valid Python expression
        return f"{self.__class__.__name__}({', '.join(field_values)})"

def idx_bin_factory(numFrames):

    class idx_bin(ctypes.Structure):
        _fields_ = [
            ("Header", idx_header),
            ("BuffIdx", idx_buff * numFrames)
        ]

    return idx_bin

def open_idx(idx_file):
    with open(idx_file, 'rb') as f:
        b = f.read()

    ih = idx_header.from_buffer_copy(b)
    idx_bin = idx_bin_factory(ih.numIdx)

    idx = idx_bin.from_buffer_copy(b)

    return idx

def get_valid_num_frames(idx_file_path: str) -> Tuple[int, int]:
    """
    Get the number of valid frames from an index file.
    
    Args:
        idx_file_path: Path to the index file
        
    Returns:
        num_valid_frames
    """
    if not os.path.exists(idx_file_path):
        raise FileNotFoundError(f"Index file not found: {idx_file_path}")
    
    idx = open_idx(idx_file_path)
    num_frames = idx.Header.numIdx
    
    return num_frames


def read_bin_file(file_path: str, frame_idx: int, num_sample_per_chirp: int,
                  num_chirp_per_loop: int, num_loops: int, num_rx_per_device: int,
                  num_devices: int) -> np.ndarray:
    """
    Read binary ADC data file for a single device.
    
    Args:
        file_path: Full path to the binary file
        frame_idx: Frame index (1-indexed as in MATLAB)
        num_sample_per_chirp: Number of samples per chirp
        num_chirp_per_loop: Number of chirps per loop
        num_loops: Number of loops
        num_rx_per_device: Number of RX channels per device
        num_devices: Number of devices
        
    Returns:
        Complex ADC data array with shape (num_sample_per_chirp, num_loops, num_rx_per_device, num_chirp_per_loop)
    """
    expected_samples_per_frame = num_sample_per_chirp * num_chirp_per_loop * num_loops * num_rx_per_device * 2
    
    with open(file_path, 'rb') as fp:
        # Seek to the frame position (frame_idx is 1-indexed)
        fp.seek((frame_idx - 1) * expected_samples_per_frame * 2)
        
        # Read data as uint16
        adc_data = np.fromfile(fp, dtype=np.uint16, count=expected_samples_per_frame)
    
    # Convert to signed (handle negative values stored as 2's complement)
    neg_mask = (adc_data & 0x8000) != 0
    adc_data = adc_data.astype(np.int32)
    adc_data[neg_mask] = adc_data[neg_mask] - 65536
    
    # Convert to complex (I + jQ)
    adc_complex = adc_data[0::2] + 1j * adc_data[1::2]
    
    # Reshape: (numLoops, numChirpPerLoop, numSamplePerChirp, numRXPerDevice)
    adc_complex = adc_complex.reshape(num_loops, num_chirp_per_loop, 
                                      num_sample_per_chirp, num_rx_per_device)
    
    # Permute to: (numSamplePerChirp, numLoops, numRXPerDevice, numChirpPerLoop)
    adc_complex = np.transpose(adc_complex, (2, 0, 3, 1))
    
    return adc_complex


def read_adc_bin_tda2_separate_files(file_name_cascade: FileNameStruct, frame_idx: int,
                                      num_sample_per_chirp: int, num_chirp_per_loop: int,
                                      num_loops: int, num_rx_per_device: int = 4,
                                      num_devices: int = 1) -> np.ndarray:
    """
    Read raw ADC data from separate files for each device (TDA2 platform).
    
    Args:
        file_name_cascade: FileNameStruct with file paths
        frame_idx: Frame index (1-indexed)
        num_sample_per_chirp: Number of samples per chirp
        num_chirp_per_loop: Number of chirps per loop  
        num_loops: Number of loops
        num_rx_per_device: Number of RX per device (typically 4)
        num_devices: Number of devices
        
    Returns:
        Combined radar data with shape (num_sample_per_chirp, num_loops, 16, num_chirp_per_loop)
    """
    data_folder = file_name_cascade.dataFolderName
    
    # Read data from each device
    master_data = read_bin_file(os.path.join(data_folder, file_name_cascade.master),
                                frame_idx, num_sample_per_chirp, num_chirp_per_loop,
                                num_loops, num_rx_per_device, num_devices)
    slave1_data = read_bin_file(os.path.join(data_folder, file_name_cascade.slave1),
                                frame_idx, num_sample_per_chirp, num_chirp_per_loop,
                                num_loops, num_rx_per_device, num_devices)
    slave2_data = read_bin_file(os.path.join(data_folder, file_name_cascade.slave2),
                                frame_idx, num_sample_per_chirp, num_chirp_per_loop,
                                num_loops, num_rx_per_device, num_devices)
    slave3_data = read_bin_file(os.path.join(data_folder, file_name_cascade.slave3),
                                frame_idx, num_sample_per_chirp, num_chirp_per_loop,
                                num_loops, num_rx_per_device, num_devices)
    
    # Combine all channels: Master (1-4), Slave1 (5-8), Slave2 (9-12), Slave3 (13-16)
    radar_data = np.concatenate([master_data, slave1_data, slave2_data, slave3_data], axis=2)
    
    return radar_data


def read_dca1000_bin_file(file_path: str, num_adc_samples: int, num_chirps: int,
                          num_rx: int, num_lanes: int = 4,
                          interleaved: bool = False) -> np.ndarray:
    """
    Read binary data captured by DCA1000 for single-chip devices.
    
    Args:
        file_path: Path to binary file
        num_adc_samples: Number of ADC samples per chirp
        num_chirps: Number of chirps per frame
        num_rx: Number of RX channels
        num_lanes: Number of LVDS lanes (2 or 4)
        interleaved: Whether data is channel-interleaved
        
    Returns:
        Complex ADC data with shape (num_chirps, num_rx, num_adc_samples)
    """
    with open(file_path, 'rb') as fp:
        raw_data = np.fromfile(fp, dtype=np.uint16)
    
    # Convert to signed
    raw_data = raw_data.astype(np.float32)
    raw_data[raw_data >= 2**15] -= 2**16
    
    if num_lanes == 4:
        # 4-lane LVDS: data arranged as [I0,I1,I2,I3,Q0,Q1,Q2,Q3,...]
        raw_data_8 = raw_data.reshape(-1, 8)
        raw_data_i = raw_data_8[:, :4].flatten()
        raw_data_q = raw_data_8[:, 4:].flatten()
    else:
        # 2-lane LVDS: data arranged as [I0,I1,Q0,Q1,...]
        raw_data_4 = raw_data.reshape(-1, 4)
        raw_data_i = raw_data_4[:, :2].flatten()
        raw_data_q = raw_data_4[:, 2:].flatten()
    
    # Combine to complex
    frame_complex = raw_data_i + 1j * raw_data_q
    
    # Reshape based on interleaving
    if interleaved:
        # Non-interleaved: [sample, channel] per chirp
        temp = frame_complex.reshape(num_chirps, num_adc_samples * num_rx)
        frame_data = np.zeros((num_chirps, num_rx, num_adc_samples), dtype=complex)
        for chirp in range(num_chirps):
            frame_data[chirp] = temp[chirp].reshape(num_adc_samples, num_rx).T
    else:
        # Interleaved: [channel, sample] per chirp
        temp = frame_complex.reshape(num_chirps, num_adc_samples * num_rx)
        frame_data = np.zeros((num_chirps, num_rx, num_adc_samples), dtype=complex)
        for chirp in range(num_chirps):
            frame_data[chirp] = temp[chirp].reshape(num_rx, num_adc_samples)
    
    return frame_data
