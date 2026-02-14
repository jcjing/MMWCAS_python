"""
plotting.py

Visualization utilities for mmWave radar data.

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/

Converted from MATLAB: plot_range_azimuth_2D.m
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def plot_range_azimuth_2d(
    range_resolution: float,
    radar_data_pre_3dfft: np.ndarray,
    TDM_MIMO_numTX: int,
    numRxAnt: int,
    antenna_azimuthonly: np.ndarray,
    LOG: bool = True,
    STATIC_ONLY: bool = True,
    PLOT_ON: bool = True,
    minRangeBinKeep: int = 5,
    rightRangeBinDiscard: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate and optionally plot range-azimuth 2D heatmap.
    This function exactly matches the MATLAB plot_range_azimuth_2D.m implementation.
    
    Args:
        range_resolution: range resolution to calculate axis to plot
        radar_data_pre_3dfft: input 3D matrix, rangeFFT x DopplerFFT x virtualArray
        TDM_MIMO_numTX: number of TXs used for processing
        numRxAnt: number of RXs used for processing
        antenna_azimuthonly: azimuth array ID
        LOG: 1:plot non-linear scale, ^0.4 by default
        STATIC_ONLY: 1 = plot heatmap for zero-Doppler; 0 = plot heatmap for nonzero-Doppler
        PLOT_ON: 1 = plot on; 0 = plot off
        minRangeBinKeep: start range index to keep
        rightRangeBinDiscard: number of right most range bins to discard
        
    Returns:
        Tuple of (mag_data_static, mag_data_dynamic, y_axis, x_axis)
    """
    dopplerFFTSize = radar_data_pre_3dfft.shape[1]
    rangeFFTSize = radar_data_pre_3dfft.shape[0]
    angleFFTSize = 256
    # ratio used to decide energy threshold used to pick non-zero Doppler bins
    ratio = 0.5
    DopplerCorrection = 0
    
    if DopplerCorrection == 1:
        # add Doppler correction before generating the heatmap
        radar_data_pre_3dfft_DopCor = np.zeros_like(radar_data_pre_3dfft)
        for dopplerInd in range(dopplerFFTSize):
            deltaPhi = 2 * np.pi * (dopplerInd - dopplerFFTSize / 2) / (TDM_MIMO_numTX * dopplerFFTSize)
            sig_bin_org = radar_data_pre_3dfft[:, dopplerInd, :]
            for i_TX in range(TDM_MIMO_numTX):
                RX_ID = slice(i_TX * numRxAnt, (i_TX + 1) * numRxAnt)
                corVec = np.exp(-1j * i_TX * deltaPhi)
                radar_data_pre_3dfft_DopCor[:, dopplerInd, RX_ID] = sig_bin_org[:, RX_ID] * corVec
        radar_data_pre_3dfft = radar_data_pre_3dfft_DopCor
    
    # Filter to azimuth-only antennas (MATLAB: radar_data_pre_3dfft(:,:,antenna_azimuthonly))
    radar_data_pre_3dfft = radar_data_pre_3dfft[:, :, antenna_azimuthonly]
    
    # Angle FFT (MATLAB: fft(radar_data_pre_3dfft, angleFFTSize, 3))
    radar_data_angle_range = np.fft.fft(radar_data_pre_3dfft, n=angleFFTSize, axis=2)
    n_angle_fft_size = radar_data_angle_range.shape[2]
    n_range_fft_size = radar_data_angle_range.shape[0]
    
    # Decide non-zero doppler bins to be used for dynamic range-azimuth heatmap
    DopplerPower = np.sum(np.mean(np.abs(radar_data_pre_3dfft), axis=2), axis=0)
    # Exclude DC and adjacent bins: [1:dopplerFFTSize/2-1, dopplerFFTSize/2+3:end]
    indices_noDC = list(range(0, dopplerFFTSize // 2 - 1)) + list(range(dopplerFFTSize // 2 + 2, dopplerFFTSize))
    DopplerPower_noDC = DopplerPower[indices_noDC]
    peakVal = np.max(DopplerPower_noDC)
    peakInd = np.argmax(DopplerPower_noDC)
    threshold = peakVal * ratio
    indSel_noDC = np.where(DopplerPower_noDC > threshold)[0]
    
    # Map back to original indices
    indSel = []
    for ii in indSel_noDC:
        if ii >= dopplerFFTSize // 2 - 1:
            indSel.append(ii + 3)
        else:
            indSel.append(ii)
    
    # Dynamic and static range-azimuth
    if len(indSel) > 0:
        radar_data_angle_range_dynamic = np.sum(np.abs(radar_data_angle_range[:, indSel, :]), axis=1)
    else:
        radar_data_angle_range_dynamic = np.zeros((n_range_fft_size, n_angle_fft_size))
    
    radar_data_angle_range_Static = np.abs(radar_data_angle_range[:, dopplerFFTSize // 2, :])
    
    # Select range indices (MATLAB: minRangeBinKeep:n_range_fft_size-rightRangeBinDiscard)
    # MATLAB colon operator is inclusive on both ends, Python arange is exclusive on upper bound
    # MATLAB: 5:492 -> [5, 6, ..., 492] (488 elements)
    # Python: need arange(5, 493) -> [5, 6, ..., 492] (488 elements)
    indices_1D = np.arange(minRangeBinKeep, n_range_fft_size - rightRangeBinDiscard + 1)
    
    # Apply fftshift (MATLAB does this on dimension 2)
    radar_data_angle_range_dynamic = np.fft.fftshift(radar_data_angle_range_dynamic, axes=1)
    radar_data_angle_range_Static = np.fft.fftshift(radar_data_angle_range_Static, axes=1)
    
    # Create sine and cosine arrays for coordinate transformation
    d = 1
    sine_theta = -2 * ((np.arange(-n_angle_fft_size / 2, n_angle_fft_size / 2 + 1)) / n_angle_fft_size) / d
    cos_theta = np.sqrt(1 - sine_theta**2)
    
    # Create meshgrid (MATLAB: meshgrid(indices_1D*range_resolution, sine_theta))
    R_mat, sine_theta_mat = np.meshgrid(indices_1D * range_resolution, sine_theta)
    _, cos_theta_mat = np.meshgrid(indices_1D, cos_theta)
    
    x_axis = R_mat * cos_theta_mat
    y_axis = R_mat * sine_theta_mat
    
    # Extract magnitude data
    # MATLAB: mag_data_static = squeeze(abs(radar_data_angle_range_Static(indices_1D+1,[1:end 1])));
    # Note: [1:end 1] means ALL columns, then append first column - so this wraps the angle dimension
    # MATLAB indices_1D+1 converts 0-based to 1-based, we use indices_1D directly in Python
    
    # First wrap the angle dimension: [1:end 1] means columns [0, 1, 2, ..., end, 0]
    # After fftshift, we have 256 columns, wrap to get 257 by appending first to end
    radar_data_angle_range_dynamic_wrapped = np.column_stack([radar_data_angle_range_dynamic, radar_data_angle_range_dynamic[:, 0]])
    radar_data_angle_range_Static_wrapped = np.column_stack([radar_data_angle_range_Static, radar_data_angle_range_Static[:, 0]])
    
    # Then select range bins and take magnitude
    mag_data_dynamic = np.abs(radar_data_angle_range_dynamic_wrapped[indices_1D, :])
    mag_data_static = np.abs(radar_data_angle_range_Static_wrapped[indices_1D, :])
    
    # Transpose and flip (MATLAB: mag_data' then flipud)
    mag_data_dynamic = mag_data_dynamic.T
    mag_data_static = mag_data_static.T
    mag_data_dynamic = np.flipud(mag_data_dynamic)
    mag_data_static = np.flipud(mag_data_static)
    
    if PLOT_ON:
        log_plot = LOG
        if STATIC_ONLY:
            if log_plot:
                data_to_plot = (mag_data_static) ** 0.4
            else:
                data_to_plot = np.abs(mag_data_static)
        else:
            if log_plot:
                data_to_plot = (mag_data_dynamic) ** 0.4
            else:
                data_to_plot = np.abs(mag_data_dynamic)
        
        # MATLAB uses surf(y_axis, x_axis, data) with view(2)
        # This is equivalent to pcolormesh
        plt.pcolormesh(y_axis, x_axis, data_to_plot, shading='auto', cmap='jet')
        plt.colorbar()
        plt.xlabel('meters')
        plt.ylabel('meters')
        plt.title('Range/Azimuth Heat Map')
    
    return mag_data_static, mag_data_dynamic, y_axis, x_axis


def plot_3d_point_cloud(
    xyz: np.ndarray,
    velocity: Optional[np.ndarray] = None,
    snr: Optional[np.ndarray] = None,
    xlim: Tuple[float, float] = (-20, 20),
    ylim: Tuple[float, float] = (1, 50),
    zlim: Tuple[float, float] = (-5, 5),
    view_angle: Tuple[float, float] = (-9, 15),
    title: str = '3D Point Cloud',
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot 3D point cloud from radar detections.
    
    Args:
        xyz: Point coordinates, shape (N, 3) or (N, 4+) where columns are [x, y, z, ...]
        velocity: Velocity values for coloring, shape (N,)
        snr: SNR values for point sizing, shape (N,)
        xlim: X-axis limits
        ylim: Y-axis limits
        zlim: Z-axis limits
        view_angle: View angle (azimuth, elevation)
        title: Plot title
        ax: Existing axes to plot on
        
    Returns:
        Matplotlib Axes object
    """
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    
    # Color by velocity if provided
    if velocity is not None:
        c = velocity
        cmap = 'RdBu'
    else:
        c = z
        cmap = 'viridis'
    
    # Size by SNR if provided
    if snr is not None:
        s = np.clip(snr, 10, 100)
    else:
        s = 45
    
    scatter = ax.scatter(x, y, z, c=c, s=s, cmap=cmap, alpha=0.8)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.view_init(elev=view_angle[1], azim=view_angle[0])
    
    plt.colorbar(scatter, ax=ax, label='Velocity (m/s)' if velocity is not None else 'Z (m)')
    
    return ax


def plot_range_doppler(
    range_doppler_map: np.ndarray,
    range_bin_size: float,
    doppler_bin_size: float,
    log_scale: bool = True,
    title: str = 'Range-Doppler Map',
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot range-Doppler map.
    
    Args:
        range_doppler_map: 2D array (num_range_bins, num_doppler_bins)
        range_bin_size: Range bin size in meters
        doppler_bin_size: Doppler bin size in m/s
        log_scale: Use log scale
        title: Plot title
        ax: Existing axes
        
    Returns:
        Matplotlib Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    num_range, num_doppler = range_doppler_map.shape
    
    # Create extent
    range_axis = np.arange(num_range) * range_bin_size
    doppler_axis = (np.arange(num_doppler) - num_doppler // 2) * doppler_bin_size
    extent = [doppler_axis[0], doppler_axis[-1], range_axis[0], range_axis[-1]]
    
    if log_scale:
        data = 10 * np.log10(np.abs(range_doppler_map) ** 2 + 1e-10)
    else:
        data = np.abs(range_doppler_map)
    
    im = ax.imshow(data, aspect='auto', origin='lower', extent=extent, cmap='jet')
    plt.colorbar(im, ax=ax, label='Power (dB)' if log_scale else 'Magnitude')
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Range (m)')
    ax.set_title(title)
    
    return ax


def plot_range_profile(
    range_fft: np.ndarray,
    range_bin_size: float,
    detection_indices: Optional[np.ndarray] = None,
    log_scale: bool = True,
    title: str = 'Range Profile',
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot 1D range profile.
    
    Args:
        range_fft: Range FFT output, shape (num_range_bins,) or (num_range_bins, num_rx)
        range_bin_size: Range bin size in meters
        detection_indices: Indices of detected targets to highlight
        log_scale: Use log scale
        title: Plot title
        ax: Existing axes
        
    Returns:
        Matplotlib Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    if range_fft.ndim == 1:
        range_fft = range_fft.reshape(-1, 1)
    
    num_range = range_fft.shape[0]
    range_axis = np.arange(num_range) * range_bin_size
    
    # Sum across RX channels
    power = np.sum(np.abs(range_fft) ** 2, axis=1)
    
    if log_scale:
        power_db = 10 * np.log10(power + 1e-10)
        ax.plot(range_axis, power_db)
        ax.set_ylabel('Power (dB)')
    else:
        ax.plot(range_axis, power)
        ax.set_ylabel('Magnitude')
    
    if detection_indices is not None:
        if log_scale:
            ax.plot(range_axis[detection_indices], 
                   10 * np.log10(power[detection_indices] + 1e-10),
                   'ro', markersize=8, label='Detections')
        else:
            ax.plot(range_axis[detection_indices], 
                   power[detection_indices],
                   'ro', markersize=8, label='Detections')
        ax.legend()
    
    ax.set_xlabel('Range (m)')
    ax.set_title(title)
    ax.grid(True)
    
    return ax
