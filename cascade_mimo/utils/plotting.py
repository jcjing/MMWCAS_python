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
    range_bin_size: float,
    doppler_fft_out: np.ndarray,
    num_tx: int,
    num_rx: int,
    antenna_azimuth_only: Optional[np.ndarray] = None,
    log_scale: bool = True,
    static_only: bool = True,
    plot_on: bool = True,
    min_range_bin_keep: int = 5,
    right_range_bin_discard: int = 20,
    azimuth_fft_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate and optionally plot range-azimuth 2D heatmap.
    
    Args:
        range_bin_size: Size of each range bin in meters
        doppler_fft_out: Doppler FFT output, shape (num_range_bins, num_doppler_bins, num_virtual_antennas)
        num_tx: Number of TX antennas
        num_rx: Number of RX antennas
        antenna_azimuth_only: Antenna indices for azimuth-only processing
        log_scale: Use log10 scale for magnitude
        static_only: Process only static (zero-Doppler) targets
        plot_on: Display the plot
        min_range_bin_keep: Minimum range bin to keep (skip close-range clutter)
        right_range_bin_discard: Number of range bins to discard from end
        azimuth_fft_size: FFT size for azimuth processing
        
    Returns:
        Tuple of (mag_data_static, mag_data_dynamic, y_axis, x_axis)
    """
    num_range_bins = doppler_fft_out.shape[0]
    num_doppler_bins = doppler_fft_out.shape[1]
    num_virtual_antennas = doppler_fft_out.shape[2]
    
    # Default antenna indices if not provided
    if antenna_azimuth_only is None:
        antenna_azimuth_only = np.arange(min(num_virtual_antennas, num_rx))
    
    # Extract zero-Doppler bin for static targets
    zero_doppler_idx = num_doppler_bins // 2
    
    # Select antennas for azimuth processing
    data_for_azimuth = doppler_fft_out[:, zero_doppler_idx, antenna_azimuth_only]
    
    # Apply azimuth FFT
    azimuth_fft = np.fft.fftshift(
        np.fft.fft(data_for_azimuth, n=azimuth_fft_size, axis=1),
        axes=1
    )
    
    # Calculate magnitude
    if log_scale:
        mag_data_static = 10 * np.log10(np.abs(azimuth_fft) ** 2 + 1e-10)
    else:
        mag_data_static = np.abs(azimuth_fft)
    
    # Process dynamic targets (non-zero Doppler)
    if not static_only:
        # Sum magnitude across all Doppler bins except zero
        dynamic_data = np.sum(np.abs(doppler_fft_out[:, :, antenna_azimuth_only]) ** 2, axis=1)
        dynamic_data -= np.abs(doppler_fft_out[:, zero_doppler_idx, antenna_azimuth_only]) ** 2
        
        # Azimuth FFT for dynamic
        dynamic_azimuth = np.fft.fftshift(
            np.fft.fft(np.sqrt(dynamic_data), n=azimuth_fft_size, axis=1),
            axes=1
        )
        
        if log_scale:
            mag_data_dynamic = 10 * np.log10(np.abs(dynamic_azimuth) ** 2 + 1e-10)
        else:
            mag_data_dynamic = np.abs(dynamic_azimuth)
    else:
        mag_data_dynamic = np.zeros_like(mag_data_static)
    
    # Create axis arrays
    range_axis = np.arange(num_range_bins) * range_bin_size
    azimuth_bins = np.arange(azimuth_fft_size) - azimuth_fft_size // 2
    azimuth_angles = np.arcsin(azimuth_bins / azimuth_fft_size) * 180 / np.pi
    
    # Calculate x,y coordinates for plotting
    max_range = range_axis[-right_range_bin_discard] if right_range_bin_discard > 0 else range_axis[-1]
    y_axis = range_axis[min_range_bin_keep:-right_range_bin_discard] if right_range_bin_discard > 0 else range_axis[min_range_bin_keep:]
    x_axis = azimuth_angles
    
    # Trim data to valid range
    if right_range_bin_discard > 0:
        mag_data_static = mag_data_static[min_range_bin_keep:-right_range_bin_discard, :]
        mag_data_dynamic = mag_data_dynamic[min_range_bin_keep:-right_range_bin_discard, :]
    else:
        mag_data_static = mag_data_static[min_range_bin_keep:, :]
        mag_data_dynamic = mag_data_dynamic[min_range_bin_keep:, :]
    
    if plot_on:
        plt.figure(figsize=(10, 8))
        
        # Convert to Cartesian for display
        extent = [x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]]
        
        plt.imshow(mag_data_static, aspect='auto', origin='lower', extent=extent)
        plt.colorbar(label='Power (dB)' if log_scale else 'Magnitude')
        plt.xlabel('Azimuth Angle (degrees)')
        plt.ylabel('Range (m)')
        plt.title('Range-Azimuth Heatmap (Static Objects)')
        
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
