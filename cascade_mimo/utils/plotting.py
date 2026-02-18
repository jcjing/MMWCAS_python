"""
plotting.py

Visualization utilities for mmWave radar data.

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/

Converted from MATLAB: plot_range_azimuth_2D.m
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from matplotlib.colors import LinearSegmentedColormap


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

# NOTE: This is a copy of "parula" from MATLAB. I don't claim any rights to this data, but The Mathworks does. Consult them and/or a lawyer if you want to use it.
cm_data = [
    [0.2422, 0.1504, 0.6603],
    [0.2444, 0.1534, 0.6728],
    [0.2464, 0.1569, 0.6847],
    [0.2484, 0.1607, 0.6961],
    [0.2503, 0.1648, 0.7071],
    [0.2522, 0.1689, 0.7179],
    [0.254, 0.1732, 0.7286],
    [0.2558, 0.1773, 0.7393],
    [0.2576, 0.1814, 0.7501],
    [0.2594, 0.1854, 0.761],
    [0.2611, 0.1893, 0.7719],
    [0.2628, 0.1932, 0.7828],
    [0.2645, 0.1972, 0.7937],
    [0.2661, 0.2011, 0.8043],
    [0.2676, 0.2052, 0.8148],
    [0.2691, 0.2094, 0.8249],
    [0.2704, 0.2138, 0.8346],
    [0.2717, 0.2184, 0.8439],
    [0.2729, 0.2231, 0.8528],
    [0.274, 0.228, 0.8612],
    [0.2749, 0.233, 0.8692],
    [0.2758, 0.2382, 0.8767],
    [0.2766, 0.2435, 0.884],
    [0.2774, 0.2489, 0.8908],
    [0.2781, 0.2543, 0.8973],
    [0.2788, 0.2598, 0.9035],
    [0.2794, 0.2653, 0.9094],
    [0.2798, 0.2708, 0.915],
    [0.2802, 0.2764, 0.9204],
    [0.2806, 0.2819, 0.9255],
    [0.2809, 0.2875, 0.9305],
    [0.2811, 0.293, 0.9352],
    [0.2813, 0.2985, 0.9397],
    [0.2814, 0.304, 0.9441],
    [0.2814, 0.3095, 0.9483],
    [0.2813, 0.315, 0.9524],
    [0.2811, 0.3204, 0.9563],
    [0.2809, 0.3259, 0.96],
    [0.2807, 0.3313, 0.9636],
    [0.2803, 0.3367, 0.967],
    [0.2798, 0.3421, 0.9702],
    [0.2791, 0.3475, 0.9733],
    [0.2784, 0.3529, 0.9763],
    [0.2776, 0.3583, 0.9791],
    [0.2766, 0.3638, 0.9817],
    [0.2754, 0.3693, 0.984],
    [0.2741, 0.3748, 0.9862],
    [0.2726, 0.3804, 0.9881],
    [0.271, 0.386, 0.9898],
    [0.2691, 0.3916, 0.9912],
    [0.267, 0.3973, 0.9924],
    [0.2647, 0.403, 0.9935],
    [0.2621, 0.4088, 0.9946],
    [0.2591, 0.4145, 0.9955],
    [0.2556, 0.4203, 0.9965],
    [0.2517, 0.4261, 0.9974],
    [0.2473, 0.4319, 0.9983],
    [0.2424, 0.4378, 0.9991],
    [0.2369, 0.4437, 0.9996],
    [0.2311, 0.4497, 0.9995],
    [0.225, 0.4559, 0.9985],
    [0.2189, 0.462, 0.9968],
    [0.2128, 0.4682, 0.9948],
    [0.2066, 0.4743, 0.9926],
    [0.2006, 0.4803, 0.9906],
    [0.195, 0.4861, 0.9887],
    [0.1903, 0.4919, 0.9867],
    [0.1869, 0.4975, 0.9844],
    [0.1847, 0.503, 0.9819],
    [0.1831, 0.5084, 0.9793],
    [0.1818, 0.5138, 0.9766],
    [0.1806, 0.5191, 0.9738],
    [0.1795, 0.5244, 0.9709],
    [0.1785, 0.5296, 0.9677],
    [0.1778, 0.5349, 0.9641],
    [0.1773, 0.5401, 0.9602],
    [0.1768, 0.5452, 0.956],
    [0.1764, 0.5504, 0.9516],
    [0.1755, 0.5554, 0.9473],
    [0.174, 0.5605, 0.9432],
    [0.1716, 0.5655, 0.9393],
    [0.1686, 0.5705, 0.9357],
    [0.1649, 0.5755, 0.9323],
    [0.161, 0.5805, 0.9289],
    [0.1573, 0.5854, 0.9254],
    [0.154, 0.5902, 0.9218],
    [0.1513, 0.595, 0.9182],
    [0.1492, 0.5997, 0.9147],
    [0.1475, 0.6043, 0.9113],
    [0.1461, 0.6089, 0.908],
    [0.1446, 0.6135, 0.905],
    [0.1429, 0.618, 0.9022],
    [0.1408, 0.6226, 0.8998],
    [0.1383, 0.6272, 0.8975],
    [0.1354, 0.6317, 0.8953],
    [0.1321, 0.6363, 0.8932],
    [0.1288, 0.6408, 0.891],
    [0.1253, 0.6453, 0.8887],
    [0.1219, 0.6497, 0.8862],
    [0.1185, 0.6541, 0.8834],
    [0.1152, 0.6584, 0.8804],
    [0.1119, 0.6627, 0.877],
    [0.1085, 0.6669, 0.8734],
    [0.1048, 0.671, 0.8695],
    [0.1009, 0.675, 0.8653],
    [0.0964, 0.6789, 0.8609],
    [0.0914, 0.6828, 0.8562],
    [0.0855, 0.6865, 0.8513],
    [0.0789, 0.6902, 0.8462],
    [0.0713, 0.6938, 0.8409],
    [0.0628, 0.6972, 0.8355],
    [0.0535, 0.7006, 0.8299],
    [0.0433, 0.7039, 0.8242],
    [0.0328, 0.7071, 0.8183],
    [0.0234, 0.7103, 0.8124],
    [0.0155, 0.7133, 0.8064],
    [0.0091, 0.7163, 0.8003],
    [0.0046, 0.7192, 0.7941],
    [0.0019, 0.722, 0.7878],
    [0.0009, 0.7248, 0.7815],
    [0.0018, 0.7275, 0.7752],
    [0.0046, 0.7301, 0.7688],
    [0.0094, 0.7327, 0.7623],
    [0.0162, 0.7352, 0.7558],
    [0.0253, 0.7376, 0.7492],
    [0.0369, 0.74, 0.7426],
    [0.0504, 0.7423, 0.7359],
    [0.0638, 0.7446, 0.7292],
    [0.077, 0.7468, 0.7224],
    [0.0899, 0.7489, 0.7156],
    [0.1023, 0.751, 0.7088],
    [0.1141, 0.7531, 0.7019],
    [0.1252, 0.7552, 0.695],
    [0.1354, 0.7572, 0.6881],
    [0.1448, 0.7593, 0.6812],
    [0.1532, 0.7614, 0.6741],
    [0.1609, 0.7635, 0.6671],
    [0.1678, 0.7656, 0.6599],
    [0.1741, 0.7678, 0.6527],
    [0.1799, 0.7699, 0.6454],
    [0.1853, 0.7721, 0.6379],
    [0.1905, 0.7743, 0.6303],
    [0.1954, 0.7765, 0.6225],
    [0.2003, 0.7787, 0.6146],
    [0.2061, 0.7808, 0.6065],
    [0.2118, 0.7828, 0.5983],
    [0.2178, 0.7849, 0.5899],
    [0.2244, 0.7869, 0.5813],
    [0.2318, 0.7887, 0.5725],
    [0.2401, 0.7905, 0.5636],
    [0.2491, 0.7922, 0.5546],
    [0.2589, 0.7937, 0.5454],
    [0.2695, 0.7951, 0.536],
    [0.2809, 0.7964, 0.5266],
    [0.2929, 0.7975, 0.517],
    [0.3052, 0.7985, 0.5074],
    [0.3176, 0.7994, 0.4975],
    [0.3301, 0.8002, 0.4876],
    [0.3424, 0.8009, 0.4774],
    [0.3548, 0.8016, 0.4669],
    [0.3671, 0.8021, 0.4563],
    [0.3795, 0.8026, 0.4454],
    [0.3921, 0.8029, 0.4344],
    [0.405, 0.8031, 0.4233],
    [0.4184, 0.803, 0.4122],
    [0.4322, 0.8028, 0.4013],
    [0.4463, 0.8024, 0.3904],
    [0.4608, 0.8018, 0.3797],
    [0.4753, 0.8011, 0.3691],
    [0.4899, 0.8002, 0.3586],
    [0.5044, 0.7993, 0.348],
    [0.5187, 0.7982, 0.3374],
    [0.5329, 0.797, 0.3267],
    [0.547, 0.7957, 0.3159],
    [0.5609, 0.7943, 0.305],
    [0.5748, 0.7929, 0.2941],
    [0.5886, 0.7913, 0.2833],
    [0.6024, 0.7896, 0.2726],
    [0.6161, 0.7878, 0.2622],
    [0.6297, 0.7859, 0.2521],
    [0.6433, 0.7839, 0.2423],
    [0.6567, 0.7818, 0.2329],
    [0.6701, 0.7796, 0.2239],
    [0.6833, 0.7773, 0.2155],
    [0.6963, 0.775, 0.2075],
    [0.7091, 0.7727, 0.1998],
    [0.7218, 0.7703, 0.1924],
    [0.7344, 0.7679, 0.1852],
    [0.7468, 0.7654, 0.1782],
    [0.759, 0.7629, 0.1717],
    [0.771, 0.7604, 0.1658],
    [0.7829, 0.7579, 0.1608],
    [0.7945, 0.7554, 0.157],
    [0.806, 0.7529, 0.1546],
    [0.8172, 0.7505, 0.1535],
    [0.8281, 0.7481, 0.1536],
    [0.8389, 0.7457, 0.1546],
    [0.8495, 0.7435, 0.1564],
    [0.86, 0.7413, 0.1587],
    [0.8703, 0.7392, 0.1615],
    [0.8804, 0.7372, 0.165],
    [0.8903, 0.7353, 0.1695],
    [0.9, 0.7336, 0.1749],
    [0.9093, 0.7321, 0.1815],
    [0.9184, 0.7308, 0.189],
    [0.9272, 0.7298, 0.1973],
    [0.9357, 0.729, 0.2061],
    [0.944, 0.7285, 0.2151],
    [0.9523, 0.7284, 0.2237],
    [0.9606, 0.7285, 0.2312],
    [0.9689, 0.7292, 0.2373],
    [0.977, 0.7304, 0.2418],
    [0.9842, 0.733, 0.2446],
    [0.99, 0.7365, 0.2429],
    [0.9946, 0.7407, 0.2394],
    [0.9966, 0.7458, 0.2351],
    [0.9971, 0.7513, 0.2309],
    [0.9972, 0.7569, 0.2267],
    [0.9971, 0.7626, 0.2224],
    [0.9969, 0.7683, 0.2181],
    [0.9966, 0.774, 0.2138],
    [0.9962, 0.7798, 0.2095],
    [0.9957, 0.7856, 0.2053],
    [0.9949, 0.7915, 0.2012],
    [0.9938, 0.7974, 0.1974],
    [0.9923, 0.8034, 0.1939],
    [0.9906, 0.8095, 0.1906],
    [0.9885, 0.8156, 0.1875],
    [0.9861, 0.8218, 0.1846],
    [0.9835, 0.828, 0.1817],
    [0.9807, 0.8342, 0.1787],
    [0.9778, 0.8404, 0.1757],
    [0.9748, 0.8467, 0.1726],
    [0.972, 0.8529, 0.1695],
    [0.9694, 0.8591, 0.1665],
    [0.9671, 0.8654, 0.1636],
    [0.9651, 0.8716, 0.1608],
    [0.9634, 0.8778, 0.1582],
    [0.9619, 0.884, 0.1557],
    [0.9608, 0.8902, 0.1532],
    [0.9601, 0.8963, 0.1507],
    [0.9596, 0.9023, 0.148],
    [0.9595, 0.9084, 0.145],
    [0.9597, 0.9143, 0.1418],
    [0.9601, 0.9203, 0.1382],
    [0.9608, 0.9262, 0.1344],
    [0.9618, 0.932, 0.1304],
    [0.9629, 0.9379, 0.1261],
    [0.9642, 0.9437, 0.1216],
    [0.9657, 0.9494, 0.1168],
    [0.9674, 0.9552, 0.1116],
    [0.9692, 0.9609, 0.1061],
    [0.9711, 0.9667, 0.1001],
    [0.973, 0.9724, 0.0938],
    [0.9749, 0.9782, 0.0872],
    [0.9769, 0.9839, 0.0805]]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)
