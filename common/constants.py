"""
constants.py

TI 4-Chip Cascade mmWave Radar Board Constants

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/
"""

import numpy as np

# ==============================================================================
# Physical Constants
# ==============================================================================

SPEED_OF_LIGHT = 3e8  # m/s

# ==============================================================================
# TI 4-Chip Cascade Board Antenna Configuration
# ==============================================================================

# 12 TX antenna azimuth positions on TI 4-chip cascade EVM
TI_CASCADE_TX_POSITION_AZI = np.array([11, 10, 9, 32, 28, 24, 20, 16, 12, 8, 4, 0])

# 12 TX antenna elevation positions on TI 4-chip cascade EVM
TI_CASCADE_TX_POSITION_ELE = np.array([6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# 16 RX antenna elevation positions (all zeros)
TI_CASCADE_RX_POSITION_ELE = np.zeros(16, dtype=int)

# 16 RX antenna azimuth positions
TI_CASCADE_RX_POSITION_AZI = np.array(
    list(range(11, 15)) + list(range(50, 54)) + list(range(46, 50)) + list(range(0, 4))
)

# RX channel order on TI 4-chip cascade EVM
TI_CASCADE_RX_ID = np.array([13, 14, 15, 16, 1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8])

# Antenna distance designed for this frequency (GHz)
TI_CASCADE_ANTENNA_DESIGN_FREQ = 76.8

# ==============================================================================
# Default Processing Parameters
# ==============================================================================

# Calibration interpolation factor (matches MATLAB default)
DEFAULT_CALIBRATION_INTERP = 7

# Scale factors for various calculations
SCALE_FACTORS = np.array([
    0.0625, 0.03125, 0.015625, 0.0078125,
    0.00390625, 0.001953125, 0.0009765625, 0.00048828125
]) * 4

# ==============================================================================
# Platform Identifiers
# ==============================================================================

PLATFORM_TI_4CHIP_CASCADE = 'TI_4Chip_CASCADE'
DATA_PLATFORM_TDA2 = 'TDA2'

# Supported mmWave devices
SUPPORTED_DEVICES = [
    'awr1642', 'iwr1642', 'awr1243', 'awr1443', 'iwr1443',
    'awr1843', 'iwr1843', 'iwr6843', 'awr2243'
]

# ==============================================================================
# Device Configuration Constants
# ==============================================================================

NUM_TX_PER_DEVICE = 3
NUM_RX_PER_DEVICE = 4
NUM_DEVICES_CASCADE = 4

# Phase shifter resolution (degrees per LSB)
PHASE_SHIFTER_RESOLUTION = 5.625
