# TI mmWave Radar Python Examples

Python signal processing examples for Texas Instruments mmWave radar platforms.

Converted from the official TI MATLAB examples.

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** numpy, scipy, matplotlib

## Directory Structure

```
PythonExamples/
├── common/                      # Shared constants & configuration
├── cascade_mimo/                # 4-chip cascade MIMO radar
│   ├── modules/                 # Signal processing modules
│   ├── utils/                   # Data parsing & visualization
│   └── txbf/                    # TX beamforming utilities
├── cascade_txbf/                # TX beamforming example
│   └── chirp_profiles/          # LRR, SMRR, USRR configs
└── singlechip_raw_data_reader/  # Single-chip DCA1000 reader
```

## Quick Start

### MIMO Signal Processing

```python
from cascade_mimo import run_mimo_signal_processing

run_mimo_signal_processing(
    data_folder_test="path/to/radar/data",
    data_folder_calib="path/to/calibration.mat",
    plot_on=True,
    save_output=True,
)
```

### TX Phase Calibration

```python
from cascade_mimo.cascade_tx_phase_calibration import run_tx_phase_calibration

run_tx_phase_calibration(
    data_folder_calib_data_path="path/to/calib/data",
    target_range=5.0,
)
```

### TX Beamforming

```python
from cascade_mimo.txbf import calculate_phase_settings

result = calculate_phase_settings(steer_angle_deg=15.0, center_freq_ghz=77.0)
print(f"Phase codes: {result['phase_shifter_codes']}")
```

### Single-Chip Data Reader

```python
from singlechip_raw_data_reader import read_raw_data

reader = read_raw_data(
    "path/to/setup.json",
    raw_data_file="output_raw.mat",
    debug_plot=True,
)
```

## Command Line Usage

```bash
# MIMO signal processing
python -m cascade_mimo.cascade_mimo_signal_processing \
    --data_folder /path/to/data --calib_file /path/to/calib.mat

# TX beamforming
python -m cascade_txbf.cascade_txbf_signal_processing \
    --data_folder /path/to/data --steer_angle 15.0

# Single-chip reader
python -m singlechip_raw_data_reader.raw_data_reader \
    --setup_json /path/to/setup.json --debug
```

## Supported Platforms

- **4-Chip Cascade:** AWR2243 cascade (MMWCAS-RF-EVM + MMWCAS-DSP-EVM)
- **Single-Chip:** AWR1642, IWR1642, AWR1843, IWR1843, IWR6843, AWR2243

## Output Formats

All scripts save results in both formats for maximum compatibility:
- `.mat` - MATLAB compatible (scipy.io.savemat)
- `.npz` - NumPy native (np.savez_compressed)

## License

Copyright (C) 2018-2020 Texas Instruments Incorporated

See individual source files for full license terms.
