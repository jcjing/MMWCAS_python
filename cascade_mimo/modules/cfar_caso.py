"""
cfar_caso.py

CFAR Cell-Averaging Smallest Of (CASO) detection.

Copyright (C) 2018-2020 Texas Instruments Incorporated - http://www.ti.com/

Converted from MATLAB: CFAR_CASO.m, CFAR_CASO_Range.m, CFAR_CASO_Doppler_overlap.m
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class CFARConfig:
    """CFAR detection configuration."""
    # Detection method
    detectMethod: int = 1  # 1 = dual-pass CASO-CFAR (range then doppler)
    
    # Number of antennas
    numAntenna: int = 192
    
    # CFAR window parameters [range, doppler]
    refWinSize: List[int] = field(default_factory=lambda: [8, 4])  # training cells
    guardWinSize: List[int] = field(default_factory=lambda: [8, 0])  # guard cells
    K0: List[float] = field(default_factory=lambda: [5.0, 3.0])  # threshold scale (LINEAR, not dB!)
    
    # Max detection parameters
    maxEnable: int = 0
    
    # Range/velocity parameters (calculated from radar config)
    rangeBinSize: float = 0.0  # meters per bin - MUST BE SET
    velocityBinSize: float = 0.0  # m/s per bin - MUST BE SET
    dopplerFFTSize: int = 64
    
    # Power threshold
    powerThre: float = 0.0
    
    # Cells to discard at edges
    discardCellLeft: int = 10  # positive frequency range bins
    discardCellRight: int = 20  # negative frequency range bins
    
    # TDM MIMO parameters
    numRxAnt: int = 4
    TDM_MIMO_numTX: int = 12
    antenna_azimuthonly: List[int] = field(default_factory=list)
    
    # Velocity extension parameters
    applyVmaxExtend: int = 0
    minDisApplyVmaxExtend: float = 10.0  # meters
    overlapAntenna_ID: np.ndarray = field(default_factory=lambda: np.array([]))
    overlapAntenna_ID_2TX: np.ndarray = field(default_factory=lambda: np.array([]))
    overlapAntenna_ID_3TX: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class Detection:
    """Single CFAR detection."""
    rangeInd: int = 0
    dopplerInd: int = 0
    dopplerInd_org: int = 0
    range: float = 0.0
    doppler: float = 0.0
    doppler_corr: float = 0.0
    doppler_corr_overlap: float = 0.0
    doppler_corr_FFT: float = 0.0
    estSNR: float = 0.0
    noise_var: float = 0.0
    bin_val: np.ndarray = field(default_factory=lambda: np.array([]))
    peakVal: float = 0.0


class CFARDetector:
    """
    CFAR-CASO (Cell Averaging Smallest Of) detector.
    Matches MATLAB implementation exactly.
    """
    
    def __init__(self, config: CFARConfig):
        """Initialize CFAR detector with configuration."""
        self.config = config
    
    def CFAR_CASO_Range(self, sig_integrate: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """
        CFAR detection along range dimension.
        Exact MATLAB translation.
        
        Args:
            sig_integrate: Range-Doppler power map (range x doppler)
            
        Returns:
            Tuple of (N_obj, Ind_obj, noise_obj, CFAR_SNR)
        """
        cellNum = self.config.refWinSize[0]
        gapNum = self.config.guardWinSize[0]
        K0 = self.config.K0[0]  # LINEAR scale, not dB!
        
        M_samp, N_pul = sig_integrate.shape
        
        gaptot = gapNum + cellNum
        N_obj = 0
        Ind_obj = []
        noise_obj = []
        CFAR_SNR = []
        
        discardCellLeft = self.config.discardCellLeft
        discardCellRight = self.config.discardCellRight
        
        # For each Doppler bin
        for k in range(N_pul):
            sigv = sig_integrate[:, k]
            vec = sigv[discardCellLeft:M_samp-discardCellRight]
            
            # Wrap edges
            vecLeft = vec[:gaptot]
            vecRight = vec[-gaptot:]
            vec = np.concatenate([vecLeft, vec, vecRight])
            
            # Process each range bin
            for j in range(M_samp - discardCellLeft - discardCellRight):
                # Cell indices (MATLAB 1-indexed, converted to 0-indexed)
                cellInda = np.arange(j - gaptot, j - gapNum) + gaptot
                cellIndb = np.arange(j + gapNum + 1, j + gaptot + 1) + gaptot
                
                # CASO: take minimum of left and right averages
                cellave1a = np.sum(vec[cellInda]) / cellNum
                cellave1b = np.sum(vec[cellIndb]) / cellNum
                cellave1 = min(cellave1a, cellave1b)
                
                # Detection logic
                if self.config.maxEnable == 1:
                    # Check if local maximum
                    cellInd = np.concatenate([cellInda, cellIndb])
                    maxInCell = np.max(vec[cellInd])
                    if vec[j + gaptot] > K0 * cellave1 and vec[j + gaptot] >= maxInCell:
                        N_obj += 1
                        Ind_obj.append([j + discardCellLeft, k])
                        noise_obj.append(cellave1)
                        CFAR_SNR.append(vec[j + gaptot] / cellave1)
                else:
                    if vec[j + gaptot] > K0 * cellave1:
                        N_obj += 1
                        Ind_obj.append([j + discardCellLeft, k])
                        noise_obj.append(cellave1)
                        CFAR_SNR.append(vec[j + gaptot] / cellave1)
        
        Ind_obj = np.array(Ind_obj) if len(Ind_obj) > 0 else np.array([]).reshape(0, 2)
        noise_obj = np.array(noise_obj)
        CFAR_SNR = np.array(CFAR_SNR)
        
        return N_obj, Ind_obj, noise_obj, CFAR_SNR
    
    def CFAR_CASO_Doppler_overlap(self, Ind_obj_Rag: np.ndarray, sigCpml: np.ndarray, 
                                   sig_integ: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        CFAR detection along Doppler dimension for cells that passed range detection.
        Exact MATLAB translation.
        
        Args:
            Ind_obj_Rag: Range detection indices (N x 2: [range_idx, doppler_idx])
            sigCpml: Complex signal (range x doppler x antenna)
            sig_integ: Integrated power (range x doppler)
            
        Returns:
            Tuple of (N_obj, Ind_obj)
        """
        maxEnable = self.config.maxEnable
        cellNum = self.config.refWinSize[1]
        gapNum = self.config.guardWinSize[1]
        K0 = self.config.K0[1]  # LINEAR scale
        
        M_samp, N_pul = sig_integ.shape
        gaptot = gapNum + cellNum
        
        # Extract unique range bins with detections
        detected_Rag_Cell = np.unique(Ind_obj_Rag[:, 0])
        sig = sig_integ[detected_Rag_Cell, :]
        
        M_samp = sig.shape[0]
        N_pul = sig.shape[1]
        
        N_obj = 0
        Ind_obj = []
        
        # For each detected range bin
        for k in range(M_samp):
            detected_Rag_Cell_i = detected_Rag_Cell[k]
            
            # Find Doppler indices detected in range CFAR for this range bin
            ind1 = np.where(Ind_obj_Rag[:, 0] == detected_Rag_Cell_i)[0]
            indR = Ind_obj_Rag[ind1, 1]
            
            # Wrap vector for circular Doppler processing
            sigv = sig[k, :]
            vec = np.zeros(N_pul + gaptot * 2)
            vec[:gaptot] = sigv[-gaptot:]
            vec[gaptot:N_pul + gaptot] = sigv
            vec[N_pul + gaptot:] = sigv[:gaptot]
            
            # Process each Doppler bin
            ind_loc_all = []
            ind_loc_Dop = []
            
            for j in range(gaptot, N_pul + gaptot):
                j0 = j - gaptot
                
                cellInda = np.arange(j - gaptot, j - gapNum)
                cellIndb = np.arange(j + gapNum + 1, j + gaptot + 1)
                
                # CASO: minimum of left and right
                cellave1a = np.sum(vec[cellInda]) / cellNum
                cellave1b = np.sum(vec[cellIndb]) / cellNum
                cellave1 = min(cellave1a, cellave1b)
                
                # Detection condition
                if maxEnable == 1:
                    cellInd = np.concatenate([cellInda, cellIndb])
                    maxInCell = np.max(vec[cellInd])
                    condition = (vec[j] > K0 * cellave1) and (vec[j] > maxInCell)
                else:
                    condition = vec[j] > K0 * cellave1
                
                # Check if this overlaps with range detection
                if condition and (j0 in indR):
                    ind_loc_all.append(detected_Rag_Cell_i)
                    ind_loc_Dop.append(j0)
            
            # Add detections, avoiding duplicates
            if len(ind_loc_all) > 0:
                ind_obj_0 = np.column_stack([ind_loc_all, ind_loc_Dop])
                
                if len(Ind_obj) == 0:
                    Ind_obj = ind_obj_0.tolist()
                else:
                    # Check for duplicates
                    Ind_obj_sum = [r[0] + 10000 * r[1] for r in Ind_obj]
                    for ii in range(len(ind_loc_all)):
                        ind_sum = ind_loc_all[ii] + 10000 * ind_loc_Dop[ii]
                        if ind_sum not in Ind_obj_sum:
                            Ind_obj.append([ind_loc_all[ii], ind_loc_Dop[ii]])
        
        N_obj = len(Ind_obj)
        Ind_obj = np.array(Ind_obj) if N_obj > 0 else np.array([]).reshape(0, 2)
        
        return N_obj, Ind_obj
    
    def datapath(self, input_signal: np.ndarray) -> List[Detection]:
        """
        Main CFAR datapath matching MATLAB.
        
        Args:
            input_signal: 3D array (range x doppler x antenna)
            
        Returns:
            List of Detection objects
        """
        # Non-coherent integration across antennas (power sum)
        sig_integrate = np.sum(np.abs(input_signal) ** 2, axis=2) + 1
        
        detection_results = []
        
        if self.config.detectMethod == 1:  # Dual-pass CASO-CFAR
            # First pass: Range CFAR
            N_obj_Rag, Ind_obj_Rag, noise_obj, CFAR_SNR = self.CFAR_CASO_Range(sig_integrate)
            
            if N_obj_Rag > 0:
                # Second pass: Doppler CFAR
                N_obj, Ind_obj = self.CFAR_CASO_Doppler_overlap(Ind_obj_Rag, input_signal, sig_integrate)
                
                # Build aggregate noise from first pass
                noise_obj_agg = np.zeros(N_obj)
                for i_obj in range(N_obj):
                    indx1R = Ind_obj[i_obj, 0]
                    indx1D = Ind_obj[i_obj, 1]
                    
                    # Find matching detection in range pass
                    ind2R = np.where(Ind_obj_Rag[:, 0] == indx1R)[0]
                    ind2D = np.where(Ind_obj_Rag[ind2R, 1] == indx1D)[0]
                    if len(ind2D) > 0:
                        noiseInd = ind2R[ind2D[0]]
                        noise_obj_agg[i_obj] = noise_obj[noiseInd]
                
                # Create Detection objects
                for i_obj in range(N_obj):
                    xind = Ind_obj[i_obj, 0]
                    
                    det = Detection()
                    det.rangeInd = Ind_obj[i_obj, 0]
                    det.range = det.rangeInd * self.config.rangeBinSize
                    
                    dopplerInd = Ind_obj[i_obj, 1]
                    det.dopplerInd_org = dopplerInd
                    det.dopplerInd = dopplerInd
                    
                    # Velocity estimation (Doppler centered at FFTSize/2)
                    det.doppler = (dopplerInd - self.config.dopplerFFTSize / 2) * self.config.velocityBinSize
                    det.doppler_corr = det.doppler
                    det.doppler_corr_overlap = det.doppler
                    det.doppler_corr_FFT = det.doppler
                    
                    det.noise_var = noise_obj_agg[i_obj]
                    
                    # Extract bin values for all antennas
                    bin_val = input_signal[xind, dopplerInd, :].reshape(-1)
                    det.bin_val = bin_val
                    
                    # Estimate SNR
                    det.estSNR = np.sum(np.abs(bin_val) ** 2) / max(det.noise_var, 1e-10)
                    det.peakVal = sig_integrate[xind, dopplerInd]
                    
                    detection_results.append(det)
        
        return detection_results


def cfar_2d(range_doppler: np.ndarray,
            guard_range: int = 4, train_range: int = 8,
            guard_doppler: int = 2, train_doppler: int = 4,
            threshold_db: float = 10.0) -> np.ndarray:
    """
    Simple 2D CFAR function.
    
    Args:
        range_doppler: 2D range-Doppler map
        guard_range: Guard cells in range
        train_range: Training cells in range
        guard_doppler: Guard cells in Doppler
        train_doppler: Training cells in Doppler
        threshold_db: Detection threshold in dB
        
    Returns:
        Boolean detection mask
    """
    config = CFARConfig(
        guardCellsRange=guard_range,
        trainCellsRange=train_range,
        guardCellsDoppler=guard_doppler,
        trainCellsDoppler=train_doppler,
        thresholdScaleRange=threshold_db,
        thresholdScaleDoppler=threshold_db,
        peakGrouping=True
    )
    detector = CFARDetector(config)
    
    # Add dummy antenna dimension
    if range_doppler.ndim == 2:
        range_doppler = range_doppler[:, :, np.newaxis]
    
    detections = detector.detect(range_doppler)
    
    # Convert to mask
    mask = np.zeros(range_doppler.shape[:2], dtype=bool)
    for det in detections:
        mask[det.rangeInd, det.dopplerInd] = True
    
    return mask
