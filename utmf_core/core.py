
"""
============================
 UTMF (Unified Temporal-Measurement Framework)
 Author: Jedi Markus Strive
 Version: UTMF-CORE v1.0
 Date: 04 dec 2025
============================
"""


import numpy as np
import pandas as pd
import h5py
import scipy.signal
import healpy as hp
from astropy.io import fits
import uproot
import pywt
from joblib import Parallel, delayed
from tqdm import tqdm
import gc
from scipy.stats import norm
import os
import warnings
import requests
from numba import jit
import dask.array as da
warnings.filterwarnings("ignore")

CONFIG = {
  
    "ligo_files": [
        "L-L1_GWOSC_O4a_16KHZ_R1-1368350720-4096.hdf5",
        "L-L1_GWOSC_04a_16KHZ_R1-1384779776-4096.hdf5",
        "L-L1_GWOSC_O4a_16KHZ_R1-1370202112-4096.hdf5",
        "L-L1_GWOSC_O4a_16KHZ_R1-1389420544-4096.hdf5"
    ],

    "cmb_files": [
        "COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits",
        "COM_CMB_IQU-smica_2048_R3.00_full.fits",
        "LFI_SkyMap_070_1024_R3.00_survey-1.fits"
    ],

    "mfdfa": {
        "q_values": np.arange(-8, 8.2, 0.2),
        "detrend_order": 0
    },

    'ligo': [
    {'sample_rate': 16384,
     'total_duration': 4096,
     'subset_duration': 4,
     'n_subsets': 100,
     'freq_range': [1, 30],
     'expected_D_f': 1.22,
     'sigma_D_f': 0.05,
     'min_std': 1e-5,
     'scales': np.logspace(np.log10(8), np.log10(4 * 16384 / 16), 20, dtype=np.int32),
     },
    {'sample_rate': 16384,
     'total_duration': 4096,
     'subset_duration': 4,
     'n_subsets': 100,
     'freq_range': [1, 30],
     'expected_D_f': 1.22,
     'sigma_D_f': 0.05,
     'min_std': 1e-5,
     'scales': np.logspace(np.log10(8), np.log10(4 * 16384 / 16), 20, dtype=np.int32),
     },
    {'sample_rate': 16384,
     'total_duration': 4096,
     'subset_duration': 4,
     'n_subsets': 100,
     'freq_range': [1, 30],
     'expected_D_f': 1.22,
     'sigma_D_f': 0.05,
     'min_std': 1e-5,
     'scales': np.logspace(np.log10(8), np.log10(4 * 16384 / 16), 20, dtype=np.int32),
     },
    {'sample_rate': 16384,
     'total_duration': 4096,
     'subset_duration': 4,
     'n_subsets': 100,
     'freq_range': [1, 30],
     'expected_D_f': 1.22,
     'sigma_D_f': 0.05,
     'min_std': 1e-5,
     'scales': np.logspace(np.log10(8), np.log10(4 * 16384 / 16), 20, dtype=np.int32),
     }
],

    "cmb": [
    {
        'nside': 2048,
        'subset_size': 100000,
        'n_subsets': 100,
        'expected_D_f': 1.19,
        'sigma_D_f': 0.04,
        'galactic_mask': True,
        'disc_radius': np.radians(20),
        'min_std': 1e-5,
        'fields': [0],
        'scales': np.array([2,4,8,16,32,64,128], dtype=np.int32),
    },
    {
        'nside': 2048,
        'subset_size': 100000,
        'n_subsets': 100,
        'expected_D_f': 1.19,
        'sigma_D_f': 0.04,
        'galactic_mask': True,
        'disc_radius': np.radians(20),
        'min_std': 1e-5,
        'fields': [0],
        'scales': np.array([2,4,8,16,32,64,128], dtype=np.int32),
    },
    {
        'nside': 1024,
        'subset_size': 45000,
        'n_subsets': 250,
        'expected_D_f': 1.19,
        'sigma_D_f': 0.04,
        'galactic_mask': True,
        'disc_radius': np.radians(30),
        'min_std': 1e-5,
        'fields': [0],
        'scales': np.array([2,4,8,16,32,64,128], dtype=np.int32),
    }
],
    "desi": {
        "subset_size": 3700,
        "n_subsets": 100,
        "expected_D_f": 1.19,
        "sigma_D_f": 0.04,
        "min_std": 1e-5,
        "columns": ["FLUX_Z", "FLUX_G", "FLUX_R"],
        "scales": np.array([2,4,8,16,32,64,128], dtype=np.int32)
    },

    "cern": {
        "subset_size": 5000,
        "n_subsets": 100,
        "expected_D_f": 1.19,
        "sigma_D_f": 0.04,
        "min_std": 1e-5,
        "tree": "mini",
        "columns": ["lep_pt", "lep_eta", "lep_phi"],
        "scales": np.array([2,4,8,16,32,64,128], dtype=np.int32)
    },

    "gaia": {
        "subset_size": 7500,
        "n_subsets": 100,
        "expected_D_f": 1.19,
        "sigma_D_f": 0.04,
        "min_std": 1e-5,
        "columns": ["RA_ICRS","DE_ICRS","pmRA","pmDE","Gmag"],
        "scales": np.array([1,2,4,6,8,16,32,64], dtype=np.int32)
    },

    "qrng": {
        "subset_size": 2560,
        "n_subsets": 100,
        "expected_D_f": 1.19,
        "sigma_D_f": 0.04,
        "min_std": 1e-5,
        "scales": np.array([2,4,8,16,32,64,128,256,512,1024], dtype=np.int32)
    }
}

@jit(nopython=True)
def polyfit_linear(x, y, lambda_reg=1e-5):
    # Linear polyfit with regularization for stability
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)
    denom = n * sum_x2 - sum_x**2 + lambda_reg
    if abs(denom) < 1e-10:
        return np.array([0.0, sum_y / n])
    m = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y * sum_x2 - sum_x * sum_xy) / denom
    return np.array([m, b])

@jit(nopython=True)
def polyval_linear(coeffs, x):
    return coeffs[0] * x + coeffs[1]

@jit(nopython=True)
def jedi_mfdfa(data, scales, q_values, detrend_order=0):
    n = len(data)
    fluct = np.zeros((len(q_values), len(scales)))
    rms_values = []
    slopes = np.zeros(len(q_values))

    for i in range(len(scales)):
        s = scales[i]
        segments = n // s
        if segments < 2:
            fluct[:, i] = np.nan
            continue
        rms = np.zeros(segments)
        valid_segments = 0
        for v in range(segments):
            segment = data[v*s:(v+1)*s]
            if len(segment) != s or np.std(segment) < 1e-10:
                continue
            x = np.arange(s, dtype=np.float64)
            if detrend_order > 0:
                try:
                    coeffs = polyfit_linear(x, segment)
                    trend = polyval_linear(coeffs, x)
                    detrended = segment - trend
                except:
                    detrended = segment - np.sum(segment) / s
            else:
                detrended = segment - np.sum(segment) / s
            sum_squares = 0.0
            for j in range(s):
                sum_squares += detrended[j]**2
            rms_val = np.sqrt(sum_squares / s + 1e-12)
            if rms_val > 1e-10:
                rms[valid_segments] = rms_val
                valid_segments += 1
        if valid_segments < 2:
            fluct[:, i] = np.nan
            continue
        rms = rms[:valid_segments]
        rms_values.append(rms)
        for j in range(len(q_values)):
            q = q_values[j]
            if q == 0:
                sum_log = 0.0
                count = 0
                for k in range(valid_segments):
                    if rms[k] > 1e-10:
                        sum_log += np.log(rms[k]**2 + 1e-12)
                        count += 1
                fluct[j, i] = np.exp(0.5 * (sum_log / count)) if count > 0 else np.nan
            else:
                sum_power = 0.0
                count = 0
                for k in range(valid_segments):
                    if rms[k] > 1e-10:
                        sum_power += (rms[k] + 1e-12)**q
                        count += 1
                fluct[j, i] = (sum_power / count)**(1/q) if count > 0 else np.nan
                if not np.isfinite(fluct[j, i]) or fluct[j, i] <= 0:
                    fluct[j, i] = np.nan
    valid_scales = np.sum(np.isfinite(fluct), axis=0)
    if np.max(valid_scales) < 4:
        return np.nan, np.full(len(q_values), np.nan), rms_values, fluct, slopes
    for j in range(len(q_values)):
        valid = np.isfinite(fluct[j, :]) & (fluct[j, :] > 0)
        if np.sum(valid) < 4:
            slopes[j] = np.nan
            continue
        coeffs = np.zeros(2)
        X = np.log(scales[valid])
        Y = np.log(fluct[j, valid] + 1e-12)
        n_valid = len(X)
        sum_x = np.sum(X)
        sum_y = np.sum(Y)
        sum_xy = np.sum(X * Y)
        sum_x2 = np.sum(X * X)
        denom = n_valid * sum_x2 - sum_x**2 + 1e-5
        if abs(denom) > 1e-10:
            coeffs[0] = (n_valid * sum_xy - sum_x * sum_y) / denom
            coeffs[1] = (sum_y * sum_x2 - sum_x * sum_xy) / denom
        slopes[j] = coeffs[0] if coeffs[0] > 0 else np.nan
    hq = slopes
    valid_hq = np.isfinite(hq)
    if np.sum(valid_hq) >= 2:
        tau = hq * q_values - 1
        alpha = np.diff(tau[valid_hq]) / np.diff(q_values[valid_hq])
        f_alpha = q_values[valid_hq][1:] * alpha - tau[valid_hq][1:]
        D_f = np.nanmean(alpha) if np.isfinite(alpha).any() else np.nan
    else:
        D_f = np.nan
    return D_f, hq, rms_values, fluct, slopes

def denoise_data(data, data_type, ligo_idx=None, cmb_idx=None):

    try:
        std_before = np.std(data)

        # --- LIGO ---------------------------------------------------------
        if data_type == 'ligo':            
            denoised = data * 1e16
            denoised = scipy.signal.savgol_filter(
                denoised,
                window_length=7,
                polyorder=1,
                mode='nearest'
            )

        # --- QRNG ---------------------------------------------------------
        elif data_type == 'qrng':
            denoised = (data - np.mean(data)) / (np.std(data) + 1e-10)

        # --- CMB, DESI, CERN, GAIA ---------------------------------------
        else:
            denoised = data / np.std(data)

        std_after = np.std(denoised)

        # Expected std checks (unchanged from v5)
        if data_type == 'ligo':
            min_std = CONFIG['ligo'][ligo_idx]['min_std']
        elif data_type == 'cmb':
            min_std = CONFIG['cmb'][cmb_idx]['min_std']
        else:
            min_std = CONFIG[data_type]['min_std']

        if std_after < min_std:
            print(f"Warning: Low variability after denoising ({data_type}, std={std_after:.2e})")

        return denoised.astype(np.float64)

    except Exception as e:
        print(f"Error in denoising {data_type}: {e}")
        return data.astype(np.float64)

def detect_data_type(file_path):
    """Detection of file data based on file name."""

    name = os.path.basename(file_path).lower()

    # ---------- LIGO ----------
    if name.endswith(".hdf5") and ("gwosc" in name or "ligo" in name):
        return "ligo"

    # ---------- CMB ----------
    if name.endswith(".fits") and (
        "cmb" in name or "smica" in name or "planck" in name
    ):
        return "cmb"

    # ---------- DESI ----------
    if "lrg" in name or "desi" in name:
        return "desi"

    # ---------- CERN ROOT ----------
    if name.endswith(".root") or "cern" in name:
        return "cern"

    # ---------- Gaia ----------
    if "gaia" in name and name.endswith(".tsv"):
        return "gaia"

    # ---------- QRNG ----------
    if "qrng" in name or "lfdr" in name or "random" in name:
        return "qrng"

    # ---------- fallback ----------
    raise ValueError(f"Could not detect data type for file: {file_path}")




def load_data_core(file_path, data_type, ligo_idx=None, cmb_idx=None, pulsar_name=None):
    """
    UTMF-core dataloader
    Gives back the 1D-reeks for MF-DFA + a small metadata-dict
    Returns
    -------
    for_mfdfa : np.ndarray or None
        1D timeseries (float64) or None.
    meta : dict or None
        Metadata over loaded data (type, length, etc.).
    """
    try:
       # =========================
        # LIGO
        # =========================
        if data_type == 'ligo':
            print(f"Loading LIGO file: {file_path}")

            # --- SAFE AUTO-DETECT ligo_idx ---
            base = os.path.basename(file_path)
            print(f"[DEBUG] Input basename = {base}")
            print("[DEBUG] CONFIG basenames:")
            for p in CONFIG['ligo_files']:
                print("   -", os.path.basename(p))


            # Fallback: match ANY file that contains "L-L1" and "16KHZ"
            if ligo_idx is None:
                for i, path in enumerate(CONFIG['ligo_files']):
                    if "16KHZ" in base.upper() and "16KHZ" in os.path.basename(path).upper():
                        ligo_idx = i
                        print(f"[DEBUG] Fallback matched index {i}")
                        break

            # Final fallback: take reference config index 0
            if ligo_idx is None:
                print("[WARNING] Could not match basename → using LIGO index 0 as default")
                ligo_idx = 0

            cfg = CONFIG['ligo'][ligo_idx]
            print(f"[DEBUG] Using LIGO config index = {ligo_idx}")

            # --- DATA LOADING ---
            with h5py.File(file_path, 'r') as f:
                if 'strain' not in f or 'Strain' not in f['strain']:
                    raise KeyError("Key 'strain/Strain' not found")
                strain = f['strain']['Strain'][:]

            # --- BANDPASS (1–30 Hz) ---
            sos = scipy.signal.butter(
                2,
                cfg['freq_range'],
                btype='band',
                fs=cfg['sample_rate'],
                output='sos'
            )
            strain_filtered = scipy.signal.sosfilt(sos, strain)

            data = da.from_array(strain_filtered, chunks='auto').compute().astype(np.float64)

            meta = {
                "data_type": "ligo",
                "index": ligo_idx,
                "path": file_path,
                "sample_rate": cfg['sample_rate'],
                "freq_range": cfg['freq_range'],
                "expected_D_f": cfg['expected_D_f'],
                "sigma_D_f": cfg['sigma_D_f'],
                "min_std": cfg['min_std'],
                "n": len(data)
            }
            return data, meta



        # =========================
        # CMB (Planck)
        # =========================
        elif data_type == 'cmb':
            print(f"Loading CMB file: {file_path}")
            cfg = CONFIG['cmb'][cmb_idx]

            cmb_maps = hp.read_map(file_path, field=cfg['fields'], verbose=False)
            if isinstance(cmb_maps, np.ndarray):
                cmb_maps = [cmb_maps]

            if cfg['galactic_mask']:
                nside = cfg['nside']
                npix = hp.nside2npix(nside)
                mask = np.ones(npix, dtype=bool)

                # plain galactic band-mask
                galactic_pixels = hp.query_strip(nside, np.radians(60), np.radians(120))
                mask[galactic_pixels] = False
                cmb_maps = [cmb_map[mask] for cmb_map in cmb_maps]

            data = da.from_array(cmb_maps[0], chunks='auto').compute().astype(np.float64)

            meta = {
                "data_type": "cmb",
                "index": cmb_idx,
                "path": file_path,
                "nside": cfg['nside'],
                "galactic_mask": cfg['galactic_mask'],
                "disc_radius": cfg['disc_radius'],
                "expected_D_f": cfg['expected_D_f'],
                "sigma_D_f": cfg['sigma_D_f'],
                "min_std": cfg['min_std'],
                "n": len(data)
            }
            return data, meta

        # =========================
        # DESI (LRG_full.dat.fits)
        # =========================
        elif data_type == 'desi':
            print(f"Loading DESI file: {file_path}")
            with fits.open(file_path) as hdul:
                signals = []
                used_cols = []

                for column in CONFIG['desi']['columns']:
                    data_col = hdul[1].data[column].astype(np.float64)
                    med = np.nanmedian(data_col)
                    data_col = np.nan_to_num(data_col, nan=med, posinf=med, neginf=med)
                    data_col = data_col[np.isfinite(data_col)]
                    if len(data_col) < 10:
                        print(f"Too few data for DESI column {column}: {len(data_col)}")
                        continue
                    signals.append(data_col)
                    used_cols.append(column)

                if not signals:
                    print("No valid data for DESI")
                    return None, None

                # For core: 1st column as 1D-signal
                for_mfdfa = signals[0]

                meta = {
                    "data_type": "desi",
                    "path": file_path,
                    "columns_used": used_cols,
                    "min_length_all_cols": min(len(s) for s in signals),
                    "n": len(for_mfdfa),
                    "expected_D_f": CONFIG['desi']['expected_D_f'],
                    "sigma_D_f": CONFIG['desi']['sigma_D_f'],
                    "min_std": CONFIG['desi']['min_std'],
                }
                return for_mfdfa, meta

        # =========================
        # Gaia DR3 (TSV)
        # =========================
        elif data_type == 'gaia':
            print(f"Loading Gaia DR3 TSV: {file_path}")
            try:
                skip_rows = 257  # zoals v5
                data = pd.read_csv(
                    file_path,
                    sep='\t',
                    skiprows=skip_rows,
                    nrows=CONFIG['gaia']['subset_size'] * 2,
                    header=None,
                    engine='python',
                    on_bad_lines='skip',
                    quoting=3,
                    comment=None
                )
                print(f"Loaded Gaia DR3: {len(data)} rows, {len(data.columns)} cols")

                col_map = {
                    'RA_ICRS': 1,
                    'DE_ICRS': 2,
                    'pmRA': 12,
                    'pmDE': 14,
                    'Gmag': 54
                }

                available_cols = []
                signals = []

                for col_name, idx in col_map.items():
                    if idx < len(data.columns):
                        col_data = pd.to_numeric(data.iloc[:, idx], errors='coerce')
                        arr = col_data.dropna().values.astype(np.float64)
                        if len(arr) < 10:
                            continue
                        signals.append(arr)
                        available_cols.append(col_name)
                        print(f"Loaded {col_name}: {len(arr)} values, mean={np.mean(arr):.3f}")
                    else:
                        print(f"Warning: Col {col_name} (idx {idx}) beyond {len(data.columns)} cols; skip")

                if not signals:
                    raise ValueError("No signals loaded – check snippet indices")

                print(f"Available cols: {available_cols}")
                min_length = min(len(s) for s in signals)
                signals = [s[:min_length] for s in signals]

                data_array = np.column_stack(signals)

                # clipping per kolom
                for i, col in enumerate(available_cols):
                    col_data = data_array[:, i]
                    q_low = np.quantile(col_data, 0.01)
                    q_high = np.quantile(col_data, 0.99)
                    data_array[:, i] = np.clip(col_data, q_low, q_high)
                    print(f"Clipped {col}: [{q_low:.3f}, {q_high:.3f}]")

                # RA/DE → radians
                if 'RA_ICRS' in available_cols and 'DE_ICRS' in available_cols:
                    ra_idx = available_cols.index('RA_ICRS')
                    de_idx = available_cols.index('DE_ICRS')
                    data_array[:, ra_idx] = np.deg2rad(data_array[:, ra_idx])
                    data_array[:, de_idx] = np.deg2rad(data_array[:, de_idx])
                    print("RA/DE converted to radians")

                if min_length < 10:
                    print("Not enough valid data for Gaia")
                    return None, None

                # for core: one 1D serie → mean over columns
                for_mfdfa = np.mean(data_array, axis=1)

                meta = {
                    "data_type": "gaia",
                    "path": file_path,
                    "columns_used": available_cols,
                    "n_rows": min_length,
                    "n_cols": len(available_cols),
                    "expected_D_f": CONFIG['gaia']['expected_D_f'],
                    "sigma_D_f": CONFIG['gaia']['sigma_D_f'],
                    "min_std": CONFIG['gaia']['min_std'],
                }

                print(f"Gaia loaded: {len(available_cols)} cols, length {min_length}")
                return for_mfdfa, meta

            except Exception as e:
                print(f"Error loading Gaia: {e}")
                import traceback
                traceback.print_exc()
                return None, None

        # =========================
        # CERN ROOT
        # =========================
        elif data_type == 'cern':
            print(f"Loading CERN file: {file_path}")
            with uproot.open(file_path) as f:
                signals = []
                used_cols = []

                for column in CONFIG['cern']['columns']:
                    data = f[CONFIG['cern']['tree']][column].array(library='np')

                    if isinstance(data, np.ndarray):
                        if column == 'lep_pt':
                            data = np.concatenate(
                                [np.array(x).flatten() for x in data if len(x) > 0]
                            )
                        else:
                            data = np.array([
                                np.mean(x) if len(x) > 0 else np.nan for x in data
                            ])

                    med = np.nanmedian(data)
                    data = np.nan_to_num(data, nan=med, posinf=med, neginf=med)
                    data = data[np.isfinite(data)]

                    if len(data) < 10:
                        print(f"Not enough data for CERN column {column}: {len(data)}")
                        continue

                    signals.append(data)
                    used_cols.append(column)

                if not signals:
                    print("No valid data for CERN")
                    return None, None

                for_mfdfa = signals[0]

                meta = {
                    "data_type": "cern",
                    "path": file_path,
                    "columns_used": used_cols,
                    "min_length_all_cols": min(len(s) for s in signals),
                    "n": len(for_mfdfa),
                    "expected_D_f": CONFIG['cern']['expected_D_f'],
                    "sigma_D_f": CONFIG['cern']['sigma_D_f'],
                    "min_std": CONFIG['cern']['min_std'],
                }
                return for_mfdfa, meta

        # =========================
        # QRNG (API)
        # =========================
        elif data_type == 'qrng':
            print("Loading LFDR QRNG via API...")
            try:
                data_list = []
                for i in range(50):  # ≈100k bits
                    for retry in range(3):
                        url = 'https://lfdr.de/qrng_api/qrng?length=256&format=HEX'
                        response = requests.get(url, timeout=5)
                        if response.status_code == 200:
                            try:
                                json_data = response.json()
                                hex_string = json_data['qrn']
                                bits = []
                                for char in hex_string:
                                    byte = int(char, 16)
                                    bits.extend([int(b) for b in f"{byte:04b}"])
                                data_list.extend(bits)
                                print(f"QRNG call {i+1}: {len(bits)} bits loaded")
                                break
                            except (json.JSONDecodeError, KeyError) as je:
                                print(f"JSON/Key error, retry {retry+1}: {je}. Response: {response.text[:100]}...")
                                if retry == 2:
                                    raise ValueError("API parse failed")
                        else:
                            print(f"HTTP {response.status_code}, retry {retry+1}. Response: {response.text[:100]}...")
                            if retry < 2:
                                time.sleep(1)
                            else:
                                raise ValueError("API failed after retries")

                if len(data_list) < 100:
                    raise ValueError(f"Not enough QRNG-data: {len(data_list)} bits")

                data = np.array(data_list).astype(np.float64)
                data += np.random.normal(0, 1e-6, len(data))
                print(f"Geladen {len(data)} quantum random bits")

                for_mfdfa = data

                meta = {
                    "data_type": "qrng",
                    "path": "lfdr_api",
                    "n": len(for_mfdfa),
                    "expected_D_f": CONFIG['qrng']['expected_D_f'],
                    "sigma_D_f": CONFIG['qrng']['sigma_D_f'],
                    "min_std": CONFIG['qrng']['min_std'],
                }
                return for_mfdfa, meta

            except Exception as e:
                print(f"Error loading QRNG (fallback to simulation): {e}")
                data = np.random.randint(0, 2, 102400).astype(np.float64)
                data += np.random.normal(0, 1e-6, len(data))
                print(f"Fallback: Simulated {len(data)} random samples")

                for_mfdfa = data
                meta = {
                    "data_type": "qrng_simulated",
                    "path": "lfdr_fallback",
                    "n": len(for_mfdfa),
                    "expected_D_f": CONFIG['qrng']['expected_D_f'],
                    "sigma_D_f": CONFIG['qrng']['sigma_D_f'],
                    "min_std": CONFIG['qrng']['min_std'],
                }
                return for_mfdfa, meta

        else:
            raise ValueError(f"Invalid data_type: {data_type}")

    except Exception as e:
        print(f"\n\n🔥 FULL ERROR inside load_data_core({data_type}):")
        import traceback
        traceback.print_exc()
        print("🔥 END ERROR\n\n")
        return None, None

# Subset processing - Enhanced logging for every subset
def process_subset_core(
    subset_idx,
    data,
    data_type,
    dataset_name,
    scales,
    ligo_idx=None,
    cmb_idx=None
):
    """
    Clean UTMF-core version of subset processor.
    """
    try:
        # === Subset extraction ===
        if data_type == "ligo":
            n_samples = int(
                CONFIG["ligo"][ligo_idx]["subset_duration"] *
                CONFIG["ligo"][ligo_idx]["sample_rate"]
            )
            max_start = len(data) - n_samples
            subset_start = np.random.randint(0, max_start + 1)
            subset_data = data[subset_start:subset_start + n_samples]

        elif data_type == 'cmb':
            nside = CONFIG['cmb'][cmb_idx]['nside']
            subset_size = CONFIG['cmb'][cmb_idx]['subset_size']
            npix = len(data)
            valid_indices = np.arange(npix)

            center_pix = np.random.choice(valid_indices)
            try:
                subset_indices = hp.query_disc(
                    nside,
                    hp.pix2vec(nside, center_pix, nest=False),
                    radius=CONFIG['cmb'][cmb_idx]['disc_radius']
                )
                subset_indices = subset_indices[subset_indices < npix]

                if len(subset_indices) < subset_size:
                    subset_indices = np.random.choice(valid_indices, size=subset_size, replace=False)
                else:
                    subset_indices = np.random.choice(subset_indices, size=subset_size, replace=False)

                subset_data = data[subset_indices]

            except Exception:
                subset_indices = np.random.choice(valid_indices, size=subset_size, replace=False)
                subset_data = data[subset_indices]

        elif data_type in ['desi', 'cern', 'nist', 'nanograv', 'qrng']:
            # Subset size
            subset_size = (
                CONFIG[data_type]['subset_size'](len(data))
                if (data_type == 'nist')
                else CONFIG[data_type]['subset_size']
            )
            max_start = len(data) - subset_size
            if max_start < 0:
                q_len = len(CONFIG['mfdfa']['q_values'])
                return (
                    np.nan,
                    np.full(q_len, np.nan),
                    np.full((q_len, len(scales)), np.nan),
                    np.full(q_len, np.nan),
                    None
                )
            subset_start = np.random.randint(0, max_start + 1)
            subset_data = data[subset_start:subset_start + subset_size]
            if len(subset_data) != subset_size:
                subset_data = np.pad(subset_data, (0, subset_size - len(subset_data)), mode='constant')

        elif data_type == 'gaia':
            subset_size = CONFIG['gaia']['subset_size']
            max_start = len(data) - subset_size
            if max_start < 0:
                q_len = len(CONFIG['mfdfa']['q_values'])
                return (
                    np.nan,
                    np.full(q_len, np.nan),
                    np.full((q_len, len(scales)), np.nan),
                    np.full(q_len, np.nan),
                    None
                )
            subset_start = np.random.randint(0, max_start + 1)
            subset_data = data[subset_start:subset_start + subset_size]
            if len(subset_data) != subset_size:
                subset_data = np.pad(
                    subset_data,
                    (0, subset_size - len(subset_data)),
                    mode='constant'
                )
        else:
            raise ValueError("Invalid data_type")

        # ============================================================
        # 2) DENOISING
        # ============================================================
        subset_data_denoised = denoise_data(subset_data, data_type, ligo_idx, cmb_idx)

        # ============================================================
        # 3) MFDFA INPUT (Gaia → pairwise distances)
        # ============================================================
        if data_type == 'gaia' and subset_data_denoised.ndim > 1:
            ra    = subset_data_denoised[:, 0]
            de    = subset_data_denoised[:, 1]
            pmra  = subset_data_denoised[:, 2]
            pmde  = subset_data_denoised[:, 3]

            sample_idx = np.random.choice(len(ra), 50, replace=False)
            ra_s, de_s, pmra_s, pmde_s = ra[sample_idx], de[sample_idx], pmra[sample_idx], pmde_s[sample_idx]

            dists = []
            for i in range(len(ra_s)):
                for j in range(i + 1, len(ra_s)):
                    d_pos = np.sqrt((ra_s[i] - ra_s[j])**2 + (de_s[i] - de_s[j])**2)
                    d_vel = np.sqrt((pmra_s[i] - pmra_s[j])**2 + (pmde_s[i] - pmde_s[j])**2)
                    dists.append(np.sqrt(d_pos**2 + d_vel**2))

            dists = np.array(dists)
            if len(dists) < 100:
                dists = np.pad(dists, (0, 100 - len(dists)), mode='constant')

            mfdfa_input = np.log(dists + 1)

        else:
            mfdfa_input = subset_data_denoised

        # ============================================================
        # 4) RUN MFDFA
        # ============================================================
        D_f, hq, rms, fluct, slopes = jedi_mfdfa(
            subset_data_denoised,
            scales,
            CONFIG['mfdfa']['q_values'],
            CONFIG['mfdfa']['detrend_order']
        )

        return D_f, hq, fluct, slopes, subset_data_denoised

    except Exception as e:
        q_len = len(CONFIG['mfdfa']['q_values'])
        return (
            np.nan,
            np.full(q_len, np.nan),
            np.full((q_len, len(scales)), np.nan),
            np.full(q_len, np.nan),
            None
        )
      
def process_dataset_core(
    data,
    data_type,
    dataset_name,
    scales,
    expected_D_f,
    sigma_D_f,
    ligo_idx=None,
    cmb_idx=None
):

    if data_type == 'ligo' and ligo_idx is None:
        ligo_idx = 0  # use first LIGO-config section
        print("[UTMF-core] ⚠️ Warning: ligo_idx was None → ligo_idx=0 is used.")

    if data_type == 'cmb' and cmb_idx is None:
        cmb_idx = 0
        print("[UTMF-core] ⚠️ Warning: cmb_idx was None → cmb_idx=0 is used.")

    if data is None:
        print(f"No valid data for {dataset_name}. Skipping.")
        return None, []

    else:
      
        if data_type == 'ligo':
            n_subsets = CONFIG['ligo'][ligo_idx]['n_subsets']
        elif data_type == 'cmb':
            n_subsets = CONFIG['cmb'][cmb_idx]['n_subsets']
        elif data_type == 'gaia':
            n_subsets = CONFIG['gaia']['n_subsets']
        else:
            n_subsets = CONFIG[data_type]['n_subsets']

        scales_arr = np.asarray(scales, dtype=np.int32)

    # === Parallel subset processing ======================================
    results = Parallel(n_jobs=-1)(
        delayed(process_subset_core)(
            subset_idx,
            data,
            data_type,
            dataset_name,
            scales_arr,
            ligo_idx=ligo_idx,
            cmb_idx=cmb_idx
        )
        for subset_idx in tqdm(range(n_subsets), desc=f"Subsets for {dataset_name}")
    )

    # Unpack resulta
    D_f_values      = np.array([r[0] for r in results], dtype=float)
    hq_values       = [r[1] for r in results if r[1] is not None]
    fluct_values    = [r[2] for r in results]
    slopes_values   = [r[3] for r in results]
    subset_data_list = [r[4] for r in results]

    print(f"\nSummary for {dataset_name}:")

    # At least 1 valid D_f?
    valid_mask = np.isfinite(D_f_values)
    if np.any(valid_mask):
        valid_D = D_f_values[valid_mask]
        mean_D_f = float(np.nanmean(valid_D))
        std_D_f  = float(np.nanstd(valid_D))

        # h(q)-mean over subsets
        valid_hq_means = [
            float(np.nanmean(hq))
            for hq in hq_values
            if hq is not None and np.any(np.isfinite(hq))
        ]
        mean_hq = float(np.nanmean(valid_hq_means)) if valid_hq_means else np.nan

        # Plain z-test
        # z = (mean_D_f - expected_D_f) / (sigma / sqrt(n))
        try:
            n_eff = len(valid_D)
            sigma_eff = sigma_D_f / max(np.sqrt(n_eff), 1e-12)
            z = (mean_D_f - expected_D_f) / max(sigma_eff, 1e-12)
            p_value = float(2 * norm.sf(abs(z)))  # two-sided
        except Exception:
            p_value = np.nan

        print(f"Mean D_f: {mean_D_f:.3f} ± {std_D_f:.3f}")
        print(f"Expected D_f: {expected_D_f:.3f} ± {sigma_D_f:.3f}")
        print(f"Difference from expected: {abs(mean_D_f - expected_D_f):.3f}")
        print(f"Mean h_q: {mean_hq:.3f}")
        print(f"Number of valid subsets: {np.sum(valid_mask)}")
        print(f"Z-test p-value: {p_value:.3f}")

        return {
            'D_f_values': D_f_values,        # np.array
            'hq_values':   hq_values,        # list of np.arrays
            'fluct':       fluct_values,     # list of np.arrays
            'slopes':      slopes_values,    # list of np.arrays
            'mean_D_f':    mean_D_f,
            'std_D_f':     std_D_f,
            'mean_hq':     mean_hq,
            'p_value':     p_value,
            'n_valid_subsets': int(np.sum(valid_mask)),
            'best_subsets': subset_data_list
        }, subset_data_list

    else:
        print("No valid D_f values computed.")
        return None, []

def run_utmf_core(file_path):

    # ======================
    # 1. Detect datatype
    # ======================
    data_type = detect_data_type(file_path)
    base = os.path.basename(file_path)
    print(f"[UTMF-core] Detected dataset type: {data_type}")

    # ======================
    # 2. Initialize indexes
    # ======================
    ligo_idx = None
    cmb_idx = None
    desi_idx = None
    cern_idx = None
    gaia_idx = None
    nist_idx = None

    # ===========================================================
    # ----------------------  LIGO  -----------------------------
    # ===========================================================
    if data_type == "ligo":
        for i, fpath in enumerate(CONFIG["ligo_files"]):
            if os.path.basename(fpath) == base:
                ligo_idx = i
                break

        if ligo_idx is None:
            print("[WARNING] LIGO file not in CONFIG['ligo_files'] — using index 0")
            ligo_idx = 0

        print(f"[DEBUG] Using LIGO index: {ligo_idx}")

        data, meta = load_data_core(
            file_path=file_path,
            data_type="ligo",
            ligo_idx=ligo_idx
        )
        if data is None:
            raise RuntimeError("load_data_core failed for LIGO")

        cfg = CONFIG["ligo"][ligo_idx]


    # ===========================================================
    # ----------------------  CMB  ------------------------------
    # ===========================================================
    elif data_type == "cmb":
        for i, fpath in enumerate(CONFIG["cmb_files"]):
            if os.path.basename(fpath) == base:
                cmb_idx = i
                break

        if cmb_idx is None:
            print("[WARNING] CMB file not matched → using index 0")
            cmb_idx = 0

        print(f"[DEBUG] Using CMB index: {cmb_idx}")

        data, meta = load_data_core(
            file_path=file_path,
            data_type="cmb",
            cmb_idx=cmb_idx
        )
        if data is None:
            raise RuntimeError("load_data_core failed for CMB")

        cfg = CONFIG["cmb"][cmb_idx]


    # ===========================================================
    # ----------------------  DESI  -----------------------------
    # ===========================================================
    elif data_type == "desi":
        print("[DEBUG] Loading DESI dataset")

        data, meta = load_data_core(
            file_path=file_path,
            data_type="desi"
        )
        if data is None:
            raise RuntimeError("load_data_core failed for DESI")

        cfg = CONFIG["desi"]


    # ===========================================================
    # ----------------------  CERN  -----------------------------
    # ===========================================================
    elif data_type == "cern":
        print("[DEBUG] Loading CERN ROOT dataset")

        data, meta = load_data_core(
            file_path=file_path,
            data_type="cern"
        )
        if data is None:
            raise RuntimeError("load_data_core failed for CERN")

        cfg = CONFIG["cern"]


    # ===========================================================
    # ----------------------  GAIA  -----------------------------
    # ===========================================================
    elif data_type == "gaia":
        print("[DEBUG] Loading Gaia DR3")

        data, meta = load_data_core(
            file_path=file_path,
            data_type="gaia"
        )
        if data is None:
            raise RuntimeError("load_data_core failed for Gaia")

        cfg = CONFIG["gaia"]


    elif data_type == "qrng":
        print("[DEBUG] Loading QRNG dataset")

        data, meta = load_data_core(
            file_path=file_path,
            data_type="qrng"
        )

        if data is None:
            raise RuntimeError("load_data_core failed for QRNG")

        cfg = CONFIG["qrng"]

    else:
        raise RuntimeError(f"Unsupported data type: {data_type}")


    # ======================
    # 3. Print summary
    # ======================
    print(f"[UTMF-core] Loaded dataset ({data_type}), length = {len(data):,}")


    # ======================
    # 4. Run MF-DFA
    # ======================
    result, _ = process_dataset_core(
        data=data,
        data_type=data_type,
        dataset_name=os.path.basename(file_path),
        scales=cfg["scales"],
        expected_D_f=cfg["expected_D_f"],
        sigma_D_f=cfg["sigma_D_f"],
        ligo_idx=ligo_idx,
        cmb_idx=cmb_idx
    )

    # Add required fields for summary
    result["data_type"] = data_type
    result["index"] = ligo_idx if data_type == "ligo" else cmb_idx
    result["expected_D_f"] = cfg["expected_D_f"]
    result["sigma_D_f"] = cfg["sigma_D_f"]

    return result



def build_utmf_summary(result: dict) -> dict:
    """
    Build a compact summary from the result of process_dataset_core.
    Verwacht een dict met o.a.:
      - 'D_f_values'
      - 'hq_values'
      - 'mean_D_f', 'std_D_f', 'p_value', 'n_valid_subsets'
      - 'data_type', 'index', 'expected_D_f', 'sigma_D_f'
    """
    # --- D_f basis ---
    D_f_values = np.array(result["D_f_values"], dtype=float)
    mean_D_f   = float(result["mean_D_f"])
    std_D_f    = float(result["std_D_f"])
    p_value    = float(result.get("p_value", np.nan))
    n_valid    = int(result.get("n_valid_subsets", np.sum(np.isfinite(D_f_values))))

    # --- h(q) statistics ---
    hq_all = result.get("hq_values", [])

    # list of arrays, only when containing data
    hq_arrays = []
    for hq in hq_all:
        if hq is None:
            continue
        arr = np.array(hq, dtype=float)
        if arr.size == 0 or not np.any(np.isfinite(arr)):
            continue
        hq_arrays.append(arr)

    if hq_arrays:
        # mean h(q) over all subsets, then mean of that
        hq_mean = float(np.nanmean([np.nanmean(a) for a in hq_arrays]))
        # breadth h(q) per subset (max-min), then mean breadth
        hq_width = float(np.nanmean([np.nanmax(a) - np.nanmin(a) for a in hq_arrays]))
    else:
        hq_mean  = np.nan
        hq_width = np.nan

    # --- expected D_f from cfg / result ---
    data_type = result.get("data_type", None)
    exp_D = np.nan
    sig_D = np.nan

    if data_type in ("ligo", "cmb"):
        idx = result.get("index", 0)
        exp_D = float(result.get("expected_D_f",
                                 CONFIG[data_type][idx]["expected_D_f"]))
        sig_D = float(result.get("sigma_D_f",
                                 CONFIG[data_type][idx]["sigma_D_f"]))
    elif data_type in ("desi", "cern", "gaia", "qrng"):
        exp_D = float(result.get("expected_D_f",
                                 CONFIG[data_type]["expected_D_f"]))
        sig_D = float(result.get("sigma_D_f",
                                 CONFIG[data_type]["sigma_D_f"]))
    else:
        # fallback: usables from result, or NaN
        exp_D = float(result.get("expected_D_f", np.nan))
        sig_D = float(result.get("sigma_D_f", np.nan))

    delta_D = float(mean_D_f - exp_D)

    # --- simpele “stability / fractality / robustness” metrics ---
    stability      = float(std_D_f) if std_D_f > 0 else np.nan
    fractality_idx = float(abs(delta_D) / (sig_D + 1e-12))  # |ΔD| / σ_D
    robustness     = float(np.exp(-fractality_idx))         


    summary = {
        "data_type": data_type,
        "index": result.get("index", None),

        "mean_D_f": mean_D_f,
        "std_D_f": std_D_f,
        "expected_D_f": exp_D,
        "sigma_D_f": sig_D,
        "delta_D_f": delta_D,

        "mean_hq": hq_mean,
        "hq_width": hq_width,

        "p_value": p_value,
        "n_valid_subsets": n_valid,

        "stability": stability,
        "fractality_index": fractality_idx,
        "robustness": robustness,
    }
    return summary


def print_utmf_summary(summary, name="Dataset"):
    print("\n====== UTMF-core Summary:", name, "======")
    print(f"D_f mean       : {summary['mean_D_f']:.3f}")
    print(f"D_f std        : {summary['std_D_f']:.3f}")
    print(f"Expected D_f   : {summary['expected_D_f']:.3f}")
    print(f"Δ D_f          : {summary['delta_D_f']:.3f}")
    print(f"Mean h(q)      : {summary['mean_hq']:.3f}")
    print(f"h(q) width     : {summary['hq_width']:.3f}")
    print(f"p-value        : {summary['p_value']:.3f}")
    print(f"Valid subsets  : {summary['n_valid_subsets']}")
    print(f"Stability      : {summary['stability']:.3f}")
    print(f"Fractality idx : {summary['fractality_index']:.3f}")
    print(f"Robustness     : {summary['robustness']:.3f}")
    print("========================================\n")

if __name__ == "__main__":
    # Simple smoke-test (user can edit this locally)
    print("UTMF-Core module imported successfully.")
