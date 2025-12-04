# UTMF-CORE

**UTMF-CORE (Unified Temporal-Measurement Framework)** is a domain-agnostic measurement system for multifractal analysis of one-dimensional physical time series. It provides a numerically stable implementation of Multifractal Detrended Fluctuation Analysis (MFDFA), combined with minimal preprocessing, strict segment validation, and a universal set of multifractal observables that remain comparable across heterogeneous datasets.

This repository accompanies the paper:

> **M. Eversdijk (2025).**
> *UTMF-Core: A Unified Temporal-Measurement Framework for Heterogeneous Physical Time Series.*
> (Preprint, forthcoming on arXiv)

---

## Key Features

* Unified MF-DFA pipeline applied identically to all datasets
* Adaptive scale and segment validation
* High-stability fluctuation estimation
* Full multifractal metric set:

  * Generalised Hölder spectrum (h(q))
  * Mass exponents (\tau(q))
  * Singularity spectrum ((\alpha, f(\alpha)))
  * Fractal dimension (D_f)
* Built-in dataset autodetection:

  * LIGO O4 strain (GWOSC)
  * Planck CMB (SMICA)
  * DESI LRG catalogs
  * CERN ATLAS 2-lepton events
  * Gaia DR3 catalog
  * ANU QRNG (API)
* Reproducible implementation matching the paper

---

## Repository Structure

```
UTMF-CORE/
│
├── utmf_core/
│   ├── __init__.py
│   └── core.py          # main implementation (pipeline, loaders, summaries)
│
├── examples/            # optional notebooks or scripts
│
├── README.md
├── LICENSE
└── CITATION.cff
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/UTMF-CORE.git
cd UTMF-CORE
```

Install required packages:

```bash
pip install numpy scipy pandas h5py dask healpy astropy uproot pywt joblib tqdm requests numba
```

Python ≥ 3.9 recommended.

---

## Basic Usage (Python)

```python
from utmf_core.core import run_utmf_core, build_utmf_summary, print_utmf_summary

file_path = "/path/to/data.hdf5"  # or .fits, .root, .tsv, etc.

result  = run_utmf_core(file_path)
summary = build_utmf_summary(result)

print_utmf_summary(summary, name="My Dataset")
```

The summary includes:

* mean & std fractal dimension
* expected vs measured (D_f)
* z-test p-value
* h(q) width
* stability & robustness indices
* number of valid subsets

---

## Usage in Google Colab

1. Create folders:

```
/MyDrive/Datasets_UTMF/
/MyDrive/Datasets_UTMF/UTMF_outputs/
```

2. Place dataset files in `Datasets_UTMF`.
3. Mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

4. Run UTMF-core:

```python
from utmf_core.core import run_utmf_core, build_utmf_summary, print_utmf_summary

file_path = "/content/drive/MyDrive/Datasets_UTMF/LRG_full.dat.fits"

try:
    result  = run_utmf_core(file_path)
    summary = build_utmf_summary(result)
    print_utmf_summary(summary, name="DESI")
except FileNotFoundError:
    print(f"❌ File not found: {file_path}")
except Exception as e:
    print(f"❌ UTMF-Core encountered an error:\n{e}")

```

---


## **All datasets used for UTMF-CORE v1.0.0 analysis listed below.** (Total 9.8GB) 
#### - Direct downloadlinks.
#### - For Colab: Create: /MyDrive/Datasets_UTMF/UTMF_outputs/
#### - Place the datasets in folder: /Datasets_UTMF/
#### - Mount Drive
#### - Run UTMF-CORE v1.0.0
#### - Results are returned in folder: /UTMF_outputs/
-----
- **[LIGO – GWOSC](https://gwosc.org/archive/links/O4a_16KHZ_R1/L1/1368195220/1389456018/simple/)**  
  HDF5 strain files (e.g., `L-L1_GWOSC_O4a_16KHZ_R1-*.hdf5`).
                                                           
  **Datasets used in UTMF-CORE v1.0.0 configuration:**
- `L-L1_GWOSC_O4a_16KHZ_R1-1384779776-4096.hdf5` [Download](https://gwosc.org/archive/data/O4a_16KHZ_R1/1384120320/L-L1_GWOSC_O4a_16KHZ_R1-1384779776-4096.hdf5) (486MB)
- `L-L1_GWOSC_O4a_16KHZ_R1-1368350720-4096.hdf5` [Download](https://gwosc.org/archive/data/O4a_16KHZ_R1/1367343104/L-L1_GWOSC_O4a_16KHZ_R1-1368350720-4096.hdf5) (486MB)
- `L-L1_GWOSC_O4a_16KHZ_R1-1370202112-4096.hdf5` [Download](https://gwosc.org/archive/data/O4a_16KHZ_R1/1369440256/L-L1_GWOSC_O4a_16KHZ_R1-1370202112-4096.hdf5) (486MB)
- `L-L1_GWOSC_O4a_16KHZ_R1-1389420544-4096.hdf5` [Download](https://gwosc.org/archive/data/O4a_16KHZ_R1/1389363200/L-L1_GWOSC_O4a_16KHZ_R1-1389420544-4096.hdf5) (486MB)
---
- **[Planck – ESA Archive](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/)**  
  FITS CMB maps (e.g., SMICA IQU maps such as `COM_CMB_IQU-smica_2048_R3.00_full.fits`).

  **Datasets used in UTMF-CORE v1.0.0 configuration:**
- `COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits` [Download](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits) (384MB)
- `COM_CMB_IQU-smica_2048_R3.00_full.fits`      [Download](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica_2048_R3.00_full.fits) (1.88GB)
---
- **[DESI – Data Release Portal](https://data.desi.lbl.gov/doc/releases/dr1/)**  
  LRG FITS catalogs (e.g., `LRG_full.dat.fits`).

  **Dataset used in UTMF-CORE v1.0.0 configuration:**
- `LRG_full.dat.fits` [Download](https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.2/LRG_full.dat.fits) (2.77GB)
---  
- **[CERN Open Data](https://opendata.cern.ch/record/15007)**  
  ROOT event files (e.g., `data_B.exactly2lep.root`).

  **Dataset used in UTMF-CORE v1.0.0 configuration:**
- `data_B.exactly2lep.root` [Download:](https://opendata.cern.ch/record/15007/files/data_B.exactly2lep.root) (451MB)
---
- **[Gaia Archive (DR3)](https://vizier.cds.unistra.fr/viz-bin/VizieR-4)**  
  Source catalogs in TSV format (e.g., `gaia_dr3.tsv`).                                                                 
  **Dataset used in UTMF-CORE v1.0.0:**                                                                                      
      **Select:**                                                                                                       
        1- 'gaiadr3'                                                                                                    
        2. Table: `I/355/gaiadr3`  
        3. Rows: `1-999999`  
        4. Format: Tab-Separated Values  
        5. Columns: All                                                                                
        6. Rename file to 'gaia_dr3' or update path in config.                                                          
---
- **[ANU Quantum Random Numbers (QRNG)](https://qrng.anu.edu.au/)**  
  API-based quantum random sequences (no download required, incorporated in UTMF-CORE v1.0.0 configuration).
---

---

# Reproducing the Paper Results

For each dataset:

```python
from utmf_core.core import run_utmf_core, build_utmf_summary, print_utmf_summary

file_path = "/content/drive/MyDrive/Datasets_UTMF/LRG_full.dat.fits"

try:
    result  = run_utmf_core(file_path)
    summary = build_utmf_summary(result)
    print_utmf_summary(summary, name="DESI")
except FileNotFoundError:
    print(f"❌ File not found: {file_path}")
except Exception as e:
    print(f"❌ UTMF-Core encountered an error:\n{e}")

```

Results include:

* (D_f) mean & std
* expected vs measured deviation
* p-value
* robustness score
* h(q) width
* subset counts

---

# Citation

```
M. Eversdijk (2025),
"UTMF-Core: A Unified Temporal-Measurement Framework for Heterogeneous Physical Time Series."
arXiv:XXXX.YYYY (preprint)
```

---

# License

Released under the **MIT License**.

---

# Contact

**Email:** [crisplatform@gmail.com](mailto:crisplatform@gmail.com)

Contributions, feedback, and independent validation studies are welcome.
UTMF-CORE is intended as the foundation for a broader family of multifractal measurement tools.
