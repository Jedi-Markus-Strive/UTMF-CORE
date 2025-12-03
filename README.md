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

# Datasets Used in UTMF-CORE v1.0

**Total size ~ 9.11 GB**
Fully reproducible via public archives.

---

## 1. LIGO – GWOSC O4a

Archive: [https://gwosc.org/archive/links/O4a_16KHZ_R1/](https://gwosc.org/archive/links/O4a_16KHZ_R1/)

Files used:

* `L-L1_GWOSC_O4a_16KHZ_R1-1384779776-4096.hdf5`
* `L-L1_GWOSC_O4a_16KHZ_R1-1368350720-4096.hdf5`
* `L-L1_GWOSC_O4a_16KHZ_R1-1370202112-4096.hdf5`
* `L-L1_GWOSC_O4a_16KHZ_R1-1389420544-4096.hdf5`

---

## 2. Planck – SMICA CMB Maps

Archive: [https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/)

Files:

* `COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits`
* `COM_CMB_IQU-smica_2048_R3.00_full.fits`

---

## 3. DESI – DR1 LRG Catalog

Dataset:
[https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.2/LRG_full.dat.fits](https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.2/LRG_full.dat.fits)

---

## 4. CERN – ATLAS Two-Lepton Events

Record: [https://opendata.cern.ch/record/15007](https://opendata.cern.ch/record/15007)

File:

* `data_B.exactly2lep.root`

---

## 5. Gaia – DR3 (TSV via VizieR)

Start: [https://vizier.cds.unistra.fr/viz-bin/VizieR-4](https://vizier.cds.unistra.fr/viz-bin/VizieR-4)

Select:

* Catalogue: `gaiadr3`
* Table: `I/355/gaiadr3`
* Rows: `1–999999`
* Format: TSV
* Columns: All
* Save as: `gaia_dr3.tsv`

---

## 6. ANU Quantum Random Numbers (QRNG)

API: [https://qrng.anu.edu.au/](https://qrng.anu.edu.au/)

Used directly by the loader; no manual download needed.

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
