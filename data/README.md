# Data Layout

The official BTE data files are not committed to this repository because they
are large. Download them from the authors' OneDrive release for Lu et al. 2022
and place them as follows:

```text
data/bte_real/
├── low_train.npz
├── low_test.npz
├── mf_train.npz
├── mf_test.npz
└── bte5x5_2iter_size10532.npz   # only needed for raw-data matching/calibration
```

Required for MF-DeepONet training/evaluation:

- `low_train.npz`
- `low_test.npz`
- `mf_train.npz`
- `mf_test.npz`

Required for OpenBTE raw-data matching/calibration:

- `bte5x5_2iter_size10532.npz`

