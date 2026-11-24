## Phase 5 - Set-level evaluation (reference = real geo-test tiles of the same classes)
- Metrics: FID and KID (Inception-v3), FD-DINOv2, precision/recall (DINOv2, k=3), Vendi score (DINOv2 and CLIP, cosine kernel).
- All images compared at 200 px grayscale. "real train (reference)" = random real geo-train tiles, same size as the test set: the real-vs-real floor.
- KID is the most reliable distance here; per-class FID with only 75-300 images is biased upward.

### Overall (all target classes pooled)
| set | Vendi_DINOv2 | Vendi_CLIP | n | FID | KID | KID_std | FD_DINOv2 | precision | recall |
|---|---|---|---|---|---|---|---|---|---|
| real test (itself) | 21.81 | 3.63 | 1295 | nan | nan | nan | nan | nan | nan |
| real train (reference) | 19.88 | 3.69 | 1295 | 64.22 | 0.0081 | 0.0035 | 0.13 | 0.5058 | 0.6919 |
| generated (all) | 15.88 | 3.32 | 900 | 145.7 | 0.0513 | 0.0089 | 0.38 | 0.1078 | 0.1861 |
| generated (filtered) | 12.97 | 3.57 | 258 | 178.11 | 0.0612 | 0.0074 | 0.48 | 0.1124 | 0.2788 |

### Per class
| group | set | Vendi_DINOv2 | Vendi_CLIP | n | FID | KID | KID_std | FD_DINOv2 | precision | recall |
|---|---|---|---|---|---|---|---|---|---|---|
| cra | real test (itself) | 4.97 | 2.07 | 129 | nan | nan | nan | nan | nan | nan |
| cra | real train (reference) | 5.02 | 2.22 | 129 | 131.84 | 0.0097 | 0.0021 | 0.13 | 0.6047 | 0.6899 |
| cra | generated (all) | 5.04 | 2.52 | 150 | 253.32 | 0.1745 | 0.0061 | 0.43 | 0.08 | 0.1705 |
| cra | generated (filtered) | 4.18 | 2.26 | 116 | 250.38 | 0.1733 | 0.0045 | 0.44 | 0.0776 | 0.155 |
| cli | real test (itself) | 11.42 | 2.45 | 106 | nan | nan | nan | nan | nan | nan |
| cli | real train (reference) | 10.36 | 2.37 | 106 | 157.3 | 0.019 | 0.0008 | 0.33 | 0.6226 | 0.7453 |
| cli | generated (all) | 12.77 | 2.56 | 150 | 240.4 | 0.084 | 0.0036 | 0.6 | 0.1133 | 0.1132 |
| rid | real test (itself) | 16.09 | 2.87 | 223 | nan | nan | nan | nan | nan | nan |
| rid | real train (reference) | 12.45 | 2.61 | 223 | 105.83 | 0.0063 | 0.0017 | 0.22 | 0.7309 | 0.5695 |
| rid | generated (all) | 11.9 | 2.55 | 150 | 212.41 | 0.0685 | 0.0064 | 0.54 | 0.1 | 0.1794 |
| rid | generated (filtered) | 4.48 | 1.73 | 12 | 280.36 | 0.0329 | 0.0121 | 0.8 | 0.25 | 0.3229 |
| ael | real test (itself) | 9.28 | 2.38 | 308 | nan | nan | nan | nan | nan | nan |
| ael | real train (reference) | 11.44 | 2.53 | 308 | 98.74 | 0.0155 | 0.0037 | 0.17 | 0.3052 | 0.7955 |
| ael | generated (all) | 8.49 | 2.68 | 75 | 207.8 | 0.058 | 0.0037 | 0.56 | 0.08 | 0.1299 |
| ael | generated (filtered) | 5.36 | 2.19 | 39 | 217.09 | 0.0748 | 0.0123 | 0.58 | 0.1282 | 0.1916 |
| aec | real test (itself) | 12.5 | 2.22 | 245 | nan | nan | nan | nan | nan | nan |
| aec | real train (reference) | 12.4 | 2.69 | 245 | 227.53 | 0.103 | 0.0093 | 0.45 | 0.1837 | 0.5224 |
| aec | generated (all) | 12.12 | 2.9 | 75 | 280.6 | 0.0975 | 0.0058 | 0.63 | 0.04 | 0.698 |
| aec | generated (filtered) | 6.46 | 2.19 | 25 | 359.69 | 0.1395 | 0.0141 | 0.71 | 0.04 | 0.7184 |
| rou | real test (itself) | 3.93 | 1.76 | 65 | nan | nan | nan | nan | nan | nan |
| rou | real train (reference) | 3.78 | 1.78 | 65 | 148.54 | 0.1043 | 0.0 | 0.27 | 0.2462 | 0.6923 |
| rou | generated (all) | 6.26 | 2.13 | 150 | 228.61 | 0.1412 | 0.014 | 0.44 | 0.1467 | 0.5846 |
| smo | real test (itself) | 4.48 | 1.76 | 219 | nan | nan | nan | nan | nan | nan |
| smo | real train (reference) | 7.71 | 2.29 | 219 | 102.3 | 0.0826 | 0.0082 | 0.33 | 0.5434 | 0.5936 |
| smo | generated (all) | 8.81 | 2.62 | 150 | 278.61 | 0.2587 | 0.0121 | 0.87 | 0.12 | 0.0822 |
| smo | generated (filtered) | 6.16 | 2.21 | 53 | 223.85 | 0.2208 | 0.0162 | 0.81 | 0.1887 | 0.1279 |

- Figure: results/phase5_metrics.png. Old-project images: not uploaded yet (put them in old_project_images/ and re-run Cell 29).
