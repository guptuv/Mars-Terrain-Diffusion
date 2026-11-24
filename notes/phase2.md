## Phase 2 - Baseline classifier (real only)
- Model: convnext_tiny.fb_in22k_ft_in1k (ImageNet-22k pretrained), 15 classes, input 200 px grayscale -> 224 px, 3 channels.
- Training: 1200 steps, batch 64, AdamW lr 0.0002, wd 0.05, cosine schedule, label smoothing 0.1, fp16. Augment: h/v flips, random crop 75-100%, brightness/contrast jitter.
- Selected on geo VAL macro-F1 (best step 1050); training time 435 s on a T4.
- **Geo-val accuracy 0.812, macro-F1 0.804.** Test set not used (scored once in Phase 6).

| class | val F1 | val precision | val images |
|---|---|---|---|
| ael - Aeolian Straight | 0.6632 | 0.5614 | 79 |
| rou - Rough Terrain | 0.5342 | 0.8125 | 98 |
| cli - Cliff | 0.787 | 0.787 | 108 |
| aec - Aeolian Curved | 0.9681 | 0.9435 | 168 |
| tex - Textured Terrain | 0.689 | 0.7574 | 489 |
| smo - Smooth Terrain | 0.7654 | 0.6884 | 159 |
| fss - Mass Wasting | 0.7399 | 0.6474 | 117 |
| rid - Ridge | 0.7756 | 0.761 | 153 |
| fse - Slope Streaks | 0.9048 | 0.9828 | 272 |
| sfe - Mounds | 0.9667 | 0.9431 | 117 |
| fsf - Channel | 0.79 | 0.7283 | 146 |
| fsg - Gullies | 0.6815 | 0.5412 | 100 |
| sfx - Crater Field | 0.9483 | 0.9375 | 172 |
| cra - Crater | 0.9839 | 0.9871 | 156 |
| mix - Mixed Terrain | 0.8575 | 0.935 | 418 |

- This model is the Phase 4 filter; its per-class precision above tells how much to trust the filter for each class.
