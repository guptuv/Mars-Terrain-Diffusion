# Mars Data Engine - Project Notes

## Phase 0 - Setup (2026-11-24)
- GPU: Tesla T4 (Colab free tier). Settings: SD 1.5 at 512x512, fp16, LoRA batch 1 x grad-accum 4, 8-bit Adam.
- Libraries: torch 2.14.1+cu130, torchvision 0.29.1+cu130, diffusers 0.43.0, peft 0.22.0, accelerate 1.16.0, transformers 5.24.1, timm 1.0.31, torchmetrics 1.9.0, open_clip_torch 4.2.2. Python 3.13.16.
- Global seed 42. All outputs saved under MyDrive/mars_data_engine/.
- Note: pip warned about cudf/pyarrow and diffusers disabled torchao; neither is used, so both are harmless.


## Phase 1 - Real data
- Dataset: DoMars16k (Mars-Bench), Hugging Face Mirali33/mb-domars16k, commit 4d79cf7a0fc98fab76d436f6434e4ce3eb78cd85, CC-BY-4.0.
- 16,150 CTX tiles (~6 m/px), 200x200 grayscale, 15 classes.
- Official Mars-Bench splits kept only for comparison. Main experiments use our leakage-free geo split (below).
- Important: imagery is CTX, not HiRISE. Captions and claims must say CTX.

### Images per class (official Mars-Bench split)
| index | train | val | test | total |
|---|---|---|---|---|
| ael - Aeolian Straight | 711 | 203 | 102 | 1016 |
| rou - Rough Terrain | 705 | 201 | 101 | 1007 |
| cli - Cliff | 700 | 200 | 100 | 1000 |
| aec - Aeolian Curved | 741 | 211 | 106 | 1058 |
| tex - Textured Terrain | 732 | 209 | 105 | 1046 |
| smo - Smooth Terrain | 811 | 232 | 116 | 1159 |
| fss - Mass Wasting | 752 | 215 | 107 | 1074 |
| rid - Ridge | 713 | 203 | 102 | 1018 |
| fse - Slope Streaks | 751 | 215 | 107 | 1073 |
| sfe - Mounds | 704 | 201 | 100 | 1005 |
| fsf - Channel | 820 | 235 | 117 | 1172 |
| fsg - Gullies | 701 | 201 | 100 | 1002 |
| sfx - Crater Field | 939 | 269 | 134 | 1342 |
| cra - Crater | 815 | 233 | 116 | 1164 |
| mix - Mixed Terrain | 710 | 203 | 101 | 1014 |
| ALL CLASSES | 11305 | 3231 | 1614 | 16150 |

### Mapping to my 8 terrain classes (train images = official train split)
| my class | DoMars16k codes | match | train images | note |
|---|---|---|---|---|
| craters | cra | direct | 815 | Crater. (sfx = Crater Field, many small craters, is kept as its own class.) |
| cliffs | cli | direct | 700 | Cliff. |
| ridges | rid | direct | 713 | Ridge. |
| sand dunes | ael, aec | direct | 1452 | Aeolian Straight + Aeolian Curved = wind-formed dunes and ripples. |
| rock fields | rou | approximate | 705 | Rough Terrain is the closest; at CTX resolution single boulders are not visible. |
| volcanic plains | smo | approximate | 811 | Smooth Terrain only. Not labelled as volcanic - smooth plains can also be sedimentary or dusty. |
| subsurface cavities | - | none | 0 | No pit / skylight / lava-tube class in DoMars16k. |
| layered sedimentary deposits | - | none | 0 | No layered-deposit class in DoMars16k. |

- Direct matches: craters, cliffs, ridges, sand dunes. Approximate: rock fields (rough terrain), volcanic plains (smooth terrain).
- No real data in DoMars16k for: subsurface cavities, layered sedimentary deposits. These need another source, or must be left out of the downstream test.

### Leakage check on the official split
{"ctx_products_train": 163, "ctx_products_test": 148, "test_products_also_in_train": 148, "fraction_test_tiles_from_train_products": 1.0}
- Every official test tile comes from a CTX image that also has tiles in train. Test scores on the official split are therefore optimistic, and a LoRA trained on train tiles would have seen the same CTX scenes as the test set.

### Leakage-free geo split (MAIN split for this project)
- All 16,150 tiles re-split so each CTX image is in exactly one split (StratifiedGroupKFold, 7 folds; best of 150 seeds by class balance -> seed 123).
- CTX images per split: {'train': 115, 'val': 21, 'test': 27}. Verified: no CTX image shared between splits.

| feature_name | train | val | test | total |
|---|---|---|---|---|
| aec - Aeolian Curved | 645 | 168 | 245 | 1058 |
| ael - Aeolian Straight | 629 | 79 | 308 | 1016 |
| cli - Cliff | 786 | 108 | 106 | 1000 |
| cra - Crater | 879 | 156 | 129 | 1164 |
| fse - Slope Streaks | 512 | 272 | 289 | 1073 |
| fsf - Channel | 888 | 146 | 138 | 1172 |
| fsg - Gullies | 657 | 100 | 245 | 1002 |
| fss - Mass Wasting | 883 | 117 | 74 | 1074 |
| mix - Mixed Terrain | 489 | 418 | 107 | 1014 |
| rid - Ridge | 642 | 153 | 223 | 1018 |
| rou - Rough Terrain | 844 | 98 | 65 | 1007 |
| sfe - Mounds | 759 | 117 | 129 | 1005 |
| sfx - Crater Field | 1018 | 172 | 152 | 1342 |
| smo - Smooth Terrain | 781 | 159 | 219 | 1159 |
| tex - Textured Terrain | 470 | 489 | 87 | 1046 |
| ALL CLASSES | 10882 | 2752 | 2516 | 16150 |

- Class shares per split are uneven because some CTX images hold hundreds of tiles of one class. Report macro-F1 and per-class F1, not only accuracy.

### Test-set lock
- Fingerprints of the official test file and of the geo-split test IDs saved in data/splits/test_lock.json. Neither test set is used until Phase 6.

### Decisions / open questions
- Phase 2 baseline: pick the model on the geo VAL set. Test-set numbers are computed once, in Phase 6, with all other runs.
- Phase 3 captions must say CTX, not HiRISE. LoRA trains on geo-train tiles only, upscaled from 200 px to 512 px.
- Generate only classes with real data: direct (craters, cliffs, ridges, sand dunes) and, marked approximate, rock fields and volcanic plains.
- Mars-Bench also ships official low-data partitions (1%, 2%, 5%, 10%, 20%, 25%, 50% of train) - planned for Phase 6.
