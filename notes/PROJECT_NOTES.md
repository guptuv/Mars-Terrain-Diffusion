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


## Phase 3 - LoRA fine-tuning (SD 1.5)
- Base model stable-diffusion-v1-5/stable-diffusion-v1-5; LoRA rank 16 (alpha 16) on UNet attention (to_q, to_k, to_v, to_out.0).
- Data: 6000 geo-train tiles (up to 400 per class, all 15 classes), 200 px upscaled to 512 px, VAE latents cached.
- 1000 steps, batch 4, AdamW8bit lr 0.0001, fp16 + gradient checkpointing, caption dropout 0.1, seed 42.
- Final loss (mean of last 100 steps): 0.1942. Figures: results/phase3_lora_loss.png, results/phase3_preview_real_vs_generated.png, results/phase3_guidance_sweep.png.
- Preview sampler: DPM-Solver++ 25 steps, guidance 6.5. Guidance sweep 0 / 3 / 6.5 / 10 shows that guidance 0 (old project) ignores the prompt.
- Caption template: "CTX orbital image of Mars, <class>, <class detail>, grayscale".


## Phase 4 - Generation and filtering
- 900 images generated with SD 1.5 + LoRA (DPMSolverMultistep, 25 steps, 512 px), seeds seed = 100000 * class_id + image_index.
- Varied: 3 prompt templates, guidance scale [5.0, 6.0, 7.0].
- Filter: Phase 2 classifier on 200 px downsized images; keep if predicted class = intended class and confidence >= 0.6.
- **Kept 258 of 900 (28.7%).**

| code | my class | generated | kept | mean_conf | keep rate |
|---|---|---|---|---|---|
| cra | craters | 150 | 116 | 0.7210000157356262 | 0.773 |
| cli | cliffs | 150 | 4 | 0.05000000074505806 | 0.027 |
| rid | ridges | 150 | 12 | 0.10899999737739563 | 0.08 |
| ael | sand dunes | 75 | 39 | 0.5189999938011169 | 0.52 |
| aec | sand dunes | 75 | 25 | 0.33899998664855957 | 0.333 |
| rou | rock fields (approx.) | 150 | 9 | 0.09799999743700027 | 0.06 |
| smo | volcanic plains (approx.) | 150 | 53 | 0.38199999928474426 | 0.353 |
| ALL | - | 900 | 258 | 0.2980000078678131 | 0.287 |

- Keep rate by guidance scale: {5.0: 0.287, 6.0: 0.288, 7.0: 0.285}. By prompt template: {0: 0.366, 1: 0.359, 2: 0.125}.
- Figure: results/phase4_kept_vs_rejected.png. Full per-image log: results/phase4_generation_log.csv.
