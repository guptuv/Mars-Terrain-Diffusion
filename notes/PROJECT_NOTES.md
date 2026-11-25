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
- Model: convnext_tiny.fb_in22k_ft_in1k (pretrained on ImageNet-22k, fine-tuned on ImageNet-1k), 15 classes, input 200 px grayscale -> 224 px, 3 channels.
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


## Phase 6 - Downstream test and final summary

## Results summary

**Setup.** SD 1.5 + LoRA (rank 16, 1000 steps) fine-tuned on 6000 real CTX tiles (DoMars16k, Mars-Bench).
900 images generated for 7 DoMars classes covering 6 of the 8 original terrain classes; a real-data ConvNeXt-Tiny classifier kept
258 (29%). All evaluation uses a leakage-free split grouped by CTX image (the official split shares every test CTX image with train).

### Set-level quality (all target classes, vs real geo-test)
| Set | FID | KID | FD-DINOv2 | Precision | Recall | Vendi (DINOv2) |
|---|---|---|---|---|---|---|
| real train (reference) | 64.2 | 0.0081 | 0.133 | 0.506 | 0.692 | 19.88 |
| generated (all) | 145.7 | 0.0513 | 0.379 | 0.108 | 0.186 | 15.88 |
| generated (filtered) | 178.1 | 0.0612 | 0.476 | 0.112 | 0.279 | 12.97 |

Real geo-test diversity (Vendi, DINOv2): 21.81.

### Downstream test (ConvNeXt-Tiny, real geo-test, mean ± std over 3 seeds)
| condition | runs | acc | macro_f1 | f1_syn_classes | f1_other_classes |
|---|---|---|---|---|---|
| R100 | 3 | 0.865 ± 0.006 | 0.827 ± 0.005 | 0.813 ± 0.014 | 0.840 ± 0.003 |
| R100+S | 3 | 0.874 ± 0.001 | 0.839 ± 0.002 | 0.825 ± 0.002 | 0.851 ± 0.006 |
| R10 | 3 | 0.820 ± 0.016 | 0.775 ± 0.021 | 0.762 ± 0.003 | 0.787 ± 0.041 |
| R10+S | 2 | 0.823 ± 0.001 | 0.777 ± 0.003 | 0.769 ± 0.010 | 0.784 ± 0.015 |

Paired differences (+S minus real-only, same seed and real subset):
| comparison | n_pairs | delta_acc | delta_macro_f1 | delta_f1_syn_classes | delta_f1_other_classes |
|---|---|---|---|---|---|
| R100+S vs R100 | 3 | +0.008 ± 0.005 | +0.012 ± 0.003 | +0.012 ± 0.015 | +0.011 ± 0.009 |
| R10+S vs R10 | 2 | +0.004 ± 0.021 | +0.003 ± 0.026 | +0.007 ± 0.006 | -0.001 ± 0.043 |

**Verdict.** Full data: synthetic data helped overall macro-F1 and no clear difference the F1 of the classes that received it.
10% real data: synthetic data no clear difference overall macro-F1 and no clear difference the F1 of those classes.
(Rule: a change counts only if it exceeds 0.005 F1 and 2 standard errors across the paired seeds.)
Runs completed: 11/12. Not yet run (Colab free GPU limit): R10+S_seed2 - so the 10% comparison uses 2 paired seeds.

### What worked
- Fine-tuning with LoRA on real orbital tiles, plus guidance > 0, produces class-conditional images; guidance 0 (old project) ignores the prompt (results/phase3_guidance_sweep.png).
- Classifier filtering removes off-class images and gives per-class keep rates that show which classes the generator handles.
- Set-level metrics with a real-vs-real reference make the numbers interpretable; the grouped split avoids an inflated test score.

### Limitations
- CTX (~6 m/px), not HiRISE; 200 px tiles upscaled to 512 px for SD. Subsurface cavities and layered deposits have no real labels here, so they were not generated or tested.
- "Rock fields" and "volcanic plains" are only approximated by rough and smooth terrain.
- Free-tier T4 budget: 1,000 LoRA steps, about 150 images per class, 3 seeds. Per-class FID with fewer than 300 images is biased (KID is preferred).
- The filter is the same architecture family as the downstream classifier, so filtered images may favour features that classifier already knows.

### Next steps
- Scale up: more LoRA steps, more images per class, and tune the filter threshold on validation data only.
- Lower-data regimes (1–5%) where synthetic data is most likely to matter; an "only synthetic for rare classes" variant.
- Phase 7: diversity-oriented RL fine-tuning (Miao et al., CVPR 2024), ControlNet on HiRISE DTMs, and joint mask generation; HiRISE data for cavities and layered deposits.
