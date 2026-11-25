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

**Interpretation (important).** With full data the paired gain is similar on classes that received synthetic images (+0.012 ± 0.015)
and on classes that did not (+0.011 ± 0.009). The improvement is therefore probably a general regularization effect of
extra, more varied training batches - not evidence that the generator taught the classifier anything specific about
craters, dunes or plains. With 10% real data the effect is within noise. **Control experiment needed before claiming
a benefit:** add the same number of (a) extra augmented real tiles and (b) synthetic images with shuffled labels; if
those give the same gain, the synthetic content itself is not what helps.

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
