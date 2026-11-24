# Mars-Terrain-Diffusion: a Mars data engine

Generating **labeled synthetic Martian terrain images** with a LoRA-fine-tuned Stable Diffusion model,
filtering them with a classifier trained on real data, judging them with set-level metrics
(FID, KID, precision/recall, Vendi), and testing whether they actually help a downstream classifier.

This project follows up an earlier prompt-only Stable Diffusion v1.4 study (about 730 images, guidance scale 0,
pairwise metrics only, no downstream test) and fixes its main weaknesses.

## Pipeline status
| Phase | What | Status |
|---|---|---|
| 0 | Setup: GPU, pinned libraries, Drive layout, seeds, run logging | done |
| 1 | Real data: DoMars16k, class mapping, leakage check, CTX-grouped split | done |
| 2 | Baseline classifier (real data only) | done |
| 3 | LoRA fine-tuning of Stable Diffusion 1.5 | to do |
| 4 | Generation + classifier filtering | to do |
| 5 | Set-level evaluation: FID, KID, precision/recall, Vendi | to do |
| 6 | Downstream test: real vs real + synthetic | to do |
| 7 | Optional extensions | to do |

Full, auto-generated notes and numbers: [notes/PROJECT_NOTES.md](notes/PROJECT_NOTES.md)

## Key decisions so far
- **Data:** DoMars16k from Mars-Bench (Hugging Face Mirali33/mb-domars16k): 16,150 CTX tiles (~6 m/px), 15 classes.
- **Leakage found:** every CTX image in the official test split also has tiles in train. We use a
  **CTX-image-grouped split** (splits/geo_split.csv) as the main split; the official split is kept only for comparison.
- **Classes without real data** (subsurface cavities, layered deposits) are not generated, because they cannot be checked.

## Repository layout
```
notebooks/   Colab notebook (run top to bottom, T4 GPU)
src/         helper code (seeding, run logging, GitHub sync)
notes/       running project notes, one section per phase
configs/     JSON settings of every run (seeds, versions, hyper-parameters)
splits/      train/val/test split files, class mapping, test-set lock (hashes)
results/     tables and figures
```
Datasets, checkpoints, and generated images are kept on Google Drive (too large for git).

## Reproduce
Open notebooks/Mars_Data_Engine.ipynb in Colab with a T4 GPU and run the cells in order.
Library versions are pinned in the install cell and recorded in configs/phase0_environment.json.

## Data credit
DoMars16k: Wilhelm et al., 2020; distributed via Mars-Bench (Purohit et al., 2025), CC-BY-4.0.

_Last synced: 2026-11-24 22:01_
