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
