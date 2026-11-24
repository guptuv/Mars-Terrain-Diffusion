## Phase 0 - Setup (2026-11-24)
- GPU: Tesla T4 (Colab free tier). Settings: SD 1.5 at 512x512, fp16, LoRA batch 1 x grad-accum 4, 8-bit Adam.
- Libraries: torch 2.14.1+cu130, torchvision 0.29.1+cu130, diffusers 0.43.0, peft 0.22.0, accelerate 1.16.0, transformers 5.24.1, timm 1.0.31, torchmetrics 1.9.0, open_clip_torch 4.2.2. Python 3.13.16.
- Global seed 42. All outputs saved under MyDrive/mars_data_engine/.
- Note: pip warned about cudf/pyarrow and diffusers disabled torchao; neither is used, so both are harmless.
