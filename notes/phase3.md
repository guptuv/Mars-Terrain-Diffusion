## Phase 3 - LoRA fine-tuning (SD 1.5)
- Base model stable-diffusion-v1-5/stable-diffusion-v1-5; LoRA rank 16 (alpha 16) on UNet attention (to_q, to_k, to_v, to_out.0).
- Data: 6000 geo-train tiles (up to 400 per class, all 15 classes), 200 px upscaled to 512 px, VAE latents cached.
- 1000 steps, batch 4, AdamW8bit lr 0.0001, fp16 + gradient checkpointing, caption dropout 0.1, seed 42.
- Final loss (mean of last 100 steps): 0.1942. Figures: results/phase3_lora_loss.png, results/phase3_preview_real_vs_generated.png, results/phase3_guidance_sweep.png.
- Preview sampler: DPM-Solver++ 25 steps, guidance 6.5. Guidance sweep 0 / 3 / 6.5 / 10 shows that guidance 0 (old project) ignores the prompt.
- Caption template: "CTX orbital image of Mars, <class>, <class detail>, grayscale".
