Aligned v3 body/EOS diagnostic A/B
================================
A canvas-mean: existing full valid-canvas CE mean.
B body-tail-balanced: 0.5*body CE mean + 0.5*EOS padding-tail CE mean.
If no padding tail exists, B uses body CE mean only. Body includes the original
response's <turn|> token. B is a diagnostic variant, NOT official section 4.

Both arms start from the same AR E2B checkpoint and conditioner seed, C=256,
K=18, same 32 train/16 validation rows, per-step noise seed, BF16 AdamW,
LR 1.125e-4, 50-step warmup, 400 steps. All 48 rows are generated at baseline
and steps 100/200/300/400 with seeds 42,43 and max 48 denoising steps.
Long outputs are capped at 768 tokens; targets here are <=471 tokens.
Recovery inputs use full 256-token EOS-padded canvases. Report body/tail
separately; do not compare total A/B loss as if objectives were identical.

Run from Colab after extracting into /content and mounting Drive:
python -u /content/scripts/run_loss_controls.py

Compact progress streams live. Raw logs and results:
/content/drive/MyDrive/saves/e2b_loss_controls_v3/{canvas-mean,body-tail-balanced}

Summarize without GPU/model:
python /content/scripts/summarize_loss_controls.py

Checkpoint saving at each evaluation includes model, conditioner, tokenizer,
Adam state, step, sample order, random states and settings. COMPLETE is written
only after all files are saved. Previous completed checkpoint is deleted only
after the new one completes. Final model is also saved as final_model.
This runner refuses pre-existing arm directories; it does not automatically
resume. Keep recovery files if interrupted; do not rerun over the old directory.
Allow roughly 40 GiB retained per arm (BF16 optimizer+model+final weights), plus
space for the old/new checkpoints during save. Drive writes and generation
checks add substantial time; no E2B wall-clock estimate has been measured for
this exact v3 A/B experiment. Do not start both arms concurrently.
