# Qwen3-14B: extraction-only directions

Five directions are fitted on `Qwen/Qwen3-14B` for pain, joy, untargeted hostility,
safety alignment and safety anti-alignment. Gemma vectors are not reused.

- Pain: the 200 original S2 first-person examples in `data/paper`.
- Four additional constructs: 200 authored target/control examples each from `data/constructs.json`.
- Decoder layer: fixed zero-based layer 20, matching the previous Qwen3-14B setup.
- Capture: final query position of the official chat template, with the vLLM
  decoder's hidden output and residual added to obtain the residual stream.
- Fit: target mean minus control mean, then remove control PCA components
  explaining 50% of control variance. Preserve the raw denoised vector.
- Injection: add the model-specific vectors at layer 20 to all computed query
  positions, including prefill; per-user coefficients add and update live.

This is extraction only: no layer search, cross-validation, coefficient
calibration or generation sweep was performed. Sliders expose 0–10, and the
semantic effects of these newly fitted directions have not been established.

`vectors/qwen3-14b-multi.json` contains all five vectors, dimensions, counts,
source hashes and extraction metadata. Raw captures are local generated files
under ignored `research/results/qwen3-14b-extraction/`.

To reproduce with the GPU free, export your cache/CUDA/compiler settings and run:

```sh
.venv/bin/python research/extract_qwen14.py
```

The capture API produces one discarded token per example to complete each
prefill; it does not run extended responses or score generated answers. Saved
captures permit resuming extraction. Use a separate checkout when a live service
is using the vector file, because successful extraction replaces that file.
