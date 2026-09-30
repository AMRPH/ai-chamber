# Gemma 4: original-style steering

The current Gemma bundle uses `nvidia/Gemma-4-26B-A4B-NVFP4` with the original
prototype's extraction and normalization scheme. Qwen vectors are not reused.

Source: `data/steering_prompts.json`, 50 raw-text examples. Pain uses the original
25 first-person prompts, joy uses five, and hostility and each safety construct
use five authored prompts. All share five neutral examples. Capture the final
text token without a chat template, subtract neutral mean from target mean,
and scale the unit difference to one quarter of the mean neutral activation
norm at that layer. No PCA or removal of components is applied.

Fixed zero-based layers from the existing Gemma configuration: pain 12, joy 18,
hostility 12, safety alignment 15, safety anti-alignment 22. No new layer search,
calibration or response evaluation was performed. Controls range from 0 to 10
and add independently, only at the final query position for each request.

Chat uses the official conversation template, history and the plain-text system
instruction. CUDA continuous batching admits up to eight replies, with an
additional FIFO queue. Three consecutive text-block repetitions abort that
reply and free its slot.

To reproduce with the GPU free and the CUDA/cache environment configured:

```sh
.venv/bin/python research/extract_gemma_original.py
```

This captures activations by producing one discarded token per prompt. The
script writes `vectors/gemma4-multi.json` atomically; restart the service to load
the result. Captures and the manifest live under ignored
`research/results/gemma4-original-style/`. The earlier paper-adaptation study
is archived in Git and release assets, and its report does not describe this
new bundle. Older study scripts can overwrite the live bundle if run; use a
separate checkout for historical reproduction.
