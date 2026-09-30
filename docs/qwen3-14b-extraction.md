# Qwen3-14B: original-style steering directions

Five directions are fitted on `Qwen/Qwen3-14B` for pain, joy, untargeted hostility,
safety alignment and safety anti-alignment. This uses the original prototype's
extraction and scaling scheme, rather than the paper's control-PCA method.

- Pain: the original 25 first-person examples.
- Joy: the original five first-person examples.
- Hostility and the two safety constructs: five authored examples each.
- All five directions share the same five neutral sentences.
- Source: `data/steering_prompts.json`, 50 sentences in total.
- Decoder layer: fixed zero-based layer 20.
- Capture: final token of raw text, without a chat template. The vLLM decoder's
  hidden output and residual are added to obtain the residual stream.
- Fit: target mean minus neutral mean, without PCA or removal of components.
- Scale: normalize that difference to unit length, then multiply it by one
  quarter of the mean norm of the neutral activations. All five directions use
  the same magnitude.
- Injection: at layer 20, add the weighted sum of directions only at the last
  query position for each user, both during prefill and token generation.
  Coefficients are independent and update live, from 0 to 10.

Ordinary chat, conversation history and the plain-text system instruction stay
in place. The extraction prompts are not added to users' conversations.

This is extraction only: no layer search, cross-validation, coefficient
calibration or generation sweep was performed. Matching the prototype method
and scale does not establish identical behavior across different model sizes.
The newly authored constructs have not been semantically validated.

`vectors/qwen3-14b-multi.json` contains all five vectors, dimensions, counts,
source hashes, normalization and injection metadata. Captures are generated
files under ignored `research/results/qwen3-14b-original-style/`; previous
control-PCA captures are separate.

To reproduce with the GPU free, export your cache/CUDA/compiler settings and run:

```sh
.venv/bin/python research/extract_qwen14.py
```

The capture API produces one discarded token per example to finish each prefill;
it does not run extended responses or score generated answers. Saved captures
are reused only if the extraction manifest matches. Successful extraction
atomically replaces the Qwen3-14B bundle. Restart the serving process afterwards
to load it. The archived Gemma study retains its original method and results.
