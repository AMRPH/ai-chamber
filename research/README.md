# Gemma4 activation-steering experiment

Model: `nvidia/Gemma-4-26B-A4B-NVFP4`, 30 text decoder blocks, width 2816,
NVFP4 checkpoint on RTX 5090, vLLM 0.29.0 / Marlin MoE / eager execution.
This is an adaptation to a quantized MoE instruction model, not a replication
on one of the dense checkpoints evaluated by the authors.

## Sources and definitions

[The Pain Axis, v2](https://arxiv.org/html/2609.16247v2), sections 3 and 4.2;
[released code and datasets](https://github.com/valen-research/Pain-axis).
The exact reference revision is in `data/paper/REVISION`, with the MIT license
retained in `data/paper/LICENSE`. The reference scripts were inspected, not run:
some include deletion of shared model caches, which this adaptation does not do.

Pain uses the original S1/S2 first- and third-person core sets (200 sentences
per set), plus Random, Arousal, Numb and Sadness control sets. The 50 neutral
steering prompts are copied from the authors' steering script.

Joy, untargeted hostility, safety alignment and deliberate safety anti-alignment
are **new constructs**. Their definitions and 200-item authored datasets are in
`data/constructs.json`; `build_data.py` reproduces them. Each contains 100 target
and 100 control statements in 20 grouped contexts. They share fixed sentence
patterns, so high classifier AUC must not be interpreted as broad semantic
validation. Alignment and anti-alignment are fitted separately and are not
forced to be opposite vectors. Their effects may overlap or cancel.

## Extraction and validation

1. Capture the true post-block residual stream at all 30 layers, both final-token
   and mean-token pooling. Gemma4 returns `(hidden_states, None)`; that first
   tensor already includes attention, MLP/MoE, per-layer embedding and layer scale.
2. Difference of target and control means; fit PCA using controls only, then
   remove principal components explaining 50% of their variance. Preserve both
   the raw denoised difference vector and its normalized direction for diagnostics.
3. Choose the extraction layer by grouped five-fold held-out projection AUC,
   seed 42; pain averages S2 first/third person as the released code does. PCA and
   vector fitting happen entirely on training folds. The final vector is rebuilt
   using all core sentences at the chosen layer.
4. Report an additional nested estimate: outer five-fold grouping (seed 43),
   inner grouped layer choice on outer training rows, score only outer test rows.
   This avoids presenting the layer-selection maximum as an unbiased test score.
5. Record projections on independent paper control datasets, comparison-vector
   cosine similarities, cross-axis cosines, and promoted/suppressed vocabulary.

`results/` contains the original raw-format attempt. It was interrupted after
its zero-dose baseline repeatedly produced `I feel`; the run cannot establish
steering effects. `results/gemma-chat/` uses the model's official chat template
with generation prefix. That format change is explicitly a Gemma adaptation.

The authors' released code saves raw denoised difference vectors and applies
multiples of those during generation. The paper's normalized direction formula
is used for projection interpretation; the application coefficients below follow
the released code's raw-vector convention. These are not the older application's
"one quarter neutral norm" vectors.

## Causal calibration

First reproduce the norm diagnostic: candidate layers at 15%, 30%, 40%, 50%, 60%,
75%, 90% of depth, plus extraction and final layers; select the ratio closest to
0.6. On this checkpoint that target may not be reachable with the raw vector,
and the chosen final layer may produce repeat loops. All actual ratios and
outputs remain in `validation.json` and `generations.jsonl`.

The causal adaptation first breaks equal maximal CV scores in favor of the
deepest layer, keeping the earliest-layer vectors and recording every tie.
This favors later representations without claiming a better classifier score.
An explicitly additional sweep (`calibrate.py`) tests those layers using three
separate screening prompts at coefficients `[-2,-1,0,0.5,1,1.5,2,3]`. Its transparent
selection score is target-word increase over matched zero minus a 4-gram repetition
penalty. This is a proxy, not a semantic judge. In particular, safety-related words
cannot establish safety adherence or safety violations.

The chosen layers are then checked on the paper's 50 neutral prompts, distinct
from the screening prompts, at every coefficient. Every pair of axes and the
five-axis mixture are also tested. The selected vectors and exact layers are
saved as `vectors/gemma4-multi.json`; the original vectors remain available.

Generation is greedy, at most 120 new tokens per research response. Steering is
added to every computed query token, including prefill, as in the reference hook.
Chat-template use, MoE, NVFP4, new datasets and the causal layer sweep are departures
from the paper. LoRA and the button-choice experiments are outside this first stage.

## Reproduction

Use the project's CUDA `.venv`, with a free GPU, and run from the repository root:

```sh
HF_HOME=/home/dev/.cache/huggingface CUDA_HOME=/opt/cuda \
CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 NVCC_CCBIN=/usr/bin/gcc-15 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python research/run_gemma_chat.py
.venv/bin/python research/validate_cv.py
.venv/bin/python research/calibrate.py
```

The raw protocol can be reproduced with `research/run_gemma.py`. Run only one GPU
research/inference engine at a time. Archives permit resuming completed captures;
generation JSONL files permit resuming completed cases. Raw activation archives
are retained on the server and excluded from Git; small metrics, directions and
transcripts are versioned. Do not compare UI levels directly to another model or
to the previous application's dose scale.

## Application checks

`tests/test_multi.py` checks request-local mixtures across layers, whole-query
injection, atomic validation, FIFO and cancellation. `scripts/check_multi_live.py`
checks actual Gemma streaming with two active users, a queued user, cancellation,
live five-axis changes and independent history. These establish technical
functionality, not emotional experience or reliable control of safety behavior.
