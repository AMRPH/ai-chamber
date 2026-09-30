# AI Chamber

A minimal chat app for experimenting with live activation steering in open language models.
Move a slider while a reply streams to change the direction added to the model's hidden activations. Sliders do not add emotion instructions to the prompt. A system instruction asks the model to roleplay a human conversation partner, stay in character without repeated AI or nonliving disclaimers, and use plain text without Markdown.

- Gemma 4: five independent controls — pain, joy, untargeted hostility, safety alignment and anti-alignment.
- Qwen3-14B: five independently fitted controls on CUDA and Transformers; Qwen3-4B retains its pain-only direction.
- Up to eight concurrent replies on CUDA by default; additional users wait in a FIFO queue.
- Generation is aborted after three consecutive copies of the same text block.
- Separate histories, controls and cancellation for each browser connection.
- Russian, English, Spanish and Chinese interface; the language selector changes only the interface.
- A fixed model display with no model picker. Compact flag-based interface language selection beside the title. No frontend build step.

**Experimental:** extracted directions do not establish subjective feelings or reliable control of safety behavior. The archived Gemma study used a different control-PCA method and did not confirm consistent amplification of pain, joy or hostility. The current Gemma bundle uses raw-text extraction with prototype scaling; no generation sweeps were performed for this version. See the [results and limitations](docs/gemma4-five-directions.md) and [research method](research/README.md).

## Requirements

| Backend | Models | Hardware / installation |
|---|---|---|
| CUDA / vLLM | Gemma 4 NVFP4, Qwen3-4B, Qwen3-14B | Linux, Python 3.13, compatible NVIDIA driver and CUDA compiler. Gemma setup tested on RTX 5090 (32 GB); NVFP4 requires Blackwell. |
| MPS / Transformers | Qwen3-4B, Qwen3-14B | Apple Silicon, Python 3.13 or 3.14. Qwen3-4B tested on a 24 GB Mac; 14B needs more than 30 GB available memory. |
| CPU / Transformers | Qwen3 | Supported for experimentation; substantial RAM and much slower generation. |

Model weights download from Hugging Face on first startup and are not included in the source archive. Allow storage for the checkpoint and any access permissions required by its model card. Use your own `HF_TOKEN` if needed.

## Download and run

```sh
git clone https://github.com/AMRPH/ai-chamber.git
cd ai-chamber
python3.13 -m venv .venv
cp .env.example .env
```

Alternatively, download and extract the [latest source ZIP](https://github.com/AMRPH/ai-chamber/archive/refs/heads/main.zip), then run the same commands inside the extracted directory, starting at `python3.13 -m venv .venv`.

### NVIDIA / Gemma 4

```sh
.venv/bin/python -m pip install -r requirements-cuda.txt
.venv/bin/python -m uvicorn chat:app --env-file .env --host 127.0.0.1 --port 8000
```

The sample `.env` loads Gemma 4 NVFP4 automatically on `cuda:0`. Set `CHAMBER_MODEL=qwen3-14b` to use Qwen instead. Open **http://127.0.0.1:8000/** and wait for the model to become ready. First load can take several minutes. If your CUDA toolkit or compiler is not on the default search path, set the corresponding entries in `.env`.

### Apple Silicon / Qwen

Set these values in `.env`:

```dotenv
CHAMBER_DEVICE=mps
CHAMBER_MODEL=qwen3-4b
```

Then run:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn chat:app --env-file .env --host 127.0.0.1 --port 8000
```

Use `CHAMBER_DEVICE=cpu` for CPU execution. Install one backend in each virtual environment; create separate environments when switching between CUDA and MPS.

## Configuration

| Variable | Meaning |
|---|---|
| `CHAMBER_DEVICE` | Explicit `cuda:0`, `mps` or `cpu`. Always set this for CUDA / vLLM. |
| `CHAMBER_MODEL` | Auto-load `gemma4-nvfp4`, `qwen3-4b` or `qwen3-14b`. Blank starts with no model. |
| `CHAMBER_PARALLEL` | Maximum concurrent replies; default `8` on CUDA and `2` on MPS/CPU. Remaining users queue. |
| `CHAMBER_GPU_MEMORY` | Fraction of GPU memory vLLM may use; default `0.95`. CUDA only. |
| `CHAMBER_CONTEXT` | Context token limit including history; default `4096`, with up to 512 new reply tokens. |
| `CHAMBER_MODEL_PASSWORD_SHA256` | SHA-256 digest of your own model-switch password. Blank disables switching. |
| `HF_HOME` | Optional model-cache location; default `.cache/huggingface` inside this directory. |
| `HF_TOKEN` | Optional Hugging Face access token. Keep it in your local `.env`. |

To generate the password digest without showing or storing the password:

```sh
.venv/bin/python -c 'import getpass, hashlib; print(hashlib.sha256(getpass.getpass("Model-switch password: ").encode()).hexdigest())'
```

Paste the result into `CHAMBER_MODEL_PASSWORD_SHA256` in `.env`. The interface has no model picker; choose the startup model through `CHAMBER_MODEL`. The password protects the administrative model-loading API; chat access itself is public unless your reverse proxy restricts it. Never commit `.env`.

Run **one Uvicorn worker** per model. Each browser connection owns an in-memory conversation; reloading starts a fresh one. When the model context fills, the oldest complete user/assistant turns are discarded automatically. The system prompt and latest user message remain, with 512 tokens reserved for the reply. The browser keeps displaying the full conversation, while the server retains only the recent context. A latest message that is too long by itself is rejected. Controls range from 0 to 10. Qwen3-14B uses raw-text target-minus-neutral directions scaled to one quarter of the mean neutral activation norm; the archived Gemma calibration study used an earlier bundle and covered coefficients up to 3. Multiple directions add, but their semantic effects may overlap. Zero stops new injection and does not erase earlier effects from the history or attention cache. Qwen3-4B uses its original pain-only direction and a different scale.

Eight is the maximum admitted CUDA concurrency, not a guarantee that eight full 4096-token histories fit at once. vLLM schedules requests according to the available KV cache. Increasing concurrency shares throughput between users.

A streaming repetition detector aborts a reply after three consecutive identical blocks, ignoring whitespace changes. Blocks can be words, phrases, sentences or paragraphs; Chinese text is supported without spaces. Punctuation alone does not trigger a stop. The detector counts only the current reply and never earlier conversation history. CUDA cancels the affected engine request immediately; Transformers stops through its token-generation stopping criterion. Other users continue normally.

CUDA uses vLLM 0.29.0 continuous batching with an isolated model process. Compilation, CUDA graphs and prefix caching are disabled so live hooks execute per request. Gemma 4 and Qwen3-14B apply all five directions at the last query position per request, including prefill, using the original prototype scale without PCA. Pain-only Qwen uses the same position rule. Earlier Gemma study results used whole-query steering and remain historical records. MPS uses Transformers with thread-local hooks.

## Deploy, test and reproduce

- [Linux service and nginx deployment](docs/deployment.md)
- [Research method, datasets and archived results](research/README.md)
- [Current Gemma original-style extraction](docs/gemma4-original-extraction.md)
- [Archived Gemma study report](docs/gemma4-five-directions.md)
- [Qwen3-14B original-style extraction](docs/qwen3-14b-extraction.md)
- [Release downloads](https://github.com/AMRPH/ai-chamber/releases/latest)

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m unittest discover -s tests
```

Tests use mocked model layers and do not download weights. GitHub Actions runs them on CPU. Checks against an already running model are optional and send real chat requests:

```sh
.venv/bin/python scripts/check_multi_live.py --url http://127.0.0.1:8000/
```

The multi-user check expects Gemma loaded with `CHAMBER_PARALLEL=2`; it verifies streaming, live levels, queueing, cancellation and independent histories. Research runs require the GPU to be free and are separate from serving the chat.

## Project layout

```text
chat.py                 HTTP and WebSocket interface, sessions and FIFO queue
runtime.py              Model registry, controls and Transformers backend
gpu_runtime.py          CUDA process supervisor
cuda_engine.py          vLLM engine and request protocol
chamber_worker.py       Activation hooks and request-to-token mapping
repetition.py           Streaming triple-repeat detection
context_window.py       Automatic trimming of old conversation turns
static/index.html       Single-file interface
data/                   Extraction datasets and source attribution
vectors/                Model-specific vectors needed to run the chat
research/               Reproducible Gemma extraction and validation scripts
deploy/                 Portable service and reverse-proxy examples
tests/                  CPU tests without model weights
```

Old prototype experiments, publication drafts and generated research outputs are excluded from the current source tree. The complete recorded Gemma study is available as a separate release asset; older prototypes remain recoverable in Git history.

## License and attribution

Project code is released under the [MIT License](LICENSE). The upstream Pain-axis dataset license is preserved in [data/paper/LICENSE](data/paper/LICENSE); see [NOTICE](NOTICE). Model weights retain their own upstream licenses. Bug reports and focused pull requests are welcome; include your backend, model, hardware and steps to reproduce, without tokens or passwords.
