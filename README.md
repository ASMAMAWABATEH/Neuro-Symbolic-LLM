# Neuro-Symbolic LLM

A neuro-symbolic continual learning framework combining a frozen foundation language model with intermediate hidden-state residual adapters and symbolic reasoning modules.

---

## Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │              Frozen Base LLM           │
                      │             (FrozenSubstrate)          │
Input Tokens ────────►│  Block 0 ──► Block 1 ──► ... Block L-1 ├─────► Logits
                      └─────────────┬──────────────────────────┘
                                    │ Intercept h_l
                                    ▼
                      ┌────────────────────────────────────────┐
                      │        ResidualAdapter (JAX)           │
                      │                                        │
                      │  z = σ(h_l · A_l)     [d -> r]         │
                      │  Δh = s · (z · B_l)   [r -> d]         │
                      └─────────────┬──────────────────────────┘
                                    │ Add Δh
                                    ▼
                             h̃_l = h_l + Δh
```

- **Base Model (`frozenllm`)**: Runs inside [`FrozenSubstrate`](frozenllm/README.md) using PyTorch/TorchAX. The base model weights remain strictly frozen ($\nabla \theta_0 = 0$) and are verified before/after execution.
- **Residual Adapter (`residual`)**: Implemented in JAX. Adds a low-rank bottleneck $\mathbb{R}^d \to \mathbb{R}^r \to \mathbb{R}^d$ to intermediate hidden states of selected layers (e.g., late layers), without modifying model weights.
- **Continual Learning Stages (`stages`)**:
  - **Stage A1**: Sequential multi-domain continual fine-tuning across legal domains (EUR-Lex &rarr; LEDGAR &rarr; SCOTUS) with catastrophic forgetting measurement.

---

## Quickstart

### Option 1: Docker (Recommended)

Docker provides an isolated, reproducible environment with all JAX, PyTorch, and TorchAX dependencies configured. It also avoids macOS Intel wheel incompatibilities for `torch>=2.6`.

#### 1. Build the Docker image
```bash
make build
# or: docker build -t neuro-symbolic-llm .
```

#### 2. Run the test suite
```bash
make test
# or: docker run --rm -v "$PWD":/app neuro-symbolic-llm pytest
```

#### 3. Run a smoke experiment
```bash
make smoke
# or:
docker run --rm -v "$PWD":/app -v nsllm-hf-cache:/root/.cache/huggingface neuro-symbolic-llm \
    python experiments/run_stage_a1.py --config configs/stage_A/a1_lora_baseline.yaml \
    --epoch 1 --rank 8 --run-name smoke --eval-batches 2
```

---

### Option 2: Local Virtual Environment

Requires Python &ge; 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

Run tests locally:
```bash
pytest
```

---

## Configuration Reference

Experiments are configured via YAML files (e.g. [`configs/stage_A/a1_lora_baseline.yaml`](configs/stage_A/a1_lora_baseline.yaml)). Unknown keys are rejected.

| Section | Parameter | Type | Description |
|---|---|---|---|
| **`adapter`** | `rank` | `int` | Bottleneck dimension $r$ (e.g. 8, 16) |
| | `activation` | `str` | Residual activation: `identity`, `gelu`, `relu`, `tanh` |
| | `alpha` | `float \| null` | Scaling factor (defaults to 1.0 when null) |
| | `layers` | `list[int] \| null` | Layer indices to adapt (null uses default late layers) |
| **`objective`** | `lambda_kl` | `float` | Weight for KL divergence against base logits |
| | `lambda_wd` | `float` | Weight decay penalty on adapter parameters |
| **`training`** | `model` | `str` | Hugging Face model identifier (e.g. `gpt2`) |
| | `seq_len` | `int` | Sequence length for token blocks (e.g. 128) |
| | `batch_size` | `int` | Training batch size (e.g. 16, 64) |
| | `epochs` | `int` | Number of epochs per domain |
| | `learning_rate`| `float` | Adam learning rate (e.g. 1e-3) |
| | `eval_batches`| `int` | Evaluation batch budget per domain |
| | `output_dir` | `str` | Base directory for experiment runs (default: `runs`) |
| **`domains`** | `list[dict]` | `list` | Sequence of domains to train continually |

---

## Running Experiments

Use [`experiments/run_stage_a1.py`](experiments/run_stage_a1.py) with CLI overrides:

```bash
# Run baseline configuration
python experiments/run_stage_a1.py --config configs/stage_A/a1_lora_baseline.yaml

# Overrides: rank, epochs (or --epoch), model, eval-batches, run-name
python experiments/run_stage_a1.py --config configs/stage_A/a1_lora_baseline.yaml \
    --rank 8 \
    --epochs 1 \
    --eval-batches 4 \
    --output-dir runs \
    --run-name experiment_smoke
```

### CLI Arguments

- `--config`: Path to the YAML configuration file.
- `--epochs` / `--epoch`: Override number of training epochs for all domains.
- `--rank`: Override residual adapter rank $r$.
- `--model`: Override the base model identifier (e.g. `gpt2`).
- `--output-dir`: Destination folder for run artifacts (default: `runs`).
- `--run-name`: Run folder name (default: auto-generated timestamped name).
- `--eval-batches`: Override evaluation batch count.
- `--eval-checkpoint`: Path to evaluate a saved `.npz` checkpoint without training.
- `--no-jit`: Disable JAX JIT compilation (useful for interactive debugging).

---

## Run Artifacts

Each experiment creates an isolated folder `<output_dir>/<run_name>/`:

| File | Content |
|---|---|
| `config.yaml` | Fully resolved experiment configuration |
| `meta.json` | Commit hash, library versions, model architecture details, parameter counts |
| `metrics.jsonl` | Step-by-step train loss, step time, and evaluation metrics |
| `results.json` | Complete cross-domain evaluation matrix, loss, perplexity, and forgetting metrics |
| `params/task_<t>_<name>.npz` | Saved adapter weights after each domain task |

---

## Developer Workflows

A `Makefile` and `docker-compose.yml` are provided for standard developer workflows:

```bash
make help          # Show available targets
make build         # Build the Docker container
make test          # Run all 225 unit and integration tests
make lint          # Run ruff static analysis
make format        # Format code with ruff and black
make smoke         # Execute Stage A1 smoke test
make shell         # Open interactive bash in the Docker environment
```

Or using Docker Compose:
```bash
docker compose run --rm test
docker compose run --rm app python experiments/run_stage_a1.py --help
```
