# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

nanochat is a minimal, hackable, full-stack implementation of an LLM training pipeline from scratch. It trains ChatGPT-like models on a single 8XH100 node for budgets ranging from $100 to $1000. The entire pipeline runs end-to-end: tokenization, pretraining, midtraining, supervised finetuning, reinforcement learning, evaluation, and web serving.

**Key Philosophy**: This is NOT a framework. It's a single, cohesive, minimal, readable, hackable baseline. There are no giant configuration objects, model factories, or if-then-else monsters. The code is designed to be forked and modified.

## Quick Start Commands

### Environment Setup
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
uv sync --extra gpu  # For GPU training
uv sync --extra cpu  # For CPU/MPS development

# Activate the virtual environment
source .venv/bin/activate

# Build the Rust tokenizer
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### Training & Running

**Full pipeline (4 hours on 8XH100, ~$100)**:
```bash
bash speedrun.sh
# Or with logging:
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

**Bigger model ($1000 tier, 41.6 hours)**:
```bash
bash run1000.sh
```

**Individual training stages**:
```bash
# Tokenizer training
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval

# Base model pretraining (distributed)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

# Midtraining (teach conversation format)
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid

# Supervised finetuning
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# Reinforcement learning (optional, GSM8K only)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i rl -a GSM8K
```

### Inference & Interaction

```bash
# Web UI (ChatGPT-style interface)
python -m scripts.chat_web

# CLI chat (interactive or single prompt)
python -m scripts.chat_cli                    # Interactive
python -m scripts.chat_cli -p "your prompt"   # Single prompt
```

### Testing

```bash
python -m pytest tests/test_rustbpe.py -v -s
```

## Architecture Overview

### Pipeline Stages

The training pipeline has four main stages, each building on the previous:

1. **Tokenizer**: Custom BPE tokenizer implemented in Rust (`rustbpe/`) for training performance, exported for use with tiktoken
2. **Base Model (Pretraining)**: Standard language modeling on web text (FineWeb dataset)
3. **Midtraining**: Teaches the model conversation format, special tokens, tool use, and multiple-choice tasks
4. **Supervised Finetuning (SFT)**: Domain adaptation to improve per-sequence performance
5. **Reinforcement Learning (RL)**: Optional GRPO-based RL on GSM8K math tasks

### Model Architecture (`nanochat/gpt.py`)

The Transformer implementation includes modern features:
- **Rotary Position Embeddings (RoPE)** instead of learned positional embeddings
- **QK Normalization** for training stability
- **Untied embeddings**: Separate token embedding and lm_head weights
- **ReLU² activation** in MLP layers
- **RMSNorm** (functional, no learnable params) applied after embeddings and before attention/MLP
- **No bias terms** in linear layers
- **Multi-Query Attention (MQA) / Group Query Attention (GQA)** support
- **Logits soft-capping** (tanh(logits/15) * 15) for stability

Model sizing follows a simple depth-based formula:
- `model_dim = depth * 64` (aspect ratio of 64)
- `num_heads = ceil(model_dim / 128)` (head_dim = 128)
- Examples: d20 = 561M params, d26 = ~1B params, d32 = 1.9B params

### Optimizer Strategy (`nanochat/muon.py`, `nanochat/adamw.py`)

Uses a **dual optimizer approach**:
- **Muon** optimizer for all linear layer weights (most parameters)
- **AdamW** for embeddings and lm_head
- Learning rates scale with model dimension: `lr *= (model_dim / 768)^-0.5`
- Momentum scheduling for Muon: ramps from 0.85 → 0.95 over first 300 steps

### Data Pipeline (`nanochat/dataloader.py`, `nanochat/dataset.py`)

- Streams FineWeb data from HuggingFace parquet files (~250MB/shard compressed)
- Tokenizes on-the-fly using multi-threaded Rust BPE tokenizer
- Distributed data loading: each rank processes different shards
- Training follows **Chinchilla scaling**: 20 tokens per parameter
- Data requirements calculation: `(num_params * 20 * 4.8 chars/token) / 250M chars/shard`

### Evaluation Tasks (`tasks/`)

- **CORE**: Custom base metric for language understanding
- **ARC-Challenge/Easy**: Reasoning and world knowledge
- **GSM8K**: Grade school math (with calculator tool use)
- **HumanEval**: Code generation (Python)
- **MMLU**: Broad knowledge across 57 subjects
- **SmolTalk**: Chat quality evaluation
- **SpellingBee**: Character-level counting tasks

Each task implements a standard interface in `tasks/common.py`.

### Configuration System (`nanochat/configurator.py`)

Uses a "poor man's configurator" that:
1. Defines defaults as script-level variables
2. Optionally loads a config file (executed as Python)
3. Parses CLI args as `--key=value` pairs
4. Updates globals() directly (no `config.` prefix needed)

Example:
```bash
python -m scripts.base_train -- --depth=26 --device_batch_size=16
```

### Report Generation (`nanochat/report.py`)

Tracks all metrics across the pipeline and generates a final `report.md` markdown report card with:
- Model configuration and training hyperparameters
- Timing and compute efficiency (MFU, tok/sec)
- Evaluation results across all tasks
- Summary table comparing BASE → MID → SFT → RL performance

## Key Files & Modules

### Training Scripts (`scripts/`)
- `base_train.py`: Pretraining script with automatic Chinchilla scheduling
- `base_loss.py`: Evaluate train/val loss on larger data samples
- `base_eval.py`: Evaluate CORE metric during pretraining
- `mid_train.py`: Midtraining with task mixtures (conversation, tool use, multiple choice)
- `chat_sft.py`: Supervised finetuning on high-quality conversations
- `chat_rl.py`: GRPO reinforcement learning (currently GSM8K only)
- `chat_eval.py`: Comprehensive evaluation across all tasks
- `tok_train.py`: Train the BPE tokenizer
- `tok_eval.py`: Evaluate tokenizer compression ratio

### Core Library (`nanochat/`)
- `gpt.py`: Transformer model implementation
- `engine.py`: Inference engine with KV caching
- `dataloader.py`: Streaming, tokenizing data loader
- `dataset.py`: Dataset downloading and parquet file management
- `tokenizer.py`: Python wrapper for Rust BPE tokenizer
- `checkpoint_manager.py`: Model checkpoint saving/loading
- `core_eval.py`: CORE metric evaluation
- `loss_eval.py`: Loss evaluation helpers
- `common.py`: Distributed training utilities, device detection
- `muon.py`, `adamw.py`: Custom optimizers

### Rust Tokenizer (`rustbpe/`)
- Lightweight BPE training implementation in Rust
- Exports vocab for use with tiktoken
- ~10x faster than minbpe, much simpler than HuggingFace tokenizers
- Built as Python extension using PyO3/maturin

## Computing Environments

### GPU Training (Primary)
- **8XH100 (80GB)**: Primary development target, ~$24/hr
- **8XA100 (80GB)**: Also supported, slightly slower
- **Single GPU**: Works without `torchrun`, uses gradient accumulation (8x slower)
- **< 80GB VRAM**: Reduce `--device_batch_size` until it fits (32→16→8→4→2→1)

### CPU/MPS (Development Only)
- Auto-detects best device: CUDA > MPS > CPU
- Use `uv sync --extra cpu` for CPU-specific PyTorch build
- See `dev/runcpu.sh` for scaled-down hyperparameters
- NOT for serious training, only for code path testing

### Memory Management
The code automatically handles gradient accumulation to maintain target batch size:
```
grad_accum_steps = total_batch_size / (device_batch_size * seq_len * world_size)
```
Reduce `device_batch_size` if you OOM—the code will compensate with more accumulation steps.

## Setting Up on Lambda Labs GPU Instances

Lambda Labs (https://cloud.lambda.ai/instances) provides on-demand GPU instances perfect for nanochat training. They offer 8XH100 and 8XA100 configurations at competitive rates with PyTorch and CUDA pre-installed.

### 1. Launch Instance

1. Go to https://cloud.lambda.ai/instances
2. Select an instance type:
   - **8x H100 (80 GB SXM5)**: Recommended, ~$24/hr, perfect for all nanochat workloads
   - **8x A100 (80 GB SXM4)**: Also works well, slightly slower
3. Choose a region with availability
4. Add your SSH key (or generate a new one)
5. Launch the instance

### 2. SSH into Instance

Once the instance is running:
```bash
ssh ubuntu@<instance-ip>
```

Lambda instances come with:
- Ubuntu with CUDA toolkit pre-installed
- PyTorch pre-installed (but we'll use our own via uv)
- Good network bandwidth for downloading datasets

### 3. Clone Repository

```bash
# Clone the nanochat repository
git clone https://github.com/karpathy/nanochat.git
cd nanochat
```

### 4. Install Dependencies

Lambda instances have Python and basic tools, but we need uv and Rust:

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env  # Make uv available

# Install Rust (needed for tokenizer)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Create virtual environment and install Python dependencies
uv venv
source .venv/bin/activate
uv sync --extra gpu

# Build the Rust tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### 5. Optional: Setup wandb

If you want experiment tracking:
```bash
wandb login
# Follow the prompts to authenticate
```

### 6. Start Training

Now you're ready to run the full pipeline:

```bash
# Run in a screen session so it continues if SSH disconnects
screen -L -Logfile speedrun.log -S speedrun

# Start the training pipeline (4 hours on 8XH100, ~$100)
bash speedrun.sh

# Detach from screen: Ctrl+A, then D
# Reattach later: screen -r speedrun
```

Or run individual stages:
```bash
# Tokenizer training (quick, ~5 min)
python -m scripts.tok_train --max_chars=2000000000

# Base model pretraining (longest stage)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20

# Continue with other stages...
```

### 7. Monitor Training

While training runs:

```bash
# Watch the log file
tail -f speedrun.log

# Check GPU utilization
nvidia-smi

# Check training progress (if using wandb)
# Visit your wandb dashboard in browser
```

### 8. Download Results

After training completes:

```bash
# On your local machine, copy the results
scp -r ubuntu@<instance-ip>:~/nanochat/report.md ./
scp -r ubuntu@<instance-ip>:~/.cache/nanochat/ ./nanochat-checkpoints/
```

### Lambda-Specific Tips

1. **Persistent Storage**: Lambda instances have ephemeral storage. Download checkpoints and results before terminating!
2. **Network Speed**: Lambda has excellent network bandwidth—dataset downloads are fast
3. **Screen/tmux**: Always use `screen` or `tmux` to keep training running if SSH disconnects
4. **Cost Monitoring**: Lambda charges by the hour. Monitor your spending in the dashboard
5. **Availability**: H100 instances can have limited availability—launch as soon as you see one available
6. **Snapshot Before Shutdown**: If you need to pause, create a persistent filesystem or snapshot your work

### Quick Start Script for Lambda

Save this as `lambda_setup.sh` for convenience:

```bash
#!/bin/bash
# Quick setup script for Lambda Labs instances

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Setup Python environment
uv venv
source .venv/bin/activate
uv sync --extra gpu

# Build tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

echo "Setup complete! Run 'bash speedrun.sh' to start training."
```

Then just:
```bash
bash lambda_setup.sh
```

## Model Scaling Guide

To train larger models, adjust three key parameters:

1. **Model depth**: `--depth=N` (determines model_dim, num_heads automatically)
2. **Batch size**: Reduce `--device_batch_size` if you OOM (default: 32)
3. **Data shards**: Calculate required shards based on Chinchilla scaling

Example for d26 (~1B params, ~$300, 12 hours):
```bash
# Download more data (450 shards for ~1B param model)
python -m nanochat.dataset -n 450 &

# Train with depth=26, halve batch size to fit in memory
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16

# Use same batch size in midtraining
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16
```

## Customization

To customize your nanochat's personality, see `dev/gen_synthetic_data.py` which generates synthetic identity conversations. This data gets mixed into midtraining and SFT stages. See [Discussion #139](https://github.com/karpathy/nanochat/discussions/139) for the full guide.

## wandb Integration

```bash
# First time setup
wandb login

# Run with logging
WANDB_RUN=my_run_name bash speedrun.sh
```

Default is `WANDB_RUN=dummy` which disables wandb logging.

## Distributed Training Notes

- Uses PyTorch's `torchrun` for distributed data parallel (DDP)
- Automatically handles gradient accumulation across ranks
- Data loading is distributed: each rank processes different shards
- Only rank 0 does logging, checkpointing, and sampling
- Remove `torchrun` to run on single GPU (automatically switches to gradient accumulation)

## Common Pitfalls

1. **OOM errors**: Reduce `--device_batch_size` progressively until it fits
2. **Not enough data**: Ensure you download enough shards based on model size (see Chinchilla calculation)
3. **Multiple epochs**: If the training script loops over the data multiple times, you need more shards
4. **Forgetting consistent batch sizes**: Use the same `--device_batch_size` in base_train and mid_train
5. **Rotary embeddings cache**: Asserts if sequence length exceeds `rotary_seq_len * 10`

## Development Workflow

1. Modify code in `nanochat/` or `scripts/`
2. The model uses `torch.compile()` for performance (adds ~30s startup time)
3. For debugging, comment out `torch.compile()` line in training scripts
4. Checkpoints saved to `~/.cache/nanochat/` by default (override with `NANOCHAT_BASE_DIR`)
5. Use `files-to-prompt` to package the entire repo for LLM analysis:
   ```bash
   files-to-prompt . -e py -e md -e rs -e html -e toml -e sh --ignore "*target*" --cxml > packaged.txt
   ```

## Performance Expectations

- **MFU (Model FLOPs Utilization)**: ~50% on H100 is good
- **Tokens/sec**: ~80K-85K tokens/sec per step on 8XH100 for d20
- **Wall clock time**: ~4 hours for $100 tier (d20), ~12 hours for d26, ~42 hours for d32
- **CORE metric**: ~0.22 after pretraining, GPT-2 is ~0.23-0.26

## Important Notes

- **Not a framework**: Intentionally simple and hackable, designed to be forked
- **Dependency-lite**: ~330KB of code, well under 100K tokens, fits in LLM context
- **No interactive git commands**: Never use `git add -i`, `git rebase -i`, etc. (not supported)
- **Device autodetection**: Automatically selects best available device (CUDA > MPS > CPU)
- **Checkpoint management**: All intermediate artifacts go to `~/.cache/nanochat/` by default
