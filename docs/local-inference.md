# Running Your Trained Model Locally

After training your nanochat model on a GPU server (like Lambda Labs), you can download the checkpoints and run inference locally on your Mac or any other machine. nanochat supports CPU and MPS (Apple Silicon GPU) inference.

## Table of Contents

- [Downloading Your Model](#downloading-your-model)
- [Local Setup](#local-setup)
- [Running Inference](#running-inference)
- [Performance Expectations](#performance-expectations)
- [Troubleshooting](#troubleshooting)

## Downloading Your Model

After training completes on your GPU instance, download the checkpoint and report:

```bash
# On your local machine, download checkpoints
scp -r ubuntu@<instance-ip>:~/.cache/nanochat/ ./nanochat-checkpoints/

# Also download the report card
scp ubuntu@<instance-ip>:~/nanochat/report.md ./
```

The checkpoint directory contains:
- `tok.model` - The trained tokenizer
- `base_*.pt` - Pretrained base model checkpoints
- `mid_*.pt` - Midtraining checkpoints
- `sft_*.pt` - Supervised finetuning checkpoints
- `rl_*.pt` - Reinforcement learning checkpoints (if trained)

## Local Setup

### 1. Install Dependencies

If you haven't already set up nanochat locally:

```bash
# Clone the repo (or use your fork)
git clone https://github.com/karpathy/nanochat.git
cd nanochat

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Install Rust (needed for tokenizer)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Create virtual environment with CPU/MPS-optimized PyTorch
uv venv
source .venv/bin/activate
uv sync --extra cpu  # Use CPU-optimized PyTorch for Mac

# Build the Rust tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### 2. Set Up Checkpoint Location

You have two options:

**Option A: Use custom checkpoint directory**
```bash
# Set environment variable to point to your downloaded checkpoints
export NANOCHAT_BASE_DIR=/path/to/nanochat-checkpoints
```

**Option B: Use default location**
```bash
# Copy checkpoints to the default location
mkdir -p ~/.cache/nanochat
cp -r nanochat-checkpoints/* ~/.cache/nanochat/
```

## Running Inference

### Web UI (Recommended)

Start the ChatGPT-style web interface:

```bash
python -m scripts.chat_web
```

Then open your browser to the URL shown (typically http://localhost:8000).

The web UI provides:
- ChatGPT-like conversation interface
- Conversation history
- Model parameter information
- Clean, modern UI

### CLI Chat

For command-line interaction:

```bash
# Interactive mode
python -m scripts.chat_cli

# Single prompt mode
python -m scripts.chat_cli -p "Tell me a story about a robot"

# With custom checkpoint
python -m scripts.chat_cli -i sft  # Use SFT checkpoint (default)
python -m scripts.chat_cli -i mid  # Use midtraining checkpoint
python -m scripts.chat_cli -i rl   # Use RL checkpoint (if available)
```

### Device Selection

nanochat automatically detects and uses the best available device:

1. **CUDA** (NVIDIA GPUs) - if available
2. **MPS** (Apple Silicon) - if on M1/M2/M3 Mac
3. **CPU** - fallback for all systems

You can verify which device is being used by checking the startup logs.

## Performance Expectations

Inference performance varies significantly by hardware:

### Apple Silicon (MPS)

- **M1/M2/M3 Base**: ~5-15 tokens/sec
- **M1/M2/M3 Pro**: ~10-25 tokens/sec
- **M1/M2/M3 Max/Ultra**: ~20-50 tokens/sec

### Intel Mac (CPU)

- **Modern Intel (8+ cores)**: ~3-10 tokens/sec
- **Older Intel**: ~1-5 tokens/sec

### Linux/Windows CPU

- **High-end Desktop (16+ cores)**: ~5-15 tokens/sec
- **Standard Desktop (8 cores)**: ~3-8 tokens/sec
- **Laptop**: ~1-5 tokens/sec

### Comparison to GPU Training

- **H100 Training**: ~80,000-85,000 tokens/sec (per training step, 8 GPUs)
- **Local Inference**: ~5-50 tokens/sec (single device)

While local inference is ~1000x slower than training on 8XH100, it's perfectly fine for interactive chat and testing your model!

## Model Sizes and Memory Requirements

Different model sizes have different memory requirements:

| Model | Parameters | RAM/VRAM Required | Typical Use |
|-------|-----------|-------------------|-------------|
| d20   | 561M      | ~4GB              | $100 speedrun, best for local |
| d26   | ~1B       | ~6GB              | $300 tier |
| d32   | 1.9B      | ~10GB             | $1000 tier |

Most modern Macs with 16GB+ unified memory can easily run d20 and d26 models.

## Advanced Usage

### Running Different Checkpoints

You can run different stages of training to compare performance:

```bash
# Base model (pretrained only)
python -m scripts.chat_cli -i base

# Midtraining (conversational format)
python -m scripts.chat_cli -i mid

# Supervised finetuning (default, best quality)
python -m scripts.chat_cli -i sft

# Reinforcement learning (if trained)
python -m scripts.chat_cli -i rl
```

### Custom Checkpoint Directory

```bash
# Run with custom checkpoint location
NANOCHAT_BASE_DIR=/path/to/checkpoints python -m scripts.chat_web
```

### Generation Parameters

You can modify generation parameters by editing the script or using environment variables (see script source for available options).

## Troubleshooting

### "Checkpoint not found" Error

```bash
# Check if checkpoints exist
ls -la ~/.cache/nanochat/

# Or if using custom directory
ls -la $NANOCHAT_BASE_DIR

# Verify you have the right checkpoint type (base, mid, sft, rl)
ls -la ~/.cache/nanochat/sft_*.pt
```

### Slow Performance

1. **Verify device selection**: Check startup logs to confirm MPS/GPU is being used
2. **Close other applications**: Free up RAM/VRAM
3. **Use smaller model**: Try d20 instead of d26/d32
4. **Check thermal throttling**: Especially on laptops

### MPS Not Being Used (Mac)

```bash
# Check if MPS is available
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# If False, make sure you:
# 1. Have Apple Silicon (M1/M2/M3)
# 2. Installed uv sync --extra cpu (includes MPS support)
# 3. Have macOS 12.3 or later
```

### Out of Memory

If you run out of memory on your local machine:

1. **Close other applications**
2. **Use a smaller model** (d20 instead of d26/d32)
3. **Reduce batch size** (if doing batch inference)
4. **Use CPU instead of MPS** (if MPS is causing issues)

### Tokenizer Errors

If you see tokenizer-related errors:

```bash
# Rebuild the tokenizer
cd rustbpe
cargo clean
cd ..
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

## Training vs. Inference Comparison

One of the unique aspects of nanochat is that you can compare the same model at different stages:

1. **Train on powerful GPUs** (~4 hours, $100)
2. **Download checkpoints** (small, ~2GB per checkpoint)
3. **Run locally** (free, any device)
4. **Compare stages**: See how base → mid → sft improves quality

This makes nanochat perfect for learning and experimentation without continuous GPU costs!

## Next Steps

- Try different prompts to explore your model's capabilities
- Compare different checkpoint stages (base, mid, sft)
- Experiment with custom personality (see [Customization Guide](https://github.com/karpathy/nanochat/discussions/139))
- Share your results with the community

## Tips for Best Experience

1. **Use SFT checkpoint by default** - It has the best conversation quality
2. **Start with simple prompts** - These micro models work best with clear, direct questions
3. **Be patient with responses** - Local inference is slower but works well
4. **Experiment with different checkpoints** - Each stage has different characteristics
5. **Monitor memory usage** - Especially on machines with limited RAM

---

**Remember**: Your model is truly yours! You trained it, you own it, and you can run it anywhere without API costs or usage limits.
