# Setting up nanochat on Lambda Labs

[Lambda Labs](https://cloud.lambda.ai/instances) provides on-demand GPU instances perfect for nanochat training. They offer 8XH100 and 8XA100 configurations at competitive rates with PyTorch and CUDA pre-installed.

## 1. Launch Instance

1. Go to https://cloud.lambda.ai/instances
2. Select an instance type:
   - **8x H100 (80 GB SXM5)**: Recommended, ~$24/hr, perfect for all nanochat workloads
   - **8x A100 (80 GB SXM4)**: Also works well, slightly slower
3. Choose a region with availability
4. Add your SSH key (or generate a new one)
5. Launch the instance

## 2. Connect and Setup

SSH into your instance:
```bash
ssh ubuntu@<instance-ip>
```

Lambda instances come with:
- Ubuntu with CUDA toolkit pre-installed
- PyTorch pre-installed (but we'll use our own via uv)
- Good network bandwidth for downloading datasets

## 3. Install Dependencies

Lambda instances have Python and basic tools, but we need uv and Rust:

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Install Rust (needed for tokenizer)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Clone nanochat (use your fork if you have one)
git clone https://github.com/karpathy/nanochat.git
cd nanochat

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv sync --extra gpu

# Build the Rust tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

## 4. Optional: Setup wandb

If you want experiment tracking:
```bash
wandb login
# Follow the prompts to authenticate
```

## 5. Start Training

Run the full pipeline in a screen session (recommended so training continues if SSH disconnects):

```bash
# Start screen session with logging
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# Detach from screen: Ctrl+A, then D
# Reattach later: screen -r speedrun
# View logs: tail -f speedrun.log
```

Or run individual stages:
```bash
# Tokenizer training (quick, ~5 min)
python -m scripts.tok_train --max_chars=2000000000

# Base model pretraining (longest stage)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20

# Continue with other stages...
```

## 6. Monitor Training

While training runs:

```bash
# Watch the log file
tail -f speedrun.log

# Check GPU utilization
nvidia-smi

# Watch GPU utilization continuously
watch -n 1 nvidia-smi

# Check training progress (if using wandb)
# Visit your wandb dashboard in browser
```

## 7. Download Results

**Important**: Lambda instances have ephemeral storage. Download your results before terminating!

```bash
# On your local machine
scp ubuntu@<instance-ip>:~/nanochat/report.md ./
scp -r ubuntu@<instance-ip>:~/.cache/nanochat/ ./nanochat-checkpoints/
```

See [Local Inference Guide](local-inference.md) for instructions on running your trained model locally.

## Quick Setup Script

Save this as `lambda_setup.sh` and run it after SSHing in for one-command setup:

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

Then just: `bash lambda_setup.sh`

## Lambda-Specific Tips

- **Persistent Storage**: Lambda instances have ephemeral storage. Download checkpoints and results before terminating!
- **Network Speed**: Lambda has excellent network bandwidth—dataset downloads are fast
- **Screen/tmux**: Always use `screen` or `tmux` to keep training running if SSH disconnects
- **Cost Monitoring**: Lambda charges by the hour. Monitor your spending in the dashboard
- **Availability**: H100 instances can have limited availability—launch as soon as you see one available
- **Snapshot Before Shutdown**: If you need to pause, create a persistent filesystem or snapshot your work

## Training Costs

- **$100 tier (d20)**: ~4 hours on 8XH100 (~$96)
- **$300 tier (d26)**: ~12 hours on 8XH100 (~$288)
- **$1000 tier (d32)**: ~41.6 hours on 8XH100 (~$998)

## Troubleshooting

### OOM (Out of Memory) Errors

If you run out of VRAM, reduce `--device_batch_size`:
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --device_batch_size=16
# Or even smaller: 8, 4, 2, 1
```

### SSH Disconnection

If your SSH connection drops:
```bash
# Reconnect and reattach to screen
ssh ubuntu@<instance-ip>
screen -r speedrun
```

### Training Not Using All GPUs

Verify with:
```bash
nvidia-smi
# All 8 GPUs should show processes

# Check if torchrun is running correctly
ps aux | grep torchrun
```

## Next Steps

After training completes:
- Check the `report.md` file for evaluation results
- Download your checkpoints
- See [Local Inference Guide](local-inference.md) to run your model locally
