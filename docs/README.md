# nanochat Documentation

Welcome to the nanochat documentation! This directory contains detailed guides for various aspects of training and running nanochat.

## Setup Guides

### [Lambda Labs Setup Guide](lambda-setup.md)
Complete walkthrough for setting up and training nanochat on Lambda Labs GPU instances (8XH100 or 8XA100). Includes:
- Instance launch and SSH setup
- Dependency installation (uv, Rust, Python packages)
- Training execution with screen sessions
- Monitoring and downloading results
- Quick setup script
- Troubleshooting common issues

**Recommended for**: First-time users, GPU training

### [Local Inference Guide](local-inference.md)
Instructions for running your trained model locally on Mac, Linux, or Windows. Includes:
- Downloading trained checkpoints
- Local environment setup with CPU/MPS support
- Running web UI and CLI chat
- Performance expectations by hardware
- Memory requirements for different model sizes
- Troubleshooting local inference issues

**Recommended for**: Running your model after training, testing without GPU costs

## Quick Links

### Training Workflows
- **$100 tier (4 hours)**: See main [README.md](../README.md) Quick Start
- **$300 tier (12 hours)**: See main README "Bigger models" section
- **$1000 tier (42 hours)**: See [run1000.sh](../run1000.sh) script

### Community Resources
- [Discussions](https://github.com/karpathy/nanochat/discussions) - Community Q&A, tips, and showcases
- [Customization Guide](https://github.com/karpathy/nanochat/discussions/139) - Infusing personality into your model
- [Introduction Post](https://github.com/karpathy/nanochat/discussions/1) - Detailed walkthrough of the $100 speedrun

## Architecture Documentation

For technical details about the architecture, see [CLAUDE.md](../CLAUDE.md) which contains:
- Pipeline stages (tokenizer, pretraining, midtraining, SFT, RL)
- Model architecture details (Transformer with RoPE, QK norm, etc.)
- Optimizer strategy (Muon + AdamW)
- Data pipeline and distributed training
- Configuration system
- Performance expectations

## Contributing

See the main [README.md](../README.md) Contributing section for information about contributing to nanochat.

## Need Help?

1. Check the relevant guide above
2. Search [Discussions](https://github.com/karpathy/nanochat/discussions) for similar questions
3. Ask in Discussions if you can't find an answer
4. For bugs, open an [Issue](https://github.com/karpathy/nanochat/issues)
